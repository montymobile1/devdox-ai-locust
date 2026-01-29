import click
import sys
import asyncio
import aiofiles  # type: ignore[import-untyped]
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Union, List, Dict, Any, TextIO, TYPE_CHECKING
from rich.console import Console
from rich.table import Table
from together import AsyncTogether

from .config import AIEnhancementConfig, Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from devdox_ai_locust.utils.debug_recorder import DebugRecorder
from .schemas.processing_result import SwaggerProcessingRequest
from devdox_ai_locust.locust_generator import LocustTestGenerator
from devdox_ai_locust.schemas.cli_dto import (
    GenerateParams,
    RunParams,
    EndpointProcessingContext,
)
from devdox_ai_locust.utils.constants import (
    DEFAULT_HOST,
    WORKFLOWS_DIR_NAME,
    FAILURES_DIR_NAME,
    LOCUSTFILE_NAME,
    BASE_WORKFLOW_FILE,
    TEST_DATA_FILE,
    ORCHESTRATOR_FILE,
    INIT_FILE,
    SCENARIO_TYPES,
    AUTH_KEYWORDS,
    DEFAULT_SCHEMA_TIMEOUT,
    MAX_LLM_WORKERS_LIMIT,
)

if TYPE_CHECKING:
    from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
    from devdox_ai_locust.utils.generation_progress import GenerationProgress

console = Console()
logger = logging.getLogger(__name__)


class TeeOutput:
    """Tee stdout/stderr to both terminal and log file."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self.original = original
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.original.write(data)
        self.original.flush()
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def fileno(self) -> int:
        return self.original.fileno()

    def isatty(self) -> bool:
        return self.original.isatty()


def _setup_logging(output_dir: Path, command_type: str) -> Tuple[Path, TextIO]:
    """Setup tee logging to capture all terminal output.

    Args:
        output_dir: Directory to save the log file
        command_type: Type of command (e.g., 'generate', 'run')

    Returns:
        Tuple of (log_path, log_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = output_dir.name
    log_filename = f"{dir_name}_{command_type}_{timestamp}.log"
    log_path = output_dir / log_filename

    log_file = open(log_path, "w", encoding="utf-8")

    # Replace stdout and stderr with tee versions
    assert sys.__stdout__ is not None
    assert sys.__stderr__ is not None
    sys.stdout = TeeOutput(sys.__stdout__, log_file)
    sys.stderr = TeeOutput(sys.__stderr__, log_file)

    # Print the full command at the start of the log
    full_command = " ".join(sys.argv)
    print(f"$ {full_command}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print()

    return log_path, log_file


def _teardown_logging(log_file: TextIO, log_path: Path) -> None:
    """Restore original stdout/stderr and close log file."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
    print(f"\n📝 Log saved to: {log_path}")


def _initialize_config(together_api_key: Optional[str]) -> Tuple[Settings, str]:
    """Initialize configuration and validate API key"""
    config_obj = Settings()
    if together_api_key:
        api_key = together_api_key
    else:
        api_key = config_obj.API_KEY

    if not api_key:
        console.print(
            "[red]Error:[/red] Together AI API key is required. "
            "Set TOGETHER_API_KEY environment variable or use --together-api-key"
        )
        sys.exit(1)

    return config_obj, api_key


def _setup_output_directory(output: Union[str, Path]) -> Path:
    """Create and return output directory path"""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _handle_cli_error(e: Exception, verbose: bool) -> None:
    """Handle CLI-level exceptions with optional traceback."""
    console.print(f"[red]Error:[/red] {e}")
    if verbose:
        import traceback

        console.print(traceback.format_exc())
    sys.exit(1)


def _get_llm_mode_label(dto: GenerateParams) -> str:
    """Get the display label for LLM mode status."""
    if dto.llm_enabled:
        return "[green]enabled[/green]"
    if dto.replay_dir:
        return f"[cyan]replay[/cyan] ({dto.replay_dir})"
    return "[yellow]disabled[/yellow]"


def _display_configuration(dto: GenerateParams, output_dir: Path) -> None:
    from rich.panel import Panel

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Source", str(dto.swagger_url))
    table.add_row("Output", str(output_dir))
    table.add_row("Host", dto.host or "auto-detect from spec")
    table.add_row(
        "Auth", "[green]enabled[/green]" if dto.auth else "[dim]disabled[/dim]"
    )
    if dto.db_type:
        table.add_row("Database", dto.db_type)
    table.add_row("Locust Users", str(dto.users))
    table.add_row("Spawn Rate", f"{dto.spawn_rate}/s")
    table.add_row("Run Time", dto.run_time)
    table.add_row("LLM", _get_llm_mode_label(dto))
    if dto.llm_enabled:
        table.add_row("LLM Timeout", f"{dto.timeout}s")
    if dto.custom_requirement:
        req_display = (
            dto.custom_requirement[:80] + "..."
            if len(dto.custom_requirement) > 80
            else dto.custom_requirement
        )
        table.add_row("Custom Req", req_display)
    if dto.dry_run:
        table.add_row("Mode", "[yellow]DRY RUN[/yellow]")
    if dto.debug:
        table.add_row("Debug", "[blue]recording intermediate states[/blue]")

    console.print(Panel(table, title="[bold]Configuration[/bold]", border_style="dim"))


def _show_results(
    created_files: List[Dict[str, Any]],
    output_dir: Path,
    start_time: datetime,
    verbose: bool,
    dry_run: bool,
    users: int,
    spawn_rate: float,
    run_time: str,
    host: Optional[str],
) -> None:
    """Display generation results and run instructions"""
    end_time = datetime.now(timezone.utc)
    processing_time = (end_time - start_time).total_seconds()

    if not created_files:
        console.print("[red]✗[/red] No test files were generated")
        sys.exit(1)

    console.print(f"[green]✓[/green] Tests generated successfully in: {output_dir}")
    console.print(f"[blue]⏱️[/blue] Processing time: {processing_time:.2f} seconds")

    _show_generated_files(created_files, verbose)

    if not dry_run:
        _show_run_instructions(output_dir, users, spawn_rate, run_time, host)


def _show_generated_files(created_files: List[Dict[str, Any]], verbose: bool) -> None:
    """Display list of generated files"""
    if verbose or len(created_files) <= 10:
        console.print("\n[bold]Generated files:[/bold]")
        for file_path in created_files:
            console.print(f"  • {file_path}")
    else:
        console.print(f"\n[bold]Generated {len(created_files)} files[/bold]")
        console.print("Use --verbose to see all file names")


def _show_run_instructions(
    output_dir: Path, users: int, spawn_rate: float, run_time: str, host: Optional[str]
) -> None:
    """Display instructions for running the generated tests"""
    console.print("\n[bold]To run tests:[/bold]")
    console.print(f"  cd {output_dir}")

    default_host = host or DEFAULT_HOST
    locustfile = output_dir / LOCUSTFILE_NAME

    if locustfile.exists():
        main_file = LOCUSTFILE_NAME
    else:
        py_files = list(output_dir.glob("*.py"))
        main_file = py_files[0].name if py_files else "generated_test.py"

    console.print(
        f"  locust -f {main_file} --users {users} --spawn-rate {spawn_rate} "
        f"--run-time {run_time} --host {default_host}"
    )

    console.print("\n[bold]Alternative: Use the run command[/bold]")
    console.print(
        f"  devdox_ai_locust run {output_dir}/{main_file} --host {default_host}"
    )


async def _fetch_schema(swagger_url: str, schema_timeout: int) -> str:
    """Fetch API schema from URL or file with timeout handling."""
    from devdox_ai_locust.utils.constants import is_url

    if is_url(swagger_url):
        source_request = SwaggerProcessingRequest(swagger_url=swagger_url)
    else:
        source_request = SwaggerProcessingRequest(swagger_file_path=swagger_url)

    source_type = "URL" if is_url(swagger_url) else "file"
    console.print(f"→ Fetching API schema from {source_type}...")
    try:
        async with asyncio.timeout(schema_timeout):
            api_schema = await get_api_schema(source_request)

            if not api_schema:
                console.print("[red]✗[/red] Failed to fetch API schema")
                sys.exit(1)

    except asyncio.TimeoutError:
        console.print(
            f"[red]✗[/red] Timeout while fetching API schema (exceeded {schema_timeout}s)"
        )
        console.print(
            "[dim]Hint: Use --schema-timeout to increase the timeout for large schemas[/dim]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error fetching API schema: {e}")
        sys.exit(1)

    if not api_schema:
        console.print("[red]✗[/red] Failed to fetch API schema")
        sys.exit(1)

    schema_kb = len(api_schema) // 1024 if api_schema else 0
    console.print(f"[green]✓[/green] API schema fetched ({schema_kb}KB)")
    return api_schema


def _parse_schema(
    api_schema: str,
) -> Tuple[Dict[str, Any], List[Endpoint], Dict[str, Any]]:
    """Parse the raw API schema into structured data."""
    console.print("→ Parsing API schema...")
    parser = OpenAPIParser()
    try:
        schema_data = parser.parse_schema(api_schema)
        endpoints = parser.parse_endpoints()
        api_info = parser.get_schema_info()

        api_title = api_info.get("title", "API")
        console.print(
            f"[green]✓[/green] Loaded [bold]{api_title}[/bold] — {len(endpoints)} endpoints"
        )
        return schema_data, endpoints, api_info

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to parse API schema: {e}")
        sys.exit(1)


class _GenerationState:
    """Mutable state shared across parallel endpoint processing."""

    def __init__(self) -> None:
        self.created_files: List[Dict[str, Any]] = []
        self.failed_endpoints: List[Dict[str, Any]] = []
        self.completed_count: int = 0
        self.failed_count: int = 0
        self.successful_endpoints: set = set()
        self.file_write_lock: asyncio.Lock = asyncio.Lock()


async def _process_api_schema(
    swagger_url: str, schema_timeout: int = DEFAULT_SCHEMA_TIMEOUT
) -> Tuple[Dict[str, Any], List[Endpoint], Dict[str, Any]]:
    """Fetch and parse API schema"""
    api_schema = await _fetch_schema(swagger_url, schema_timeout)
    return _parse_schema(api_schema)


async def _generate_and_create_tests(
    dto: GenerateParams,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    output_dir: Path,
    debug_recorder: Optional[DebugRecorder] = None,
) -> List[Dict[str, Any]]:
    """Generate tests using scenario-based approach (positive/negative/security per tag)"""
    together_client = (
        AsyncTogether(api_key=dto.together_api_key) if dto.llm_enabled else None
    )

    # Create AI config with custom timeout
    ai_config = AIEnhancementConfig(timeout=dto.timeout)

    # Always use scenario-based generation for better results
    return await _generate_scenario_based_tests(
        dto=dto,
        ai_client=together_client,
        ai_config=ai_config,
        endpoints=endpoints,
        api_info=api_info,
        output_dir=output_dir,
        debug_recorder=debug_recorder,
    )


async def _generate_scenario_based_tests(
    dto: GenerateParams,
    ai_client: Optional[AsyncTogether],
    ai_config: AIEnhancementConfig,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    output_dir: Path,
    debug_recorder: Optional[DebugRecorder] = None,
) -> List[Dict[str, Any]]:
    """Generate tests using per-endpoint approach (5 scenarios per endpoint)"""
    ctx, base_files = _setup_generation_context(
        dto, ai_client, ai_config, endpoints, api_info, output_dir, debug_recorder
    )

    state = _GenerationState()

    progress = _init_progress(ctx.scenario_gen, len(endpoints), dto.verbose)

    await _process_all_endpoints(endpoints, state, ctx, progress)

    await _finalize_generation(endpoints, state, ctx, progress, output_dir, base_files)

    return state.created_files


def _setup_generation_context(
    dto: GenerateParams,
    ai_client: Optional[AsyncTogether],
    ai_config: AIEnhancementConfig,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    output_dir: Path,
    debug_recorder: Optional[DebugRecorder],
) -> Tuple[EndpointProcessingContext, Dict[str, str]]:
    """Build the EndpointProcessingContext for endpoint processing."""
    grouped_endpoints = _group_endpoints_by_tag(endpoints)
    num_tags = len(grouped_endpoints)

    scenario_gen = _init_scenario_generator(
        ai_client, ai_config, dto.max_llm_workers, debug_recorder
    )
    if dto.replay_dir:
        scenario_gen.replay_dir = Path(dto.replay_dir)
    _print_generation_plan(ai_config, scenario_gen, len(endpoints), num_tags)

    template_gen = LocustTestGenerator()
    base_files = _generate_base_files(
        template_gen,
        endpoints,
        api_info,
        dto.auth,
        dto.host,
        dto.db_type,
        debug_recorder,
    )
    base_workflow_content = base_files.get(BASE_WORKFLOW_FILE, "")
    test_data_content = base_files.get(TEST_DATA_FILE, "")

    auth_endpoints = _detect_auth_endpoints(endpoints) if dto.auth else None
    workflows_dir = _prepare_workflows_dir(output_dir)

    pre_llm_templates = _generate_pre_llm_templates(
        endpoints, scenario_gen, template_gen
    )

    endpoint_to_tag = _build_endpoint_tag_mapping(grouped_endpoints)

    endpoint_ctx = EndpointProcessingContext(
        scenario_gen=scenario_gen,
        template_gen=template_gen,
        workflows_dir=workflows_dir,
        base_workflow_content=base_workflow_content,
        test_data_content=test_data_content,
        auth_endpoints=auth_endpoints,
        all_endpoints=endpoints,
        custom_requirement=dto.custom_requirement,
        db_type=dto.db_type,
        pre_llm_templates=pre_llm_templates,
        endpoint_to_tag=endpoint_to_tag,
        no_llm=dto.no_llm,
    )

    return endpoint_ctx, base_files


async def _process_all_endpoints(
    endpoints: List[Endpoint],
    state: _GenerationState,
    ctx: EndpointProcessingContext,
    progress: "GenerationProgress",
) -> List[Dict[str, Any]]:
    """Process all endpoints in parallel."""
    with progress:
        tasks = [
            _process_and_save_endpoint(
                endpoint=ep,
                state=state,
                ctx=ctx,
                progress=progress,
            )
            for ep in endpoints
        ]
        await asyncio.gather(*tasks)
    return state.created_files


async def _finalize_generation(
    endpoints: List[Endpoint],
    state: _GenerationState,
    ctx: EndpointProcessingContext,
    progress: "GenerationProgress",
    output_dir: Path,
    base_files: Dict[str, str],
) -> None:
    """Generate orchestrators, init files, and base files after endpoint processing."""
    grouped_endpoints = _group_endpoints_by_tag(endpoints)
    num_tags = len(grouped_endpoints)

    if ctx.llm_enabled or ctx.replay_dir:
        orchestrator_files, orchestrator_failures = await _generate_orchestrators(
            grouped_endpoints=grouped_endpoints,
            successful_endpoints=state.successful_endpoints,
            scenario_gen=ctx.scenario_gen,
            workflows_dir=ctx.workflows_dir,
            base_workflow_content=ctx.base_workflow_content,
            test_data_content=ctx.test_data_content,
            auth_endpoints=ctx.auth_endpoints,
            custom_requirement=ctx.custom_requirement,
            db_type=ctx.db_type,
            progress=progress,
        )
        state.created_files.extend(orchestrator_files)
        _report_orchestrator_results(
            orchestrator_files, orchestrator_failures, num_tags
        )
    else:
        console.print("[dim]→ Skipping orchestrator generation (LLM disabled)[/dim]")

    _report_generation_summary(
        ctx.scenario_gen,
        state.failed_endpoints,
        state.completed_count,
        state.failed_count,
        len(endpoints),
    )

    _create_init_files(
        ctx.workflows_dir,
        grouped_endpoints,
        ctx.scenario_gen,
    )

    _write_base_files(base_files, output_dir, ctx.workflows_dir, state.created_files)


def _group_endpoints_by_tag(endpoints: List[Endpoint]) -> Dict[str, List[Endpoint]]:
    """Group endpoints by their first tag."""
    grouped: Dict[str, List[Endpoint]] = {}
    for ep in endpoints:
        tag = ep.tags[0] if ep.tags else "default"
        grouped.setdefault(tag, []).append(ep)
    return grouped


def _init_scenario_generator(
    ai_client: Optional[AsyncTogether],
    ai_config: AIEnhancementConfig,
    max_llm_workers: int,
    debug_recorder: Optional[DebugRecorder],
) -> "ScenarioWorkflowGenerator":
    """Initialize the ScenarioWorkflowGenerator."""
    from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

    prompt_dir = Path(__file__).parent / "prompt"
    return ScenarioWorkflowGenerator(
        prompt_dir=prompt_dir,
        ai_client=ai_client,  # type: ignore[arg-type]
        ai_config=ai_config,
        max_concurrency=max_llm_workers,
        debug_recorder=debug_recorder,
    )


def _print_generation_plan(
    ai_config: AIEnhancementConfig,
    scenario_gen: "ScenarioWorkflowGenerator",
    num_endpoints: int,
    num_tags: int,
) -> None:
    """Print the generation plan to console."""
    scenario_filenames = ", ".join(scenario_gen.SCENARIO_FILES.values())
    time_estimate = scenario_gen.estimate_time(num_endpoints)

    console.print("\n[bold]→ Generation Plan[/bold]")
    console.print(f"  Model: [cyan]{ai_config.model}[/cyan]")
    console.print(f"  Concurrency: {scenario_gen.current_concurrency} workers")
    console.print(f"  Endpoints: {num_endpoints} across {num_tags} tags")
    console.print(
        f"  Scenarios: {scenario_gen.num_scenarios} per endpoint ({scenario_filenames})"
    )
    console.print(f"  Total LLM calls: {time_estimate.total_calls}")
    console.print()


def _generate_base_files(
    template_gen: LocustTestGenerator,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    auth: bool,
    host: Optional[str],
    db_type: str,
    debug_recorder: Optional[DebugRecorder],
) -> Dict[str, str]:
    """Generate base template files and record debug info."""
    base_files, _, _ = template_gen.generate_from_endpoints(
        endpoints, api_info, include_auth=auth, target_host=host, db_type=db_type
    )
    base_files = template_gen.fix_indent(base_files)

    if debug_recorder and debug_recorder.enabled:
        for filename, content in base_files.items():
            debug_recorder.record_static_file(
                file_name=filename,
                context={
                    "endpoint_count": len(endpoints),
                    "api_info": api_info,
                    "include_auth": auth,
                    "target_host": host,
                    "db_type": db_type,
                },
                rendered_content=content,
            )
    return base_files


def _detect_auth_endpoints(endpoints: List[Endpoint]) -> List[Endpoint]:
    """Detect auth-related endpoints by keyword matching."""
    # TODO: Auth endpoint detection is overly broad - matches any path containing
    # "auth", "login", "token", or "session" as substrings. Consider using
    # OpenAPI security schemes instead.
    return [
        ep for ep in endpoints if any(kw in ep.path.lower() for kw in AUTH_KEYWORDS)
    ]


def _prepare_workflows_dir(output_dir: Path) -> Path:
    """Clean and create the workflows output directory."""
    import shutil

    workflows_dir = output_dir / WORKFLOWS_DIR_NAME
    if workflows_dir.exists():
        shutil.rmtree(workflows_dir)
        logger.info("Cleaned previous workflow files")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    return workflows_dir


def _sanitize_dir_name(name: str) -> str:
    """Sanitize a string for use as a directory name."""
    import re

    name = (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    # Note: Using [^a-z0-9_] instead of \W to ensure ASCII-only identifiers
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def _to_class_name(name: str) -> str:
    """Convert a name to PascalCase class name."""
    sanitized = _sanitize_dir_name(name)
    words = sanitized.replace("_", " ").split()
    return "".join(word.capitalize() for word in words) or "Unnamed"


def _generate_pre_llm_workflow(
    endpoint: Endpoint,
    scenario_type: str,
    scenario_gen: "ScenarioWorkflowGenerator",
    template_gen: LocustTestGenerator,
) -> str:
    """Generate a pre-LLM fallback workflow for an endpoint and scenario type."""
    operation_id = scenario_gen.get_endpoint_dir_name(endpoint)
    class_name = _to_class_name(operation_id)
    method = endpoint.method.lower()
    path = endpoint.path

    task_method = template_gen._generate_task_method(endpoint)
    indented_task = "\n".join(
        f"    {line}" if line.strip() else line for line in task_method.split("\n")
    )

    return f'''"""
Pre-LLM workflow for {method.upper()} {path}
Generated using template generator.
"""
from locust import task
from workflows.base_workflow import BaseWorkflow

import logging

logger = logging.getLogger(__name__)


class {class_name}{scenario_type.capitalize()}Workflow(BaseWorkflow):
    """{scenario_type.capitalize()} tests for {method.upper()} {path}"""

{indented_task}
'''


def _generate_pre_llm_templates(
    endpoints: List[Endpoint],
    scenario_gen: "ScenarioWorkflowGenerator",
    template_gen: LocustTestGenerator,
) -> Dict[Tuple[int, str], str]:
    """Generate all pre-LLM fallback templates. Exits on failure."""
    console.print("→ Generating base templates...")
    templates: Dict[Tuple[int, str], str] = {}
    try:
        for endpoint in endpoints:
            for st in SCENARIO_TYPES:
                templates[(id(endpoint), st)] = _generate_pre_llm_workflow(
                    endpoint, st, scenario_gen, template_gen
                )
    except Exception as e:
        console.print(
            "[bold red]CRITICAL ERROR: Failed to generate base templates[/bold red]"
        )
        console.print("[red]This indicates a bug or corrupted installation.[/red]")
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Pre-LLM template generation failed: {e}", exc_info=True)
        sys.exit(1)
    console.print("[green]✓[/green] Base templates generated")
    return templates


def _build_endpoint_tag_mapping(
    grouped_endpoints: Dict[str, List[Endpoint]],
) -> Dict[int, str]:
    """Build a mapping from endpoint id to tag name."""
    mapping: Dict[int, str] = {}
    for tag_name, tag_endpoints in grouped_endpoints.items():
        for ep in tag_endpoints:
            mapping[id(ep)] = tag_name
    return mapping


def _init_progress(
    scenario_gen: "ScenarioWorkflowGenerator", num_endpoints: int, verbose: bool
) -> "GenerationProgress":
    """Initialize and attach the GenerationProgress display."""
    from devdox_ai_locust.utils.generation_progress import GenerationProgress

    progress = GenerationProgress(
        total=num_endpoints,
        num_workers=scenario_gen.current_concurrency,
        console=console,
        verbose=verbose,
    )
    scenario_gen.progress = progress
    return progress


async def _process_and_save_endpoint(
    endpoint: Endpoint,
    state: _GenerationState,
    ctx: EndpointProcessingContext,
    progress: "GenerationProgress",
) -> List[Dict[str, Any]]:
    """Process a single endpoint: generate scenarios and save files."""
    tag_name = ctx.endpoint_to_tag.get(id(endpoint), "default")
    tag_dir_name = _sanitize_dir_name(tag_name)
    operation_id = ctx.scenario_gen.get_endpoint_dir_name(endpoint)
    endpoint_info = f"{endpoint.method} {endpoint.path}"
    endpoint_dir = ctx.workflows_dir / tag_dir_name / operation_id

    progress.endpoint_start(endpoint_info)

    try:
        return await _save_generated_scenarios(
            endpoint=endpoint,
            state=state,
            ctx=ctx,
            endpoint_dir=endpoint_dir,
            progress=progress,
        )
    except Exception as e:
        return await _handle_endpoint_failure(
            e=e,
            state=state,
            endpoint=endpoint,
            endpoint_dir=endpoint_dir,
            ctx=ctx,
            progress=progress,
        )


async def _save_generated_scenarios(
    endpoint: Endpoint,
    state: _GenerationState,
    ctx: EndpointProcessingContext,
    endpoint_dir: Path,
    progress: "GenerationProgress",
) -> List[Dict[str, Any]]:
    """Generate and save scenario workflows for a single endpoint."""
    tag_name = ctx.endpoint_to_tag.get(id(endpoint), "default")
    operation_id = ctx.scenario_gen.get_endpoint_dir_name(endpoint)
    endpoint_info = f"{endpoint.method} {endpoint.path}"

    endpoint_dir.mkdir(parents=True, exist_ok=True)

    if not ctx.llm_enabled and not ctx.replay_dir:
        return await _write_fallback_files(
            endpoint=endpoint,
            endpoint_dir=endpoint_dir,
            tag_name=tag_name,
            operation_id=operation_id,
            pre_llm_templates=ctx.pre_llm_templates,
            scenario_gen=ctx.scenario_gen,
            template_gen=ctx.template_gen,
        )

    scenarios = await ctx.scenario_gen.generate_endpoint_workflows(
        endpoint=endpoint,
        base_workflow_content=ctx.base_workflow_content,
        test_data_content=ctx.test_data_content,
        auth_endpoints=ctx.auth_endpoints,
        tag_name=tag_name,
        all_endpoints=ctx.all_endpoints,
        custom_requirement=ctx.custom_requirement,
        db_type=ctx.db_type,
    )

    local_files = []
    for scenario_type, content in scenarios.items():
        if content:
            filename = ctx.scenario_gen.SCENARIO_FILES[scenario_type]
            file_path = endpoint_dir / filename
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)
            local_files.append(
                {
                    "path": str(file_path),
                    "size": len(content),
                    "tag": tag_name,
                    "operation_id": operation_id,
                    "scenario": scenario_type.value,
                }
            )

    async with state.file_write_lock:
        state.completed_count += 1
        state.successful_endpoints.add(id(endpoint))
        state.created_files.extend(local_files)

    progress.endpoint_done(endpoint_info)
    return local_files


async def _handle_endpoint_failure(
    e: Exception,
    state: _GenerationState,
    endpoint: Endpoint,
    endpoint_dir: Path,
    ctx: EndpointProcessingContext,
    progress: "GenerationProgress",
) -> List[Dict[str, Any]]:
    """Handle a failed endpoint by saving failure info and writing fallback files."""
    operation_id = ctx.scenario_gen.get_endpoint_dir_name(endpoint)
    endpoint_info = f"{endpoint.method} {endpoint.path}"
    tag_name = ctx.endpoint_to_tag.get(id(endpoint), "default")

    saved_failure_path = await _save_failure_code(
        e, ctx.workflows_dir, operation_id, endpoint_info
    )

    fallback_files = await _write_fallback_files(
        endpoint=endpoint,
        endpoint_dir=endpoint_dir,
        tag_name=tag_name,
        operation_id=operation_id,
        pre_llm_templates=ctx.pre_llm_templates,
        scenario_gen=ctx.scenario_gen,
        template_gen=ctx.template_gen,
    )

    async with state.file_write_lock:
        state.failed_count += 1
        state.failed_endpoints.append(
            {
                "endpoint": endpoint_info,
                "operation_id": operation_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "saved_code": saved_failure_path,
            }
        )
        state.created_files.extend(fallback_files)

    progress.endpoint_failed(endpoint_info, e)
    return fallback_files


async def _save_failure_code(
    e: Exception,
    workflows_dir: Path,
    operation_id: str,
    endpoint_info: str,
) -> Optional[str]:
    """Save failed LLM-generated code to the failures directory."""
    if not (hasattr(e, "code") and e.code):
        return None

    failures_dir = workflows_dir / FAILURES_DIR_NAME
    failures_dir.mkdir(parents=True, exist_ok=True)
    failure_filename = f"{operation_id}_{getattr(e, 'scenario_type', 'unknown')}.py"
    failure_path = failures_dir / failure_filename
    try:
        async with aiofiles.open(failure_path, "w", encoding="utf-8") as f:
            error_header = (
                f"# FAILED CODE - {endpoint_info}\n"
                f"# Error: {getattr(e, 'error', str(e))}\n"
                f"# Scenario: {getattr(e, 'scenario_type', 'unknown')}\n"
                f"# {'─' * 57}\n\n"
            )
            await f.write(error_header + e.code)
        logger.debug(f"Saved failed code to {failure_path}")
        return str(failure_path)
    except Exception as save_error:
        logger.debug(f"Could not save failed code: {save_error}")
        return None


async def _write_fallback_files(
    endpoint: Endpoint,
    endpoint_dir: Path,
    tag_name: str,
    operation_id: str,
    pre_llm_templates: Dict[Tuple[int, str], str],
    scenario_gen: "ScenarioWorkflowGenerator",
    template_gen: LocustTestGenerator,
) -> List[Dict[str, Any]]:
    """Write pre-LLM fallback templates when LLM generation fails."""
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    fallback_files = []

    for scenario_type in SCENARIO_TYPES:
        fallback_content = pre_llm_templates.get((id(endpoint), scenario_type), "")
        if not fallback_content:
            fallback_content = _generate_pre_llm_workflow(
                endpoint, scenario_type, scenario_gen, template_gen
            )
        filename = f"{scenario_type}_workflow.py"
        file_path = endpoint_dir / filename
        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(fallback_content)
            fallback_files.append(
                {
                    "path": str(file_path),
                    "size": len(fallback_content),
                    "tag": tag_name,
                    "operation_id": operation_id,
                    "scenario": scenario_type,
                    "fallback": True,
                }
            )
        except Exception as write_error:
            logger.error(f"Failed to write fallback file {file_path}: {write_error}")

    return fallback_files


async def _generate_orchestrators(
    grouped_endpoints: Dict[str, List[Endpoint]],
    successful_endpoints: set,
    scenario_gen: "ScenarioWorkflowGenerator",
    workflows_dir: Path,
    base_workflow_content: str,
    test_data_content: str,
    auth_endpoints: Optional[List[Endpoint]],
    custom_requirement: Optional[str],
    db_type: str,
    progress: "GenerationProgress",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate orchestrator workflows for each tag."""
    num_tags = len(grouped_endpoints)
    console.print(f"\n→ Generating orchestrators ({num_tags} tags)...")

    orchestrator_files: List[Dict[str, Any]] = []
    orchestrator_failures: List[Dict[str, Any]] = []

    for tag_name, tag_endpoints in grouped_endpoints.items():
        result = await _generate_single_orchestrator(
            tag_name=tag_name,
            tag_endpoints=tag_endpoints,
            successful_endpoints=successful_endpoints,
            scenario_gen=scenario_gen,
            workflows_dir=workflows_dir,
            base_workflow_content=base_workflow_content,
            test_data_content=test_data_content,
            auth_endpoints=auth_endpoints,
            custom_requirement=custom_requirement,
            db_type=db_type,
            progress=progress,
        )
        if result is not None:
            if "error" in result:
                orchestrator_failures.append(result)
            else:
                orchestrator_files.append(result)

    return orchestrator_files, orchestrator_failures


async def _generate_single_orchestrator(
    tag_name: str,
    tag_endpoints: List[Endpoint],
    successful_endpoints: set,
    scenario_gen: "ScenarioWorkflowGenerator",
    workflows_dir: Path,
    base_workflow_content: str,
    test_data_content: str,
    auth_endpoints: Optional[List[Endpoint]],
    custom_requirement: Optional[str],
    db_type: str,
    progress: "GenerationProgress",
) -> Optional[Dict[str, Any]]:
    """Generate a single orchestrator for one tag. Returns file info, failure info, or None."""
    tag_dir_name = _sanitize_dir_name(tag_name)
    valid_endpoints = [ep for ep in tag_endpoints if id(ep) in successful_endpoints]

    if not valid_endpoints:
        progress.orchestrator_skipped(tag_dir_name, "no valid endpoints")
        return None

    try:
        orchestrator_code = await scenario_gen.generate_tag_orchestrator(
            tag_name=tag_name,
            tag_endpoints=valid_endpoints,
            base_workflow_content=base_workflow_content,
            test_data_content=test_data_content,
            auth_endpoints=auth_endpoints,
            custom_requirement=custom_requirement,
            db_type=db_type,
        )

        tag_dir = workflows_dir / tag_dir_name
        tag_dir.mkdir(parents=True, exist_ok=True)
        orchestrator_path = tag_dir / ORCHESTRATOR_FILE
        async with aiofiles.open(orchestrator_path, "w", encoding="utf-8") as f:
            await f.write(orchestrator_code)
        progress.orchestrator_done(tag_dir_name)
        return {
            "path": str(orchestrator_path),
            "size": len(orchestrator_code),
            "tag": tag_name,
        }

    except Exception as e:
        progress.orchestrator_failed(tag_dir_name, e)
        return {"tag": tag_name, "error": str(e)}


def _report_orchestrator_results(
    orchestrator_files: List[Dict[str, Any]],
    orchestrator_failures: List[Dict[str, Any]],
    num_tags: int,
) -> None:
    """Print orchestrator generation results."""
    if orchestrator_failures:
        console.print(
            f"[yellow]⚠[/yellow] Orchestrators: {len(orchestrator_files)}/{num_tags} succeeded"
        )
    else:
        console.print(f"[green]✓[/green] All {num_tags} orchestrators generated")


def _report_generation_summary(
    scenario_gen: "ScenarioWorkflowGenerator",
    failed_endpoints: List[Dict[str, Any]],
    completed_count: int,
    failed_count: int,
    num_endpoints: int,
) -> None:
    """Print final generation summary with rate limit info and failure details."""
    rate_info = scenario_gen.get_rate_limit_info()
    console.print(
        f"\n[dim]Final rate limit: {rate_info.requests_per_minute} RPM, "
        f"Concurrency used: {scenario_gen.current_concurrency}[/dim]"
    )

    if failed_endpoints:
        console.print(
            f"\n[bold yellow]⚠ Generation completed with {failed_count} failures[/bold yellow]"
        )
        console.print(
            f"   [green]✓ Succeeded:[/green] {completed_count}/{num_endpoints}"
        )
        console.print(f"   [red]✗ Failed:[/red] {failed_count}/{num_endpoints}")
        _print_failure_details(failed_endpoints)
    else:
        console.print(
            f"\n[bold green]✓ All {num_endpoints} endpoints generated successfully[/bold green]"
        )


def _print_failure_details(failed_endpoints: List[Dict[str, Any]]) -> None:
    """Print details of failed endpoints."""
    console.print("\n[bold red]Failed Endpoints:[/bold red]")
    for failure in failed_endpoints[:10]:
        console.print(f"   • {failure.get('endpoint', 'unknown')}")
        error_msg = failure.get("error", "Unknown error") or "Unknown error"
        error_type = failure.get("error_type", "Error")
        console.print(f"     [dim]{error_type}: {error_msg[:200]}[/dim]")
    if len(failed_endpoints) > 10:
        console.print(f"   ... and {len(failed_endpoints) - 10} more failures")


def _create_init_files(
    workflows_dir: Path,
    grouped_endpoints: Dict[str, List[Endpoint]],
    scenario_gen: "ScenarioWorkflowGenerator",
) -> None:
    """Generate __init__.py files for workflow directories."""
    console.print("→ Creating workflow __init__.py files...")

    # Main workflows/__init__.py
    tag_imports = [
        f"from .{_sanitize_dir_name(tag)} import *" for tag in grouped_endpoints.keys()
    ]
    (workflows_dir / INIT_FILE).write_text(
        "\n".join(tag_imports) + "\n", encoding="utf-8"
    )

    # Per-tag __init__.py
    for tag_name, tag_endpoints in grouped_endpoints.items():
        tag_dir_name = _sanitize_dir_name(tag_name)
        tag_dir = workflows_dir / tag_dir_name
        if not tag_dir.exists():
            continue

        init_lines = _build_tag_init_lines(
            tag_dir, tag_dir_name, tag_endpoints, scenario_gen
        )
        (tag_dir / INIT_FILE).write_text("\n".join(init_lines) + "\n", encoding="utf-8")

    console.print("[green]✓[/green] Workflow __init__.py files created")


def _build_tag_init_lines(
    tag_dir: Path,
    tag_dir_name: str,
    tag_endpoints: List[Endpoint],
    scenario_gen: "ScenarioWorkflowGenerator",
) -> List[str]:
    """Build import lines for a tag's __init__.py."""
    init_lines = ['"""Auto-generated workflow exports"""']

    for ep in tag_endpoints:
        op_id = scenario_gen.get_endpoint_dir_name(ep)
        class_name = _to_class_name(op_id)
        endpoint_dir = tag_dir / op_id
        for scenario in SCENARIO_TYPES:
            if (endpoint_dir / f"{scenario}_workflow.py").exists():
                init_lines.append(
                    f"from .{op_id}.{scenario}_workflow "
                    f"import {class_name}{scenario.capitalize()}Workflow"
                )

    orchestrator_path = tag_dir / ORCHESTRATOR_FILE
    if orchestrator_path.exists():
        orchestrator_class = _to_class_name(tag_dir_name) + "Orchestrator"
        init_lines.append(f"from .orchestrator_workflow import {orchestrator_class}")

    return init_lines


def _write_base_files(
    base_files: Dict[str, str],
    output_dir: Path,
    workflows_dir: Path,
    created_files: List[Dict[str, Any]],
) -> None:
    """Write base template files to the output directory."""
    console.print("→ Creating base files...")
    for filename, content in base_files.items():
        if filename == BASE_WORKFLOW_FILE:
            file_path = workflows_dir / filename
        else:
            file_path = output_dir / filename
        file_path.write_text(content, encoding="utf-8")
        created_files.append({"path": str(file_path), "size": len(content)})
    console.print(f"[green]✓[/green] Base files created ({len(base_files)} files)")


@click.group()
@click.version_option(version="0.1.9")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress and extra information while running.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """DevDox AI LoadTest - Automatically generate and run load tests from your API documentation."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        console.print("[green]Verbose mode enabled[/green]")


@cli.command()
@click.argument("swagger_url")  # Can be URL or file path
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output",
    help=(
        "Where to save the generated test files. "
        "A folder will be created at this path if it does not exist. "
        "Defaults to 'output' in the current directory."
    ),
)
@click.option(
    "--users",
    "-u",
    type=int,
    default=10,
    help=(
        "How many virtual users to simulate during the load test. "
        "Each user runs the generated test scenarios independently. "
        "Defaults to 10."
    ),
)
@click.option(
    "--spawn-rate",
    "-r",
    type=float,
    default=2,
    help=(
        "How quickly new virtual users are added per second when the test starts. "
        "For example, a rate of 2 means 2 new users join every second "
        "until the total number of users is reached. Defaults to 2."
    ),
)
@click.option(
    "--run-time",
    "-t",
    type=str,
    default="5m",
    help=(
        "How long the load test should run. "
        "Use a number followed by 's' for seconds, 'm' for minutes, or 'h' for hours. "
        "Examples: '30s', '5m', '1h'. Defaults to '5m'."
    ),
)
@click.option(
    "--host",
    "-H",
    type=str,
    help=(
        "The base URL of the API server to test against "
        "(e.g., 'http://localhost:8000'). "
        "If not provided, the tool will try to detect it from the API specification."
    ),
)
@click.option(
    "--auth/--no-auth",
    default=True,
    help=(
        "Whether to include authentication handling in the generated tests. "
        "When enabled, the tool detects login/token endpoints in your API "
        "and adds authentication flows to the tests. "
        "Use --no-auth if your API does not require authentication."
    ),
)
@click.option(
    "--db-type",
    type=click.Choice(["", "mongo", "postgresql"], case_sensitive=False),
    default="",
    help=(
        "If your API uses a database, specify the type here so the generated tests "
        "can include appropriate setup and teardown logic. "
        "Leave empty if your API does not use a database or you do not need "
        "database-aware tests."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=(
        "Generate the test files but do not show run instructions afterwards. "
        "Useful when you only want to inspect the generated code without running it."
    ),
)
@click.option(
    "--custom-requirement",
    type=str,
    help=(
        "A plain-text instruction that guides how the AI generates tests. "
        "For example: 'Focus on pagination edge cases' or "
        "'All POST requests should include an authorization header'. "
        "The AI will incorporate this into every generated test scenario."
    ),
)
@click.option(
    "--together-api-key",
    type=str,
    envvar="TOGETHER_API_KEY",
    help=(
        "Your Together AI API key, used to power the AI-based test generation. "
        "You can pass it here or set the TOGETHER_API_KEY environment variable. "
        "Required unless --no-llm is used."
    ),
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help=(
        "Maximum number of seconds to wait for each AI response. "
        "Increase this if you have a large API with many endpoints, "
        "as the AI may need more time to generate complex test scenarios. "
        "Defaults to 120 seconds."
    ),
)
@click.option(
    "--schema-timeout",
    type=int,
    default=30,
    help=(
        "Maximum number of seconds to wait when fetching your API specification. "
        "Increase this if your spec is hosted on a slow server "
        "or is a very large file. Defaults to 30 seconds."
    ),
)
@click.option(
    "--max-llm-workers",
    type=int,
    default=1,
    help=(
        "How many AI requests to run at the same time. "
        "Higher values speed up generation but use more API credits. "
        "Maximum allowed is 10. Defaults to 1."
    ),
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help=(
        "Save every intermediate step of the generation process to disk. "
        "This includes the raw AI prompts, responses, extracted code, "
        "and validation results. Useful for troubleshooting or auditing "
        "how the tests were generated."
    ),
)
@click.option(
    "--no-llm",
    default=None,
    required=False,
    is_flag=False,
    flag_value="",
    help=(
        "Skip all AI calls. When used without a value (--no-llm), "
        "generates basic template-only tests from the API schema — "
        "no API key is needed. When used with a directory path "
        "(--no-llm ./fixtures), replays pre-recorded AI responses "
        "from that folder, running the full generation pipeline "
        "without making any real AI requests."
    ),
)
@click.pass_context
def generate(ctx: click.Context, **kwargs: Any) -> None:
    """Generate load test files from an API specification.

    SWAGGER_URL is the URL or local file path to your OpenAPI/Swagger specification
    (e.g., 'http://localhost:8000/openapi.json' or './spec.yaml').

    The tool reads your API specification, uses AI to create realistic test
    scenarios (positive, negative, and security), and outputs ready-to-run
    Locust test files.
    """
    ctx.ensure_object(dict)
    params = {**kwargs, "verbose": ctx.obj.get("verbose", False)}
    dto = GenerateParams(**params)

    # Validate max_llm_workers
    if dto.max_llm_workers > MAX_LLM_WORKERS_LIMIT:
        raise click.BadParameter(
            f"--max-llm-workers cannot exceed {MAX_LLM_WORKERS_LIMIT} (got {dto.max_llm_workers})",
            param_hint="'--max-llm-workers'",
        )

    output_dir = _setup_output_directory(dto.output)
    try:
        asyncio.run(run_generate(dto, output_dir))
    except Exception as e:
        _handle_cli_error(e, dto.verbose)


async def run_generate(dto: GenerateParams, output_dir: Path) -> None:
    """Async entry point for the generate command."""
    start_time = datetime.now(timezone.utc)
    debug_recorder = _init_generation(dto, output_dir)

    try:
        raw_schema, endpoints, api_info = await _process_api_schema(
            dto.swagger_url, dto.schema_timeout
        )

        _record_debug_parsed_schema(
            dto.debug,
            debug_recorder,
            raw_schema,
            endpoints,
            api_info,
            dto.host,
            dto.auth,
            dto.db_type,
            dto.timeout,
            dto.custom_requirement,
        )

        # Resolve API key (skip when LLM is disabled)
        if dto.llm_enabled:
            _, api_key = _initialize_config(dto.together_api_key)
            resolved_dto = dto.model_copy(update={"together_api_key": api_key})
        else:
            resolved_dto = dto

        created_files = await _generate_and_create_tests(
            resolved_dto,
            endpoints,
            api_info,
            output_dir,
            debug_recorder,
        )

        if dto.debug:
            await debug_recorder.finalize()
            console.print(
                f"[blue]🔍 Debug info saved to:[/blue] {debug_recorder.debug_root}"
            )

        _show_results(
            created_files,
            output_dir,
            start_time,
            dto.verbose,
            dto.dry_run,
            dto.users,
            dto.spawn_rate,
            dto.run_time,
            dto.host,
        )

    except Exception as e:
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        console.print(
            f"[red]✗[/red] Generation failed after {processing_time:.2f}s: {e}"
        )
        raise


def _init_generation(dto: GenerateParams, output_dir: Path) -> DebugRecorder:
    """Initialize config display and debug recorder for generation."""
    debug_recorder = DebugRecorder(output_dir, enabled=dto.debug)

    _record_debug_cli_args(dto, debug_recorder)
    _display_configuration(dto, output_dir)

    return debug_recorder


def _record_debug_cli_args(
    dto: GenerateParams,
    debug_recorder: DebugRecorder,
) -> None:
    """Record CLI arguments to debug recorder if debug mode is enabled."""
    if not dto.debug:
        return
    console.print("[blue]🔍 Debug mode enabled[/blue] - recording intermediate states")
    debug_recorder.record_cli_args(dto.model_dump())


def _record_debug_parsed_schema(
    debug: bool,
    debug_recorder: DebugRecorder,
    raw_schema: dict,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    host: Optional[str],
    auth: bool,
    db_type: str,
    timeout: int,
    custom_requirement: Optional[str],
) -> None:
    """Record parsed OpenAPI data to debug recorder if debug mode is enabled."""
    if not debug:
        return
    debug_recorder.record_openapi_raw(raw_schema)
    debug_recorder.record_openapi_parsed(endpoints, api_info)
    debug_recorder.record_resolved_config(
        {
            "api_info": api_info,
            "host": host,
            "auth": auth,
            "db_type": db_type,
            "timeout": timeout,
            "custom_requirement": custom_requirement,
        }
    )


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option(
    "--users",
    "-u",
    type=int,
    default=10,
    help=(
        "How many virtual users to simulate during the load test. "
        "Each user runs the test scenarios independently. Defaults to 10."
    ),
)
@click.option(
    "--spawn-rate",
    "-r",
    type=float,
    default=2,
    help="How quickly new virtual users are added per second. Defaults to 2.",
)
@click.option(
    "--run-time",
    "-t",
    type=str,
    default="5m",
    help=(
        "How long the load test should run. "
        "Examples: '30s', '5m', '1h'. Defaults to '5m'."
    ),
)
@click.option(
    "--host",
    "-H",
    type=str,
    required=True,
    help=(
        "The base URL of the API server to test against "
        "(e.g., 'http://localhost:8000'). Required."
    ),
)
@click.option(
    "--headless",
    is_flag=True,
    help=(
        "Run the load test without the Locust web dashboard. "
        "Results are printed to the terminal instead. "
        "Useful for CI/CD pipelines or automated testing."
    ),
)
@click.pass_context
def run(ctx: click.Context, **kwargs: Any) -> None:
    """Run previously generated Locust load tests against your API.

    TEST_FILE is the path to the generated locustfile.py
    (e.g., 'output/locustfile.py').

    This command starts Locust with the specified test file and
    automatically captures logs for later analysis.
    """
    ctx.ensure_object(dict)
    params = {**kwargs, "verbose": ctx.obj.get("verbose", False)}
    dto = RunParams(**params)
    run_locust(dto)


def run_locust(dto: RunParams) -> None:
    """Execute locust tests with logging."""

    test_file_path = Path(dto.test_file)
    test_suite_dir = test_file_path.parent

    # Setup logging
    log_path, log_file = _setup_logging(test_suite_dir, "locust_run")

    try:
        cmd = _build_locust_command(dto)

        if dto.verbose:
            console.print(f"[blue]Running command:[/blue] {' '.join(cmd)}")

        console.print("[green]Starting Locust test...[/green]")

        _execute_locust_process(cmd)

    except FileNotFoundError:
        console.print(
            "[red]Locust not found. Please install locust: pip install locust[/red]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error running Locust:[/red] {e}")
        sys.exit(1)
    finally:
        _teardown_logging(log_file, log_path)


def _build_locust_command(dto: RunParams) -> List[str]:
    """Build the locust CLI command from RunParams."""
    cmd = [
        "locust",
        "-f",
        str(dto.test_file),
        "--users",
        str(dto.users),
        "--spawn-rate",
        str(dto.spawn_rate),
        "--run-time",
        dto.run_time,
        "--host",
        dto.host,
    ]

    if dto.headless:
        cmd.append("--headless")

    return cmd


def _execute_locust_process(cmd: List[str]) -> None:
    """Run the locust subprocess with real-time output."""
    import subprocess

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        process.terminate()
        process.wait()

    return_code = process.wait()

    if return_code != 0:
        console.print(f"[red]Test execution failed with exit code {return_code}[/red]")
        sys.exit(return_code)


def main() -> None:
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
