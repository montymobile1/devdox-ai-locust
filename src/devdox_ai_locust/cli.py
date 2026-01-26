import click
import sys
import asyncio
import aiofiles
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Union, List, Dict, Any, TextIO
from rich.console import Console
from rich.table import Table
from together import AsyncTogether

from .ai_config import AIEnhancementConfig
from .config import Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from devdox_ai_locust.utils.debug_recorder import DebugRecorder
from .schemas.processing_result import SwaggerProcessingRequest

console = Console()


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


def _display_configuration(
    swagger_url: str,
    output_dir: Path,
    users: int,
    spawn_rate: float,
    run_time: str,
    host: Optional[str],
    auth: bool,
    custom_requirement: Optional[str],
    dry_run: bool,
) -> None:
    table = Table(title="Generation Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Source", str(swagger_url))
    table.add_row("Output Directory", str(output_dir))

    table.add_row("Users", str(users))
    table.add_row("Spawn Rate", str(spawn_rate))
    table.add_row("Run Time", run_time)
    table.add_row("Host", host or "Auto-detect")
    table.add_row("Authentication", "Enabled" if auth else "Disabled")
    table.add_row("Custom Requirement", custom_requirement or "None")
    table.add_row("Dry Run", "Yes" if dry_run else "No")

    console.print(table)


def _show_results(
    created_files: List[Dict[Any, Any]],
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


def _show_generated_files(created_files: List[Dict[Any, Any]], verbose: bool) -> None:
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

    default_host = host or "http://localhost:8000"
    locustfile = output_dir / "locustfile.py"

    if locustfile.exists():
        main_file = "locustfile.py"
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


async def _process_api_schema(
    swagger_url: str, verbose: bool
) -> Tuple[Dict[str, Any], List[Endpoint], Dict[str, Any]]:
    """Fetch and parse API schema"""
    source_request = SwaggerProcessingRequest(swagger_url=swagger_url)
    api_schema = None

    # Fetch API schema
    source_type = 'URL' if swagger_url.startswith(('http://', 'https://')) else 'file'
    console.print(f"→ Fetching API schema from {source_type}...")
    try:
        async with asyncio.timeout(30):
            api_schema = await get_api_schema(source_request)

            if not api_schema:
                console.print("[red]✗[/red] Failed to fetch API schema")
                sys.exit(1)

    except asyncio.TimeoutError:
        console.print("[red]✗[/red] Timeout while fetching API schema")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error fetching API schema: {e}")
        sys.exit(1)

    if not api_schema:
        console.print("[red]✗[/red] Failed to fetch API schema")
        sys.exit(1)

    schema_kb = len(api_schema) // 1024 if api_schema else 0
    console.print(f"[green]✓[/green] API schema fetched ({schema_kb}KB)")

    # Parse schema
    console.print("→ Parsing API schema...")
    parser = OpenAPIParser()
    try:
        schema_data = parser.parse_schema(api_schema)
        endpoints = parser.parse_endpoints()
        api_info = parser.get_schema_info()

        api_title = api_info.get('title', 'API')
        console.print(
            f"[green]✓[/green] Loaded [bold]{api_title}[/bold] — {len(endpoints)} endpoints"
        )
        return schema_data, endpoints, api_info

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to parse API schema: {e}")
        sys.exit(1)


async def _generate_and_create_tests(
    api_key: str,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    output_dir: Path,
    custom_requirement: Optional[str] = "",
    host: Optional[str] = "0.0.0.0",
    auth: bool = False,
    db_type: str = "",
    timeout: int = 120,
    debug_recorder: Optional[DebugRecorder] = None,
) -> List[Dict[Any, Any]]:
    """Generate tests using scenario-based approach (positive/negative/security per tag)"""
    together_client = AsyncTogether(api_key=api_key)

    # Create AI config with custom timeout
    ai_config = AIEnhancementConfig(timeout=timeout)

    # Always use scenario-based generation for better results
    return await _generate_scenario_based_tests(
        together_client,
        ai_config,
        endpoints,
        api_info,
        output_dir,
        auth,
        db_type,
        host,
        custom_requirement,
        debug_recorder,
    )


async def _generate_scenario_based_tests(
    ai_client: AsyncTogether,
    ai_config: AIEnhancementConfig,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    output_dir: Path,
    auth: bool,
    db_type: str,
    host: Optional[str] = None,
    custom_requirement: Optional[str] = None,
    debug_recorder: Optional[DebugRecorder] = None,
) -> List[Dict[Any, Any]]:
    """Generate tests using per-endpoint approach (5 scenarios per endpoint)"""
    from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
    from devdox_ai_locust.locust_generator import LocustTestGenerator

    # Group endpoints by tag for directory organization
    grouped_endpoints: Dict[str, List[Endpoint]] = {}
    for ep in endpoints:
        tag = ep.tags[0] if ep.tags else "default"
        if tag not in grouped_endpoints:
            grouped_endpoints[tag] = []
        grouped_endpoints[tag].append(ep)

    num_tags = len(grouped_endpoints)
    num_endpoints = len(endpoints)

    # Setup prompt directory
    prompt_dir = Path(__file__).parent / "prompt"

    # Initialize scenario generator
    scenario_gen = ScenarioWorkflowGenerator(
        prompt_dir=prompt_dir,
        ai_client=ai_client,
        ai_config=ai_config,
        debug_recorder=debug_recorder,
    )

    # Get dynamic counts from generator
    num_scenarios = scenario_gen.num_scenarios
    scenario_filenames = ", ".join(scenario_gen.SCENARIO_FILES.values())

    # Show time estimate (using generator's estimate_time with num_endpoints)
    time_estimate = scenario_gen.estimate_time(num_endpoints)
    estimated_rpm = time_estimate.rpm

    console.print(f"\n📊 [bold]Generation Plan[/bold]")
    console.print(f"   {num_endpoints} endpoints × {num_scenarios} scenarios = {time_estimate.total_calls} LLM calls")
    console.print(f"   Estimated: {time_estimate} (across {num_tags} tags)\n")

    # Generate base files first using template generator
    template_gen = LocustTestGenerator()
    base_files, _, _ = template_gen.generate_from_endpoints(
        endpoints, api_info, include_auth=auth, target_host=host, db_type=db_type
    )
    base_files = template_gen.fix_indent(base_files)

    base_workflow_content = base_files.get("base_workflow.py", "")
    test_data_content = base_files.get("test_data.py", "")

    # Record static file generation
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

    # Get auth endpoints
    auth_endpoints = [ep for ep in endpoints if any(
        kw in ep.path.lower() for kw in ["auth", "login", "token", "session"]
    )]

    created_files: List[Dict[str, Any]] = []
    failed_endpoints: List[Dict[str, Any]] = []  # Track failures
    workflows_dir = output_dir / "workflows"
    completed_count = 0
    failed_count = 0
    file_write_lock = asyncio.Lock()

    # Helper to sanitize directory names
    def sanitize_dir_name(name: str) -> str:
        import re
        name = name.lower().replace("-", "_").replace(" ", "_").replace(".", "_").replace("/", "_")
        name = re.sub(r'[^a-z0-9_]', '', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name or "unnamed"

    # Helper to convert to PascalCase class name
    def to_class_name(name: str) -> str:
        sanitized = sanitize_dir_name(name)
        words = sanitized.replace("_", " ").split()
        return "".join(word.capitalize() for word in words) or "Unnamed"

    # Generate pre-LLM template for a single endpoint and scenario
    def generate_pre_llm_workflow(endpoint: Any, scenario_type: str) -> str:
        """Generate a pre-LLM workflow using template generator"""
        operation_id = scenario_gen.get_endpoint_dir_name(endpoint)
        class_name = to_class_name(operation_id)
        method = endpoint.method.lower()
        path = endpoint.path

        # Use template generator to create proper task method
        task_method = template_gen._generate_task_method(endpoint)
        # Indent task method for class body (4 spaces)
        indented_task = "\n".join(
            f"    {line}" if line.strip() else line
            for line in task_method.split("\n")
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

    # Generate base templates first (LLM will enhance these)
    console.print("→ Generating base templates...")
    pre_llm_templates: Dict[Tuple[int, str], str] = {}
    scenario_types = ["positive", "negative", "security"]
    for endpoint in endpoints:
        for scenario_type in scenario_types:
            pre_llm_templates[(id(endpoint), scenario_type)] = generate_pre_llm_workflow(
                endpoint, scenario_type
            )
    console.print("[green]✓[/green] Base templates generated")

    # Build endpoint to tag mapping
    endpoint_to_tag = {}
    for tag_name, tag_endpoints in grouped_endpoints.items():
        for ep in tag_endpoints:
            endpoint_to_tag[id(ep)] = tag_name

    # Track active workers for concurrent progress display
    active_workers: set = set()  # Set of active endpoint short_info strings
    active_lock = asyncio.Lock()

    # Progress tracking - milestones + heartbeat for user feedback
    printed_milestones: set = set()
    milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    last_heartbeat_count = 0
    heartbeat_interval = max(5, num_endpoints // 20)  # Every 5% or minimum 5 endpoints

    def check_and_print_progress():
        """Print progress - milestones at 10% intervals, heartbeat between them"""
        nonlocal last_heartbeat_count
        total_processed = completed_count + failed_count
        if num_endpoints <= 0:
            return

        percent = (total_processed * 100) // num_endpoints

        # Check for milestone
        milestone_printed = False
        for milestone in milestones:
            if percent >= milestone and milestone not in printed_milestones:
                printed_milestones.add(milestone)
                console.print(f"  ✓ {milestone}% complete ({total_processed}/{num_endpoints} endpoints)")
                last_heartbeat_count = total_processed
                milestone_printed = True

        # Heartbeat between milestones to show system is working
        if not milestone_printed and (total_processed - last_heartbeat_count) >= heartbeat_interval:
            console.print(f"    ... processing ({total_processed}/{num_endpoints})")
            last_heartbeat_count = total_processed

    # Process endpoint and save files (resilient - catches and tracks errors)
    async def process_and_save_endpoint(endpoint: Any) -> List[Dict[str, Any]]:
        nonlocal completed_count, failed_count
        tag_name = endpoint_to_tag.get(id(endpoint), "default")
        tag_dir_name = sanitize_dir_name(tag_name)
        operation_id = scenario_gen.get_endpoint_dir_name(endpoint)
        endpoint_info = f"{endpoint.method} {endpoint.path}"
        short_info = f"{endpoint.method} {endpoint.path[:30]}..." if len(endpoint.path) > 30 else endpoint_info

        # Track active worker
        async with active_lock:
            active_workers.add(short_info)

        try:
            # Create endpoint directory
            endpoint_dir = workflows_dir / tag_dir_name / operation_id
            endpoint_dir.mkdir(parents=True, exist_ok=True)

            # Generate scenario workflows
            scenarios = await scenario_gen.generate_endpoint_workflows(
                endpoint=endpoint,
                base_workflow_content=base_workflow_content,
                test_data_content=test_data_content,
                auth_endpoints=auth_endpoints if auth else None,
                tag_name=tag_name,
                all_endpoints=endpoints,
                custom_requirement=custom_requirement,
                db_type=db_type,
            )

            # Save files using async I/O
            local_files = []
            for scenario_type, content in scenarios.items():
                if content:
                    filename = scenario_gen.SCENARIO_FILES[scenario_type]
                    file_path = endpoint_dir / filename
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(content)
                    local_files.append({
                        "path": str(file_path),
                        "size": len(content),
                        "tag": tag_name,
                        "operation_id": operation_id,
                        "scenario": scenario_type.value,
                    })

            # Update progress (success) and release worker
            async with file_write_lock:
                completed_count += 1
                created_files.extend(local_files)
                active_workers.discard(short_info)
                check_and_print_progress()

            return local_files

        except Exception as e:
            # Use pre-generated pre-LLM templates as fallback
            fallback_files = []
            for scenario_type in ["positive", "negative", "security"]:
                # Get pre-generated template (already exists before LLM processing)
                fallback_content = pre_llm_templates.get((id(endpoint), scenario_type), "")
                if not fallback_content:
                    # Should never happen, but generate on-the-fly as last resort
                    fallback_content = generate_pre_llm_workflow(endpoint, scenario_type)
                filename = f"{scenario_type}_workflow.py"
                file_path = endpoint_dir / filename
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(fallback_content)
                fallback_files.append({
                    "path": str(file_path),
                    "size": len(fallback_content),
                    "tag": tag_name,
                    "operation_id": operation_id,
                    "scenario": scenario_type,
                    "fallback": True,
                })

            async with file_write_lock:
                failed_count += 1
                failed_endpoints.append({
                    "endpoint": endpoint_info,
                    "operation_id": operation_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                created_files.extend(fallback_files)  # Use pre-LLM templates
                active_workers.discard(short_info)
                check_and_print_progress()
                # Print full traceback so user can debug (no truncation)
                console.print(
                    f"\n   [yellow]⚠[/yellow] {tag_dir_name}/{operation_id} failed, using base template:"
                )
                # Use format_exception with no limit to avoid Python 3.11+ truncation
                exc_lines = traceback.format_exception(type(e), e, e.__traceback__)
                console.print(f"[red]{''.join(exc_lines)}[/red]")
            return fallback_files

    # Process all endpoints in parallel
    num_workers = scenario_gen.current_concurrency
    console.print(f"\n→ Generating workflows ({num_workers} concurrent)...")

    # Create all tasks
    tasks = [process_and_save_endpoint(ep) for ep in endpoints]

    # Run all concurrently (semaphore in generator limits actual API calls)
    await asyncio.gather(*tasks)

    console.print(f"[green]✓[/green] Workflows generated ({completed_count + failed_count}/{num_endpoints} endpoints)")

    # Generate orchestrator for each tag
    console.print(f"\n→ Generating orchestrators ({num_tags} tags)...")
    orchestrator_files = []
    orchestrator_failures = []

    for tag_name, tag_endpoints in grouped_endpoints.items():
        tag_dir_name = sanitize_dir_name(tag_name)
        tag_dir = workflows_dir / tag_dir_name

        try:
            orchestrator_code = await scenario_gen.generate_tag_orchestrator(
                tag_name=tag_name,
                tag_endpoints=tag_endpoints,
                base_workflow_content=base_workflow_content,
                test_data_content=test_data_content,
                auth_endpoints=auth_endpoints if auth else None,
                custom_requirement=custom_requirement,
                db_type=db_type,
            )

            # Save orchestrator file
            orchestrator_path = tag_dir / "orchestrator_workflow.py"
            async with aiofiles.open(orchestrator_path, 'w', encoding='utf-8') as f:
                await f.write(orchestrator_code)
            orchestrator_files.append({
                "path": str(orchestrator_path),
                "size": len(orchestrator_code),
                "tag": tag_name,
            })
            console.print(f"  [green]✓[/green] {tag_dir_name}/orchestrator_workflow.py")

        except Exception as e:
            orchestrator_failures.append({
                "tag": tag_name,
                "error": str(e),
            })
            console.print(f"  [yellow]⚠[/yellow] {tag_dir_name}/orchestrator_workflow.py failed: {str(e)[:100]}")

    created_files.extend(orchestrator_files)

    if orchestrator_failures:
        console.print(f"[yellow]⚠[/yellow] Orchestrators: {len(orchestrator_files)}/{num_tags} succeeded")
    else:
        console.print(f"[green]✓[/green] All {num_tags} orchestrators generated")

    # Show summary
    rate_info = scenario_gen.get_rate_limit_info()
    console.print(f"\n[dim]Final rate limit: {rate_info.requests_per_minute} RPM, "
                  f"Concurrency used: {scenario_gen.current_concurrency}[/dim]")

    # Report results
    if failed_endpoints:
        console.print(f"\n[bold yellow]⚠ Generation completed with {failed_count} failures[/bold yellow]")
        console.print(f"   [green]✓ Succeeded:[/green] {completed_count}/{num_endpoints}")
        console.print(f"   [red]✗ Failed:[/red] {failed_count}/{num_endpoints}")

        # Show failure details
        console.print(f"\n[bold red]Failed Endpoints:[/bold red]")
        for failure in failed_endpoints[:10]:  # Show first 10
            console.print(f"   • {failure['endpoint']}")
            console.print(f"     [dim]{failure['error_type']}: {failure['error'][:200]}[/dim]")

        if len(failed_endpoints) > 10:
            console.print(f"   ... and {len(failed_endpoints) - 10} more failures")
    else:
        console.print(f"\n[bold green]✓ All {num_endpoints} endpoints generated successfully[/bold green]")

    # Generate __init__.py files for each tag directory to enable imports
    console.print("→ Creating workflow __init__.py files...")
    # Create main workflows/__init__.py
    workflows_init = workflows_dir / "__init__.py"
    tag_imports = []
    for tag_name in grouped_endpoints.keys():
        tag_dir_name = sanitize_dir_name(tag_name)
        tag_imports.append(f"from .{tag_dir_name} import *")
    workflows_init.write_text("\n".join(tag_imports) + "\n", encoding='utf-8')

    # Create __init__.py for each tag directory
    for tag_name, tag_endpoints in grouped_endpoints.items():
        tag_dir_name = sanitize_dir_name(tag_name)
        tag_dir = workflows_dir / tag_dir_name
        if tag_dir.exists():
            init_lines = ['"""Auto-generated workflow exports"""']
            for ep in tag_endpoints:
                op_id = scenario_gen.get_endpoint_dir_name(ep)
                class_name = to_class_name(op_id)
                # Import all three scenario types
                for scenario in ["positive", "negative", "security"]:
                    init_lines.append(
                        f"from .{op_id}.{scenario}_workflow import {class_name}{scenario.capitalize()}Workflow"
                    )
            # Import orchestrator if it exists
            orchestrator_path = tag_dir / "orchestrator_workflow.py"
            if orchestrator_path.exists():
                orchestrator_class = to_class_name(tag_dir_name) + "Orchestrator"
                init_lines.append(
                    f"from .orchestrator_workflow import {orchestrator_class}"
                )
            tag_init = tag_dir / "__init__.py"
            tag_init.write_text("\n".join(init_lines) + "\n", encoding='utf-8')
    console.print("[green]✓[/green] Workflow __init__.py files created")

    # Write base files
    console.print("→ Creating base files...")
    for filename, content in base_files.items():
        # base_workflow.py goes in workflows/ to match locustfile.py import
        if filename == "base_workflow.py":
            file_path = workflows_dir / filename
        else:
            file_path = output_dir / filename
        file_path.write_text(content, encoding='utf-8')
        created_files.append({
            "path": str(file_path),
            "size": len(content),
        })
    console.print(f"[green]✓[/green] Base files created ({len(base_files)} files)")

    return created_files


@click.group()
@click.version_option(version="0.1.9")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """DevDox AI LoadTest - Generate Locust tests from API documentation"""
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
    help="Output directory for generated tests (default: output)",
)
@click.option("--users", "-u", type=int, default=10, help="Number of simulated users")
@click.option(
    "--spawn-rate",
    "-r",
    type=float,
    default=2,
    help="Rate to spawn users (users per second)",
)
@click.option(
    "--run-time", "-t", type=str, default="5m", help="Test run time (e.g., 5m, 1h)"
)
@click.option("--host", "-H", type=str, help="Target host URL")
@click.option("--auth/--no-auth", default=True, help="Include authentication in tests")
@click.option(
    "--db-type",
    type=click.Choice(["", "mongo", "postgresql"], case_sensitive=False),
    default="",
    help="Database type for testing (empty for no database, mongo, or postgresql)",
)
@click.option("--dry-run", is_flag=True, help="Generate tests without running them")
@click.option(
    "--custom-requirement", type=str, help="Custom requirements for test generation"
)
@click.option(
    "--together-api-key",
    type=str,
    envvar="TOGETHER_API_KEY",
    help="Together AI API key (can also be set via TOGETHER_API_KEY env var)",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help="Timeout in seconds for AI API calls (default: 120, increase for large APIs)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode to record all intermediate states for auditing",
)
@click.pass_context
def generate(
    ctx: click.Context,
    swagger_url: str,
    output: str,
    users: int,
    spawn_rate: float,
    run_time: str,
    host: Optional[str],
    auth: bool,
    db_type: str,
    dry_run: bool,
    custom_requirement: Optional[str],
    together_api_key: Optional[str],
    timeout: int,
    debug: bool,
) -> None:
    """Generate Locust test files from API documentation URL or file"""

    # Setup output directory and logging
    output_dir = _setup_output_directory(output)
    log_path, log_file = _setup_logging(output_dir, "generate")

    try:
        # Run the async generation
        asyncio.run(
            _async_generate(
                ctx,
                swagger_url,
                output,
                users,
                spawn_rate,
                run_time,
                host,
                auth,
                db_type,
                dry_run,
                custom_requirement,
                together_api_key,
                timeout,
                debug,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if ctx.obj["verbose"]:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)
    finally:
        _teardown_logging(log_file, log_path)


async def _async_generate(
    ctx: click.Context,
    swagger_url: str,
    output: str,
    users: int,
    spawn_rate: float,
    run_time: str,
    host: Optional[str],
    auth: bool,
    db_type: str,
    dry_run: bool,
    custom_requirement: Optional[str],
    together_api_key: Optional[str],
    timeout: int = 120,
    debug: bool = False,
) -> None:
    """Async function to handle the generation process"""
    start_time = datetime.now(timezone.utc)

    try:
        _, api_key = _initialize_config(together_api_key)
        output_dir = _setup_output_directory(output)

        # Initialize debug recorder
        debug_recorder = DebugRecorder(output_dir, enabled=debug)

        if debug:
            console.print("[blue]🔍 Debug mode enabled[/blue] - recording intermediate states")
            # Record CLI args
            debug_recorder.record_cli_args({
                "swagger_url": swagger_url,
                "output": output,
                "users": users,
                "spawn_rate": spawn_rate,
                "run_time": run_time,
                "host": host,
                "auth": auth,
                "db_type": db_type,
                "dry_run": dry_run,
                "custom_requirement": custom_requirement,
                "timeout": timeout,
            })

        # Display configuration
        if ctx.obj["verbose"]:
            _display_configuration(
                swagger_url,
                output_dir,
                users,
                spawn_rate,
                run_time,
                host,
                auth,
                custom_requirement,
                dry_run,
            )

        raw_schema, endpoints, api_info = await _process_api_schema(
            swagger_url, ctx.obj["verbose"]
        )

        # Record parsed OpenAPI data
        if debug:
            debug_recorder.record_openapi_raw(raw_schema)
            debug_recorder.record_openapi_parsed(endpoints, api_info)
            debug_recorder.record_resolved_config({
                "api_info": api_info,
                "host": host,
                "auth": auth,
                "db_type": db_type,
                "timeout": timeout,
                "custom_requirement": custom_requirement,
            })

        created_files = await _generate_and_create_tests(
            api_key,
            endpoints,
            api_info,
            output_dir,
            custom_requirement,
            host,
            auth,
            db_type,
            timeout,
            debug_recorder,
        )

        # Finalize debug recording
        if debug:
            await debug_recorder.finalize()
            console.print(f"[blue]🔍 Debug info saved to:[/blue] {debug_recorder.debug_root}")

        # Show results
        _show_results(
            created_files,
            output_dir,
            start_time,
            ctx.obj["verbose"],
            dry_run,
            users,
            spawn_rate,
            run_time,
            host,
        )

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()
        console.print(
            f"[red]✗[/red] Generation failed after {processing_time:.2f}s: {e}"
        )
        raise


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--users", "-u", type=int, default=10, help="Number of simulated users")
@click.option("--spawn-rate", "-r", type=float, default=2, help="Rate to spawn users")
@click.option("--run-time", "-t", type=str, default="5m", help="Test run time")
@click.option("--host", "-H", type=str, required=True, help="Target host URL")
@click.option("--headless", is_flag=True, help="Run in headless mode (no web UI)")
@click.pass_context
def run(
    ctx: click.Context,
    test_file: str,
    users: int,
    spawn_rate: float,
    run_time: str,
    host: str,
    headless: bool,
) -> None:
    """Run generated Locust tests with automatic logging"""
    import subprocess

    test_file_path = Path(test_file)
    test_suite_dir = test_file_path.parent

    # Setup logging
    log_path, log_file = _setup_logging(test_suite_dir, "locust_run")

    try:
        cmd = [
            "locust",
            "-f",
            str(test_file),
            "--users",
            str(users),
            "--spawn-rate",
            str(spawn_rate),
            "--run-time",
            run_time,
            "--host",
            host,
        ]

        if headless:
            cmd.append("--headless")

        if ctx.obj["verbose"]:
            console.print(f"[blue]Running command:[/blue] {' '.join(cmd)}")

        console.print("[green]Starting Locust test...[/green]")

        # Run locust with real-time output capture via tee
        # Since we've already setup tee on stdout/stderr, subprocess output
        # needs to be captured and printed (which will tee to log)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read and display output in real-time (tee will capture it)
        try:
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


def main() -> None:
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
