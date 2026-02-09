import click
import shutil
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Union, List, Dict, Any
from rich.console import Console
from rich.table import Table
from together import AsyncTogether

from .hybrid_loctus_generator import HybridLocustGenerator
from .locust_enhancer import LocustTestEnhancer, EnhanceResult
from .config import Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from .schemas.processing_result import SwaggerProcessingRequest

console = Console()

LOCUSTFILE_MODULE_NAME = "locustfile.py"

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
    locustfile = output_dir / LOCUSTFILE_MODULE_NAME

    if locustfile.exists():
        main_file = LOCUSTFILE_MODULE_NAME
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


def _display_enhance_configuration(
    swagger_url: str,
    suite_dir: Path,
    custom_requirement: str,
    dry_run: bool,
) -> None:
    """Display enhancement configuration as a Rich table."""
    table = Table(title="Enhancement Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("API Spec Source", str(swagger_url))
    table.add_row("Test Suite", str(suite_dir))
    table.add_row("Custom Requirement", custom_requirement)
    table.add_row("Dry Run", "Yes" if dry_run else "No")

    console.print(table)


def _discover_suite_files(suite_dir: Path, verbose: bool) -> Dict[str, Any]:
    """Scan a test suite directory and categorise its files.

    Returns a dict with keys:
        locustfile: Path or None
        test_data: Path or None
        workflows: list of Path
        suite_dir: Path
    """
    result: Dict[str, Any] = {
        "locustfile": None,
        "test_data": None,
        "workflows": [],
        "suite_dir": suite_dir,
    }

    locustfile = suite_dir / LOCUSTFILE_MODULE_NAME
    if locustfile.exists():
        result["locustfile"] = locustfile

    test_data = suite_dir / "test_data.py"
    if test_data.exists():
        result["test_data"] = test_data

    workflows_dir = suite_dir / "workflows"
    if workflows_dir.is_dir():
        result["workflows"] = sorted(
            p for p in workflows_dir.glob("*.py") if p.name != "__init__.py"
        )

    if verbose:
        wf_count = len(result["workflows"])
        console.print(
            f"[blue]Suite discovery:[/blue] "
            f"locustfile={'found' if result['locustfile'] else 'missing'}, "
            f"test_data={'found' if result['test_data'] else 'missing'}, "
            f"{wf_count} workflow file(s)"
        )
        for wf in result["workflows"]:
            console.print(f"  [dim]-[/dim] {wf.name}")

    return result


def _identify_coverage_gaps(
    endpoints: List[Endpoint],
    existing_workflows: List[Path],
    verbose: bool,
) -> List[str]:
    """Find swagger tags not covered by existing workflow files."""
    all_tags: set = set()
    for ep in endpoints:
        for tag in ep.tags:
            all_tags.add(tag.lower().replace(" ", "_"))

    covered_tags: set = set()
    for wf_path in existing_workflows:
        stem = wf_path.stem
        if stem.endswith("_workflow"):
            covered_tags.add(stem[: -len("_workflow")])

    gaps = sorted(all_tags - covered_tags)

    if verbose:
        console.print(
            f"[blue]Tag coverage:[/blue] {len(all_tags)} tag(s) in spec, "
            f"{len(covered_tags)} covered, {len(gaps)} uncovered"
        )
        for gap in gaps:
            console.print(f"  [yellow]-[/yellow] {gap} (no workflow file)")

    return gaps


def _build_results_table(
    results: Dict[str, "EnhanceResult"],
    created_files: List[str],
    dry_run: bool,
) -> Tuple[Table, Dict[str, int]]:
    """Build the Rich table and compute aggregate counters for enhancement results.

    Returns:
        A tuple of (table, counters) where counters is a dict with keys:
        total_added, total_replaced, total_imports, updated, unchanged, failed.
    """
    table = Table(title="Enhancement Results")
    table.add_column("File", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Tasks +", justify="right")
    table.add_column("Tasks ~", justify="right")
    table.add_column("Imports +", justify="right")
    table.add_column("Warnings", justify="right", style="yellow")

    counters: Dict[str, int] = {
        "total_added": 0,
        "total_replaced": 0,
        "total_imports": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
    }

    for file_path, result in results.items():
        _add_result_row(table, file_path, result, dry_run, counters)

    for created in created_files:
        name = Path(created).name
        action = "[DRY RUN]" if dry_run else "[blue]CREATED[/blue]"
        table.add_row(name, action, "-", "-", "-", "0")

    return table, counters


def _add_result_row(
    table: Table,
    file_path: str,
    result: "EnhanceResult",
    dry_run: bool,
    counters: Dict[str, int],
) -> None:
    """Add a single result row to the table and update counters."""
    name = Path(file_path).name

    if not result.success:
        table.add_row(
            name, "[red]FAILED[/red]", "-", "-", "-", result.error or "Unknown",
        )
        counters["failed"] += 1
        return

    if result.enhanced_source == result.original_source:
        table.add_row(name, "[dim]UNCHANGED[/dim]", "0", "0", "0", "0")
        counters["unchanged"] += 1
        return

    tasks_added = len(result.added_tasks)
    tasks_replaced = len(getattr(result, "replaced_tasks", []))
    imports_added = len(result.added_imports)
    warn_count = len(result.warnings)

    action = "[DRY RUN]" if dry_run else "[green]UPDATED[/green]"
    table.add_row(
        name,
        action,
        str(tasks_added),
        str(tasks_replaced),
        str(imports_added),
        str(warn_count) if warn_count else "0",
    )
    counters["total_added"] += tasks_added
    counters["total_replaced"] += tasks_replaced
    counters["total_imports"] += imports_added
    counters["updated"] += 1


def _print_summary(
    counters: Dict[str, int],
    created_count: int,
    processing_time: float,
) -> None:
    """Print the summary, totals, and processing time lines."""
    console.print(
        f"\n[bold]Summary:[/bold] {counters['updated']} updated, "
        f"{created_count} created, {counters['unchanged']} unchanged, "
        f"{counters['failed']} failed"
    )
    console.print(
        f"[bold]Totals:[/bold] +{counters['total_added']} tasks added, "
        f"~{counters['total_replaced']} tasks replaced, +{counters['total_imports']} imports"
    )
    console.print(f"[blue]Processing time:[/blue] {processing_time:.2f}s")


def _print_verbose_details(results: Dict[str, "EnhanceResult"]) -> None:
    """Print per-file verbose details (added tasks, replaced tasks, warnings)."""
    for file_path, result in results.items():
        name = Path(file_path).name

        if result.success and result.added_tasks:
            console.print(
                f"\n[bold]{name}[/bold] added tasks: "
                f"{', '.join(result.added_tasks)}"
            )

        replaced = getattr(result, "replaced_tasks", [])
        if replaced:
            console.print(
                f"[bold]{name}[/bold] replaced tasks: "
                f"{', '.join(replaced)}"
            )

        if result.warnings:
            for w in result.warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")


def _show_enhance_results(
    results: Dict[str, "EnhanceResult"],
    created_files: List[str],
    start_time: datetime,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Display enhancement results summary."""
    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

    table, counters = _build_results_table(results, created_files, dry_run)
    console.print(table)

    _print_summary(counters, len(created_files), processing_time)

    if verbose:
        _print_verbose_details(results)


async def _process_api_schema(
    swagger_url: str, verbose: bool
) -> Tuple[Dict[str, Any], List[Endpoint], Dict[str, Any]]:
    """Fetch and parse API schema"""
    source_request = SwaggerProcessingRequest(swagger_url=swagger_url)
    api_schema = None
    with console.status(
        f"[bold green]Fetching API schema from {'URL' if swagger_url.startswith(('http://', 'https://')) else 'file'}..."
    ):
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
    schema_length = len(api_schema) if api_schema else 0
    console.print(
        f"[green]✓[/green] Successfully fetched API schema ({schema_length} characters)"
    )

    # Parse schema
    with console.status("[bold green]Parsing API schema..."):
        parser = OpenAPIParser()
        try:
            schema_data = parser.parse_schema(api_schema)
            if verbose:
                console.print("✓ Schema data parsed successfully")

            endpoints = parser.parse_endpoints()
            api_info = parser.get_schema_info()

            console.print(
                f"[green]📋 Parsed {len(endpoints)} endpoints from {api_info.get('title', 'API')}[/green]"
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
) -> List[Dict[Any, Any]]:
    """Generate tests using AI and create test files"""
    together_client = AsyncTogether(api_key=api_key)

    with console.status("[bold green]Generating Locust tests with AI..."):
        generator = HybridLocustGenerator(ai_client=together_client)
        test_files, test_directories = await generator.generate_from_endpoints(
            endpoints=endpoints,
            api_info=api_info,
            custom_requirement=custom_requirement,
            target_host=host,
            include_auth=auth,
            db_type=db_type,
        )

    # Create test files
    with console.status("[bold green]Creating test files..."):
        created_files = []

        # Create workflow files
        if test_directories:
            workflows_dir = output_dir / "workflows"
            workflows_dir.mkdir(exist_ok=True)
            for file_workflow in test_directories:
                workflow_files = await generator._create_test_files_safely(
                    file_workflow, workflows_dir
                )
                created_files.extend(workflow_files)

        # Create main test files
        if test_files:
            main_files = await generator._create_test_files_safely(
                test_files, output_dir
            )
            created_files.extend(main_files)

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
) -> None:  # Added return type annotation
    """Generate Locust test files from API documentation URL or file"""

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
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if ctx.obj["verbose"]:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


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
) -> None:
    """Async function to handle the generation process"""

    start_time = datetime.now(timezone.utc)

    try:
        _, api_key = _initialize_config(together_api_key)
        output_dir = _setup_output_directory(output)
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

        _, endpoints, api_info = await _process_api_schema(
            swagger_url, ctx.obj["verbose"]
        )

        created_files = await _generate_and_create_tests(
            api_key,
            endpoints,
            api_info,
            output_dir,
            custom_requirement,
            host,
            auth,
            db_type,
        )

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
    """Run generated Locust tests"""

    try:
        import subprocess

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
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Test execution failed:[/red] {e}")
        sys.exit(1)
    except FileNotFoundError:
        console.print(
            "[red]Locust not found. Please install locust: pip install locust[/red]"
        )
        sys.exit(1)


@cli.command()
@click.argument("swagger_url")
@click.option(
    "--test-suite",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to the existing generated test suite directory",
)
@click.option(
    "--custom-requirement",
    type=str,
    required=True,
    help="What test scenarios to generate — the primary driver for enhancement",
)
@click.option(
    "--together-api-key",
    type=str,
    required=True,
    envvar="TOGETHER_API_KEY",
    help="Together AI API key (required; also accepted via TOGETHER_API_KEY env var)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without writing to disk",
)
@click.pass_context
def enhance(
    ctx: click.Context,
    swagger_url: str,
    test_suite: str,
    custom_requirement: str,
    together_api_key: str,
    dry_run: bool,
) -> None:
    """Enhance an existing Locust test suite with new AI-generated scenarios.

    Reads the existing test suite, analyzes its structure, and uses AI to
    generate new test scenarios based on the custom requirement. Updates
    existing files where applicable and creates new workflow files for
    uncovered API tags.

    SWAGGER_URL is the URL or file path to the OpenAPI/Swagger specification.

    \b
    Examples:
        dal enhance https://api.example.com/openapi.json \\
            --test-suite ./output \\
            --custom-requirement "Add concurrent user registration tests" \\
            --together-api-key sk-xxx

        dal enhance ./spec.yaml \\
            --test-suite ./output \\
            --custom-requirement "Add edge cases for payment processing" \\
            --together-api-key sk-xxx --dry-run
    """
    try:
        asyncio.run(
            _async_enhance(
                ctx,
                swagger_url,
                test_suite,
                custom_requirement,
                together_api_key,
                dry_run,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if ctx.obj["verbose"]:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


async def _async_enhance(
    ctx: click.Context,
    swagger_url: str,
    test_suite: str,
    custom_requirement: str,
    together_api_key: str,
    dry_run: bool,
) -> None:
    """Async handler for the enhance command."""
    start_time = datetime.now(timezone.utc)
    verbose = ctx.obj["verbose"]

    try:
        # 1. Validate API key
        _, api_key = _initialize_config(together_api_key)
        suite_dir = Path(test_suite)

        # 2. Display configuration
        if verbose:
            _display_enhance_configuration(
                swagger_url, suite_dir, custom_requirement, dry_run
            )

        # 3. Discover suite structure
        console.print("[bold]Discovering test suite structure...[/bold]")
        suite = _discover_suite_files(suite_dir, verbose)

        if not suite["workflows"] and not suite["locustfile"]:
            console.print(
                "[red]Error:[/red] No Locust test files found in "
                f"suite directory: {suite_dir}"
            )
            sys.exit(1)

        file_count = (
            len(suite["workflows"])
            + (1 if suite["locustfile"] else 0)
            + (1 if suite["test_data"] else 0)
        )
        console.print(
            f"[green]\u2713[/green] Found {file_count} enhanceable file(s) "
            f"in {suite_dir}"
        )

        # 4. Fetch and parse swagger spec
        _, endpoints, api_info = await _process_api_schema(
            swagger_url, verbose
        )

        if verbose:
            # Show a per-tag endpoint breakdown
            tag_counts: Dict[str, int] = {}
            for ep in endpoints:
                for tag in ep.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            console.print("[blue]Endpoints per tag:[/blue]")
            for tag, count in sorted(tag_counts.items()):
                console.print(f"  [dim]-[/dim] {tag}: {count} endpoint(s)")

        # 5. Identify coverage gaps
        gaps = _identify_coverage_gaps(
            endpoints, suite["workflows"], verbose
        )

        # 6. Enhance existing files
        console.print(
            f"\n[bold]Enhancing test suite based on requirement:[/bold] "
            f"{custom_requirement}\n"
        )

        # -- Verbose: configure logging level for enhancer & merger --
        if verbose:
            import logging as _logging

            _logging.getLogger("devdox_ai_locust.locust_enhancer").setLevel(
                _logging.DEBUG
            )
            _logging.getLogger("devdox_ai_locust.utils.code_merger").setLevel(
                _logging.DEBUG
            )
            # Ensure there's a handler that can show debug output
            root = _logging.getLogger()
            if not root.handlers:
                handler = _logging.StreamHandler()
                handler.setFormatter(
                    _logging.Formatter("%(name)s %(levelname)s: %(message)s")
                )
                root.addHandler(handler)
            root.setLevel(_logging.DEBUG)

        enhancer = LocustTestEnhancer(
            together_api_key=api_key, verbose=verbose
        )

        results: Dict[str, EnhanceResult] = {}

        # --- Workflow files (primary targets) ---
        for wf_path in suite["workflows"]:
            results[str(wf_path)] = await _enhance_single_file(
                enhancer, wf_path, custom_requirement, swagger_url, verbose
            )

        # --- locustfile.py ---
        if suite["locustfile"]:
            lf_path = suite["locustfile"]
            results[str(lf_path)] = await _enhance_single_file(
                enhancer, lf_path, custom_requirement, swagger_url, verbose
            )

        # --- test_data.py ---
        if suite["test_data"]:
            td_path = suite["test_data"]
            results[str(td_path)] = await _enhance_single_file(
                enhancer, td_path, custom_requirement, swagger_url, verbose
            )

        # 7. Create new workflow files for uncovered tags
        created_files: List[str] = []
        if gaps:
            # Get a reference workflow for style matching
            reference_source = None
            if suite["workflows"]:
                try:
                    with open(suite["workflows"][0], "r", encoding="utf-8") as f:
                        reference_source = f.read()
                except IOError:
                    pass

            # Group endpoints by tag for gap generation
            tag_endpoint_map: Dict[str, List[Endpoint]] = {}
            for ep in endpoints:
                for tag in ep.tags:
                    normalised = tag.lower().replace(" ", "_")
                    if normalised in gaps:
                        tag_endpoint_map.setdefault(normalised, []).append(ep)

            if verbose:
                console.print(
                    f"\n[bold]Generating new workflows for "
                    f"{len(tag_endpoint_map)} uncovered tag(s)...[/bold]"
                )

            for gap_tag, gap_endpoints in tag_endpoint_map.items():
                file_start = datetime.now(timezone.utc)

                if verbose:
                    ep_list = ", ".join(
                        f"{getattr(e, 'method', '?').upper()} "
                        f"{getattr(e, 'path', '?')}"
                        for e in gap_endpoints
                    )
                    console.print(
                        f"  [blue]>[/blue] {gap_tag}: "
                        f"{len(gap_endpoints)} endpoint(s) — {ep_list}"
                    )

                with console.status(
                    f"[bold blue]Generating new workflow for "
                    f"'{gap_tag}'..."
                ):
                    try:
                        gen_result = await enhancer.generate_new_workflow(
                            tag_name=gap_tag,
                            tag_endpoints=gap_endpoints,
                            custom_requirement=custom_requirement,
                            swagger_url=swagger_url,
                            reference_workflow_source=reference_source,
                        )
                    except Exception as e:
                        gen_result = EnhanceResult(
                            success=False,
                            enhanced_source="",
                            original_source="",
                            error=str(e),
                        )

                elapsed = (
                    datetime.now(timezone.utc) - file_start
                ).total_seconds()

                if gen_result.success and gen_result.enhanced_source.strip():
                    workflows_dir = suite_dir / "workflows"
                    workflows_dir.mkdir(exist_ok=True)
                    new_path = workflows_dir / f"{gap_tag}_workflow.py"

                    if not dry_run:
                        with open(new_path, "w", encoding="utf-8") as f:
                            f.write(gen_result.enhanced_source)

                    created_files.append(str(new_path))
                    if verbose:
                        new_lines = gen_result.enhanced_source.count("\n") + 1
                        console.print(
                            f"  {new_path.name}: [blue]created[/blue] "
                            f"({elapsed:.1f}s, {new_lines} lines)"
                        )
                    else:
                        console.print(
                            f"  [blue]+[/blue] {gap_tag}_workflow.py "
                            f"({elapsed:.1f}s)"
                        )
                else:
                    console.print(
                        f"  [red]\u2717[/red] {gap_tag}_workflow.py: "
                        f"{gen_result.error} ({elapsed:.1f}s)"
                    )

        # 8. Write results to disk
        if not dry_run:
            _write_enhance_results(results, verbose)
        elif verbose:
            console.print("[dim]Dry run — no files written.[/dim]")

        # 9. Show results summary
        _show_enhance_results(
            results, created_files, start_time, verbose, dry_run
        )

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()
        console.print(
            f"[red]\u2717[/red] Enhancement failed after "
            f"{processing_time:.2f}s: {e}"
        )
        raise


async def _enhance_single_file(
    enhancer: LocustTestEnhancer,
    file_path: Path,
    custom_requirement: str,
    swagger_url: str,
    verbose: bool,
) -> EnhanceResult:
    """Enhance a single file, returning the result and logging progress."""
    file_start = datetime.now(timezone.utc)

    if verbose:
        file_size = file_path.stat().st_size
        line_count = file_path.read_text(encoding="utf-8").count("\n") + 1
        console.print(
            f"  [blue]>[/blue] {file_path.name} "
            f"({line_count} lines, {file_size} bytes)"
        )

    with console.status(f"[bold green]Enhancing {file_path.name}..."):
        try:
            result = await enhancer.enhance_file(
                str(file_path), custom_requirement, swagger_url
            )
        except Exception as e:
            result = EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="",
                error=str(e),
            )

    elapsed = (datetime.now(timezone.utc) - file_start).total_seconds()

    if verbose:
        if result.success:
            added_tasks = len(result.added_tasks)
            replaced_tasks = len(getattr(result, "replaced_tasks", []))
            added_imports = len(result.added_imports)
            added_helpers = len(getattr(result, "added_helpers", []))
            added_classes = len(getattr(result, "added_classes", []))
            replaced_helpers = len(getattr(result, "replaced_helpers", []))
            replaced_classes = len(getattr(result, "replaced_classes", []))
            warn_count = len(result.warnings)

            parts = []
            if added_tasks:
                parts.append(f"+{added_tasks} tasks")
            if replaced_tasks:
                parts.append(f"~{replaced_tasks} tasks replaced")
            if added_imports:
                parts.append(f"+{added_imports} imports")
            if added_helpers:
                parts.append(f"+{added_helpers} helpers")
            if added_classes:
                parts.append(f"+{added_classes} classes")
            if replaced_helpers:
                parts.append(f"~{replaced_helpers} helpers replaced")
            if replaced_classes:
                parts.append(f"~{replaced_classes} classes replaced")
            if warn_count:
                parts.append(f"{warn_count} warning(s)")

            detail = ", ".join(parts) if parts else "no changes"

            if result.enhanced_source != result.original_source:
                orig_lines = result.original_source.count("\n") + 1
                new_lines = result.enhanced_source.count("\n") + 1
                diff = new_lines - orig_lines
                sign = "+" if diff >= 0 else ""
                detail += f" | {orig_lines}->{new_lines} lines ({sign}{diff})"

            console.print(
                f"  {file_path.name}: [green]done[/green] "
                f"({elapsed:.1f}s) [{detail}]"
            )
        else:
            console.print(
                f"  {file_path.name}: [red]failed[/red] "
                f"({elapsed:.1f}s) {result.error}"
            )
    else:
        # Even in non-verbose mode, show basic per-file progress
        if result.success:
            console.print(
                f"  [green]\u2713[/green] {file_path.name} ({elapsed:.1f}s)"
            )
        else:
            console.print(
                f"  [red]\u2717[/red] {file_path.name}: {result.error}"
            )

    return result


def _write_enhance_results(
    results: Dict[str, EnhanceResult], verbose: bool
) -> None:
    """Write enhanced sources back to disk with backups."""
    for file_path, result in results.items():
        if not result.success:
            continue
        if result.enhanced_source == result.original_source:
            continue

        # Create backup
        backup_path = file_path + ".bak"
        shutil.copy2(file_path, backup_path)
        if verbose:
            console.print(
                f"  [dim]Backup:[/dim] {Path(file_path).name} -> "
                f"{Path(backup_path).name}"
            )

        # Write enhanced file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result.enhanced_source)

        if verbose:
            original_lines = result.original_source.count("\n")
            new_lines = result.enhanced_source.count("\n")
            diff = new_lines - original_lines
            sign = "+" if diff >= 0 else ""
            console.print(
                f"  [green]Wrote:[/green] {Path(file_path).name} "
                f"({new_lines} lines, {sign}{diff} from original)"
            )


def main() -> None:
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
