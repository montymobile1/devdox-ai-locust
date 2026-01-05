import click
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Union, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.status import Status

from .modular_generator import ModularGenerator
from .schemas.progress import ProgressStatus, ProgressPhase
from .config import Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from devdox_ai_locust.utils.patch_tracker import PatchTracker
from devdox_ai_locust.utils.metadata_manager import MetadataManager
from .schemas.processing_result import SwaggerProcessingRequest

console = Console(force_terminal=True)  # Force terminal mode for immediate output


class ProgressDisplay:
    """Manages live progress display for generation"""

    PHASE_ICONS = {
        ProgressPhase.INITIALIZING: "🔧",
        ProgressPhase.PARSING_SCHEMA: "📖",
        ProgressPhase.GENERATING_TEMPLATES: "📝",
        ProgressPhase.ANALYZING_CODEBASE: "🔍",
        ProgressPhase.ENHANCING_LOCUSTFILE: "🤖",
        ProgressPhase.ENHANCING_TEST_DATA: "🤖",
        ProgressPhase.ENHANCING_VALIDATION: "🤖",
        ProgressPhase.ENHANCING_DOMAIN_FLOWS: "🤖",
        ProgressPhase.ENHANCING_WORKFLOWS: "🤖",
        ProgressPhase.MERGING_CODE: "🔀",
        ProgressPhase.VALIDATING_OUTPUT: "✅",
        ProgressPhase.WRITING_FILES: "💾",
        ProgressPhase.FINALIZING: "📦",
        ProgressPhase.COMPLETE: "🎉",
        ProgressPhase.FAILED: "❌",
    }

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.fields[detail]}"),
            console=console,
            transient=False,
        )
        self.main_task: Optional[TaskID] = None
        self.current_status = ""
        self.completed_phases: List[str] = []

    def start(self) -> None:
        """Start the progress display"""
        self.main_task = self.progress.add_task(
            "Generating tests...",
            total=100,
            detail="Initializing..."
        )
        self.progress.start()

    def stop(self) -> None:
        """Stop the progress display"""
        self.progress.stop()

    async def update(self, status: ProgressStatus) -> None:
        """Update the progress display with new status"""
        if self.main_task is None:
            return

        icon = self.PHASE_ICONS.get(status.phase, "⏳")
        message = f"{icon} {status.message}"

        phase_progress = {
            ProgressPhase.INITIALIZING: 5,
            ProgressPhase.GENERATING_TEMPLATES: 15,
            ProgressPhase.ANALYZING_CODEBASE: 20,
            ProgressPhase.ENHANCING_LOCUSTFILE: 35,
            ProgressPhase.ENHANCING_TEST_DATA: 50,
            ProgressPhase.ENHANCING_VALIDATION: 60,
            ProgressPhase.ENHANCING_DOMAIN_FLOWS: 70,
            ProgressPhase.ENHANCING_WORKFLOWS: 85,
            ProgressPhase.VALIDATING_OUTPUT: 90,
            ProgressPhase.WRITING_FILES: 95,
            ProgressPhase.COMPLETE: 100,
            ProgressPhase.FAILED: 100,
        }

        progress_value = phase_progress.get(status.phase, 0)
        detail = status.detail or ""

        if status.is_ai_call:
            detail = f"[yellow]AI[/yellow] {detail}"

        self.progress.update(
            self.main_task,
            completed=progress_value,
            description=message,
            detail=detail,
        )

        if status.phase not in [ProgressPhase.COMPLETE, ProgressPhase.FAILED]:
            phase_str = f"{icon} {status.message}"
            if phase_str not in self.completed_phases:
                self.completed_phases.append(phase_str)


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

    _show_generated_files(created_files, verbose, output_dir)

    if not dry_run:
        _show_run_instructions(output_dir, users, spawn_rate, run_time, host)


def _show_generated_files(created_files: List[Dict[Any, Any]], verbose: bool, output_dir: Optional[Path] = None) -> None:
    """Display list of generated files in a clean format"""
    from rich.tree import Tree
    from rich.text import Text

    # Get output directory name for stripping from paths
    output_dir_name = str(output_dir.name) if output_dir else ""

    # Extract just the file paths (relative to output directory)
    file_paths = []
    for f in created_files:
        if isinstance(f, dict):
            path = f.get("path", str(f))
            if isinstance(path, str):
                # Normalize path separators
                path = path.replace("\\", "/")
                # Strip output directory prefix if present
                if output_dir_name:
                    # Find and remove output_dir_name from path
                    parts = path.split("/")
                    for i, part in enumerate(parts):
                        if part == output_dir_name:
                            path = "/".join(parts[i+1:])
                            break
            file_paths.append(path)
        else:
            file_paths.append(str(f))

    # Group files by directory
    file_tree: Dict[str, List[str]] = {}
    root_files = []

    for fp in file_paths:
        fp = str(fp)  # Ensure it's a string
        if "/" in fp:
            parts = fp.split("/")
            dir_name = parts[0]
            file_name = "/".join(parts[1:])
            if dir_name not in file_tree:
                file_tree[dir_name] = []
            file_tree[dir_name].append(file_name)
        else:
            root_files.append(fp)

    console.print()
    console.print(f"[bold green]📦 Generated {len(created_files)} files:[/bold green]")

    if verbose or len(created_files) <= 15:
        # Show tree structure with actual output directory name
        tree_root = f"📁 [bold]{output_dir_name}/[/bold]" if output_dir_name else "📁 [bold]output/[/bold]"
        tree = Tree(tree_root)

        # Add root files first
        for f in sorted(root_files):
            icon = "🐍" if f.endswith(".py") else ("📄" if f.endswith(".txt") else "📋")
            tree.add(f"{icon} {f}")

        # Add directories
        for dir_name in sorted(file_tree.keys()):
            dir_branch = tree.add(f"📁 [cyan]{dir_name}/[/cyan]")
            for f in sorted(file_tree[dir_name])[:5]:  # Limit files shown per dir
                icon = "🐍" if f.endswith(".py") else "📄"
                dir_branch.add(f"{icon} {f}")
            if len(file_tree[dir_name]) > 5:
                dir_branch.add(f"[dim]... and {len(file_tree[dir_name]) - 5} more[/dim]")

        console.print(tree)
    else:
        # Summary view for many files
        console.print(f"  📄 Root files: {len(root_files)}")
        for dir_name, files in sorted(file_tree.items()):
            console.print(f"  📁 {dir_name}/: {len(files)} files")
        console.print("\n[dim]Use --verbose to see all file names[/dim]")


def _show_run_instructions(
    output_dir: Path, users: int, spawn_rate: float, run_time: str, host: Optional[str]
) -> None:
    """Display instructions for running the generated tests"""

    default_host = host or "http://localhost:8000"
    locustfile = output_dir / "locustfile.py"

    if locustfile.exists():
        main_file = "locustfile.py"
    else:
        py_files = list(output_dir.glob("*.py"))
        main_file = py_files[0].name if py_files else "generated_test.py"

    console.print()
    console.print("[bold green]🚀 Next Steps[/bold green]")
    console.print("──────────────────────────────────────────────────")
    console.print("\n[cyan]1) Prepare your environment[/cyan]")
    console.print(f"  • cd {output_dir}")
    console.print(f"  • pip install -r requirements.txt")

    console.print("\n[cyan]2) Choose how you want to run[/cyan]")
    console.print("  Option A — Launch with Locust web UI")
    console.print(
        "    • "
        + (
            "locust -f {main_file} --users {users} --spawn-rate {spawn_rate} "
            "--run-time {run_time} --host {default_host}"
        ).format(
            main_file=main_file,
            users=users,
            spawn_rate=spawn_rate,
            run_time=run_time,
            default_host=default_host,
        )
    )

    console.print()
    console.print("  Option B — Run headless via devdox CLI")
    console.print(
        f"    • devdox_ai_locust run {output_dir}/{main_file} --host {default_host}"
    )

    console.print("\n[bold red]Heads up[/bold red]")
    console.print(
        "  [yellow]Copy .env.example to .env and configure your API credentials before running tests[/yellow]"
    )


def _is_url(source: str) -> bool:
    """Check if the source is a URL or a file path"""
    source = source.strip()
    return source.startswith(('http://', 'https://'))


async def _process_api_schema(
    swagger_source: str, verbose: bool
) -> Tuple[Dict[str, Any], List[Endpoint], Dict[str, Any]]:
    """Fetch and parse API schema from URL or file path"""

    # Determine if source is URL or file path
    is_url = _is_url(swagger_source)

    # Create appropriate request based on source type
    if is_url:
        source_request = SwaggerProcessingRequest(swagger_url=swagger_source)
        source_type = "URL"
    else:
        source_request = SwaggerProcessingRequest(swagger_path=swagger_source)
        source_type = "file"

    api_schema = None
    with console.status(f"[bold green]Fetching API schema from {source_type}..."):
        try:
            async with asyncio.timeout(30):
                api_schema = await get_api_schema(source_request)

                if not api_schema:
                    console.print("[red]✗[/red] Failed to fetch API schema")
                    sys.exit(1)

        except asyncio.TimeoutError:
            console.print("[red]✗[/red] Timeout while fetching API schema")
            sys.exit(1)
        except FileNotFoundError as e:
            console.print(f"[red]✗[/red] File not found: {e}")
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


async def _generate_modular_tests(
    api_key: str,
    endpoints: List[Endpoint],
    schemas: Dict[str, Any],
    api_info: Dict[str, Any],
    output_dir: Path,
    host: Optional[str] = "http://localhost",
    auth: bool = True,
    db_type: str = "",
    retry_on_invalid: int = 0,
    enable_patch_tracking: bool = True,
    custom_requirement: Optional[str] = None,
) -> List[Dict[Any, Any]]:
    """Generate tests using the SOLID-based modular generator.

    This generator produces focused, single-responsibility files
    that are easier to maintain and enhance.
    """
    from rich.text import Text
    from rich.table import Table
    import time as time_module

    # Phase icons and descriptions for visual feedback
    PHASE_INFO = {
        "INIT": ("🔧", "Initializing", "cyan"),
        "ANALYZE": ("🔍", "Analyzing API", "yellow"),
        "SETUP": ("📁", "Setting up", "blue"),
        "CONTEXT": ("📋", "Building context", "blue"),
        "STATIC": ("📝", "Static files", "magenta"),
        "AI": ("🤖", "AI Enhancement", "yellow"),
        "WRITE": ("💾", "Writing files", "blue"),
        "COMPLETE": ("✅", "Complete", "green"),
        "ERROR": ("❌", "Error", "red"),
    }

    # Current progress state
    start_gen_time = time_module.time()
    progress_state = {
        "phase": "INIT",
        "message": "Initializing...",
        "detail": "",
        "progress": 0,
        "files_written": 0,
        "ai_calls": 0,
        "last_printed_phase": "",
    }

    status: Optional[Status] = None

    def progress_callback(phase: str, message: str, detail: str, pct: int) -> None:
        """Callback for ModularGenerator progress updates - prints to console."""
        nonlocal status
        progress_state["phase"] = phase
        progress_state["message"] = message
        progress_state["detail"] = detail
        progress_state["progress"] = pct
        icon, _, color = PHASE_INFO.get(phase, ("⏳", "", "white"))

        if status is None:
            status = console.status("[cyan]Preparing generation...[/cyan]", spinner="dots")
            status.start()

        if status:
            status_text = f"[{color}]{message}[/]"
            if detail:
                status_text += f" [dim]→ {detail}[/dim]"
            status.update(status=status_text)

        # Track AI calls and file writes
        if phase == "AI":
            progress_state["ai_calls"] += 1
        elif phase == "WRITE" and detail:
            progress_state["files_written"] += 1
            # Only print every 10th file to reduce noise
            if progress_state["files_written"] % 10 == 0:
                console.print(f"[dim]   💾 Written {progress_state['files_written']} files...[/dim]")
                sys.stdout.flush()  # Force immediate display on Windows
            return

        # Print phase updates (avoid duplicate phase messages)
        if phase != progress_state["last_printed_phase"] or phase in ("AI", "ERROR", "COMPLETE"):
            progress_state["last_printed_phase"] = phase

            # Format the message
            elapsed = time_module.time() - start_gen_time
            time_str = f"[dim][{elapsed:5.1f}s][/dim]"

            if phase == "COMPLETE":
                console.print(f"{time_str} [{color}]{icon} {message}[/{color}]")
            elif phase == "ERROR":
                console.print(f"{time_str} [{color}]{icon} {message}: {detail}[/{color}]")
            elif detail:
                console.print(f"{time_str} [{color}]{icon} {message}[/{color}] [dim]→ {detail}[/dim]")
            else:
                console.print(f"{time_str} [{color}]{icon} {message}[/{color}]")

            # Force immediate display on Windows (buffering can delay output)
            sys.stdout.flush()

    # Print initial configuration using a simple, dash-separated section (no boxes)
    console.print()
    console.print("[bold cyan]⚙️ Configuration[/bold cyan]")
    console.print("[dim]" + "─" * 50 + "[/dim]")

    def _conf_line(label: str, value: str, icon: str = "") -> None:
        padded = label.ljust(12)
        console.print(f"{icon} [bold]{padded}[/bold] {value}")

    _conf_line("Output", str(output_dir), "📁")
    _conf_line("Host", host or "http://localhost", "🌐")
    _conf_line("Endpoints", str(len(endpoints)), "📡")
    if db_type:
        _conf_line("Database", db_type, "🗄️")

    # Get security information from OpenAPI spec
    global_security = api_info.get("global_security", [])
    security_schemes = api_info.get("security_schemes", {})

    # Find secured endpoints using OpenAPI security specification
    secured_endpoints = []
    public_endpoints = []
    for ep in endpoints:
        if ep.requires_auth(global_security):
            secured_endpoints.append(ep.path)
        else:
            public_endpoints.append(ep.path)

    # Add security info to config section
    if security_schemes:
        scheme_names = ", ".join(security_schemes.keys())
        _conf_line("Security", scheme_names, "🔐")
    _conf_line("Secured", f"{len(secured_endpoints)} endpoints", "🔒")
    _conf_line("Public", f"{len(public_endpoints)} endpoints", "🔓")

    console.print()

    # For backwards compatibility, pass secured endpoint paths as auth_endpoints
    auth_endpoints = secured_endpoints

    # Initialize metadata manager and patch tracker
    metadata_manager = MetadataManager(output_dir)

    # Initialize session with API info - this populates metadata.json properly
    swagger_source = api_info.get("swagger_source", "")
    source_type = api_info.get("source_type", "url")
    metadata_manager.initialize_session(
        api_info=api_info,
        swagger_source=swagger_source,
        source_type=source_type,
    )

    # Update generation config
    metadata_manager.update_generation_config(
        host=host,
        auth_enabled=auth,
        db_type=db_type,
    )
    metadata_manager.update_api_endpoints_count(len(endpoints))

    # Initialize patch tracker with metadata manager for proper WAL tracking
    patch_tracker: Optional[PatchTracker] = None
    if enable_patch_tracking:
        patch_tracker = PatchTracker.from_metadata_manager(metadata_manager)
        patch_tracker.start_session()

    # Print generation header
    console.print()
    console.print("[bold blue]🚀 Generating Load Tests[/bold blue]")
    console.print("[dim]─" * 50 + "[/dim]")
    sys.stdout.flush()  # Ensure header is displayed before generation starts

    # Create modular generator with progress callback and patch tracker
    progress_callback("INIT", "Initializing generator", "", 0)
    try:
        generator = ModularGenerator(
            output_dir=str(output_dir),
            api_key=api_key,
            target_host=host or "http://localhost",
            auth_enabled=auth,
            db_type=db_type,
            retry_on_invalid=retry_on_invalid,
            progress_callback=progress_callback,
            patch_tracker=patch_tracker,  # Pass tracker for WAL-style patching
            custom_requirement=custom_requirement,
        )
    except Exception as e:
        if status:
            status.stop()
        console.print(f"[red]✗ Failed to initialize ModularGenerator: {e}[/red]")
        raise

    # Generate all modular files
    generated_files = {}
    generation_error = None

    try:
        generated_files = await generator.generate(
            endpoints=endpoints,
            schemas=schemas,
            api_info=api_info,
            auth_endpoints=auth_endpoints if auth else None,
        )
    except Exception as e:
        generation_error = e
    finally:
        if status:
            status.stop()

    # Print separator and final summary
    console.print("[dim]─" * 50 + "[/dim]")

    if generation_error:
        console.print(f"[red]❌ Generation failed: {generation_error}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        raise generation_error

    # Show completion stats
    elapsed = time_module.time() - start_gen_time
    console.print(f"[green]✅ Generated {len(generated_files)} files in {elapsed:.1f}s[/green]")
    console.print(f"[dim]   🤖 {progress_state['ai_calls']} AI calls  │  📄 {progress_state['files_written']} files written[/dim]")

    if not generated_files:
        console.print("[yellow]⚠ Warning: No files were generated![/yellow]")

    # Convert to format expected by _show_results
    created_files = []
    for file_path, content in generated_files.items():
        full_path = output_dir / file_path
        created_files.append({
            "path": str(full_path),
            "content": content,
            "type": "modular",
        })
        # Register file in metadata manager
        metadata_manager.register_file(file_path, content)

    # Finalize patch tracking
    if patch_tracker:
        summary = patch_tracker.get_summary()
        patch_tracker.finalize()
        session_id = summary.get('session_id', '')
        total = summary.get('total_patches', 0)
        if total > 0:
            console.print(f"[blue]📋 Patches saved to: .devdox_ai_locust/{session_id}/.patches/ ({total} patches)[/blue]")

    # Finalize metadata
    metadata_manager.finalize_session()
    console.print("[blue]📄 Metadata saved to: .devdox_ai_locust/metadata.json[/blue]")

    return created_files


@click.group()
@click.version_option(version="0.1.9")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """DevDox AI LoadTest - Generate Locust tests from API documentation"""
    import logging

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug

    if verbose:
        console.print("[green]Verbose mode enabled[/green]")

    if debug:
        # Enable debug logging for ModularGenerator and related modules
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.getLogger("devdox_ai_locust").setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled - see detailed output[/yellow]")


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
    "--retry-on-invalid",
    type=int,
    default=0,
    help="Number of retries if AI generates invalid code (default: 0, no retry)",
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
    retry_on_invalid: int,
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
                retry_on_invalid,
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
    retry_on_invalid: int = 0,
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

        schemas, endpoints, api_info = await _process_api_schema(
            swagger_url, ctx.obj["verbose"]
        )

        # Use SOLID-based modular generator for reliable output
        created_files = await _generate_modular_tests(
            api_key=api_key,
            endpoints=endpoints,
            schemas=schemas,
            api_info=api_info,
            output_dir=output_dir,
            host=host,
            auth=auth,
            db_type=db_type,
            retry_on_invalid=retry_on_invalid,
            custom_requirement=custom_requirement,
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


def main() -> None:
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
