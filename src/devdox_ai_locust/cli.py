import click
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Union, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.status import Status

from .modular_generator import ModularGenerator
from .config import Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from devdox_ai_locust.utils.patch_tracker import PatchTracker
from devdox_ai_locust.utils.metadata_manager import MetadataManager
from .schemas.processing_result import SwaggerProcessingRequest

console = Console(force_terminal=True)  # Force terminal mode for immediate output

DEFAULT_HOST = "http://localhost"
DEFAULT_API_HOST = "http://localhost:8000"
DIM = "[/dim]"


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


def _normalize_file_paths(
    created_files: List[Dict[Any, Any]],
    output_dir: Optional[Path],
) -> List[str]:
    output_dir_name = str(output_dir.name) if output_dir else ""
    file_paths = []

    for file_entry in created_files:
        path = file_entry
        if isinstance(file_entry, dict):
            path = file_entry.get("path", str(file_entry))
        path_str = str(path)
        path_str = path_str.replace("\\", "/")
        if output_dir_name:
            parts = path_str.split("/")
            for i, part in enumerate(parts):
                if part == output_dir_name:
                    path_str = "/".join(parts[i + 1 :])
                    break
        file_paths.append(path_str)

    return file_paths


def _group_files_by_directory(file_paths: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    file_tree: Dict[str, List[str]] = {}
    root_files: List[str] = []

    for fp in file_paths:
        if "/" in fp:
            dir_name, file_name = fp.split("/", 1)
            file_tree.setdefault(dir_name, []).append(file_name)
        else:
            root_files.append(fp)

    return file_tree, root_files


def _file_icon(filename: str) -> str:
    if filename.endswith(".py"):
        return "🐍"
    if filename.endswith(".txt"):
        return "📄"
    return "📋"


def _render_file_tree(
    output_dir_name: str,
    file_tree: Dict[str, List[str]],
    root_files: List[str],
) -> None:
    from rich.tree import Tree

    tree_root = f"📁 [bold]{output_dir_name}/[/bold]" if output_dir_name else "📁 [bold]output/[/bold]"
    tree = Tree(tree_root)

    for filename in sorted(root_files):
        tree.add(f"{_file_icon(filename)} {filename}")

    for dir_name in sorted(file_tree.keys()):
        dir_branch = tree.add(f"📁 [cyan]{dir_name}/[/cyan]")
        for filename in sorted(file_tree[dir_name])[:5]:
            icon = "🐍" if filename.endswith(".py") else "📄"
            dir_branch.add(f"{icon} {filename}")
        extra_count = len(file_tree[dir_name]) - 5
        if extra_count > 0:
            dir_branch.add(f"[dim]... and {extra_count} more{DIM}")

    console.print(tree)


def _show_generated_files(
    created_files: List[Dict[Any, Any]],
    verbose: bool,
    output_dir: Optional[Path] = None,
) -> None:
    """Display list of generated files in a clean format"""
    output_dir_name = str(output_dir.name) if output_dir else ""
    file_paths = _normalize_file_paths(created_files, output_dir)
    file_tree, root_files = _group_files_by_directory(file_paths)

    console.print()
    console.print(f"[bold green]📦 Generated {len(created_files)} files:[/bold green]")

    if verbose or len(created_files) <= 15:
        _render_file_tree(output_dir_name, file_tree, root_files)
        return

    console.print(f"  📄 Root files: {len(root_files)}")
    for dir_name, files in sorted(file_tree.items()):
        console.print(f"  📁 {dir_name}/: {len(files)} files")
    console.print(f"\n[dim]Use --verbose to see all file names{DIM}")


def _show_run_instructions(
    output_dir: Path, users: int, spawn_rate: float, run_time: str, host: Optional[str]
) -> None:
    """Display instructions for running the generated tests"""

    default_host = host or DEFAULT_API_HOST
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
    console.print("  • pip install -r requirements.txt")

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


def _collect_security_info(
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    global_security = api_info.get("global_security", [])
    security_schemes = api_info.get("security_schemes", {})

    secured_endpoints = []
    public_endpoints = []
    for ep in endpoints:
        if ep.requires_auth(global_security):
            secured_endpoints.append(ep.path)
        else:
            public_endpoints.append(ep.path)

    return secured_endpoints, public_endpoints, security_schemes


def _print_generation_config(
    output_dir: Path,
    host: Optional[str],
    endpoints: List[Endpoint],
    db_type: str,
    security_schemes: Dict[str, Any],
    secured_endpoints: List[str],
    public_endpoints: List[str],
) -> None:
    console.print()
    console.print("[bold cyan]⚙️ Configuration[/bold cyan]")
    console.print("[dim]" + "─" * 50 + DIM)

    def _conf_line(label: str, value: str, icon: str = "") -> None:
        padded = label.ljust(12)
        console.print(f"{icon} [bold]{padded}[/bold] {value}")

    _conf_line("Output", str(output_dir), "📁")
    _conf_line("Host", host or DEFAULT_HOST, "🌐")
    _conf_line("Endpoints", str(len(endpoints)), "📡")
    if db_type:
        _conf_line("Database", db_type, "🗄️")

    if security_schemes:
        scheme_names = ", ".join(security_schemes.keys())
        _conf_line("Security", scheme_names, "🔐")
    _conf_line("Secured", f"{len(secured_endpoints)} endpoints", "🔒")
    _conf_line("Public", f"{len(public_endpoints)} endpoints", "🔓")

    console.print()


def _init_metadata_and_patch_tracking(
    output_dir: Path,
    api_info: Dict[str, Any],
    host: Optional[str],
    auth: bool,
    db_type: str,
    endpoints: List[Endpoint],
    enable_patch_tracking: bool,
) -> Tuple[MetadataManager, Optional[PatchTracker]]:
    metadata_manager = MetadataManager(output_dir)
    swagger_source = api_info.get("swagger_source", "")
    source_type = api_info.get("source_type", "url")
    metadata_manager.initialize_session(
        api_info=api_info,
        swagger_source=swagger_source,
        source_type=source_type,
    )

    metadata_manager.update_generation_config(
        host=host,
        auth_enabled=auth,
        db_type=db_type,
    )
    metadata_manager.update_api_endpoints_count(len(endpoints))

    patch_tracker: Optional[PatchTracker] = None
    if enable_patch_tracking:
        patch_tracker = PatchTracker.from_metadata_manager(metadata_manager)
        patch_tracker.start_session()

    return metadata_manager, patch_tracker


def _build_created_files(
    generated_files: Dict[str, str],
    output_dir: Path,
    metadata_manager: MetadataManager,
) -> List[Dict[Any, Any]]:
    created_files: List[Dict[Any, Any]] = []
    for file_path, content in generated_files.items():
        full_path = output_dir / file_path
        created_files.append({
            "path": str(full_path),
            "content": content,
            "type": "modular",
        })
        metadata_manager.register_file(file_path, content)
    return created_files


def _finalize_patch_tracking(patch_tracker: Optional[PatchTracker]) -> None:
    if not patch_tracker:
        return
    summary = patch_tracker.get_summary()
    patch_tracker.finalize()
    session_id = summary.get('session_id', '')
    total = summary.get('total_patches', 0)
    if total > 0:
        console.print(
            f"[blue]📋 Patches saved to: .devdox_ai_locust/{session_id}/.patches/ ({total} patches)[/blue]"
        )


class _ProgressReporter:
    def __init__(self, phase_info: Dict[str, Tuple[str, str, str]]):
        self.phase_info = phase_info
        self.start_gen_time = None
        self.status: Optional[Status] = None
        self.state = {
            "phase": "INIT",
            "message": "Initializing...",
            "detail": "",
            "progress": 0,
            "files_written": 0,
            "ai_calls": 0,
            "last_printed_phase": "",
        }

    def start(self) -> None:
        import time as time_module

        self.start_gen_time = time_module.time()

    def stop(self) -> None:
        if self.status:
            self.status.stop()

    def progress_callback(self, phase: str, message: str, detail: str, pct: int) -> None:
        if self.start_gen_time is None:
            self.start()

        self.state["phase"] = phase
        self.state["message"] = message
        self.state["detail"] = detail
        self.state["progress"] = pct

        self._update_status_text(phase, message, detail)

        if self._track_progress_metrics(phase, detail):
            return

        if phase != self.state["last_printed_phase"] or phase in ("AI", "ERROR", "COMPLETE"):
            self.state["last_printed_phase"] = phase
            self._print_phase_update(phase, message, detail)
            sys.stdout.flush()

    def _update_status_text(self, phase: str, message: str, detail: str) -> None:
        _, _, color = self.phase_info.get(phase, ("⏳", "", "white"))

        if self.status is None:
            self.status = console.status("[cyan]Preparing generation...[/cyan]", spinner="dots")
            self.status.start()

        status_text = f"[{color}]{message}[/]"
        if detail:
            status_text += f" [dim]→ {detail}{DIM}"
        self.status.update(status=status_text)

    def _print_phase_update(self, phase: str, message: str, detail: str) -> None:
        import time as time_module

        icon, _, color = self.phase_info.get(phase, ("⏳", "", "white"))
        elapsed = time_module.time() - (self.start_gen_time or 0)
        time_str = f"[dim][{elapsed:5.1f}s]{DIM}"
        if phase == "COMPLETE":
            console.print(f"{time_str} [{color}]{icon} {message}[/{color}]")
            return
        if phase == "ERROR":
            console.print(f"{time_str} [{color}]{icon} {message}: {detail}[/{color}]")
            return
        if detail:
            console.print(f"{time_str} [{color}]{icon} {message}[/{color}] [dim]→ {detail}{DIM}")
            return
        console.print(f"{time_str} [{color}]{icon} {message}[/{color}]")

    def _track_progress_metrics(self, phase: str, detail: str) -> bool:
        if phase == "AI":
            self.state["ai_calls"] += 1
            return False
        if phase == "WRITE" and detail:
            self.state["files_written"] += 1
            if self.state["files_written"] % 10 == 0:
                console.print(f"[dim]   💾 Written {self.state['files_written']} files...{DIM}")
                sys.stdout.flush()
            return True
        return False


def _print_generation_header() -> None:
    console.print()
    console.print("[bold blue]🚀 Generating Load Tests[/bold blue]")
    console.print("[dim]─" * 50 + DIM)
    sys.stdout.flush()


def _create_modular_generator(
    output_dir: Path,
    api_key: str,
    host: Optional[str],
    auth: bool,
    db_type: str,
    retry_on_invalid: int,
    progress_callback,
    patch_tracker: Optional[PatchTracker],
    custom_requirement: Optional[str],
    progress_reporter: _ProgressReporter,
) -> ModularGenerator:
    progress_callback("INIT", "Initializing generator", "", 0)
    try:
        return ModularGenerator(
            output_dir=str(output_dir),
            api_key=api_key,
            target_host=host or DEFAULT_HOST,
            auth_enabled=auth,
            db_type=db_type,
            retry_on_invalid=retry_on_invalid,
            progress_callback=progress_callback,
            patch_tracker=patch_tracker,
            custom_requirement=custom_requirement,
        )
    except Exception as e:
        progress_reporter.stop()
        console.print(f"[red]✗ Failed to initialize ModularGenerator: {e}[/red]")
        raise


async def _execute_generation(
    generator: ModularGenerator,
    endpoints: List[Endpoint],
    schemas: Dict[str, Any],
    api_info: Dict[str, Any],
    auth_endpoints: List[str],
    auth: bool,
    progress_reporter: _ProgressReporter,
) -> Dict[str, str]:
    try:
        return await generator.generate(
            endpoints=endpoints,
            schemas=schemas,
            api_info=api_info,
            auth_endpoints=auth_endpoints if auth else None,
        )
    except Exception as e:
        console.print(f"[red]❌ Generation failed: {e}[/red]")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
        raise
    finally:
        progress_reporter.stop()


def _print_generation_summary(
    progress_reporter: _ProgressReporter,
    generated_files: Dict[str, str],
) -> None:
    import time as time_module

    elapsed = time_module.time() - (progress_reporter.start_gen_time or 0)
    console.print(f"[green]✅ Generated {len(generated_files)} files in {elapsed:.1f}s[/green]")
    console.print(
        f"[dim]   🤖 {progress_reporter.state['ai_calls']} AI calls  │  "
        f"📄 {progress_reporter.state['files_written']} files written{DIM}"
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
    host: Optional[str] = DEFAULT_HOST,
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

    progress_reporter = _ProgressReporter(PHASE_INFO)
    progress_callback = progress_reporter.progress_callback

    secured_endpoints, public_endpoints, security_schemes = _collect_security_info(
        endpoints,
        api_info,
    )
    _print_generation_config(
        output_dir,
        host,
        endpoints,
        db_type,
        security_schemes,
        secured_endpoints,
        public_endpoints,
    )

    auth_endpoints = secured_endpoints

    metadata_manager, patch_tracker = _init_metadata_and_patch_tracking(
        output_dir=output_dir,
        api_info=api_info,
        host=host,
        auth=auth,
        db_type=db_type,
        endpoints=endpoints,
        enable_patch_tracking=enable_patch_tracking,
    )

    _print_generation_header()

    generator = _create_modular_generator(
        output_dir=output_dir,
        api_key=api_key,
        host=host,
        auth=auth,
        db_type=db_type,
        retry_on_invalid=retry_on_invalid,
        progress_callback=progress_callback,
        patch_tracker=patch_tracker,
        custom_requirement=custom_requirement,
        progress_reporter=progress_reporter,
    )

    generated_files = await _execute_generation(
        generator=generator,
        endpoints=endpoints,
        schemas=schemas,
        api_info=api_info,
        auth_endpoints=auth_endpoints,
        auth=auth,
        progress_reporter=progress_reporter,
    )

    console.print("[dim]─" * 50 + DIM)
    _print_generation_summary(progress_reporter, generated_files)

    if not generated_files:
        console.print("[yellow]⚠ Warning: No files were generated![/yellow]")

    created_files = _build_created_files(generated_files, output_dir, metadata_manager)

    _finalize_patch_tracking(patch_tracker)

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
