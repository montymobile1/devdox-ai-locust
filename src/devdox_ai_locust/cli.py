import click
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Union, List, Dict, Any
from rich.console import Console
from rich.table import Table
from together import AsyncTogether

from .ai_config import AIEnhancementConfig
from .config import Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from .schemas.processing_result import SwaggerProcessingRequest

console = Console()


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
    diagnostics: bool = False,
    timeout: int = 120,
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
    )


async def _generate_scenario_based_tests(
    ai_client: AsyncTogether,
    ai_config: AIEnhancementConfig,
    endpoints: List[Endpoint],
    api_info: Dict[str, Any],
    output_dir: Path,
    auth: bool,
    db_type: str,
) -> List[Dict[Any, Any]]:
    """Generate tests using scenario-based approach (positive/negative/security per tag)"""
    from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator, ScenarioType
    from devdox_ai_locust.locust_generator import LocustTestGenerator

    # Group endpoints by tag
    grouped_endpoints: Dict[str, List[Endpoint]] = {}
    for ep in endpoints:
        tag = ep.tags[0] if ep.tags else "default"
        if tag not in grouped_endpoints:
            grouped_endpoints[tag] = []
        grouped_endpoints[tag].append(ep)

    num_tags = len(grouped_endpoints)

    # Show time estimate
    # 3 LLM calls per tag (positive + negative + security)
    total_calls = num_tags * 3
    # Estimate RPM (will be updated after first call)
    estimated_rpm = 60  # Conservative default
    estimated_minutes = total_calls / estimated_rpm

    console.print(f"\n📊 [bold]Scenario-based Generation[/bold]")
    console.print(f"   Tags to process: {num_tags}")
    console.print(f"   API calls needed: {total_calls} (3 per tag)")
    console.print(f"   Estimated time: ~{estimated_minutes:.1f} minutes (at {estimated_rpm} RPM)")
    console.print(f"   Files per tag: positive_workflow.py, negative_workflow.py, security_workflow.py\n")

    # Setup prompt directory
    prompt_dir = Path(__file__).parent / "prompt"

    # Initialize scenario generator
    scenario_gen = ScenarioWorkflowGenerator(
        prompt_dir=prompt_dir,
        ai_client=ai_client,
        ai_config=ai_config,
    )

    # Generate base files first using template generator
    template_gen = LocustTestGenerator()
    base_files, _, _ = template_gen.generate_from_endpoints(
        endpoints, api_info, include_auth=auth, db_type=db_type
    )
    base_files = template_gen.fix_indent(base_files)

    base_workflow_content = base_files.get("base_workflow.py", "")
    test_data_content = base_files.get("test_data.py", "")

    # Get auth endpoints
    auth_endpoints = [ep for ep in endpoints if any(
        kw in ep.path.lower() for kw in ["auth", "login", "token", "session"]
    )]

    created_files = []
    workflows_dir = output_dir / "workflows"

    # Process each tag
    with console.status("[bold green]Generating scenario-based workflows...") as status:
        for idx, (tag_name, tag_endpoints) in enumerate(grouped_endpoints.items(), 1):
            status.update(f"[bold green]Processing tag {idx}/{num_tags}: {tag_name}...")

            # Create tag directory
            tag_dir = workflows_dir / tag_name.lower().replace(" ", "_")
            tag_dir.mkdir(parents=True, exist_ok=True)

            # Generate scenario workflows
            scenarios = await scenario_gen.generate_scenario_workflows(
                tag_name=tag_name,
                endpoints=tag_endpoints,
                base_workflow_content=base_workflow_content,
                test_data_content=test_data_content,
                auth_endpoints=auth_endpoints if auth else None,
            )

            # Save generated files
            for scenario_type, content in scenarios.items():
                if content:
                    filename = scenario_gen.SCENARIO_FILES[scenario_type]
                    file_path = tag_dir / filename
                    file_path.write_text(content, encoding='utf-8')
                    created_files.append({
                        "path": str(file_path),
                        "size": len(content),
                        "tag": tag_name,
                        "scenario": scenario_type.value,
                    })
                    console.print(f"   ✓ {tag_name}/{filename}")

            # Update rate limit info after first call
            if idx == 1:
                rate_info = scenario_gen.get_rate_limit_info()
                if rate_info.requests_per_minute != estimated_rpm:
                    console.print(f"\n📊 Updated rate limit: {rate_info.requests_per_minute} RPM")
                    remaining_calls = (num_tags - 1) * 2
                    new_estimate = remaining_calls / rate_info.requests_per_minute
                    console.print(f"   Revised estimate: ~{new_estimate:.1f} minutes remaining\n")

    # Write base files
    with console.status("[bold green]Creating base files..."):
        for filename, content in base_files.items():
            file_path = output_dir / filename
            file_path.write_text(content, encoding='utf-8')
            created_files.append({
                "path": str(file_path),
                "size": len(content),
            })

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
    "--diagnostics",
    is_flag=True,
    default=False,
    help="Enable diagnostics mode: saves pre/post LLM patches and prompts for debugging",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help="Timeout in seconds for AI API calls (default: 120, increase for large APIs)",
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
    diagnostics: bool,
    timeout: int,
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
                diagnostics,
                timeout,
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
    diagnostics: bool = False,
    timeout: int = 120,
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
            diagnostics,
            timeout,
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
