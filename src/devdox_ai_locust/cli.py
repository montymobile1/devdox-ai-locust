import click
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from together import Together

from .hybrid_loctus_generator import HybridLocustGenerator
from .config import Settings
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser
from .schemas.processing_result import SwaggerProcessingRequest

console = Console()


def _initialize_config(config, together_api_key):
    """Initialize configuration and validate API key"""
    config_obj = Settings()
    if config:
        config_obj.load_from_file(config)

    api_key = together_api_key or config_obj.API_KEY
    if not api_key:
        console.print(
            "[red]Error:[/red] Together AI API key is required. Set TOGETHER_API_KEY environment variable or use --together-api-key")
        sys.exit(1)

    return config_obj, api_key


def _setup_output_directory(output):
    """Create and return output directory path"""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _display_configuration(swagger_url, output_dir, users, spawn_rate,
                                   run_time, host, auth, custom_requirement, dry_run, timeout):

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
            table.add_row("Timeout", f"{timeout}s")

            console.print(table)


def _show_results(created_files, output_dir, start_time, verbose, dry_run,
                  users, spawn_rate, run_time, host):
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


def _show_generated_files(created_files, verbose):
    """Display list of generated files"""
    if verbose or len(created_files) <= 10:
        console.print("\n[bold]Generated files:[/bold]")
        for file_path in created_files:
            console.print(f"  • {file_path}")
    else:
        console.print(f"\n[bold]Generated {len(created_files)} files[/bold]")
        console.print("Use --verbose to see all file names")


def _show_run_instructions(output_dir, users, spawn_rate, run_time, host):
    """Display instructions for running the generated tests"""
    console.print("\n[bold]To run tests:[/bold]")
    console.print(f"  cd {output_dir}")

    default_host = host or 'http://localhost:8000'
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
    console.print(f"  devdox_ai_locust run {output_dir}/{main_file} --host {default_host}")



async def _process_api_schema(swagger_url, timeout, verbose):
    """Fetch and parse API schema"""
    # Fetch API schema
    source_request = SwaggerProcessingRequest(swagger_url=swagger_url)

    is_url = swagger_url.startswith(('http://', 'https://'))
    with console.status(f"[bold green]Fetching API schema from {'URL' if is_url else 'file'}..."):
        api_schema = await get_api_schema(source_request, timeout=timeout)

    if not api_schema:
        console.print("[red]✗[/red] Failed to fetch API schema")
        sys.exit(1)

    console.print(f"[green]✓[/green] Successfully fetched API schema ({len(api_schema)} characters)")

    # Parse schema
    with console.status("[bold green]Parsing API schema..."):
        parser = OpenAPIParser()
        try:
            schema_data = parser.parse_schema(api_schema)
            if verbose:
                console.print("✓ Schema data parsed successfully")

            endpoints = parser.parse_endpoints()
            api_info = parser.get_schema_info()

            console.print(f"[green]📋 Parsed {len(endpoints)} endpoints from {api_info.get('title', 'API')}[/green]")
            return schema_data, endpoints, api_info

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to parse API schema: {e}")
            sys.exit(1)


async def _generate_and_create_tests(api_key, endpoints, api_info, output_dir,
                                     custom_requirement, host, auth):
    """Generate tests using AI and create test files"""
    together_client = Together(api_key=api_key)

    with console.status("[bold green]Generating Locust tests with AI..."):
        generator = HybridLocustGenerator(ai_client=together_client)

        # Build generation kwargs
        generation_kwargs = {}
        if custom_requirement:
            generation_kwargs['custom_requirement'] = custom_requirement
        if host:
            generation_kwargs['target_host'] = host
        if not auth:
            generation_kwargs['include_auth'] = False

        test_files, test_directories = await generator.generate_from_endpoints(
            endpoints=endpoints,
            api_info=api_info,
            output_dir=str(output_dir),
            **generation_kwargs
        )

    # Create test files
    with console.status("[bold green]Creating test files..."):
        created_files = []

        # Create workflow files
        if test_directories:
            workflows_dir = output_dir / "workflows"
            workflows_dir.mkdir(exist_ok=True)
            for file_workflow in test_directories:
                workflow_files = await generator._create_test_files_safely(file_workflow, workflows_dir)
                created_files.extend(workflow_files)

        # Create main test files
        if test_files:
            main_files = await generator._create_test_files_safely(test_files, output_dir)
            created_files.extend(main_files)

    return created_files


@click.group()
@click.version_option(version="0.1.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """DevDox AI LoadTest - Generate Locust tests from API documentation"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    if verbose:
        console.print("[green]Verbose mode enabled[/green]")


@cli.command()
@click.argument('swagger_url')  # Can be URL or file path
@click.option('--output', '-o', type=click.Path(), default="output",
              help='Output directory for generated tests (default: output)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--users', '-u', type=int, default=10, help='Number of simulated users')
@click.option('--spawn-rate', '-r', type=float, default=2, help='Rate to spawn users (users per second)')
@click.option('--run-time', '-t', type=str, default='5m', help='Test run time (e.g., 5m, 1h)')
@click.option('--host', '-H', type=str, help='Target host URL')
@click.option('--auth/--no-auth', default=True, help='Include authentication in tests')
@click.option('--dry-run', is_flag=True, help='Generate tests without running them')
@click.option('--custom-requirement', type=str, help='Custom requirements for test generation')
@click.option('--together-api-key', type=str, envvar='TOGETHER_API_KEY',
              help='Together AI API key (can also be set via TOGETHER_API_KEY env var)')
@click.option('--timeout', type=int, default=30, help='Timeout for URL requests (seconds)')
@click.pass_context
def generate(ctx, swagger_url, output, config,  users, spawn_rate, run_time, host, auth, dry_run,
             custom_requirement, together_api_key, timeout):
    """Generate Locust test files from API documentation URL or file"""

    try:
        # Run the async generation
        asyncio.run(_async_generate(
            ctx, swagger_url, output, config,  users, spawn_rate,
            run_time, host, auth, dry_run, custom_requirement, together_api_key, timeout
        ))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


async def _async_generate(ctx, swagger_url, output, config,  users, spawn_rate,
                          run_time, host, auth, dry_run, custom_requirement, together_api_key, timeout):
    """Async function to handle the generation process"""

    start_time = datetime.now(timezone.utc)

    try:

        _, api_key = _initialize_config(config, together_api_key)
        output_dir = _setup_output_directory(output)



        # Display configuration
        if ctx.obj['verbose']:
            _display_configuration(swagger_url, output_dir, users, spawn_rate,
                                   run_time, host, auth, custom_requirement, dry_run, timeout)

        _, endpoints, api_info = await _process_api_schema(
            swagger_url, timeout, ctx.obj['verbose']
        )

        created_files = await _generate_and_create_tests(
            api_key, endpoints, api_info, output_dir, custom_requirement, host, auth
        )

        # Show results
        _show_results(created_files, output_dir, start_time, ctx.obj['verbose'],
                      dry_run, users, spawn_rate, run_time, host)



    except Exception as e:
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()
        console.print(f"[red]✗[/red] Generation failed after {processing_time:.2f}s: {e}")
        raise


@cli.command()
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--users', '-u', type=int, default=10, help='Number of simulated users')
@click.option('--spawn-rate', '-r', type=float, default=2, help='Rate to spawn users')
@click.option('--run-time', '-t', type=str, default='5m', help='Test run time')
@click.option('--host', '-H', type=str, required=True, help='Target host URL')
@click.option('--headless', is_flag=True, help='Run in headless mode (no web UI)')
@click.pass_context
def run(ctx, test_file, users, spawn_rate, run_time, host, headless):
    """Run generated Locust tests"""

    try:
        import subprocess

        cmd = [
            "locust",
            "-f", str(test_file),
            "--users", str(users),
            "--spawn-rate", str(spawn_rate),
            "--run-time", run_time,
            "--host", host
        ]

        if headless:
            cmd.append("--headless")

        if ctx.obj['verbose']:
            console.print(f"[blue]Running command:[/blue] {' '.join(cmd)}")

        console.print("[green]Starting Locust test...[/green]")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Test execution failed:[/red] {e}")
        sys.exit(1)
    except FileNotFoundError:
        console.print("[red]Locust not found. Please install locust: pip install locust[/red]")
        sys.exit(1)





def main():
    """Main entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()