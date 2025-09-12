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
        # Initialize configuration
        config_obj = Settings()
        if config:
            config_obj.load_from_file(config)
        
        # Determine API key from multiple sources (CLI arg > env var > config)
        api_key = together_api_key or config_obj.api_key
        
        # Validate Together AI API key
        if not api_key:
            console.print(
                "[red]Error:[/red] Together AI API key is required. Set TOGETHER_API_KEY environment variable or use --together-api-key")
            sys.exit(1)

        # Set output directory
        relative_path = Path(output)
        relative_path.mkdir(parents=True, exist_ok=True)

        # Display configuration
        if ctx.obj['verbose']:
            table = Table(title="Generation Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Input Source", str(swagger_url))
            table.add_row("Output Directory", str(relative_path))

            table.add_row("Users", str(users))
            table.add_row("Spawn Rate", str(spawn_rate))
            table.add_row("Run Time", run_time)
            table.add_row("Host", host or "Auto-detect")
            table.add_row("Authentication", "Enabled" if auth else "Disabled")
            table.add_row("Custom Requirement", custom_requirement or "None")
            table.add_row("Dry Run", "Yes" if dry_run else "No")
            table.add_row("Timeout", f"{timeout}s")

            console.print(table)

        # Determine if input is URL or file
        is_url = swagger_url.startswith(('http://', 'https://'))

        # Create SwaggerProcessingRequest
        source_request = SwaggerProcessingRequest(
            swagger_url=swagger_url
        )

        # Fetch API schema
        with console.status(f"[bold green]Fetching API schema from {'URL' if is_url else 'file'}..."):
            api_schema = await get_api_schema(source_request, timeout=timeout)

        if not api_schema:
            console.print("[red]✗[/red] Failed to fetch API schema")
            sys.exit(1)

        console.print(f"[green]✓[/green] Successfully fetched API schema ({len(api_schema)} characters)")

        # Parse schema with OpenAPIParser
        with console.status("[bold green]Parsing API schema..."):
            parser = OpenAPIParser()
            try:
                schema_data = parser.parse_schema(api_schema)
                if ctx.obj['verbose']:
                    console.print("✓ Schema data parsed successfully")

                endpoints = parser.parse_endpoints()
                api_info = parser.get_schema_info()

                console.print(f"[green]📋 Parsed {len(endpoints)} endpoints from {api_info.get('title', 'API')}[/green]")

            except Exception as e:
                console.print(f"[red]✗[/red] Failed to parse API schema: {e}")
                sys.exit(1)

        # Initialize Together AI client
        together_client = Together(api_key=api_key)

        # Generate tests with HybridLocustGenerator
        with console.status("[bold green]Generating Locust tests with AI..."):
            generator = HybridLocustGenerator(ai_client=together_client)

            # Add custom requirement to generator if provided
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
                output_dir=str(relative_path),
                **generation_kwargs
            )

        # Create test files
        with console.status("[bold green]Creating test files..."):
            created_files = []

            # Create workflow files in workflows subdirectory
            if test_directories:
                workflows_dir = relative_path / "workflows"
                workflows_dir.mkdir(exist_ok=True)
                for file_workflow in test_directories:
                    workflow_files = await generator._create_test_files_safely(file_workflow, workflows_dir)
                    created_files += workflow_files

            # Create main test files
            if test_files:
                main_files = await generator._create_test_files_safely(test_files, relative_path)
                created_files += main_files

        # Calculate processing time
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        if created_files:
            console.print(f"[green]✓[/green] Tests generated successfully in: {relative_path}")
            console.print(f"[blue]⏱️[/blue] Processing time: {processing_time:.2f} seconds")

            # Show generated files
            if ctx.obj['verbose'] or len(created_files) <= 10:
                console.print("\n[bold]Generated files:[/bold]")
                for file_path in created_files:
                    console.print(f"  • {file_path}")
            else:
                console.print(f"\n[bold]Generated {len(created_files)} files[/bold]")
                console.print("Use --verbose to see all file names")

            # Show run instructions
            if not dry_run:
                console.print(f"\n[bold]To run tests:[/bold]")
                console.print(f"  cd {relative_path}")

                # Find the main locustfile
                locustfile = relative_path / "locustfile.py"
                if locustfile.exists():
                    console.print(
                        f"  locust -f locustfile.py --users {users} --spawn-rate {spawn_rate} --run-time {run_time} --host {host or 'http://localhost:8000'}")
                else:
                    # Find any .py file that might be the main test file
                    py_files = list(relative_path.glob("*.py"))
                    if py_files:
                        main_file = py_files[0].name
                        console.print(
                            f"  locust -f {main_file} --users {users} --spawn-rate {spawn_rate} --run-time {run_time} --host {host or 'http://localhost:8000'}")

                console.print(f"\n[bold]Alternative: Use the run command[/bold]")
                main_file = "locustfile.py" if locustfile.exists() else (
                    py_files[0].name if py_files else "generated_test.py")
                console.print(
                    f"  devdox-loadtest run {relative_path}/{main_file} --host {host or 'http://localhost:8000'}")
        else:
            console.print("[red]✗[/red] No test files were generated")
            sys.exit(1)

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

        console.print(f"[green]Starting Locust test...[/green]")
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