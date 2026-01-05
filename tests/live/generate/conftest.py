"""
Generate command specific fixtures and configuration.

These fixtures are specific to testing the `generate` CLI command.
Configuration is loaded with priority: CLI > ENV > .env.test
"""

import os
import subprocess
import pytest
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Import the config helper from parent
from tests.live.conftest import get_config_value


def pytest_addoption(parser):
    """Add generate-specific command-line options."""
    parser.addoption(
        "--swagger-url",
        action="store",
        default=None,
        help="Swagger/OpenAPI URL (or SWAGGER_URL env var, or .env.test)",
    )
    parser.addoption(
        "--swagger-file",
        action="store",
        default=None,
        help="Path to local Swagger/OpenAPI file",
    )
    parser.addoption(
        "--mongodb-uri",
        action="store",
        default=None,
        help="MongoDB connection URI (optional, or MONGODB_URI)",
    )
    parser.addoption(
        "--postgresql-uri",
        action="store",
        default=None,
        help="PostgreSQL connection URI (optional, or POSTGRESQL_URI)",
    )
    parser.addoption(
        "--target-host",
        action="store",
        default=None,
        help="Target host URL for generated tests (optional, or TARGET_HOST)",
    )


# Mark all tests in this directory as live tests
pytestmark = pytest.mark.live


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def swagger_url(request) -> str:
    """
    Get Swagger URL with priority: CLI > ENV > .env.test

    Uses: --swagger-url, SWAGGER_URL, or .env.test
    """
    url = get_config_value(
        request.config.getoption("--swagger-url"),
        "SWAGGER_URL"
    )
    if not url:
        pytest.skip(
            "No Swagger URL provided. Set via:\n"
            "  --swagger-url FLAG\n"
            "  SWAGGER_URL env var\n"
            "  SWAGGER_URL in .env.test"
        )
    return url


@pytest.fixture
def swagger_file(request) -> Optional[str]:
    """Get local Swagger file path from command line or .env.test."""
    file_path = get_config_value(
        request.config.getoption("--swagger-file"),
        "SWAGGER_FILE"
    )
    if file_path and not Path(file_path).exists():
        pytest.skip(f"Swagger file not found: {file_path}")
    return file_path


@pytest.fixture
def swagger_source(swagger_url, swagger_file) -> str:
    """Get Swagger source - prefers file over URL if both provided."""
    if swagger_file:
        return swagger_file
    return swagger_url


@pytest.fixture
def mongodb_uri(request) -> Optional[str]:
    """Get MongoDB URI (optional) with priority: CLI > ENV > .env.test"""
    return get_config_value(
        request.config.getoption("--mongodb-uri"),
        "MONGODB_URI"
    )


@pytest.fixture
def postgresql_uri(request) -> Optional[str]:
    """Get PostgreSQL URI (optional) with priority: CLI > ENV > .env.test"""
    return get_config_value(
        request.config.getoption("--postgresql-uri"),
        "POSTGRESQL_URI"
    )


@pytest.fixture
def target_host(request) -> Optional[str]:
    """Get target host URL (optional) with priority: CLI > ENV > .env.test"""
    return get_config_value(
        request.config.getoption("--target-host"),
        "TARGET_HOST"
    )


# =============================================================================
# Command Runner
# =============================================================================

def run_generate_command(
    swagger_source: str,
    output_dir: Path,
    api_key: str,
    host: Optional[str] = None,
    auth: bool = True,
    db_type: Optional[str] = None,
    custom_requirement: Optional[str] = None,
    users: int = 10,
    spawn_rate: float = 2,
    run_time: str = "5m",
    dry_run: bool = False,
    verbose: bool = False,
    timeout: int = 300,
) -> Tuple[int, str, str]:
    """
    Run the generate command and return exit code, stdout, stderr.

    Args:
        swagger_source: URL or file path to Swagger/OpenAPI spec
        output_dir: Output directory for generated tests
        api_key: Together API key
        host: Target host URL (optional)
        auth: Include authentication (default: True)
        db_type: Database type (None, "mongo", "postgresql")
        custom_requirement: Custom requirements string
        users: Number of simulated users
        spawn_rate: Rate to spawn users
        run_time: Test run time
        dry_run: Generate without running
        verbose: Enable verbose output
        timeout: Command timeout in seconds (default: 300)

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    cmd = [
        "uv", "run", "devdox-ai-locust", "generate",
        swagger_source,
        "-o", str(output_dir),
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", run_time,
    ]

    if host:
        cmd.extend(["--host", host])

    if auth:
        cmd.append("--auth")
    else:
        cmd.append("--no-auth")

    if db_type:
        cmd.extend(["--db-type", db_type])

    if custom_requirement:
        cmd.extend(["--custom-requirement", custom_requirement])

    if dry_run:
        cmd.append("--dry-run")

    if verbose:
        cmd.append("-v")

    env = os.environ.copy()
    env["TOGETHER_API_KEY"] = api_key

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )

    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def generate_runner(api_key, output_dir):
    """Fixture that returns a configured generate command runner."""
    def runner(swagger_source: str, **kwargs) -> Tuple[int, str, str]:
        return run_generate_command(
            swagger_source=swagger_source,
            output_dir=output_dir,
            api_key=api_key,
            **kwargs
        )
    return runner


# =============================================================================
# Output Analysis Utilities
# =============================================================================

class GeneratedOutput:
    """Helper class for analyzing generated output."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.files: Dict[str, str] = {}
        self._load_files()

    def _load_files(self):
        """Load all Python files from output directory."""
        for file_path in self.output_dir.rglob("*.py"):
            relative_path = file_path.relative_to(self.output_dir)
            try:
                self.files[str(relative_path)] = file_path.read_text()
            except Exception:
                pass

    @property
    def locustfile(self) -> Optional[str]:
        """Get locustfile.py content."""
        return self.files.get("locustfile.py")

    @property
    def config(self) -> Optional[str]:
        """Get config.py content."""
        return self.files.get("config.py")

    def has_file(self, name: str) -> bool:
        """Check if a file exists (supports partial match)."""
        return name in self.files or any(name in f for f in self.files)

    def get_all_content(self) -> str:
        """Get all file contents concatenated."""
        return "\n".join(self.files.values())

    def search_content(self, pattern: str, case_sensitive: bool = False) -> List[Tuple[str, List[str]]]:
        """Search for pattern in all files, return matches."""
        import re
        flags = 0 if case_sensitive else re.IGNORECASE
        results = []
        for filename, content in self.files.items():
            matches = re.findall(pattern, content, flags)
            if matches:
                results.append((filename, matches))
        return results

    def contains(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if any file contains the text."""
        all_content = self.get_all_content()
        if case_sensitive:
            return text in all_content
        return text.lower() in all_content.lower()


@pytest.fixture
def analyze_output(output_dir):
    """Fixture that returns an output analyzer."""
    def analyzer() -> GeneratedOutput:
        return GeneratedOutput(output_dir)
    return analyzer
