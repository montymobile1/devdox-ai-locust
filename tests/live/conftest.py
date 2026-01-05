"""
Shared live test configuration and fixtures.

Live tests are SKIPPED by default to prevent them from running in CI/CD.
To run live tests, use: pytest tests/live --run-live

Configuration Priority:
    1. Command-line arguments (--api-key, --swagger-url, etc.)
    2. Environment variables (TOGETHER_API_KEY, SWAGGER_URL, etc.)
    3. .env.test file in project root (loaded automatically)

Directory Structure:
    tests/live/
    ├── conftest.py           # This file - shared fixtures
    ├── README.md             # Documentation
    └── generate/             # Tests for `generate` command
        ├── conftest.py       # Generate-specific fixtures
        └── test_*.py         # Test modules
"""

import os
import pytest
from pathlib import Path
from typing import Optional, Dict, Any


# =============================================================================
# .env.test Loading
# =============================================================================

def load_env_test_file() -> Dict[str, str]:
    """
    Load configuration from .env.test file.

    Looks for .env.test in:
    1. Current directory
    2. Project root (parent of tests/)
    3. tests/live/ directory

    Returns:
        Dict of environment variable names to values
    """
    env_vars = {}

    # Possible locations for .env.test
    possible_paths = [
        Path.cwd() / ".env.test",
        Path(__file__).parent.parent.parent / ".env.test",  # Project root
        Path(__file__).parent / ".env.test",  # tests/live/
    ]

    env_file = None
    for path in possible_paths:
        if path.exists():
            env_file = path
            break

    if not env_file:
        return env_vars

    # Parse .env.test file
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    env_vars[key] = value
    except Exception as e:
        print(f"Warning: Failed to load .env.test: {e}")

    return env_vars


# Load .env.test at module import
_ENV_TEST_VARS = load_env_test_file()


def get_config_value(
    cli_value: Optional[str],
    env_var: str,
    env_test_key: Optional[str] = None
) -> Optional[str]:
    """
    Get configuration value with priority: CLI > ENV > .env.test

    Args:
        cli_value: Value from command line (highest priority)
        env_var: Environment variable name
        env_test_key: Key in .env.test (defaults to env_var)

    Returns:
        Configuration value or None
    """
    if cli_value:
        return cli_value

    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    test_key = env_test_key or env_var
    return _ENV_TEST_VARS.get(test_key)


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_addoption(parser):
    """Add custom command-line options for live tests."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live integration tests (requires API key)",
    )
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="Together API key (or TOGETHER_API_KEY env var, or .env.test)",
    )
    parser.addoption(
        "--output-dir",
        action="store",
        default=None,
        help="Output directory for generated files (default: temp directory)",
    )
    parser.addoption(
        "--keep-output",
        action="store_true",
        default=False,
        help="Keep generated output files after tests (for inspection)",
    )
    parser.addoption(
        "--env-test",
        action="store",
        default=None,
        help="Path to .env.test file (default: auto-detect)",
    )


def pytest_configure(config):
    """Register custom markers and load .env.test."""
    global _ENV_TEST_VARS

    # Load custom .env.test path if specified
    custom_env_path = config.getoption("--env-test", None)
    if custom_env_path:
        path = Path(custom_env_path)
        if path.exists():
            _ENV_TEST_VARS = load_env_test_file_from_path(path)
        else:
            print(f"Warning: Specified .env.test not found: {custom_env_path}")

    # Register markers
    config.addinivalue_line(
        "markers", "live: mark test as live integration test (requires --run-live)"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database connection"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (long-running)"
    )
    config.addinivalue_line(
        "markers", "extensive: mark test as extensive validation"
    )


def load_env_test_file_from_path(path: Path) -> Dict[str, str]:
    """Load .env.test from specific path."""
    env_vars = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    env_vars[key] = value
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
    return env_vars


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless --run-live is specified."""
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="Need --run-live option to run live tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def env_test_config() -> Dict[str, str]:
    """Get the loaded .env.test configuration."""
    return _ENV_TEST_VARS.copy()


@pytest.fixture
def api_key(request) -> str:
    """
    Get API key with priority: CLI > ENV > .env.test

    Uses: --api-key, TOGETHER_API_KEY, or .env.test
    """
    key = get_config_value(
        request.config.getoption("--api-key"),
        "TOGETHER_API_KEY"
    )
    if not key:
        pytest.skip(
            "No API key provided. Set via:\n"
            "  --api-key FLAG\n"
            "  TOGETHER_API_KEY env var\n"
            "  TOGETHER_API_KEY in .env.test"
        )
    return key


@pytest.fixture
def output_dir(request, tmp_path) -> Path:
    """Get output directory - uses temp dir by default."""
    custom_dir = get_config_value(
        request.config.getoption("--output-dir"),
        "TEST_OUTPUT_DIR"
    )
    if custom_dir:
        path = Path(custom_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path


@pytest.fixture
def keep_output(request) -> bool:
    """Whether to keep output files after tests."""
    cli_value = request.config.getoption("--keep-output")
    if cli_value:
        return True
    env_value = get_config_value(None, "KEEP_TEST_OUTPUT")
    return env_value and env_value.lower() in ("true", "1", "yes")
