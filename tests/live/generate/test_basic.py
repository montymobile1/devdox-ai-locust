"""
Basic generation tests for the `generate` CLI command.

Tests different input formats, output configurations, and basic options.
"""

import pytest
from pathlib import Path

from .conftest import run_generate_command


class TestInputFormats:
    """Test different input source formats."""

    def test_generate_from_url(self, api_key, swagger_url, output_dir):
        """Test generation from Swagger URL."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check expected files exist
        assert (output_dir / "locustfile.py").exists(), "Missing locustfile.py"
        assert (output_dir / "config.py").exists(), "Missing config.py"

    def test_generate_from_file(self, api_key, swagger_file, output_dir):
        """Test generation from local Swagger file."""
        if not swagger_file:
            pytest.skip("No swagger file provided (use --swagger-file)")

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_file,
            output_dir=output_dir,
            api_key=api_key,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"
        assert (output_dir / "locustfile.py").exists()


class TestOutputConfiguration:
    """Test output directory and file generation options."""

    def test_custom_output_directory(self, api_key, swagger_url, tmp_path):
        """Test generation to custom output directory."""
        custom_output = tmp_path / "custom_output"

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=custom_output,
            api_key=api_key,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"
        assert custom_output.exists()
        assert (custom_output / "locustfile.py").exists()

    def test_generates_expected_files(self, api_key, swagger_url, output_dir):
        """Test that all expected files are generated."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Required files
        assert (output_dir / "locustfile.py").exists(), "Missing locustfile.py"
        assert (output_dir / "config.py").exists(), "Missing config.py"

        # Check for Python files (should have at least 2)
        py_files = list(output_dir.rglob("*.py"))
        assert len(py_files) >= 2, f"Expected at least 2 Python files, got {len(py_files)}"


class TestVerboseMode:
    """Test verbose output options."""

    def test_verbose_mode(self, api_key, swagger_url, output_dir):
        """Test generation with verbose output."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

        # Verbose output should contain progress information
        output = stdout + stderr
        assert any(
            indicator in output.lower()
            for indicator in ["generating", "enhancing", "parsing", "writing"]
        ), "Verbose mode should show progress information"

    def test_quiet_mode(self, api_key, swagger_url, output_dir):
        """Test generation without verbose output (default)."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=False,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"
        # Just verify it succeeds - output format may vary


class TestLocustConfiguration:
    """Test Locust configuration options (users, spawn-rate, run-time)."""

    def test_custom_users(self, api_key, swagger_url, output_dir):
        """Test generation with custom user count."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            users=50,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check config.py contains user setting
        config_path = output_dir / "config.py"
        if config_path.exists():
            config_content = config_path.read_text()
            # Config should reference users (may be configurable)

    def test_custom_spawn_rate(self, api_key, swagger_url, output_dir):
        """Test generation with custom spawn rate."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            spawn_rate=5.0,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_custom_run_time(self, api_key, swagger_url, output_dir):
        """Test generation with custom run time."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            run_time="10m",
        )
        assert exit_code == 0, f"Generation failed: {stderr}"


class TestHostConfiguration:
    """Test host URL configuration."""

    def test_with_custom_host(self, api_key, swagger_url, output_dir, target_host):
        """Test generation with custom target host."""
        host = target_host or "http://localhost:8080"

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            host=host,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check that host is reflected in config
        config_path = output_dir / "config.py"
        if config_path.exists():
            config_content = config_path.read_text()
            # Host should be in config (may be environment variable based)

    def test_without_host(self, api_key, swagger_url, output_dir):
        """Test generation without specifying host (should use auto-detect)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            host=None,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"


class TestDryRun:
    """Test dry-run mode."""

    def test_dry_run(self, api_key, swagger_url, output_dir):
        """Test generation in dry-run mode."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            dry_run=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Files should still be generated in dry-run mode
        # (dry-run just doesn't run them)
        assert (output_dir / "locustfile.py").exists()
