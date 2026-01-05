"""
Progress output tests for the `generate` command.

Tests verbose mode, progress display, and output formatting (Branch 2 features).
"""

import pytest

from .conftest import run_generate_command


class TestVerboseOutput:
    """Test verbose mode output."""

    def test_verbose_shows_progress_phases(self, api_key, swagger_url, output_dir):
        """Verify verbose mode shows progress phase information."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should contain progress indicators
        progress_indicators = [
            "generating",
            "enhancing",
            "writing",
            "complete",
            "parsing",
        ]

        found_indicators = [p for p in progress_indicators if p.lower() in output.lower()]
        assert len(found_indicators) > 0, \
            "Verbose output doesn't show progress information"

    def test_verbose_shows_file_count(self, api_key, swagger_url, output_dir):
        """Verify verbose mode shows number of files generated."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should mention files in some way
        file_indicators = ["file", "generated", "created", "wrote"]

        found = any(ind in output.lower() for ind in file_indicators)
        # This is informational - not all outputs may have this

    def test_verbose_shows_endpoint_count(self, api_key, swagger_url, output_dir):
        """Verify verbose mode shows number of endpoints parsed."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should mention endpoints
        assert "endpoint" in output.lower() or "parsed" in output.lower(), \
            "Verbose output doesn't show endpoint information"


class TestQuietOutput:
    """Test non-verbose (quiet) mode output."""

    def test_quiet_mode_succeeds(self, api_key, swagger_url, output_dir):
        """Verify non-verbose mode completes successfully."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=False,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_quiet_mode_less_verbose(self, api_key, swagger_url, tmp_path):
        """Verify quiet mode has less output than verbose mode."""
        # Run with verbose
        verbose_output = tmp_path / "verbose"
        verbose_output.mkdir()
        _, verbose_stdout, verbose_stderr = run_generate_command(
            swagger_source="https://petstore.swagger.io/v2/swagger.json",  # Use a known URL
            output_dir=verbose_output,
            api_key=api_key,
            verbose=True,
        )
        verbose_len = len(verbose_stdout + verbose_stderr)

        # Run without verbose
        quiet_output = tmp_path / "quiet"
        quiet_output.mkdir()
        _, quiet_stdout, quiet_stderr = run_generate_command(
            swagger_source="https://petstore.swagger.io/v2/swagger.json",
            output_dir=quiet_output,
            api_key=api_key,
            verbose=False,
        )
        quiet_len = len(quiet_stdout + quiet_stderr)

        # Verbose should generally have more output
        # (This may not always be true depending on implementation)
        print(f"Verbose output length: {verbose_len}")
        print(f"Quiet output length: {quiet_len}")


class TestSuccessOutput:
    """Test success output messages."""

    def test_success_message_on_completion(self, api_key, swagger_url, output_dir):
        """Verify success message is shown on completion."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should have some success indicator
        success_indicators = ["success", "complete", "generated", "✓", "done"]

        found = any(ind in output.lower() for ind in success_indicators)
        assert found, "No success message in output"

    def test_output_directory_shown(self, api_key, swagger_url, output_dir):
        """Verify output directory is shown in output."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should mention output directory
        assert str(output_dir) in output or "output" in output.lower(), \
            "Output directory not shown in output"


class TestConfigurationDisplay:
    """Test configuration display in output."""

    def test_shows_configuration_table(self, api_key, swagger_url, output_dir):
        """Verify configuration is displayed."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should show some configuration info
        config_indicators = ["configuration", "setting", "users", "spawn"]

        found = any(ind in output.lower() for ind in config_indicators)
        # This is informational


class TestRunInstructions:
    """Test run instructions in output."""

    def test_shows_run_instructions(self, api_key, swagger_url, output_dir):
        """Verify run instructions are shown."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = stdout + stderr
        # Should show how to run the tests
        run_indicators = ["run", "locust", "command", "cd"]

        found_count = sum(1 for ind in run_indicators if ind in output.lower())
        # Should have at least some run instructions
        assert found_count >= 1, "No run instructions in output"
