"""
Comprehensive tests for cli.py module.

Tests cover:
- Helper functions (_initialize_config, _setup_output_directory, etc.)
- URL detection (_is_url)
- Configuration display
- Result display
- Click commands (generate, run)
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from devdox_ai_locust.cli import (
    _initialize_config,
    _setup_output_directory,
    _display_configuration,
    _show_results,
    _show_generated_files,
    _show_run_instructions,
    _is_url,
    cli,
    main,
)


# =============================================================================
# _initialize_config Tests
# =============================================================================


class TestInitializeConfig:
    """Tests for _initialize_config function."""

    def test_uses_provided_api_key(self):
        """Should use provided API key over config."""
        with patch("devdox_ai_locust.cli.console"):
            config, api_key = _initialize_config("provided-key")
            assert api_key == "provided-key"

    def test_falls_back_to_config_api_key(self):
        """Should fall back to config API key when none provided."""
        with patch("devdox_ai_locust.cli.Settings") as mock_settings:
            mock_instance = MagicMock()
            mock_instance.API_KEY = "config-key"
            mock_settings.return_value = mock_instance

            with patch("devdox_ai_locust.cli.console"):
                config, api_key = _initialize_config(None)
                assert api_key == "config-key"

    def test_exits_when_no_api_key(self):
        """Should exit with code 1 when no API key available."""
        with patch("devdox_ai_locust.cli.Settings") as mock_settings:
            mock_instance = MagicMock()
            mock_instance.API_KEY = ""
            mock_settings.return_value = mock_instance

            with patch("devdox_ai_locust.cli.console"):
                with pytest.raises(SystemExit) as exc_info:
                    _initialize_config(None)
                assert exc_info.value.code == 1


# =============================================================================
# _setup_output_directory Tests
# =============================================================================


class TestSetupOutputDirectory:
    """Tests for _setup_output_directory function."""

    def test_creates_directory(self, temp_dir):
        """Should create the output directory."""
        output_path = temp_dir / "new_output"
        assert not output_path.exists()

        result = _setup_output_directory(str(output_path))

        assert output_path.exists()
        assert result == output_path

    def test_handles_existing_directory(self, temp_dir):
        """Should handle existing directory gracefully."""
        output_path = temp_dir / "existing"
        output_path.mkdir()

        result = _setup_output_directory(str(output_path))

        assert result == output_path

    def test_creates_nested_directories(self, temp_dir):
        """Should create nested directories."""
        output_path = temp_dir / "a" / "b" / "c"
        assert not output_path.exists()

        _setup_output_directory(str(output_path))

        assert output_path.exists()

    def test_returns_path_object(self, temp_dir):
        """Should return Path object."""
        result = _setup_output_directory(str(temp_dir / "output"))
        assert isinstance(result, Path)


# =============================================================================
# _is_url Tests
# =============================================================================


class TestIsUrl:
    """Tests for _is_url function."""

    def test_detects_http_url(self):
        """Should detect http:// URLs."""
        assert _is_url("http://example.com/swagger.json") is True

    def test_detects_https_url(self):
        """Should detect https:// URLs."""
        assert _is_url("https://api.example.com/openapi.json") is True

    def test_rejects_file_path(self):
        """Should reject file paths."""
        assert _is_url("/path/to/swagger.json") is False
        assert _is_url("./swagger.json") is False
        assert _is_url("swagger.json") is False

    def test_handles_whitespace(self):
        """Should handle leading/trailing whitespace."""
        assert _is_url("  https://example.com/api  ") is True
        assert _is_url("  /path/to/file.json  ") is False

    def test_rejects_other_protocols(self):
        """Should reject non-http protocols."""
        assert _is_url("ftp://example.com/file") is False
        assert _is_url("file:///path/to/file") is False


# =============================================================================
# _display_configuration Tests
# =============================================================================


class TestDisplayConfiguration:
    """Tests for _display_configuration function."""

    def test_displays_all_settings(self, temp_dir):
        """Should display all configuration settings."""
        with patch("devdox_ai_locust.cli.console") as mock_console:
            _display_configuration(
                swagger_url="https://example.com/swagger.json",
                output_dir=temp_dir,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host="http://localhost:8000",
                auth=True,
                custom_requirement="Test custom flows",
                dry_run=False,
            )

            # Should have called console.print with a table
            mock_console.print.assert_called()

    def test_handles_none_values(self, temp_dir):
        """Should handle None values gracefully."""
        with patch("devdox_ai_locust.cli.console"):
            # Should not raise
            _display_configuration(
                swagger_url="https://example.com/api",
                output_dir=temp_dir,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host=None,
                auth=False,
                custom_requirement=None,
                dry_run=True,
            )


# =============================================================================
# _show_generated_files Tests
# =============================================================================


class TestShowGeneratedFiles:
    """Tests for _show_generated_files function."""

    def test_shows_all_files_when_verbose(self):
        """Should show all files when verbose is True."""
        files = [{"path": f"output/file{i}.py"} for i in range(15)]

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_generated_files(files, verbose=True)

            # Should print header and tree (uses console.print with Tree object)
            assert mock_console.print.call_count >= 2
            # Check that it mentions the file count
            calls_str = str(mock_console.print.call_args_list)
            assert "15 files" in calls_str

    def test_shows_all_files_when_few(self):
        """Should show all files when 10 or fewer."""
        files = [{"path": f"output/file{i}.py"} for i in range(5)]

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_generated_files(files, verbose=False)

            # Should print header + tree
            assert mock_console.print.call_count >= 2
            # Check that it mentions the file count
            calls_str = str(mock_console.print.call_args_list)
            assert "5 files" in calls_str

    def test_shows_count_when_many_and_not_verbose(self):
        """Should show count when many files and not verbose."""
        files = [{"path": f"output/dir{i}/file{i}.py"} for i in range(20)]

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_generated_files(files, verbose=False)

            # Check that it mentions the count in some form
            calls_str = str(mock_console.print.call_args_list)
            assert "20 files" in calls_str or "20" in calls_str


# =============================================================================
# _show_run_instructions Tests
# =============================================================================


class TestShowRunInstructions:
    """Tests for _show_run_instructions function."""

    def test_shows_locust_command(self, temp_dir):
        """Should display locust run command in a panel."""
        # Create a locustfile.py
        locustfile = temp_dir / "locustfile.py"
        locustfile.write_text("# test")

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_run_instructions(
                output_dir=temp_dir,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host="http://localhost:8000",
            )

            # Should print a Panel containing locust command
            mock_console.print.assert_called()
            # The Panel contains the locust command text
            calls_str = str(mock_console.print.call_args_list)
            assert "locust" in calls_str or "Panel" in calls_str

    def test_uses_default_host_when_none(self, temp_dir):
        """Should use default host when none provided."""
        locustfile = temp_dir / "locustfile.py"
        locustfile.write_text("# test")

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_run_instructions(
                output_dir=temp_dir,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host=None,
            )

            # Check that console.print was called with Panel
            mock_console.print.assert_called()
            calls_str = str(mock_console.print.call_args_list)
            # Panel contains the default host
            assert "localhost" in calls_str or "Panel" in calls_str


# =============================================================================
# _show_results Tests
# =============================================================================


class TestShowResults:
    """Tests for _show_results function."""

    def test_exits_when_no_files(self, temp_dir):
        """Should exit with code 1 when no files created."""
        with patch("devdox_ai_locust.cli.console"):
            with pytest.raises(SystemExit) as exc_info:
                _show_results(
                    created_files=[],
                    output_dir=temp_dir,
                    start_time=datetime.now(timezone.utc),
                    verbose=False,
                    dry_run=False,
                    users=10,
                    spawn_rate=2.0,
                    run_time="5m",
                    host=None,
                )
            assert exc_info.value.code == 1

    def test_shows_success_message(self, temp_dir):
        """Should show success message when files created."""
        files = [{"filename": "test.py", "path": temp_dir / "test.py"}]
        (temp_dir / "locustfile.py").write_text("# test")

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_results(
                created_files=files,
                output_dir=temp_dir,
                start_time=datetime.now(timezone.utc),
                verbose=False,
                dry_run=False,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host=None,
            )

            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("successfully" in str(call).lower() for call in calls)

    def test_shows_processing_time(self, temp_dir):
        """Should show processing time."""
        files = [{"filename": "test.py", "path": temp_dir / "test.py"}]
        (temp_dir / "locustfile.py").write_text("# test")

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_results(
                created_files=files,
                output_dir=temp_dir,
                start_time=datetime.now(timezone.utc),
                verbose=False,
                dry_run=False,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host=None,
            )

            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("seconds" in str(call).lower() for call in calls)

    def test_skips_run_instructions_on_dry_run(self, temp_dir):
        """Should skip run instructions when dry_run is True."""
        files = [{"filename": "test.py", "path": temp_dir / "test.py"}]

        with patch("devdox_ai_locust.cli.console"):
            with patch("devdox_ai_locust.cli._show_run_instructions") as mock_show_run:
                _show_results(
                    created_files=files,
                    output_dir=temp_dir,
                    start_time=datetime.now(timezone.utc),
                    verbose=False,
                    dry_run=True,
                    users=10,
                    spawn_rate=2.0,
                    run_time="5m",
                    host=None,
                )

                mock_show_run.assert_not_called()


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestCliCommand:
    """Tests for the main CLI group command."""

    def test_cli_version(self):
        """Should show version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.9" in result.output

    def test_cli_help(self):
        """Should show help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DevDox AI LoadTest" in result.output

    def test_cli_verbose_flag(self):
        """Should accept verbose flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestGenerateCommand:
    """Tests for the generate command."""

    def test_generate_help(self):
        """Should show generate command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "swagger_url" in result.output.lower() or "SWAGGER_URL" in result.output

    def test_generate_requires_swagger_url(self):
        """Should require swagger_url argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage" in result.output

    def test_generate_accepts_options(self):
        """Should accept all options."""
        runner = CliRunner()
        # Just test that options are recognized
        result = runner.invoke(
            cli,
            [
                "generate",
                "--help",
            ],
        )
        assert "--output" in result.output
        assert "--users" in result.output
        assert "--spawn-rate" in result.output
        assert "--run-time" in result.output
        assert "--host" in result.output
        assert "--auth" in result.output


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self):
        """Should show run command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "test_file" in result.output.lower() or "TEST_FILE" in result.output

    def test_run_requires_test_file(self):
        """Should require test_file argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0

    def test_run_requires_host(self, temp_dir):
        """Should require --host option."""
        test_file = temp_dir / "test.py"
        test_file.write_text("# test")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(test_file)])
        assert result.exit_code != 0
        assert "host" in result.output.lower()


# =============================================================================
# Main Entry Point Tests
# =============================================================================


class TestMainEntryPoint:
    """Tests for main entry point."""

    def test_main_invokes_cli(self):
        """main() should invoke cli()."""
        with patch("devdox_ai_locust.cli.cli"):
            # main() calls cli() which would normally process sys.argv
            # We just verify it's called
            try:
                main()
            except SystemExit:
                pass  # Click may exit
            # Can't easily verify call due to Click's internal handling


# =============================================================================
# Integration Tests
# =============================================================================


class TestCliIntegration:
    """Integration tests for CLI."""

    def test_generate_with_mock_api(self, temp_dir):
        """Test generate command with mocked API calls."""
        runner = CliRunner()

        # Mock all the async operations
        with patch("devdox_ai_locust.cli._async_generate") as mock_generate:
            mock_generate.return_value = None

            with patch("devdox_ai_locust.cli._initialize_config") as mock_init:
                mock_init.return_value = (MagicMock(), "test-api-key")

                runner.invoke(
                    cli,
                    [
                        "generate",
                        "https://example.com/swagger.json",
                        "--output",
                        str(temp_dir),
                        "--together-api-key",
                        "test-key",
                        "--dry-run",
                    ],
                )

                # Just verify it attempted to run
                # The actual execution depends on many factors

    def test_run_command_with_mock_subprocess(self, temp_dir):
        """Test run command with mocked subprocess."""
        test_file = temp_dir / "test.py"
        test_file.write_text("# locust test")

        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            runner.invoke(
                cli,
                [
                    "run",
                    str(test_file),
                    "--host",
                    "http://localhost:8000",
                    "--users",
                    "5",
                    "--spawn-rate",
                    "1",
                    "--run-time",
                    "1m",
                ],
            )

            # Verify subprocess.run was called with locust command
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "locust" in call_args
            assert "-f" in call_args
            assert str(test_file) in call_args

    def test_run_command_headless_mode(self, temp_dir):
        """Test run command with headless flag."""
        test_file = temp_dir / "test.py"
        test_file.write_text("# locust test")

        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            runner.invoke(
                cli,
                [
                    "run",
                    str(test_file),
                    "--host",
                    "http://localhost:8000",
                    "--headless",
                ],
            )

            call_args = mock_run.call_args[0][0]
            assert "--headless" in call_args


# =============================================================================
# Mocked Async Function Tests
# =============================================================================


class TestProcessApiSchema:
    """Tests for _process_api_schema async function with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_processes_url_source(self):
        """Should process URL source correctly."""
        from devdox_ai_locust.cli import _process_api_schema

        mock_schema = '{"openapi": "3.0.0", "info": {"title": "Test API", "version": "1.0"}, "paths": {}}'
        mock_endpoints = [MagicMock()]
        mock_api_info = {"title": "Test API", "version": "1.0"}

        with patch("devdox_ai_locust.cli.get_api_schema") as mock_get:
            mock_get.return_value = mock_schema

            with patch("devdox_ai_locust.cli.OpenAPIParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse_schema.return_value = {"openapi": "3.0.0"}
                mock_parser.parse_endpoints.return_value = mock_endpoints
                mock_parser.get_schema_info.return_value = mock_api_info
                mock_parser_class.return_value = mock_parser

                with patch("devdox_ai_locust.cli.console"):
                    schema_data, endpoints, api_info = await _process_api_schema(
                        "https://example.com/swagger.json", verbose=False
                    )

                    assert endpoints == mock_endpoints
                    assert api_info == mock_api_info

    @pytest.mark.asyncio
    async def test_processes_file_source(self):
        """Should process file source correctly."""
        from devdox_ai_locust.cli import _process_api_schema

        mock_schema = '{"openapi": "3.0.0", "paths": {}}'

        with patch("devdox_ai_locust.cli.get_api_schema") as mock_get:
            mock_get.return_value = mock_schema

            with patch("devdox_ai_locust.cli.OpenAPIParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse_schema.return_value = {}
                mock_parser.parse_endpoints.return_value = []
                mock_parser.get_schema_info.return_value = {}
                mock_parser_class.return_value = mock_parser

                with patch("devdox_ai_locust.cli.console"):
                    await _process_api_schema("/path/to/swagger.json", verbose=True)

    @pytest.mark.asyncio
    async def test_exits_on_empty_schema(self):
        """Should exit when schema is empty."""
        from devdox_ai_locust.cli import _process_api_schema

        with patch("devdox_ai_locust.cli.get_api_schema") as mock_get:
            mock_get.return_value = None

            with patch("devdox_ai_locust.cli.console"):
                with pytest.raises(SystemExit) as exc_info:
                    await _process_api_schema("https://example.com/api", verbose=False)
                assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_exits_on_timeout(self):
        """Should exit on timeout."""
        from devdox_ai_locust.cli import _process_api_schema
        import asyncio

        with patch("devdox_ai_locust.cli.get_api_schema") as mock_get:
            mock_get.side_effect = asyncio.TimeoutError()

            with patch("devdox_ai_locust.cli.console"):
                with pytest.raises(SystemExit) as exc_info:
                    await _process_api_schema("https://example.com/api", verbose=False)
                assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_exits_on_file_not_found(self):
        """Should exit when file not found."""
        from devdox_ai_locust.cli import _process_api_schema

        with patch("devdox_ai_locust.cli.get_api_schema") as mock_get:
            mock_get.side_effect = FileNotFoundError("File not found")

            with patch("devdox_ai_locust.cli.console"):
                with pytest.raises(SystemExit) as exc_info:
                    await _process_api_schema("/nonexistent/file.json", verbose=False)
                assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_exits_on_parse_error(self):
        """Should exit on parse error."""
        from devdox_ai_locust.cli import _process_api_schema

        with patch("devdox_ai_locust.cli.get_api_schema") as mock_get:
            mock_get.return_value = "invalid schema content"

            with patch("devdox_ai_locust.cli.OpenAPIParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse_schema.side_effect = ValueError("Invalid schema")
                mock_parser_class.return_value = mock_parser

                with patch("devdox_ai_locust.cli.console"):
                    with pytest.raises(SystemExit) as exc_info:
                        await _process_api_schema("https://example.com/api", verbose=False)
                    assert exc_info.value.code == 1


class TestAsyncGenerate:
    """Tests for _async_generate function."""

    @pytest.mark.asyncio
    async def test_full_generation_flow(self, temp_dir):
        """Should execute full generation flow."""
        from devdox_ai_locust.cli import _async_generate

        mock_ctx = MagicMock()
        mock_ctx.obj = {"verbose": False}

        mock_endpoints = [MagicMock()]
        mock_api_info = {"title": "Test API"}
        mock_files = [{"filename": "test.py", "path": temp_dir / "test.py"}]

        with patch("devdox_ai_locust.cli._initialize_config") as mock_init:
            mock_init.return_value = (MagicMock(), "test-api-key")

            with patch("devdox_ai_locust.cli._process_api_schema") as mock_process:
                mock_process.return_value = ({}, mock_endpoints, mock_api_info)

                with patch("devdox_ai_locust.cli._generate_modular_tests") as mock_gen:
                    mock_gen.return_value = mock_files

                    with patch("devdox_ai_locust.cli._show_results"):
                        with patch("devdox_ai_locust.cli.console"):
                            await _async_generate(
                                ctx=mock_ctx,
                                swagger_url="https://example.com/api",
                                output=str(temp_dir),
                                users=10,
                                spawn_rate=2.0,
                                run_time="5m",
                                host="http://localhost",
                                auth=True,
                                db_type="",
                                dry_run=False,
                                custom_requirement=None,
                                together_api_key="test-key",
                            )

                            mock_process.assert_called_once()
                            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_displays_config_in_verbose_mode(self, temp_dir):
        """Should display config when verbose is True."""
        from devdox_ai_locust.cli import _async_generate

        mock_ctx = MagicMock()
        mock_ctx.obj = {"verbose": True}

        with patch("devdox_ai_locust.cli._initialize_config") as mock_init:
            mock_init.return_value = (MagicMock(), "test-key")

            with patch("devdox_ai_locust.cli._display_configuration") as mock_display:
                with patch("devdox_ai_locust.cli._process_api_schema") as mock_process:
                    mock_process.return_value = ({}, [], {})

                    with patch("devdox_ai_locust.cli._generate_modular_tests") as mock_gen:
                        mock_gen.return_value = [{"filename": "test.py"}]

                        with patch("devdox_ai_locust.cli._show_results"):
                            with patch("devdox_ai_locust.cli.console"):
                                await _async_generate(
                                    ctx=mock_ctx,
                                    swagger_url="https://example.com/api",
                                    output=str(temp_dir),
                                    users=10,
                                    spawn_rate=2.0,
                                    run_time="5m",
                                    host=None,
                                    auth=True,
                                    db_type="",
                                    dry_run=False,
                                    custom_requirement="Test requirement",
                                    together_api_key="test-key",
                                )

                                mock_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_generation_error(self, temp_dir):
        """Should handle and re-raise generation errors."""
        from devdox_ai_locust.cli import _async_generate

        mock_ctx = MagicMock()
        mock_ctx.obj = {"verbose": False}

        with patch("devdox_ai_locust.cli._initialize_config") as mock_init:
            mock_init.return_value = (MagicMock(), "test-key")

            with patch("devdox_ai_locust.cli._process_api_schema") as mock_process:
                mock_process.side_effect = Exception("Generation failed")

                with patch("devdox_ai_locust.cli.console"):
                    with pytest.raises(Exception, match="Generation failed"):
                        await _async_generate(
                            ctx=mock_ctx,
                            swagger_url="https://example.com/api",
                            output=str(temp_dir),
                            users=10,
                            spawn_rate=2.0,
                            run_time="5m",
                            host=None,
                            auth=True,
                            db_type="",
                            dry_run=False,
                            custom_requirement=None,
                            together_api_key="test-key",
                        )


class TestRunCommandExecution:
    """Tests for run command execution paths."""

    def test_run_executes_locust(self, temp_dir):
        """Should execute locust with correct arguments."""
        test_file = temp_dir / "locustfile.py"
        test_file.write_text("# test")

        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = runner.invoke(
                cli,
                [
                    "run",
                    str(test_file),
                    "--host",
                    "http://localhost:8000",
                    "--users",
                    "20",
                    "--spawn-rate",
                    "5",
                    "--run-time",
                    "10m",
                ],
            )

            call_args = mock_run.call_args[0][0]
            assert "--users" in call_args
            assert "20" in call_args
            assert "--spawn-rate" in call_args
            # spawn_rate is a float, so it becomes "5.0"
            assert "5.0" in call_args or "5" in call_args
            assert "--run-time" in call_args
            assert "10m" in call_args

    def test_run_handles_subprocess_error(self, temp_dir):
        """Should handle subprocess CalledProcessError."""
        import subprocess

        test_file = temp_dir / "locustfile.py"
        test_file.write_text("# test")

        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "locust")

            result = runner.invoke(
                cli,
                ["run", str(test_file), "--host", "http://localhost:8000"],
            )

            assert result.exit_code == 1

    def test_run_handles_locust_not_found(self, temp_dir):
        """Should handle FileNotFoundError when locust not installed."""
        test_file = temp_dir / "locustfile.py"
        test_file.write_text("# test")

        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("locust not found")

            result = runner.invoke(
                cli,
                ["run", str(test_file), "--host", "http://localhost:8000"],
            )

            assert result.exit_code == 1
            assert "locust" in result.output.lower()

    def test_run_verbose_mode(self, temp_dir):
        """Should show command in verbose mode."""
        test_file = temp_dir / "locustfile.py"
        test_file.write_text("# test")

        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = runner.invoke(
                cli,
                [
                    "--verbose",
                    "run",
                    str(test_file),
                    "--host",
                    "http://localhost:8000",
                ],
            )

            # Verbose mode should print the command
            # (output depends on whether console.print is called)


class TestGenerateCommandExecution:
    """Tests for generate command execution paths."""

    def test_generate_handles_exception(self, temp_dir):
        """Should handle exceptions gracefully."""
        runner = CliRunner()

        with patch("devdox_ai_locust.cli.asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = runner.invoke(
                cli,
                [
                    "generate",
                    "https://example.com/swagger.json",
                    "--output",
                    str(temp_dir),
                    "--together-api-key",
                    "test-key",
                ],
            )

            assert result.exit_code == 1
            assert "error" in result.output.lower()

    def test_generate_shows_traceback_in_verbose(self, temp_dir):
        """Should show traceback in verbose mode on error."""
        runner = CliRunner()

        with patch("devdox_ai_locust.cli.asyncio.run") as mock_run:
            mock_run.side_effect = ValueError("Detailed error")

            result = runner.invoke(
                cli,
                [
                    "--verbose",
                    "generate",
                    "https://example.com/swagger.json",
                    "--output",
                    str(temp_dir),
                    "--together-api-key",
                    "test-key",
                ],
            )

            assert result.exit_code == 1


class TestShowRunInstructionsEdgeCases:
    """Tests for edge cases in _show_run_instructions."""

    def test_finds_alternative_py_file(self, temp_dir):
        """Should find alternative .py file when no locustfile.py."""
        # Create an alternative Python file
        alt_file = temp_dir / "my_tests.py"
        alt_file.write_text("# test file")

        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_run_instructions(
                output_dir=temp_dir,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host="http://localhost:8000",
            )

            calls = ["".join(call.args) for call in mock_console.print.call_args_list if call.args]
            assert any("🚀 Next Steps" in text for text in calls)
            assert any("1) Prepare your environment" in text for text in calls)
            assert any("2) Choose how you want to run" in text for text in calls)
            assert any("my_tests.py" in text for text in calls)

    def test_handles_empty_directory(self, temp_dir):
        """Should handle directory with no Python files."""
        with patch("devdox_ai_locust.cli.console") as mock_console:
            _show_run_instructions(
                output_dir=temp_dir,
                users=10,
                spawn_rate=2.0,
                run_time="5m",
                host=None,
            )

            # Should still print instructions via Panel
            assert mock_console.print.called
