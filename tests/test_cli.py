"""
Tests for CLI module
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
from datetime import datetime, timezone
from pathlib import Path

from devdox_ai_locust.cli import (
    cli,
    _async_generate,
    _initialize_config,
    _setup_output_directory,
    _display_configuration,
    _show_results,
    _show_generated_files,
    _show_run_instructions,
    _process_api_schema,
    _generate_and_create_tests,
)
from devdox_ai_locust.config import Settings


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    settings = Mock(spec=Settings)
    settings.API_KEY = "test-api-key"
    return settings


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_endpoints():
    """Sample endpoints for testing."""
    from devdox_ai_locust.utils.open_ai_parser import Endpoint

    return [
        Endpoint(
            path="/api/users",
            method="GET",
            operation_id="getUsers",
            summary="Get all users",
            parameters=[],
            request_body=None,
            responses={},
            description="Description of the endpoint",
            tags=["users"],
        ),
        Endpoint(
            path="/api/users/{id}",
            method="GET",
            operation_id="getUserById",
            summary="Get user by ID",
            parameters=[],
            request_body=None,
            responses={},
            description="Description of the endpoint",
            tags=["users"],
        ),
    ]


@pytest.fixture
def sample_api_info():
    """Sample API info for testing."""
    return {
        "title": "Test API",
        "description": "A test API for testing",
        "version": "1.0.0",
    }


class TestCLI:
    """Test CLI functionality."""

    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DevDox AI LoadTest" in result.output

    def test_cli_version(self, cli_runner):
        """Test CLI version command."""
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_cli_verbose_flag(self, cli_runner):
        """Test CLI verbose flag."""
        result = cli_runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_generate_command_help(self, cli_runner):
        """Test generate command help."""
        result = cli_runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate Locust test files" in result.output

    def test_run_command_help(self, cli_runner):
        """Test run command help."""
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run generated Locust tests" in result.output

    @patch("devdox_ai_locust.cli.Settings")
    def test_initialize_config_with_api_key(self, mock_settings_class):
        """Test config initialization with API key."""
        mock_settings = Mock()
        mock_settings.API_KEY = "test-key"
        mock_settings_class.return_value = mock_settings

        config, api_key = _initialize_config("provided-key")

        assert api_key == "provided-key"
        assert config == mock_settings

    @patch("devdox_ai_locust.cli.Settings")
    def test_initialize_config_from_settings(self, mock_settings_class):
        """Test config initialization from settings."""
        mock_settings = Mock()
        mock_settings.API_KEY = "settings-key"
        mock_settings_class.return_value = mock_settings

        config, api_key = _initialize_config(None)

        assert api_key == "settings-key"
        assert config == mock_settings

    @patch("devdox_ai_locust.cli.Settings")
    @patch("devdox_ai_locust.cli.sys.exit")
    def test_initialize_config_no_api_key(self, mock_exit, mock_settings_class):
        """Test config initialization without API key."""
        mock_settings = Mock()
        mock_settings.API_KEY = ""
        mock_settings_class.return_value = mock_settings

        _initialize_config(None)

        mock_exit.assert_called_once_with(1)

    def test_setup_output_directory(self, temp_dir):
        """Test output directory setup."""
        output_dir = temp_dir / "test_output"
        result = _setup_output_directory(str(output_dir))

        assert result == output_dir
        assert output_dir.exists()

    @patch("devdox_ai_locust.cli.asyncio.run")
    @patch("devdox_ai_locust.cli._async_generate")
    def test_generate_command_basic(
        self, mock_async_generate, mock_asyncio_run, cli_runner
    ):
        """Test basic generate command."""
        mock_async_generate.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            _ = cli_runner.invoke(
                cli,
                [
                    "generate",
                    "https://api.example.com/swagger.json",
                    "--output",
                    temp_dir,
                    "--together-api-key",
                    "test-key",
                ],
            )

            # Should not crash, might fail due to async issues in testing
            mock_asyncio_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_command_basic(self, mock_subprocess, cli_runner, temp_dir):
        """Test basic run command."""
        # Create a dummy test file
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        _ = cli_runner.invoke(
            cli,
            [
                "run",
                str(test_file),
                "--host",
                "http://localhost:8000",
                "--users",
                "10",
                "--spawn-rate",
                "2",
            ],
        )

        # Should call subprocess.run with locust command
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "locust" in args
        assert str(test_file) in args
        assert "http://localhost:8000" in args

    @patch("subprocess.run")
    def test_run_command_headless(self, mock_subprocess, cli_runner, temp_dir):
        """Test run command with headless flag."""
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        _ = cli_runner.invoke(
            cli,
            ["run", str(test_file), "--host", "http://localhost:8000", "--headless"],
        )

        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "--headless" in args

    def test_run_command_file_not_found(self, cli_runner):
        """Test run command with non-existent file."""
        result = cli_runner.invoke(
            cli, ["run", "/non/existent/file.py", "--host", "http://localhost:8000"]
        )

        assert result.exit_code != 0

    @patch("subprocess.run")
    def test_run_command_subprocess_error(self, mock_subprocess, cli_runner, temp_dir):
        """Test run command with subprocess error."""
        from subprocess import CalledProcessError

        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        mock_subprocess.side_effect = CalledProcessError(1, "locust")

        result = cli_runner.invoke(
            cli, ["run", str(test_file), "--host", "http://localhost:8000"]
        )

        assert result.exit_code == 1

    @patch("subprocess.run")
    def test_run_command_locust_not_found(self, mock_subprocess, cli_runner, temp_dir):
        """Test run command when locust is not installed."""
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        mock_subprocess.side_effect = FileNotFoundError()

        result = cli_runner.invoke(
            cli, ["run", str(test_file), "--host", "http://localhost:8000"]
        )

        assert result.exit_code == 1


class TestAsyncGenerate:
    """Test async generate functionality."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._setup_output_directory")
    @patch("devdox_ai_locust.cli._process_api_schema")
    @patch("devdox_ai_locust.cli._generate_and_create_tests")
    @patch("devdox_ai_locust.cli._show_results")
    async def test_async_generate_success(
        self,
        mock_show_results,
        mock_generate_tests,
        mock_process_schema,
        mock_setup_output,
        mock_init_config,
        temp_dir,
        sample_endpoints,
        sample_api_info,
    ):
        """Test successful async generate."""
        # Setup mocks
        mock_init_config.return_value = (Mock(), "test-api-key")
        mock_setup_output.return_value = temp_dir
        mock_process_schema.return_value = (None, sample_endpoints, sample_api_info)
        mock_generate_tests.return_value = [{"path": "test_file.py"}]

        # Create mock context
        mock_ctx = Mock()
        mock_ctx.obj = {"verbose": False}

        # Test the function
        await _async_generate(
            mock_ctx,
            "https://api.example.com/swagger.json",
            "output",
            10,  # users
            2,  # spawn_rate
            "5m",  # run_time
            None,  # host
            "",
            True,  # auth
            False,  # dry_run
            None,  # custom_requirement
            "test-api-key",
        )

        # Verify calls
        mock_init_config.assert_called_once()
        mock_setup_output.assert_called_once()
        mock_process_schema.assert_called_once()
        mock_generate_tests.assert_called_once()
        mock_show_results.assert_called_once()

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._setup_output_directory")
    @patch("devdox_ai_locust.cli._process_api_schema")
    async def test_async_generate_schema_error(
        self, mock_process_schema, mock_setup_output, mock_init_config, temp_dir
    ):
        """Test async generate with schema processing error."""
        # Setup mocks
        mock_init_config.return_value = (Mock(), "test-api-key")
        mock_setup_output.return_value = temp_dir
        mock_process_schema.side_effect = Exception("Schema error")

        # Create mock context
        mock_ctx = Mock()
        mock_ctx.obj = {"verbose": False}

        # Test should raise exception
        with pytest.raises(Exception, match="Schema error"):
            await _async_generate(
                mock_ctx,
                "https://api.example.com/swagger.json",
                "output",
                10,
                2,
                "5m",
                None,
                True,
                "",
                False,
                None,
                "test-api-key",
            )

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._setup_output_directory")
    @patch("devdox_ai_locust.cli._process_api_schema")
    @patch("devdox_ai_locust.cli._generate_and_create_tests")
    async def test_async_generate_with_verbose(
        self,
        mock_generate_tests,
        mock_process_schema,
        mock_setup_output,
        mock_init_config,
        temp_dir,
        sample_endpoints,
        sample_api_info,
    ):
        """Test async generate with verbose output."""
        # Setup mocks
        mock_init_config.return_value = (Mock(), "test-api-key")
        mock_setup_output.return_value = temp_dir
        mock_process_schema.return_value = (None, sample_endpoints, sample_api_info)
        mock_generate_tests.return_value = [{"path": "test_file.py"}]

        # Create mock context with verbose=True
        mock_ctx = Mock()
        mock_ctx.obj = {"verbose": True}

        # Test the function
        with patch("devdox_ai_locust.cli._display_configuration") as mock_display:
            with patch("devdox_ai_locust.cli._show_results"):
                await _async_generate(
                    mock_ctx,
                    "https://api.example.com/swagger.json",
                    "output",
                    10,
                    2,
                    "5m",
                    None,
                    True,
                    "",
                    False,
                    None,
                    "test-api-key",
                )

                # Should call display configuration when verbose
                mock_display.assert_called_once()


class TestProcessApiSchema:
    """Test API schema processing functionality."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.get_api_schema")
    @patch("devdox_ai_locust.cli.OpenAPIParser")
    async def test_process_api_schema_success(
        self, mock_parser_class, mock_get_schema, sample_endpoints, sample_api_info
    ):
        """Test successful API schema processing."""
        # Mock the schema fetching
        mock_get_schema.return_value = '{"openapi": "3.0.0"}'

        # Mock the parser
        mock_parser = Mock()
        mock_parser.parse_schema.return_value = {"openapi": "3.0.0"}
        mock_parser.parse_endpoints.return_value = sample_endpoints
        mock_parser.get_schema_info.return_value = sample_api_info
        mock_parser_class.return_value = mock_parser

        # Test the function
        schema_data, endpoints, api_info = await _process_api_schema(
            "https://api.example.com/swagger.json", verbose=False
        )

        # Verify results
        assert endpoints == sample_endpoints
        assert api_info == sample_api_info
        mock_get_schema.assert_called_once()
        mock_parser.parse_schema.assert_called_once()
        mock_parser.parse_endpoints.assert_called_once()
        mock_parser.get_schema_info.assert_called_once()


class TestGenerateAndCreateTests:
    """Test test generation and creation functionality."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.AsyncTogether")
    @patch("devdox_ai_locust.cli.HybridLocustGenerator")
    async def test_generate_and_create_tests_success(
        self,
        mock_generator_class,
        mock_together_class,
        temp_dir,
        sample_endpoints,
        sample_api_info,
    ):
        """Test successful test generation and creation."""
        # Mock Together client
        mock_client = AsyncMock()
        mock_together_class.return_value = mock_client

        # Mock generator
        mock_generator = Mock()
        mock_generator.generate_from_endpoints = AsyncMock(
            return_value=(
                {"test_file.py": "test content"},
                [{"workflow_file.py": "workflow content"}],
            )
        )
        mock_generator._create_test_files_safely = AsyncMock(
            return_value=[{"path": "created_file.py"}]
        )
        mock_generator_class.return_value = mock_generator

        # Test the function
        created_files = await _generate_and_create_tests(
            api_key="test-api-key",
            endpoints=sample_endpoints,
            api_info=sample_api_info,
            output_dir=temp_dir,
            custom_requirement="test requirement",
            host="http://localhost:8000",
            auth=True,
        )

        # Verify calls
        mock_together_class.assert_called_once_with(api_key="test-api-key")
        mock_generator_class.assert_called_once_with(ai_client=mock_client)
        mock_generator.generate_from_endpoints.assert_called_once()
        assert len(created_files) > 0


class TestCLIHelperFunctions:
    """Test CLI helper functions."""

    def test_display_configuration(self, temp_dir):
        """Test display configuration function."""
        # This should not raise an exception
        _display_configuration(
            swagger_url="https://api.example.com/swagger.json",
            output_dir=temp_dir,
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host="http://localhost:8000",
            auth=True,
            custom_requirement="test requirement",
            dry_run=False,
        )

    def test_show_generated_files_verbose(self):
        """Test showing generated files in verbose mode."""
        files = [{"path": "file1.py"}, {"path": "file2.py"}, {"path": "file3.py"}]

        # Should not raise an exception
        _show_generated_files(files, verbose=True)

    def test_show_generated_files_non_verbose(self):
        """Test showing generated files in non-verbose mode."""
        files = [{"path": f"file{i}.py"} for i in range(15)]  # More than 10 files

        # Should not raise an exception
        _show_generated_files(files, verbose=False)

    def test_show_run_instructions(self, temp_dir):
        """Test showing run instructions."""
        # Create a locustfile.py
        locustfile = temp_dir / "locustfile.py"
        locustfile.write_text("# Locust file")

        # Should not raise an exception
        _show_run_instructions(
            output_dir=temp_dir,
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host="http://localhost:8000",
        )

    def test_show_run_instructions_no_locustfile(self, temp_dir):
        """Test showing run instructions when no locustfile.py exists."""
        # Create some other Python file
        test_file = temp_dir / "test.py"
        test_file.write_text("# Test file")

        # Should not raise an exception
        _show_run_instructions(
            output_dir=temp_dir,
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host=None,  # Test with None host
        )

    @patch("devdox_ai_locust.cli.sys.exit")
    def test_show_results_no_files(self, mock_exit, temp_dir):
        """Test show results when no files were created."""
        start_time = datetime.now(timezone.utc)

        _show_results(
            created_files=[],
            output_dir=temp_dir,
            start_time=start_time,
            verbose=False,
            dry_run=False,
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host="http://localhost:8000",
        )

        mock_exit.assert_called_once_with(1)

    def test_show_results_success(self, temp_dir):
        """Test show results with successful file creation."""
        start_time = datetime.now(timezone.utc)
        created_files = [{"path": "file1.py"}, {"path": "file2.py"}]

        # Should not raise an exception
        _show_results(
            created_files=created_files,
            output_dir=temp_dir,
            start_time=start_time,
            verbose=True,
            dry_run=True,  # Test with dry run
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host="http://localhost:8000",
        )


class TestCLIEdgeCases:
    """Test CLI edge cases and error conditions."""

    def test_generate_command_exception_handling(self, cli_runner):
        """Test generate command exception handling."""
        # Test with invalid arguments that should cause an error
        result = cli_runner.invoke(
            cli, ["generate", "invalid-url", "--together-api-key", "test-key"]
        )

        # Should exit with error code
        assert result.exit_code != 0

    def test_main_function(self):
        """Test main function entry point."""
        from devdox_ai_locust.cli import main

        # Test that main function exists and can be called
        # (We can't actually call it without arguments as it would invoke Click)
        assert callable(main)
