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
    run_generate,
    _initialize_config,
    _setup_output_directory,
    _display_configuration,
    _show_results,
    _show_generated_files,
    _show_run_instructions,
    _process_api_schema,
    _generate_and_create_tests,
)
from devdox_ai_locust.schemas.cli_dto import GenerateParams, EndpointProcessingContext
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


def _make_generate_dto(**overrides):
    """Helper to build a GenerateParams with sensible defaults."""
    defaults = dict(
        swagger_url="https://api.example.com/swagger.json",
        output="output",
        users=10,
        spawn_rate=2.0,
        run_time="5m",
        host=None,
        auth=True,
        db_type="",
        dry_run=False,
        custom_requirement=None,
        together_api_key="test-api-key",
        timeout=120,
        schema_timeout=30,
        max_llm_workers=1,
        debug=False,
        verbose=False,
    )
    defaults.update(overrides)
    return GenerateParams(**defaults)


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
        assert "Generate load test files from an API specification" in result.output

    def test_run_command_help(self, cli_runner):
        """Test run command help."""
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run previously generated Locust load tests" in result.output

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
    def test_generate_command_basic(self, mock_asyncio_run, cli_runner):
        """Test basic generate command."""
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

    @patch("devdox_ai_locust.cli._teardown_logging")
    @patch("devdox_ai_locust.cli._setup_logging")
    @patch("devdox_ai_locust.cli._execute_locust_process")
    def test_run_command_basic(
        self, mock_execute, mock_setup_log, mock_teardown_log, cli_runner, temp_dir
    ):
        """Test basic run command."""
        # Create a dummy test file
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        # Mock logging setup to not modify stdout
        mock_log_file = Mock()
        mock_setup_log.return_value = (temp_dir / "test.log", mock_log_file)

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

        mock_execute.assert_called_once()

    @patch("devdox_ai_locust.cli._teardown_logging")
    @patch("devdox_ai_locust.cli._setup_logging")
    @patch("devdox_ai_locust.cli._execute_locust_process")
    def test_run_command_headless(
        self, mock_execute, mock_setup_log, mock_teardown_log, cli_runner, temp_dir
    ):
        """Test run command with headless flag."""
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        mock_log_file = Mock()
        mock_setup_log.return_value = (temp_dir / "test.log", mock_log_file)

        _ = cli_runner.invoke(
            cli,
            ["run", str(test_file), "--host", "http://localhost:8000", "--headless"],
        )

        mock_execute.assert_called_once()

    def test_run_command_file_not_found(self, cli_runner):
        """Test run command with non-existent file."""
        result = cli_runner.invoke(
            cli, ["run", "/non/existent/file.py", "--host", "http://localhost:8000"]
        )

        assert result.exit_code != 0

    @patch("devdox_ai_locust.cli._teardown_logging")
    @patch("devdox_ai_locust.cli._setup_logging")
    @patch("devdox_ai_locust.cli._execute_locust_process")
    def test_run_command_subprocess_error(
        self, mock_execute, mock_setup_log, mock_teardown_log, cli_runner, temp_dir
    ):
        """Test run command with subprocess error."""
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        mock_log_file = Mock()
        mock_setup_log.return_value = (temp_dir / "test.log", mock_log_file)

        mock_execute.side_effect = SystemExit(1)

        result = cli_runner.invoke(
            cli, ["run", str(test_file), "--host", "http://localhost:8000"]
        )

        assert result.exit_code == 1

    @patch("devdox_ai_locust.cli._teardown_logging")
    @patch("devdox_ai_locust.cli._setup_logging")
    @patch("devdox_ai_locust.cli._build_locust_command")
    def test_run_command_locust_not_found(
        self, mock_build_cmd, mock_setup_log, mock_teardown_log, cli_runner, temp_dir
    ):
        """Test run command when locust is not installed."""
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test locust file")

        mock_log_file = Mock()
        mock_setup_log.return_value = (temp_dir / "test.log", mock_log_file)

        # Make _execute_locust_process raise FileNotFoundError via the actual path
        with patch(
            "devdox_ai_locust.cli._execute_locust_process",
            side_effect=FileNotFoundError(),
        ):
            result = cli_runner.invoke(
                cli, ["run", str(test_file), "--host", "http://localhost:8000"]
            )

        assert result.exit_code == 1


class TestRunGenerate:
    """Test run_generate (was _async_generate) functionality."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.DebugRecorder")
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._process_api_schema")
    @patch("devdox_ai_locust.cli._generate_and_create_tests")
    @patch("devdox_ai_locust.cli._show_results")
    async def test_run_generate_success(
        self,
        mock_show_results,
        mock_generate_tests,
        mock_process_schema,
        mock_init_config,
        mock_debug_recorder,
        temp_dir,
        sample_endpoints,
        sample_api_info,
    ):
        """Test successful run_generate."""
        mock_init_config.return_value = (Mock(), "test-api-key")
        mock_process_schema.return_value = (None, sample_endpoints, sample_api_info)
        mock_generate_tests.return_value = [{"path": "test_file.py"}]

        dto = _make_generate_dto(together_api_key="test-api-key")

        await run_generate(dto, temp_dir)

        mock_init_config.assert_called_once()
        mock_process_schema.assert_called_once()
        mock_generate_tests.assert_called_once()
        mock_show_results.assert_called_once()

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.DebugRecorder")
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._process_api_schema")
    async def test_run_generate_schema_error(
        self,
        mock_process_schema,
        mock_init_config,
        mock_debug_recorder,
        temp_dir,
    ):
        """Test run_generate with schema processing error."""
        mock_init_config.return_value = (Mock(), "test-api-key")
        mock_process_schema.side_effect = Exception("Schema error")

        dto = _make_generate_dto(together_api_key="test-api-key")

        with pytest.raises(Exception, match="Schema error"):
            await run_generate(dto, temp_dir)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.DebugRecorder")
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._process_api_schema")
    @patch("devdox_ai_locust.cli._generate_and_create_tests")
    async def test_run_generate_with_verbose(
        self,
        mock_generate_tests,
        mock_process_schema,
        mock_init_config,
        mock_debug_recorder,
        temp_dir,
        sample_endpoints,
        sample_api_info,
    ):
        """Test run_generate with verbose output."""
        mock_init_config.return_value = (Mock(), "test-api-key")
        mock_process_schema.return_value = (None, sample_endpoints, sample_api_info)
        mock_generate_tests.return_value = [{"path": "test_file.py"}]

        dto = _make_generate_dto(verbose=True, together_api_key="test-api-key")

        with patch("devdox_ai_locust.cli._display_configuration") as mock_display:
            with patch("devdox_ai_locust.cli._show_results"):
                await run_generate(dto, temp_dir)

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
    @patch("devdox_ai_locust.cli._generate_scenario_based_tests")
    @patch("devdox_ai_locust.cli.AsyncTogether")
    async def test_generate_and_create_tests_success(
        self,
        mock_together_class,
        mock_scenario_tests,
        temp_dir,
        sample_endpoints,
        sample_api_info,
    ):
        """Test successful test generation and creation."""
        mock_client = AsyncMock()
        mock_together_class.return_value = mock_client

        mock_scenario_tests.return_value = [{"path": "created_file.py"}]

        dto = _make_generate_dto(
            together_api_key="test-api-key",
            host="http://localhost:8000",
            custom_requirement="test requirement",
        )

        created_files = await _generate_and_create_tests(
            dto=dto,
            endpoints=sample_endpoints,
            api_info=sample_api_info,
            output_dir=temp_dir,
        )

        mock_together_class.assert_called_once_with(api_key="test-api-key")
        mock_scenario_tests.assert_called_once()
        assert created_files == [{"path": "created_file.py"}]


class TestCLIHelperFunctions:
    """Test CLI helper functions."""

    def test_display_configuration(self, temp_dir):
        """Test display configuration function."""
        dto = _make_generate_dto(
            swagger_url="https://api.example.com/swagger.json",
            host="http://localhost:8000",
            custom_requirement="test requirement",
        )
        _display_configuration(dto, temp_dir)

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


class TestTeeOutput:
    """Test TeeOutput class."""

    def test_write(self):
        from devdox_ai_locust.cli import TeeOutput
        from io import StringIO

        original = StringIO()
        log_file = StringIO()
        tee = TeeOutput(original, log_file)

        result = tee.write("hello")
        assert result == 5
        assert original.getvalue() == "hello"
        assert log_file.getvalue() == "hello"

    def test_flush(self):
        from devdox_ai_locust.cli import TeeOutput

        original = Mock()
        log_file = Mock()
        tee = TeeOutput(original, log_file)
        tee.flush()
        original.flush.assert_called_once()
        log_file.flush.assert_called_once()

    def test_fileno(self):
        from devdox_ai_locust.cli import TeeOutput

        original = Mock()
        original.fileno.return_value = 1
        log_file = Mock()
        tee = TeeOutput(original, log_file)
        assert tee.fileno() == 1

    def test_isatty(self):
        from devdox_ai_locust.cli import TeeOutput

        original = Mock()
        original.isatty.return_value = False
        log_file = Mock()
        tee = TeeOutput(original, log_file)
        assert tee.isatty() is False


class TestSetupAndTeardownLogging:
    """Test logging setup/teardown functions."""

    def test_setup_logging(self, temp_dir):
        from devdox_ai_locust.cli import _setup_logging
        import sys

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            log_path, log_file = _setup_logging(temp_dir, "test")
            assert log_path.exists()
            assert "test" in log_path.name
            log_file.close()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def test_teardown_logging(self, temp_dir):
        from devdox_ai_locust.cli import _teardown_logging
        import sys

        log_path = temp_dir / "test.log"
        log_file = open(log_path, "w")
        # Set stdout/stderr to something else to verify restore
        sys.stdout = Mock()
        sys.stderr = Mock()
        _teardown_logging(log_file, log_path)
        assert sys.stdout is sys.__stdout__
        assert sys.stderr is sys.__stderr__


class TestHelperFunctions:
    """Test various helper functions in cli.py."""

    def test_sanitize_dir_name_basic(self):
        from devdox_ai_locust.cli import _sanitize_dir_name

        assert _sanitize_dir_name("Hello-World") == "hello_world"
        assert _sanitize_dir_name("foo.bar") == "foo_bar"
        assert _sanitize_dir_name("a/b/c") == "a_b_c"
        assert _sanitize_dir_name("UPPER") == "upper"

    def test_sanitize_dir_name_special_chars(self):
        from devdox_ai_locust.cli import _sanitize_dir_name

        assert _sanitize_dir_name("a@b#c") == "abc"
        assert _sanitize_dir_name("") == "unnamed"
        assert _sanitize_dir_name("___") == "unnamed"
        assert _sanitize_dir_name("a__b") == "a_b"

    def test_to_class_name(self):
        from devdox_ai_locust.cli import _to_class_name

        assert _to_class_name("get_users") == "GetUsers"
        assert _to_class_name("hello-world") == "HelloWorld"
        assert _to_class_name("") == "Unnamed"

    def test_group_endpoints_by_tag(self, sample_endpoints):
        from devdox_ai_locust.cli import _group_endpoints_by_tag

        grouped = _group_endpoints_by_tag(sample_endpoints)
        assert "users" in grouped
        assert len(grouped["users"]) == 2

    def test_group_endpoints_no_tags(self):
        from devdox_ai_locust.cli import _group_endpoints_by_tag
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        ep = Endpoint(
            path="/test",
            method="GET",
            operation_id="test",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=[],
        )
        grouped = _group_endpoints_by_tag([ep])
        assert "default" in grouped

    def test_detect_auth_endpoints(self, sample_endpoints):
        from devdox_ai_locust.cli import _detect_auth_endpoints
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        # Add an auth-like endpoint
        auth_ep = Endpoint(
            path="/auth/login",
            method="POST",
            operation_id="login",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["auth"],
        )
        token_ep = Endpoint(
            path="/api/token",
            method="POST",
            operation_id="token",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["auth"],
        )
        non_auth = Endpoint(
            path="/api/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )
        result = _detect_auth_endpoints([auth_ep, token_ep, non_auth])
        assert len(result) == 2

    def test_prepare_workflows_dir(self, temp_dir):
        from devdox_ai_locust.cli import _prepare_workflows_dir

        workflows_dir = _prepare_workflows_dir(temp_dir)
        assert workflows_dir.exists()
        assert workflows_dir.name == "workflows"

    def test_prepare_workflows_dir_cleans_existing(self, temp_dir):
        from devdox_ai_locust.cli import _prepare_workflows_dir

        # Create existing workflows dir with content
        existing = temp_dir / "workflows"
        existing.mkdir()
        (existing / "old_file.txt").write_text("old")

        workflows_dir = _prepare_workflows_dir(temp_dir)
        assert workflows_dir.exists()
        assert not (workflows_dir / "old_file.txt").exists()

    def test_build_endpoint_tag_mapping(self):
        from devdox_ai_locust.cli import _build_endpoint_tag_mapping

        ep1 = Mock()
        ep2 = Mock()
        grouped = {"users": [ep1], "auth": [ep2]}
        mapping = _build_endpoint_tag_mapping(grouped)
        assert mapping[id(ep1)] == "users"
        assert mapping[id(ep2)] == "auth"

    def test_generation_state(self):
        from devdox_ai_locust.cli import _GenerationState

        state = _GenerationState()
        assert state.created_files == []
        assert state.failed_endpoints == []
        assert state.completed_count == 0
        assert state.failed_count == 0
        assert len(state.successful_endpoints) == 0

    def test_display_configuration_all_options(self, temp_dir):
        """Test display configuration with all optional params."""
        dto = _make_generate_dto(
            swagger_url="https://api.example.com/swagger.json",
            host=None,
            auth=False,
            custom_requirement="A" * 100,
            dry_run=True,
            db_type="mongo",
            timeout=300,
            debug=True,
        )
        _display_configuration(dto, temp_dir)

    def test_show_results_success_not_dry_run(self, temp_dir):
        """Test show results without dry run shows run instructions."""
        start_time = datetime.now(timezone.utc)
        created_files = [{"path": "file1.py"}]
        # Create a locustfile so _show_run_instructions finds it
        (temp_dir / "locustfile.py").write_text("# locust")

        _show_results(
            created_files=created_files,
            output_dir=temp_dir,
            start_time=start_time,
            verbose=False,
            dry_run=False,
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host="http://localhost:8000",
        )

    def test_show_run_instructions_no_py_files(self, temp_dir):
        """Test run instructions with no Python files."""
        _show_run_instructions(
            output_dir=temp_dir,
            users=10,
            spawn_rate=2.0,
            run_time="5m",
            host="http://localhost:8000",
        )

    def test_write_base_files(self, temp_dir):
        from devdox_ai_locust.cli import _write_base_files

        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()
        base_files = {
            "base_workflow.py": "# base",
            "locustfile.py": "# locust",
        }
        created = []
        _write_base_files(base_files, temp_dir, workflows_dir, created)
        assert len(created) == 2
        assert (workflows_dir / "base_workflow.py").exists()
        assert (temp_dir / "locustfile.py").exists()

    def test_report_orchestrator_results_all_success(self, capsys):
        from devdox_ai_locust.cli import _report_orchestrator_results

        _report_orchestrator_results(
            orchestrator_files=[{"path": "a.py"}],
            orchestrator_failures=[],
            num_tags=1,
        )

    def test_report_orchestrator_results_with_failures(self):
        from devdox_ai_locust.cli import _report_orchestrator_results

        _report_orchestrator_results(
            orchestrator_files=[{"path": "a.py"}],
            orchestrator_failures=[{"tag": "x", "error": "fail"}],
            num_tags=2,
        )

    def test_print_failure_details(self):
        from devdox_ai_locust.cli import _print_failure_details

        failures = [
            {"endpoint": f"EP{i}", "error": f"err{i}", "error_type": "ValueError"}
            for i in range(12)
        ]
        _print_failure_details(failures)

    def test_record_debug_cli_args_disabled(self):
        from devdox_ai_locust.cli import _record_debug_cli_args

        recorder = Mock()
        dto = _make_generate_dto(debug=False)
        _record_debug_cli_args(dto, recorder)
        recorder.record_cli_args.assert_not_called()

    def test_record_debug_cli_args_enabled(self):
        from devdox_ai_locust.cli import _record_debug_cli_args

        recorder = Mock()
        dto = _make_generate_dto(debug=True)
        _record_debug_cli_args(dto, recorder)
        recorder.record_cli_args.assert_called_once()

    def test_record_debug_parsed_schema_disabled(self):
        from devdox_ai_locust.cli import _record_debug_parsed_schema

        recorder = Mock()
        _record_debug_parsed_schema(
            False,
            recorder,
            {},
            [],
            {},
            None,
            True,
            "",
            120,
            None,
        )
        recorder.record_openapi_raw.assert_not_called()

    def test_record_debug_parsed_schema_enabled(self):
        from devdox_ai_locust.cli import _record_debug_parsed_schema

        recorder = Mock()
        _record_debug_parsed_schema(
            True,
            recorder,
            {"raw": True},
            [],
            {"title": "API"},
            "host",
            True,
            "mongo",
            120,
            "req",
        )
        recorder.record_openapi_raw.assert_called_once()
        recorder.record_openapi_parsed.assert_called_once()
        recorder.record_resolved_config.assert_called_once()


class TestGenerateCommand:
    """Test the generate CLI command edge cases."""

    def test_generate_max_workers_exceeds_10(self, cli_runner):
        """Test generate command rejects max-llm-workers > 10."""
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "https://api.example.com/swagger.json",
                "--together-api-key",
                "test-key",
                "--max-llm-workers",
                "11",
            ],
        )
        assert result.exit_code != 0

    @patch("devdox_ai_locust.cli.asyncio.run")
    def test_generate_verbose_exception(self, mock_run, cli_runner, temp_dir):
        """Test generate command with verbose mode and exception."""
        mock_run.side_effect = RuntimeError("async fail")
        result = cli_runner.invoke(
            cli,
            [
                "--verbose",
                "generate",
                "https://api.example.com/swagger.json",
                "--output",
                str(temp_dir),
                "--together-api-key",
                "test-key",
            ],
        )
        assert result.exit_code == 1


class TestProcessApiSchemaEdgeCases:
    """Test _process_api_schema edge cases."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.get_api_schema")
    async def test_process_api_schema_none_result(self, mock_get_schema):
        """Test schema processing when get_api_schema returns None."""
        mock_get_schema.return_value = None
        with pytest.raises(SystemExit):
            await _process_api_schema("https://api.example.com/swagger.json", False)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.get_api_schema")
    async def test_process_api_schema_timeout(self, mock_get_schema):
        """Test schema processing timeout."""
        import asyncio as aio

        async def slow_fetch(*args, **kwargs):
            await aio.sleep(10)

        mock_get_schema.side_effect = slow_fetch
        with pytest.raises(SystemExit):
            await _process_api_schema(
                "https://api.example.com/swagger.json", False, schema_timeout=0
            )

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.get_api_schema")
    async def test_process_api_schema_exception(self, mock_get_schema):
        """Test schema processing with generic exception."""
        mock_get_schema.side_effect = ValueError("bad schema")
        with pytest.raises(SystemExit):
            await _process_api_schema("https://api.example.com/swagger.json", False)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.get_api_schema")
    @patch("devdox_ai_locust.cli.OpenAPIParser")
    async def test_process_api_schema_parse_failure(
        self, mock_parser_class, mock_get_schema
    ):
        """Test schema processing when parser fails."""
        mock_get_schema.return_value = '{"openapi": "3.0.0"}'
        mock_parser = Mock()
        mock_parser.parse_schema.side_effect = ValueError("parse error")
        mock_parser_class.return_value = mock_parser
        with pytest.raises(SystemExit):
            await _process_api_schema("https://api.example.com/swagger.json", False)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli.get_api_schema")
    @patch("devdox_ai_locust.cli.OpenAPIParser")
    async def test_process_api_schema_file_source(
        self, mock_parser_class, mock_get_schema
    ):
        """Test schema processing from file path."""
        mock_get_schema.return_value = '{"openapi": "3.0.0"}'
        mock_parser = Mock()
        mock_parser.parse_schema.return_value = {}
        mock_parser.parse_endpoints.return_value = []
        mock_parser.get_schema_info.return_value = {"title": "Test"}
        mock_parser_class.return_value = mock_parser

        result = await _process_api_schema("/path/to/file.json", False)
        assert result[2]["title"] == "Test"


class TestRunCommandVerbose:
    """Test run command verbose mode."""

    @patch("devdox_ai_locust.cli._teardown_logging")
    @patch("devdox_ai_locust.cli._setup_logging")
    @patch("devdox_ai_locust.cli._execute_locust_process")
    def test_run_command_verbose(
        self, mock_execute, mock_setup_log, mock_teardown_log, cli_runner, temp_dir
    ):
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test")

        mock_log_file = Mock()
        mock_setup_log.return_value = (temp_dir / "test.log", mock_log_file)

        cli_runner.invoke(
            cli,
            [
                "--verbose",
                "run",
                str(test_file),
                "--host",
                "http://localhost:8000",
            ],
        )

    @patch("devdox_ai_locust.cli._teardown_logging")
    @patch("devdox_ai_locust.cli._setup_logging")
    @patch("devdox_ai_locust.cli._execute_locust_process")
    def test_run_command_generic_exception(
        self, mock_execute, mock_setup_log, mock_teardown_log, cli_runner, temp_dir
    ):
        test_file = temp_dir / "test_locustfile.py"
        test_file.write_text("# Test")

        mock_log_file = Mock()
        mock_setup_log.return_value = (temp_dir / "test.log", mock_log_file)
        mock_execute.side_effect = RuntimeError("unexpected")

        result = cli_runner.invoke(
            cli,
            ["run", str(test_file), "--host", "http://localhost:8000"],
        )
        assert result.exit_code == 1


class TestGeneratePreLlmWorkflow:
    """Test _generate_pre_llm_workflow function."""

    def test_generates_valid_python(self):
        from devdox_ai_locust.cli import _generate_pre_llm_workflow
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="Get users",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        mock_scenario_gen = Mock()
        mock_scenario_gen.get_endpoint_dir_name.return_value = "get_users"

        mock_template_gen = Mock()
        mock_template_gen._generate_task_method.return_value = (
            "@task\ndef test_get(self):\n    pass"
        )

        result = _generate_pre_llm_workflow(
            ep, "positive", mock_scenario_gen, mock_template_gen
        )
        assert "class GetUsersPositiveWorkflow" in result
        assert "Pre-LLM workflow" in result
        assert "BaseWorkflow" in result


class TestGeneratePreLlmTemplates:
    """Test _generate_pre_llm_templates function."""

    @patch("devdox_ai_locust.cli._generate_pre_llm_workflow")
    def test_generates_all_scenario_types(self, mock_gen_workflow):
        from devdox_ai_locust.cli import _generate_pre_llm_templates
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        mock_gen_workflow.return_value = "# workflow code"

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        mock_scenario_gen = Mock()
        mock_template_gen = Mock()

        result = _generate_pre_llm_templates([ep], mock_scenario_gen, mock_template_gen)
        # 1 endpoint * 3 scenario types = 3 templates
        assert len(result) == 3

    @patch("devdox_ai_locust.cli._generate_pre_llm_workflow")
    def test_exits_on_error(self, mock_gen_workflow):
        from devdox_ai_locust.cli import _generate_pre_llm_templates
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        mock_gen_workflow.side_effect = RuntimeError("template error")

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        with pytest.raises(SystemExit):
            _generate_pre_llm_templates([ep], Mock(), Mock())


class TestCreateInitFiles:
    """Test _create_init_files function."""

    def test_creates_init_files(self, temp_dir):
        from devdox_ai_locust.cli import _create_init_files
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        # Create tag dir and endpoint dir with scenario files
        tag_dir = workflows_dir / "users"
        tag_dir.mkdir()
        ep_dir = tag_dir / "get_users"
        ep_dir.mkdir()
        (ep_dir / "positive_workflow.py").write_text("# positive")
        (tag_dir / "orchestrator_workflow.py").write_text("# orch")

        mock_scenario_gen = Mock()
        mock_scenario_gen.get_endpoint_dir_name.return_value = "get_users"

        grouped = {"users": [ep]}
        _create_init_files(workflows_dir, grouped, mock_scenario_gen)

        assert (workflows_dir / "__init__.py").exists()
        assert (tag_dir / "__init__.py").exists()
        init_content = (tag_dir / "__init__.py").read_text()
        assert "positive_workflow" in init_content
        assert "orchestrator_workflow" in init_content

    def test_skips_nonexistent_tag_dir(self, temp_dir):
        from devdox_ai_locust.cli import _create_init_files
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        mock_scenario_gen = Mock()
        grouped = {"nonexistent": [ep]}
        _create_init_files(workflows_dir, grouped, mock_scenario_gen)
        assert (workflows_dir / "__init__.py").exists()


class TestSaveFailureCode:
    """Test _save_failure_code async function."""

    @pytest.mark.asyncio
    async def test_no_code_attribute(self):
        from devdox_ai_locust.cli import _save_failure_code

        e = ValueError("plain error")
        result = await _save_failure_code(e, Path("."), "op", "GET /x")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_code_attribute(self):
        from devdox_ai_locust.cli import _save_failure_code

        e = ValueError("error")
        e.code = ""
        result = await _save_failure_code(e, Path("."), "op", "GET /x")
        assert result is None

    @pytest.mark.asyncio
    async def test_saves_failure_code(self, temp_dir):
        from devdox_ai_locust.cli import _save_failure_code

        e = ValueError("error")
        e.code = "bad python code"
        e.error = "syntax error"
        e.scenario_type = "positive"

        result = await _save_failure_code(e, temp_dir, "get_users", "GET /users")
        assert result is not None
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "FAILED CODE" in content


class TestWriteFallbackFiles:
    """Test _write_fallback_files async function."""

    @pytest.mark.asyncio
    async def test_writes_fallback_templates(self, temp_dir):
        from devdox_ai_locust.cli import _write_fallback_files
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        endpoint_dir = temp_dir / "workflows" / "users" / "get_users"

        mock_scenario_gen = Mock()
        mock_scenario_gen.get_endpoint_dir_name.return_value = "get_users"

        mock_template_gen = Mock()
        mock_template_gen._generate_task_method.return_value = (
            "@task\ndef test(self):\n    pass"
        )

        # Provide pre_llm_templates for positive, empty for others
        pre_llm_templates = {
            (id(ep), "positive"): "# positive fallback",
            (id(ep), "negative"): "# negative fallback",
        }

        result = await _write_fallback_files(
            endpoint=ep,
            endpoint_dir=endpoint_dir,
            tag_name="users",
            operation_id="get_users",
            pre_llm_templates=pre_llm_templates,
            scenario_gen=mock_scenario_gen,
            template_gen=mock_template_gen,
        )
        assert len(result) == 3
        assert all(f.get("fallback") for f in result)


class TestReportGenerationSummary:
    """Test _report_generation_summary function."""

    def test_all_success(self):
        from devdox_ai_locust.cli import _report_generation_summary

        mock_gen = Mock()
        mock_gen.get_rate_limit_info.return_value = Mock(requests_per_minute=60)
        mock_gen.current_concurrency = 1

        _report_generation_summary(mock_gen, [], 5, 0, 5)

    def test_with_failures(self):
        from devdox_ai_locust.cli import _report_generation_summary

        mock_gen = Mock()
        mock_gen.get_rate_limit_info.return_value = Mock(requests_per_minute=60)
        mock_gen.current_concurrency = 1

        failures = [{"endpoint": "GET /x", "error": "fail", "error_type": "Err"}]
        _report_generation_summary(mock_gen, failures, 4, 1, 5)


class TestInitProgress:
    """Test _init_progress function."""

    def test_creates_progress(self):
        from devdox_ai_locust.cli import _init_progress

        mock_gen = Mock()
        mock_gen.current_concurrency = 2
        progress = _init_progress(mock_gen, 10, False)
        assert progress.total == 10
        assert mock_gen.progress == progress


class TestGenerateBaseFiles:
    """Test _generate_base_files function."""

    def test_with_debug_recorder(self):
        from devdox_ai_locust.cli import _generate_base_files

        mock_template_gen = Mock()
        mock_template_gen.generate_from_endpoints.return_value = (
            {"base_workflow.py": "# base", "test_data.py": "# data"},
            None,
            None,
        )
        mock_template_gen.fix_indent.return_value = {
            "base_workflow.py": "# base",
            "test_data.py": "# data",
        }

        mock_recorder = Mock()
        mock_recorder.enabled = True

        result = _generate_base_files(
            mock_template_gen,
            [],
            {"title": "API"},
            True,
            "host",
            "mongo",
            mock_recorder,
        )
        assert "base_workflow.py" in result
        assert mock_recorder.record_static_file.call_count == 2

    def test_without_debug_recorder(self):
        from devdox_ai_locust.cli import _generate_base_files

        mock_template_gen = Mock()
        mock_template_gen.generate_from_endpoints.return_value = (
            {"base.py": "# base"},
            None,
            None,
        )
        mock_template_gen.fix_indent.return_value = {"base.py": "# base"}

        result = _generate_base_files(
            mock_template_gen,
            [],
            {},
            False,
            None,
            "",
            None,
        )
        assert "base.py" in result


class TestProcessAndSaveEndpoint:
    """Test _process_and_save_endpoint async function."""

    @pytest.mark.asyncio
    async def test_success_path(self, temp_dir):
        from devdox_ai_locust.cli import _process_and_save_endpoint, _GenerationState
        from devdox_ai_locust.utils.open_ai_parser import Endpoint
        from enum import Enum

        class ScenarioType(Enum):
            POSITIVE = "positive"

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        state = _GenerationState()
        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        mock_scenario_gen = Mock()
        mock_scenario_gen.get_endpoint_dir_name.return_value = "get_users"
        mock_scenario_gen.generate_endpoint_workflows = AsyncMock(
            return_value={ScenarioType.POSITIVE: "# positive code"}
        )
        mock_scenario_gen.SCENARIO_FILES = {
            ScenarioType.POSITIVE: "positive_workflow.py"
        }

        mock_progress = Mock()

        ctx = EndpointProcessingContext(
            scenario_gen=mock_scenario_gen,
            template_gen=Mock(),
            workflows_dir=workflows_dir,
            base_workflow_content="# base",
            test_data_content="# data",
            auth_endpoints=None,
            all_endpoints=[ep],
            custom_requirement=None,
            db_type="",
            pre_llm_templates={},
            endpoint_to_tag={id(ep): "users"},
        )

        result = await _process_and_save_endpoint(
            endpoint=ep,
            state=state,
            ctx=ctx,
            progress=mock_progress,
        )
        assert state.completed_count == 1
        assert id(ep) in state.successful_endpoints
        assert len(result) == 1
        mock_progress.endpoint_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_path(self, temp_dir):
        from devdox_ai_locust.cli import _process_and_save_endpoint, _GenerationState
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        state = _GenerationState()
        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        mock_scenario_gen = Mock()
        mock_scenario_gen.get_endpoint_dir_name.return_value = "get_users"
        mock_scenario_gen.generate_endpoint_workflows = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )

        mock_template_gen = Mock()
        mock_template_gen._generate_task_method.return_value = (
            "@task\ndef test(self):\n    pass"
        )

        mock_progress = Mock()

        pre_llm = {
            (id(ep), "positive"): "# pos",
            (id(ep), "negative"): "# neg",
            (id(ep), "security"): "# sec",
        }

        ctx = EndpointProcessingContext(
            scenario_gen=mock_scenario_gen,
            template_gen=mock_template_gen,
            workflows_dir=workflows_dir,
            base_workflow_content="# base",
            test_data_content="# data",
            auth_endpoints=None,
            all_endpoints=[ep],
            custom_requirement=None,
            db_type="",
            pre_llm_templates=pre_llm,
            endpoint_to_tag={id(ep): "users"},
        )

        await _process_and_save_endpoint(
            endpoint=ep,
            state=state,
            ctx=ctx,
            progress=mock_progress,
        )
        assert state.failed_count == 1
        assert len(state.failed_endpoints) == 1
        mock_progress.endpoint_failed.assert_called_once()


class TestGenerateOrchestrators:
    """Test _generate_orchestrators async function."""

    @pytest.mark.asyncio
    async def test_success(self, temp_dir):
        from devdox_ai_locust.cli import _generate_orchestrators
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        ep = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )

        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        mock_scenario_gen = Mock()
        mock_scenario_gen.generate_tag_orchestrator = AsyncMock(
            return_value="# orchestrator"
        )

        mock_progress = Mock()

        files, failures = await _generate_orchestrators(
            grouped_endpoints={"users": [ep]},
            successful_endpoints={id(ep)},
            scenario_gen=mock_scenario_gen,
            workflows_dir=workflows_dir,
            base_workflow_content="# base",
            test_data_content="# data",
            auth_endpoints=None,
            custom_requirement=None,
            db_type="",
            progress=mock_progress,
        )
        assert len(files) == 1
        assert len(failures) == 0
        mock_progress.orchestrator_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_skipped_no_valid(self, temp_dir):
        from devdox_ai_locust.cli import _generate_orchestrators

        ep = Mock()
        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        mock_progress = Mock()

        files, failures = await _generate_orchestrators(
            grouped_endpoints={"users": [ep]},
            successful_endpoints=set(),  # no valid
            scenario_gen=Mock(),
            workflows_dir=workflows_dir,
            base_workflow_content="",
            test_data_content="",
            auth_endpoints=None,
            custom_requirement=None,
            db_type="",
            progress=mock_progress,
        )
        assert len(files) == 0
        mock_progress.orchestrator_skipped.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure(self, temp_dir):
        from devdox_ai_locust.cli import _generate_orchestrators

        ep = Mock()
        workflows_dir = temp_dir / "workflows"
        workflows_dir.mkdir()

        mock_scenario_gen = Mock()
        mock_scenario_gen.generate_tag_orchestrator = AsyncMock(
            side_effect=RuntimeError("orch error")
        )

        mock_progress = Mock()

        files, failures = await _generate_orchestrators(
            grouped_endpoints={"users": [ep]},
            successful_endpoints={id(ep)},
            scenario_gen=mock_scenario_gen,
            workflows_dir=workflows_dir,
            base_workflow_content="",
            test_data_content="",
            auth_endpoints=None,
            custom_requirement=None,
            db_type="",
            progress=mock_progress,
        )
        assert len(failures) == 1
        mock_progress.orchestrator_failed.assert_called_once()
