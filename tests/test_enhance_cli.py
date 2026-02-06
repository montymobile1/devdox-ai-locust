"""
Tests for enhance CLI command.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
from pathlib import Path

from devdox_ai_locust.cli import (
    cli,
    enhance,
    _async_enhance,
    _discover_suite_files,
    _identify_coverage_gaps,
    _display_enhance_configuration,
    _show_enhance_results,
    _enhance_single_file,
    _write_enhance_results,
)
from devdox_ai_locust.locust_enhancer import EnhanceResult


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_suite_dir():
    """Create a temporary test suite directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite_dir = Path(tmpdir)

        # Create locustfile.py
        (suite_dir / "locustfile.py").write_text('''
from locust import HttpUser, task

class MainUser(HttpUser):
    @task
    def test(self):
        self.client.get("/api")
''')

        # Create test_data.py
        (suite_dir / "test_data.py").write_text('''
TEST_DATA = {"user": "test"}
''')

        # Create workflows directory
        workflows_dir = suite_dir / "workflows"
        workflows_dir.mkdir()

        # Create a workflow file
        (workflows_dir / "users_workflow.py").write_text('''
from locust import HttpUser, task

class UsersWorkflow(HttpUser):
    @task
    def get_users(self):
        self.client.get("/users")
''')

        (workflows_dir / "__init__.py").write_text("")

        yield suite_dir


@pytest.fixture
def sample_endpoints():
    """Sample endpoints for testing."""
    from devdox_ai_locust.utils.open_ai_parser import Endpoint

    return [
        Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="Get users",
            parameters=[],
            request_body=None,
            responses=[],
            description="",
            tags=["users"],
        ),
        Endpoint(
            path="/orders",
            method="POST",
            operation_id="createOrder",
            summary="Create order",
            parameters=[],
            request_body=None,
            responses=[],
            description="",
            tags=["orders"],
        ),
    ]


class TestEnhanceCommand:
    """Tests for enhance CLI command."""

    def test_enhance_help(self, cli_runner):
        """Test enhance command help."""
        result = cli_runner.invoke(cli, ["enhance", "--help"])

        assert result.exit_code == 0
        assert "Enhance an existing Locust test suite" in result.output
        assert "--test-suite" in result.output
        assert "--custom-requirement" in result.output
        assert "--together-api-key" in result.output

    def test_enhance_missing_required_args(self, cli_runner):
        """Test enhance command with missing required arguments."""
        result = cli_runner.invoke(cli, ["enhance", "http://example.com/swagger.json"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_enhance_nonexistent_suite(self, cli_runner):
        """Test enhance with non-existent test suite directory."""
        result = cli_runner.invoke(
            cli,
            [
                "enhance",
                "http://example.com/swagger.json",
                "--test-suite", "/nonexistent/path",
                "--custom-requirement", "Add tests",
                "--together-api-key", "test-key",
            ],
        )

        assert result.exit_code != 0

    @patch("devdox_ai_locust.cli.asyncio.run")
    def test_enhance_command_invokes_async(
        self, mock_run, cli_runner, temp_suite_dir
    ):
        """Test that enhance command invokes async handler."""
        cli_runner.invoke(
            cli,
            [
                "enhance",
                "http://example.com/swagger.json",
                "--test-suite", str(temp_suite_dir),
                "--custom-requirement", "Add edge cases",
                "--together-api-key", "test-key",
            ],
        )

        mock_run.assert_called_once()


class TestDiscoverSuiteFiles:
    """Tests for suite file discovery."""

    def test_discover_full_suite(self, temp_suite_dir):
        """Test discovering a complete test suite."""
        result = _discover_suite_files(temp_suite_dir, verbose=False)

        assert result["suite_dir"] == temp_suite_dir
        assert result["locustfile"] == temp_suite_dir / "locustfile.py"
        assert result["test_data"] == temp_suite_dir / "test_data.py"
        assert len(result["workflows"]) == 1
        assert result["workflows"][0].name == "users_workflow.py"

    def test_discover_missing_locustfile(self, temp_suite_dir):
        """Test discovering suite without locustfile."""
        (temp_suite_dir / "locustfile.py").unlink()

        result = _discover_suite_files(temp_suite_dir, verbose=False)

        assert result["locustfile"] is None

    def test_discover_missing_test_data(self, temp_suite_dir):
        """Test discovering suite without test_data."""
        (temp_suite_dir / "test_data.py").unlink()

        result = _discover_suite_files(temp_suite_dir, verbose=False)

        assert result["test_data"] is None

    def test_discover_no_workflows(self, temp_suite_dir):
        """Test discovering suite without workflow files."""
        import shutil
        shutil.rmtree(temp_suite_dir / "workflows")

        result = _discover_suite_files(temp_suite_dir, verbose=False)

        assert result["workflows"] == []

    def test_discover_verbose(self, temp_suite_dir, capsys):
        """Test verbose discovery output."""
        _discover_suite_files(temp_suite_dir, verbose=True)

        # Verbose mode should print discovery info
        # (Rich console output may not be captured by capsys)


class TestIdentifyCoverageGaps:
    """Tests for coverage gap identification."""

    def test_identify_gaps_with_uncovered_tag(self, sample_endpoints, temp_suite_dir):
        """Test identifying uncovered API tags."""
        workflows = [temp_suite_dir / "workflows" / "users_workflow.py"]

        gaps = _identify_coverage_gaps(sample_endpoints, workflows, verbose=False)

        assert "orders" in gaps
        assert "users" not in gaps  # Already covered

    def test_identify_gaps_all_covered(self, temp_suite_dir):
        """Test when all tags are covered."""
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        endpoints = [
            Endpoint(
                path="/users",
                method="GET",
                operation_id="getUsers",
                summary="",
                parameters=[],
                request_body=None,
                responses=[],
                description="",
                tags=["users"],
            ),
        ]
        workflows = [temp_suite_dir / "workflows" / "users_workflow.py"]

        gaps = _identify_coverage_gaps(endpoints, workflows, verbose=False)

        assert gaps == []

    def test_identify_gaps_tag_normalization(self, temp_suite_dir):
        """Test that tag names are normalized."""
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        endpoints = [
            Endpoint(
                path="/users",
                method="GET",
                operation_id="getUsers",
                summary="",
                parameters=[],
                request_body=None,
                responses=[],
                description="",
                tags=["Users"],  # Capital U
            ),
        ]
        workflows = [temp_suite_dir / "workflows" / "users_workflow.py"]

        gaps = _identify_coverage_gaps(endpoints, workflows, verbose=False)

        assert gaps == []  # Should match despite case difference


class TestDisplayEnhanceConfiguration:
    """Tests for configuration display."""

    def test_display_configuration(self, temp_suite_dir):
        """Test displaying enhancement configuration."""
        # Should not raise
        _display_enhance_configuration(
            swagger_url="http://example.com/swagger.json",
            suite_dir=temp_suite_dir,
            custom_requirement="Add edge cases",
            dry_run=False,
        )

    def test_display_configuration_dry_run(self, temp_suite_dir):
        """Test displaying configuration with dry run."""
        _display_enhance_configuration(
            swagger_url="http://example.com/swagger.json",
            suite_dir=temp_suite_dir,
            custom_requirement="Add tests",
            dry_run=True,
        )


class TestShowEnhanceResults:
    """Tests for results display."""

    def test_show_results_success(self):
        """Test showing successful enhancement results."""
        from datetime import datetime, timezone

        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new code",
                original_source="old code",
                added_tasks=["new_task"],
                added_imports=["import json"],
            ),
        }

        _show_enhance_results(
            results=results,
            created_files=[],
            start_time=datetime.now(timezone.utc),
            verbose=False,
            dry_run=False,
        )

    def test_show_results_with_failures(self):
        """Test showing results with failures."""
        from datetime import datetime, timezone

        results = {
            "/path/to/file.py": EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="code",
                error="Something went wrong",
            ),
        }

        _show_enhance_results(
            results=results,
            created_files=[],
            start_time=datetime.now(timezone.utc),
            verbose=True,
            dry_run=False,
        )

    def test_show_results_with_created_files(self):
        """Test showing results with newly created files."""
        from datetime import datetime, timezone

        results = {}
        created_files = ["/path/to/new_workflow.py"]

        _show_enhance_results(
            results=results,
            created_files=created_files,
            start_time=datetime.now(timezone.utc),
            verbose=False,
            dry_run=False,
        )

    def test_show_results_unchanged(self):
        """Test showing results when file is unchanged."""
        from datetime import datetime, timezone

        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="same code",
                original_source="same code",  # Same as enhanced
            ),
        }

        _show_enhance_results(
            results=results,
            created_files=[],
            start_time=datetime.now(timezone.utc),
            verbose=False,
            dry_run=False,
        )


class TestEnhanceSingleFile:
    """Tests for single file enhancement."""

    @pytest.mark.asyncio
    async def test_enhance_single_file_success(self, temp_suite_dir):
        """Test successful single file enhancement."""
        from devdox_ai_locust.locust_enhancer import LocustTestEnhancer

        mock_enhancer = Mock(spec=LocustTestEnhancer)
        mock_enhancer.enhance_file = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="enhanced",
                original_source="original",
                added_tasks=["new_task"],
            )
        )

        file_path = temp_suite_dir / "locustfile.py"

        result = await _enhance_single_file(
            enhancer=mock_enhancer,
            file_path=file_path,
            custom_requirement="Add tests",
            swagger_url="http://example.com/swagger.json",
            verbose=False,
        )

        assert result.success is True
        mock_enhancer.enhance_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_single_file_failure(self, temp_suite_dir):
        """Test single file enhancement failure."""
        from devdox_ai_locust.locust_enhancer import LocustTestEnhancer

        mock_enhancer = Mock(spec=LocustTestEnhancer)
        mock_enhancer.enhance_file = AsyncMock(
            side_effect=Exception("Enhancement failed")
        )

        file_path = temp_suite_dir / "locustfile.py"

        result = await _enhance_single_file(
            enhancer=mock_enhancer,
            file_path=file_path,
            custom_requirement="Add tests",
            swagger_url="http://example.com/swagger.json",
            verbose=False,
        )

        assert result.success is False
        assert "Enhancement failed" in result.error


class TestWriteEnhanceResults:
    """Tests for writing enhancement results."""

    def test_write_results_creates_backup(self, temp_suite_dir):
        """Test that writing results creates backup files."""
        file_path = str(temp_suite_dir / "locustfile.py")
        original_content = (temp_suite_dir / "locustfile.py").read_text()

        results = {
            file_path: EnhanceResult(
                success=True,
                enhanced_source="# Enhanced content\n" + original_content,
                original_source=original_content,
            ),
        }

        _write_enhance_results(results, verbose=False)

        # Check backup was created
        backup_path = Path(file_path + ".bak")
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

        # Check file was updated
        assert "Enhanced content" in Path(file_path).read_text()

    def test_write_results_skips_unchanged(self, temp_suite_dir):
        """Test that unchanged files are not written."""
        file_path = str(temp_suite_dir / "locustfile.py")
        content = (temp_suite_dir / "locustfile.py").read_text()

        results = {
            file_path: EnhanceResult(
                success=True,
                enhanced_source=content,  # Same as original
                original_source=content,
            ),
        }

        _write_enhance_results(results, verbose=False)

        # No backup should be created
        backup_path = Path(file_path + ".bak")
        assert not backup_path.exists()

    def test_write_results_skips_failed(self, temp_suite_dir):
        """Test that failed results are not written."""
        file_path = str(temp_suite_dir / "locustfile.py")
        original = (temp_suite_dir / "locustfile.py").read_text()

        results = {
            file_path: EnhanceResult(
                success=False,
                enhanced_source="",
                original_source=original,
                error="Failed",
            ),
        }

        _write_enhance_results(results, verbose=False)

        # File should remain unchanged
        assert Path(file_path).read_text() == original


class TestAsyncEnhance:
    """Tests for async enhance handler."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli._initialize_config")
    @patch("devdox_ai_locust.cli._process_api_schema")
    @patch("devdox_ai_locust.cli.LocustTestEnhancer")
    async def test_async_enhance_success(
        self,
        mock_enhancer_class,
        mock_process_schema,
        mock_init_config,
        temp_suite_dir,
        sample_endpoints,
    ):
        """Test successful async enhancement."""
        mock_init_config.return_value = (Mock(), "test-key")
        mock_process_schema.return_value = (
            {},
            sample_endpoints,
            {"title": "Test API"},
        )

        mock_enhancer = Mock()
        mock_enhancer.enhance_file = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="enhanced",
                original_source="original",
            )
        )
        mock_enhancer.generate_new_workflow = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="new workflow",
                original_source="",
            )
        )
        mock_enhancer_class.return_value = mock_enhancer

        mock_ctx = Mock()
        mock_ctx.obj = {"verbose": False}

        await _async_enhance(
            ctx=mock_ctx,
            swagger_url="http://example.com/swagger.json",
            test_suite=str(temp_suite_dir),
            custom_requirement="Add tests",
            together_api_key="test-key",
            dry_run=True,
        )

        mock_enhancer.enhance_file.assert_called()

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.cli._initialize_config")
    async def test_async_enhance_empty_suite(
        self, mock_init_config, temp_suite_dir
    ):
        """Test async enhancement with empty suite."""
        import shutil

        # Remove all files
        (temp_suite_dir / "locustfile.py").unlink()
        (temp_suite_dir / "test_data.py").unlink()
        shutil.rmtree(temp_suite_dir / "workflows")

        mock_init_config.return_value = (Mock(), "test-key")

        mock_ctx = Mock()
        mock_ctx.obj = {"verbose": False}

        with pytest.raises(SystemExit):
            await _async_enhance(
                ctx=mock_ctx,
                swagger_url="http://example.com/swagger.json",
                test_suite=str(temp_suite_dir),
                custom_requirement="Add tests",
                together_api_key="test-key",
                dry_run=False,
            )
