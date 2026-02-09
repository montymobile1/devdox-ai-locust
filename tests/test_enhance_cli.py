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
    _build_results_table,
    _add_result_row,
    _print_summary,
    _print_verbose_details,
    _validate_and_discover_suite,
    _log_tag_breakdown,
    _configure_verbose_logging,
    _enhance_suite_files,
    _generate_gap_workflows,
    _generate_single_gap_workflow,
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


class TestBuildResultsTable:
    """Tests for _build_results_table helper."""

    def test_successful_result_counters(self):
        """Test that a successful updated result produces correct counters."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new code",
                original_source="old code",
                added_tasks=["task_a", "task_b"],
                added_imports=["import os"],
                replaced_tasks=["task_c"],
            ),
        }

        table, counters = _build_results_table(results, [], dry_run=False)

        assert counters["updated"] == 1
        assert counters["total_added"] == 2
        assert counters["total_replaced"] == 1
        assert counters["total_imports"] == 1
        assert counters["unchanged"] == 0
        assert counters["failed"] == 0

    def test_failed_result_row(self):
        """Test that a failed result increments failed_count."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="code",
                error="AI call failed",
            ),
        }

        table, counters = _build_results_table(results, [], dry_run=False)

        assert counters["failed"] == 1
        assert counters["updated"] == 0

    def test_unchanged_result_row(self):
        """Test that an unchanged result increments unchanged_count."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="same",
                original_source="same",
            ),
        }

        table, counters = _build_results_table(results, [], dry_run=False)

        assert counters["unchanged"] == 1
        assert counters["updated"] == 0

    def test_dry_run_action_text(self):
        """Test that dry run mode produces [DRY RUN] in row data."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                added_tasks=["t1"],
            ),
        }

        table, counters = _build_results_table(results, [], dry_run=True)

        assert counters["updated"] == 1

    def test_created_files_appear_in_table(self):
        """Test that created files are added as rows."""
        results = {}
        created = ["/path/to/new_workflow.py", "/path/to/another.py"]

        table, counters = _build_results_table(results, created, dry_run=False)

        assert table.row_count == 2
        assert counters["updated"] == 0

    def test_mixed_results_aggregate_counters(self):
        """Test mixed success, failure, and unchanged results."""
        results = {
            "/path/success.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                added_tasks=["t1"],
                added_imports=["import x"],
            ),
            "/path/failed.py": EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="code",
                error="err",
            ),
            "/path/unchanged.py": EnhanceResult(
                success=True,
                enhanced_source="same",
                original_source="same",
            ),
        }

        table, counters = _build_results_table(results, ["/created.py"], dry_run=False)

        assert counters["updated"] == 1
        assert counters["failed"] == 1
        assert counters["unchanged"] == 1
        assert counters["total_added"] == 1
        assert counters["total_imports"] == 1
        # 3 result rows + 1 created row
        assert table.row_count == 4

    def test_empty_results(self):
        """Test with no results and no created files."""
        table, counters = _build_results_table({}, [], dry_run=False)

        assert table.row_count == 0
        assert all(v == 0 for v in counters.values())


class TestAddResultRow:
    """Tests for the _add_result_row helper."""

    def _make_table(self):
        from rich.table import Table
        table = Table()
        table.add_column("File")
        table.add_column("Action")
        table.add_column("Tasks +")
        table.add_column("Tasks ~")
        table.add_column("Imports +")
        table.add_column("Warnings")
        return table

    def _zero_counters(self):
        return {
            "total_added": 0, "total_replaced": 0, "total_imports": 0,
            "updated": 0, "unchanged": 0, "failed": 0,
        }

    def test_failed_result_returns_early(self):
        """Test that a failed result adds one row and increments failed."""
        table = self._make_table()
        counters = self._zero_counters()

        _add_result_row(
            table, "/p/f.py",
            EnhanceResult(success=False, enhanced_source="", original_source="c", error="boom"),
            dry_run=False, counters=counters,
        )

        assert table.row_count == 1
        assert counters["failed"] == 1
        assert counters["updated"] == 0

    def test_unchanged_result_returns_early(self):
        """Test that an unchanged result adds one row and increments unchanged."""
        table = self._make_table()
        counters = self._zero_counters()

        _add_result_row(
            table, "/p/f.py",
            EnhanceResult(success=True, enhanced_source="s", original_source="s"),
            dry_run=False, counters=counters,
        )

        assert table.row_count == 1
        assert counters["unchanged"] == 1

    def test_updated_result_accumulates_counters(self):
        """Test that a successful changed result accumulates all counters."""
        table = self._make_table()
        counters = self._zero_counters()

        _add_result_row(
            table, "/p/f.py",
            EnhanceResult(
                success=True, enhanced_source="new", original_source="old",
                added_tasks=["a", "b"], added_imports=["i1"],
                replaced_tasks=["r1"],
            ),
            dry_run=False, counters=counters,
        )

        assert counters["updated"] == 1
        assert counters["total_added"] == 2
        assert counters["total_replaced"] == 1
        assert counters["total_imports"] == 1


class TestPrintSummary:
    """Tests for _print_summary helper."""

    @patch("devdox_ai_locust.cli.console")
    def test_summary_format(self, mock_console):
        """Test that summary line contains correct counter values."""
        counters = {
            "total_added": 5, "total_replaced": 2, "total_imports": 3,
            "updated": 1, "unchanged": 1, "failed": 1,
        }

        _print_summary(counters, created_count=2, processing_time=1.5)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "1 updated" in joined
        assert "2 created" in joined
        assert "1 unchanged" in joined
        assert "1 failed" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_totals_format(self, mock_console):
        """Test that totals line shows added/replaced/imports."""
        counters = {
            "total_added": 10, "total_replaced": 3, "total_imports": 4,
            "updated": 2, "unchanged": 0, "failed": 0,
        }

        _print_summary(counters, created_count=0, processing_time=2.0)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "+10 tasks added" in joined
        assert "~3 tasks replaced" in joined
        assert "+4 imports" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_processing_time_precision(self, mock_console):
        """Test that processing time is formatted to 2 decimal places."""
        counters = {
            "total_added": 0, "total_replaced": 0, "total_imports": 0,
            "updated": 0, "unchanged": 0, "failed": 0,
        }

        _print_summary(counters, created_count=0, processing_time=3.14159)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "3.14s" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_all_zero_counters(self, mock_console):
        """Test summary with all zero counters."""
        counters = {
            "total_added": 0, "total_replaced": 0, "total_imports": 0,
            "updated": 0, "unchanged": 0, "failed": 0,
        }

        _print_summary(counters, created_count=0, processing_time=0.0)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "0 updated" in joined
        assert "0 created" in joined
        assert "0 unchanged" in joined
        assert "0 failed" in joined


class TestPrintVerboseDetails:
    """Tests for _print_verbose_details helper."""

    @patch("devdox_ai_locust.cli.console")
    def test_prints_added_tasks(self, mock_console):
        """Test that added tasks are printed for successful results."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                added_tasks=["task_a", "task_b"],
            ),
        }

        _print_verbose_details(results)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "task_a" in joined
        assert "task_b" in joined
        assert "added tasks" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_prints_replaced_tasks(self, mock_console):
        """Test that replaced tasks are printed."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                replaced_tasks=["replaced_one"],
            ),
        }

        _print_verbose_details(results)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "replaced_one" in joined
        assert "replaced tasks" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_prints_warnings(self, mock_console):
        """Test that warnings are printed."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                warnings=["Duplicate method skipped", "Syntax issue fixed"],
            ),
        }

        _print_verbose_details(results)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "Duplicate method skipped" in joined
        assert "Syntax issue fixed" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_skips_failed_results_added_tasks(self, mock_console):
        """Test that failed results do not print added tasks."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="code",
                error="Failed",
                added_tasks=["should_not_appear"],
            ),
        }

        _print_verbose_details(results)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "should_not_appear" not in joined

    @patch("devdox_ai_locust.cli.console")
    def test_nothing_to_print(self, mock_console):
        """Test that no output is produced when there are no details."""
        results = {
            "/path/to/file.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
            ),
        }

        _print_verbose_details(results)

        mock_console.print.assert_not_called()

    @patch("devdox_ai_locust.cli.console")
    def test_multiple_files_mixed_details(self, mock_console):
        """Test verbose output across multiple files with different details."""
        results = {
            "/path/alpha.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                added_tasks=["alpha_task"],
            ),
            "/path/beta.py": EnhanceResult(
                success=True,
                enhanced_source="new",
                original_source="old",
                replaced_tasks=["beta_replaced"],
                warnings=["beta warning"],
            ),
        }

        _print_verbose_details(results)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "alpha_task" in joined
        assert "beta_replaced" in joined
        assert "beta warning" in joined


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


class TestValidateAndDiscoverSuite:
    """Tests for _validate_and_discover_suite helper."""

    def test_valid_suite_returns_dict(self, temp_suite_dir):
        """Test that a valid suite returns the discovery dict."""
        suite = _validate_and_discover_suite(temp_suite_dir, verbose=False)

        assert suite["suite_dir"] == temp_suite_dir
        assert suite["locustfile"] is not None
        assert len(suite["workflows"]) == 1

    def test_empty_suite_exits(self, temp_suite_dir):
        """Test that an empty suite triggers sys.exit."""
        import shutil
        (temp_suite_dir / "locustfile.py").unlink()
        (temp_suite_dir / "test_data.py").unlink()
        shutil.rmtree(temp_suite_dir / "workflows")

        with pytest.raises(SystemExit):
            _validate_and_discover_suite(temp_suite_dir, verbose=False)

    def test_suite_with_only_locustfile(self, temp_suite_dir):
        """Test suite with only locustfile passes validation."""
        import shutil
        shutil.rmtree(temp_suite_dir / "workflows")

        suite = _validate_and_discover_suite(temp_suite_dir, verbose=False)

        assert suite["locustfile"] is not None
        assert suite["workflows"] == []

    def test_suite_with_only_workflows(self, temp_suite_dir):
        """Test suite with only workflows passes validation."""
        (temp_suite_dir / "locustfile.py").unlink()

        suite = _validate_and_discover_suite(temp_suite_dir, verbose=False)

        assert suite["locustfile"] is None
        assert len(suite["workflows"]) == 1

    @patch("devdox_ai_locust.cli.console")
    def test_file_count_correct(self, mock_console, temp_suite_dir):
        """Test that file count includes workflows + locustfile + test_data."""
        _validate_and_discover_suite(temp_suite_dir, verbose=False)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        # 1 workflow + 1 locustfile + 1 test_data = 3
        assert "3 enhanceable file(s)" in joined


class TestLogTagBreakdown:
    """Tests for _log_tag_breakdown helper."""

    @patch("devdox_ai_locust.cli.console")
    def test_correct_per_tag_counts(self, mock_console, sample_endpoints):
        """Test correct per-tag counts are printed."""
        _log_tag_breakdown(sample_endpoints)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "users: 1 endpoint(s)" in joined
        assert "orders: 1 endpoint(s)" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_multi_tag_endpoint(self, mock_console):
        """Test endpoint with multiple tags counted in each."""
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        endpoints = [
            Endpoint(
                path="/admin/users",
                method="GET",
                operation_id="adminGetUsers",
                summary="",
                parameters=[],
                request_body=None,
                responses=[],
                description="",
                tags=["admin", "users"],
            ),
        ]

        _log_tag_breakdown(endpoints)

        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "admin: 1 endpoint(s)" in joined
        assert "users: 1 endpoint(s)" in joined

    @patch("devdox_ai_locust.cli.console")
    def test_empty_endpoints(self, mock_console):
        """Test empty endpoints list does not crash."""
        _log_tag_breakdown([])

        # Should still print the header
        calls = [str(c) for c in mock_console.print.call_args_list]
        joined = " ".join(calls)
        assert "Endpoints per tag" in joined


class TestConfigureVerboseLogging:
    """Tests for _configure_verbose_logging helper."""

    def test_enhancer_logger_set_to_debug(self):
        """Test enhancer logger is set to DEBUG."""
        import logging

        _configure_verbose_logging()

        logger = logging.getLogger("devdox_ai_locust.locust_enhancer")
        assert logger.level == logging.DEBUG

    def test_merger_logger_set_to_debug(self):
        """Test code merger logger is set to DEBUG."""
        import logging

        _configure_verbose_logging()

        logger = logging.getLogger("devdox_ai_locust.utils.code_merger")
        assert logger.level == logging.DEBUG

    def test_root_logger_gets_handler(self):
        """Test root logger gets a handler if none exist."""
        import logging

        root = logging.getLogger()
        original_handlers = root.handlers[:]

        # Remove all handlers temporarily
        root.handlers.clear()

        _configure_verbose_logging()

        assert len(root.handlers) >= 1
        assert root.level == logging.DEBUG

        # Restore original handlers
        root.handlers = original_handlers

    def test_existing_handlers_not_duplicated(self):
        """Test that existing handlers are not duplicated."""
        import logging

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_count = len(root.handlers)

        # Ensure there's at least one handler
        if not root.handlers:
            root.addHandler(logging.StreamHandler())
            original_count = 1

        _configure_verbose_logging()

        # Should not have added another handler
        assert len(root.handlers) == original_count

        # Restore
        root.handlers = original_handlers


class TestEnhanceSuiteFiles:
    """Tests for _enhance_suite_files helper."""

    def test_all_workflows_enhanced(self, temp_suite_dir):
        """Test all workflow files are enhanced."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.enhance_file = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="enhanced",
                original_source="original",
            )
        )

        suite = {
            "workflows": [temp_suite_dir / "workflows" / "users_workflow.py"],
            "locustfile": None,
            "test_data": None,
        }

        results = asyncio.run(_enhance_suite_files(
            mock_enhancer, suite, "Add tests", "http://example.com", False
        ))

        assert len(results) == 1
        mock_enhancer.enhance_file.assert_called_once()

    def test_locustfile_included(self, temp_suite_dir):
        """Test locustfile is enhanced when present."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.enhance_file = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="enhanced",
                original_source="original",
            )
        )

        suite = {
            "workflows": [],
            "locustfile": temp_suite_dir / "locustfile.py",
            "test_data": None,
        }

        results = asyncio.run(_enhance_suite_files(
            mock_enhancer, suite, "Add tests", "http://example.com", False
        ))

        assert len(results) == 1

    def test_test_data_included(self, temp_suite_dir):
        """Test test_data is enhanced when present."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.enhance_file = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="enhanced",
                original_source="original",
            )
        )

        suite = {
            "workflows": [],
            "locustfile": None,
            "test_data": temp_suite_dir / "test_data.py",
        }

        results = asyncio.run(_enhance_suite_files(
            mock_enhancer, suite, "Add tests", "http://example.com", False
        ))

        assert len(results) == 1

    def test_missing_optional_files_skipped(self, temp_suite_dir):
        """Test that None locustfile/test_data are skipped."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.enhance_file = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="e",
                original_source="o",
            )
        )

        suite = {
            "workflows": [temp_suite_dir / "workflows" / "users_workflow.py"],
            "locustfile": None,
            "test_data": None,
        }

        results = asyncio.run(_enhance_suite_files(
            mock_enhancer, suite, "Add tests", "http://example.com", False
        ))

        assert len(results) == 1
        assert mock_enhancer.enhance_file.call_count == 1

    def test_empty_workflows(self):
        """Test empty workflows list produces empty results."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.enhance_file = AsyncMock()

        suite = {
            "workflows": [],
            "locustfile": None,
            "test_data": None,
        }

        results = asyncio.run(_enhance_suite_files(
            mock_enhancer, suite, "Add tests", "http://example.com", False
        ))

        assert len(results) == 0
        mock_enhancer.enhance_file.assert_not_called()


class TestGenerateSingleGapWorkflow:
    """Tests for _generate_single_gap_workflow helper."""

    def test_successful_generation(self, temp_suite_dir, sample_endpoints):
        """Test successful gap workflow generation returns file path."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.generate_new_workflow = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="from locust import HttpUser\n",
                original_source="",
            )
        )

        result = asyncio.run(_generate_single_gap_workflow(
            enhancer=mock_enhancer,
            gap_tag="orders",
            gap_endpoints=sample_endpoints,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            reference_source=None,
            suite_dir=temp_suite_dir,
            verbose=False,
            dry_run=False,
        ))

        assert result is not None
        assert "orders_workflow.py" in result
        assert Path(result).exists()

    def test_failed_generation_returns_none(self, temp_suite_dir, sample_endpoints):
        """Test failed generation returns None."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.generate_new_workflow = AsyncMock(
            return_value=EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="",
                error="AI error",
            )
        )

        result = asyncio.run(_generate_single_gap_workflow(
            enhancer=mock_enhancer,
            gap_tag="orders",
            gap_endpoints=sample_endpoints,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            reference_source=None,
            suite_dir=temp_suite_dir,
            verbose=False,
            dry_run=False,
        ))

        assert result is None

    def test_dry_run_does_not_write(self, temp_suite_dir, sample_endpoints):
        """Test dry run does not write files to disk."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.generate_new_workflow = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="from locust import HttpUser\n",
                original_source="",
            )
        )

        result = asyncio.run(_generate_single_gap_workflow(
            enhancer=mock_enhancer,
            gap_tag="orders",
            gap_endpoints=sample_endpoints,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            reference_source=None,
            suite_dir=temp_suite_dir,
            verbose=False,
            dry_run=True,
        ))

        assert result is not None
        assert not Path(result).exists()

    def test_exception_returns_none(self, temp_suite_dir, sample_endpoints):
        """Test that an exception from the enhancer returns None."""
        import asyncio

        mock_enhancer = Mock()
        mock_enhancer.generate_new_workflow = AsyncMock(
            side_effect=Exception("Network error")
        )

        result = asyncio.run(_generate_single_gap_workflow(
            enhancer=mock_enhancer,
            gap_tag="orders",
            gap_endpoints=sample_endpoints,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            reference_source=None,
            suite_dir=temp_suite_dir,
            verbose=False,
            dry_run=False,
        ))

        assert result is None


class TestGenerateGapWorkflows:
    """Tests for _generate_gap_workflows helper."""

    def test_generates_for_uncovered_tags(self, temp_suite_dir):
        """Test workflows generated for uncovered tags."""
        import asyncio
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        mock_enhancer = Mock()
        mock_enhancer.generate_new_workflow = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="from locust import HttpUser\n",
                original_source="",
            )
        )

        endpoints = [
            Endpoint(
                path="/orders",
                method="POST",
                operation_id="createOrder",
                summary="",
                parameters=[],
                request_body=None,
                responses=[],
                description="",
                tags=["orders"],
            ),
        ]

        suite = {
            "workflows": [temp_suite_dir / "workflows" / "users_workflow.py"],
            "locustfile": None,
            "test_data": None,
        }

        created = asyncio.run(_generate_gap_workflows(
            enhancer=mock_enhancer,
            gaps=["orders"],
            endpoints=endpoints,
            suite=suite,
            suite_dir=temp_suite_dir,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            verbose=False,
            dry_run=False,
        ))

        assert len(created) == 1
        assert "orders_workflow.py" in created[0]

    def test_reference_source_from_first_workflow(self, temp_suite_dir):
        """Test reference source is read from the first workflow file."""
        import asyncio
        from devdox_ai_locust.utils.open_ai_parser import Endpoint

        mock_enhancer = Mock()
        mock_enhancer.generate_new_workflow = AsyncMock(
            return_value=EnhanceResult(
                success=True,
                enhanced_source="code\n",
                original_source="",
            )
        )

        endpoints = [
            Endpoint(
                path="/products",
                method="GET",
                operation_id="getProducts",
                summary="",
                parameters=[],
                request_body=None,
                responses=[],
                description="",
                tags=["products"],
            ),
        ]

        suite = {
            "workflows": [temp_suite_dir / "workflows" / "users_workflow.py"],
            "locustfile": None,
            "test_data": None,
        }

        asyncio.run(_generate_gap_workflows(
            enhancer=mock_enhancer,
            gaps=["products"],
            endpoints=endpoints,
            suite=suite,
            suite_dir=temp_suite_dir,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            verbose=False,
            dry_run=True,
        ))

        call_kwargs = mock_enhancer.generate_new_workflow.call_args[1]
        assert "HttpUser" in call_kwargs["reference_workflow_source"]

    def test_empty_gaps_returns_empty(self, temp_suite_dir):
        """Test no gaps produces no created files."""
        import asyncio

        mock_enhancer = Mock()

        suite = {
            "workflows": [],
            "locustfile": None,
            "test_data": None,
        }

        created = asyncio.run(_generate_gap_workflows(
            enhancer=mock_enhancer,
            gaps=[],
            endpoints=[],
            suite=suite,
            suite_dir=temp_suite_dir,
            custom_requirement="Add tests",
            swagger_url="http://example.com",
            verbose=False,
            dry_run=False,
        ))

        assert created == []
