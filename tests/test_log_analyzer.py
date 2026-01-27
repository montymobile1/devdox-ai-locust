"""
Tests for log_analyzer module
"""

import io
from collections import Counter
from pathlib import Path

from devdox_ai_locust.log_analyzer import (
    _extract_api_path,
    _line_iterator,
    _normalize_error,
    _normalize_exception,
    _record_error,
    _write_reduced_log,
    analyze_log,
)


class TestLineIterator:
    """Test _line_iterator function."""

    def test_basic_iteration(self):
        """Test that lines are yielded with newlines stripped."""
        f = io.StringIO("line1\nline2\nline3\n")
        result = list(_line_iterator(f))
        assert result == ["line1", "line2", "line3"]

    def test_empty_file(self):
        """Test iteration over empty file."""
        f = io.StringIO("")
        result = list(_line_iterator(f))
        assert result == []

    def test_strips_carriage_return(self):
        """Test that \\r\\n endings are stripped."""
        f = io.StringIO("line1\r\nline2\r\n")
        result = list(_line_iterator(f))
        assert result == ["line1", "line2"]


class TestNormalizeError:
    """Test error normalization for deduplication."""

    def test_removes_timestamps(self):
        """Test that timestamps are removed."""
        line = "2024-01-15T10:30:00.123 ERROR something failed"
        normalized = _normalize_error(line)
        assert "2024-01-15" not in normalized

    def test_replaces_uuids(self):
        """Test that UUIDs are replaced with placeholder."""
        line = "Error for resource 550e8400-e29b-41d4-a716-446655440000"
        normalized = _normalize_error(line)
        assert "<UUID>" in normalized
        assert "550e8400" not in normalized

    def test_replaces_api_paths(self):
        """Test that API paths are normalized."""
        line = "GET /users/123/orders failed"
        normalized = _normalize_error(line)
        assert "<PATH>" in normalized

    def test_replaces_numeric_path_segments(self):
        """Test that numeric path segments are replaced."""
        line = "Error at /items/9876"
        normalized = _normalize_error(line)
        assert "/<ID>" in normalized


class TestNormalizeException:
    """Test exception normalization."""

    def test_replaces_uuids(self):
        """Test UUID replacement in exception lines."""
        line = "KeyError: '550e8400-e29b-41d4-a716-446655440000'"
        normalized = _normalize_exception(line)
        assert "<UUID>" in normalized

    def test_replaces_long_values(self):
        """Test that long string values are replaced."""
        long_val = "x" * 60
        line = f"ValueError: '{long_val}'"
        normalized = _normalize_exception(line)
        assert "<LONG_VALUE>" in normalized


class TestExtractApiPath:
    """Test API path extraction from context lines."""

    def test_extracts_get_path(self):
        """Test extraction of GET path."""
        lines = ["some context", "GET /api/v1/users response", "more context"]
        result = _extract_api_path(lines)
        assert result == "GET /api/v1/users"

    def test_no_api_path(self):
        """Test returns empty string when no API path found."""
        lines = ["just some log line", "another line"]
        result = _extract_api_path(lines)
        assert result == ""


class TestRecordError:
    """Test _record_error function."""

    def test_first_occurrence_returns_zero(self):
        """Test that first occurrence of an error returns 0 duplicates."""
        sigs = Counter()
        samples = {}
        apis = {}
        result = _record_error("ERROR: something broke", [], sigs, samples, apis)
        assert result == 0
        assert len(sigs) == 1
        assert len(samples) == 1

    def test_duplicate_returns_one(self):
        """Test that duplicate error returns 1."""
        sigs = Counter()
        samples = {}
        apis = {}
        _record_error("ERROR: something broke", [], sigs, samples, apis)
        result = _record_error("ERROR: something broke", [], sigs, samples, apis)
        assert result == 1
        assert sigs.most_common(1)[0][1] == 2

    def test_context_with_api_path_tracked(self):
        """Test that API paths from context are tracked."""
        from collections import defaultdict

        sigs = Counter()
        samples = {}
        apis = defaultdict(set)
        context = ["REQUEST: GET /users", "Response status: 500"]
        _record_error("ERROR: server error", context, sigs, samples, apis)
        # Should have tracked the API path
        sig = list(apis.keys())[0]
        assert len(apis[sig]) > 0


class TestWriteReducedLog:
    """Test _write_reduced_log output formatting."""

    def test_writes_header(self, tmp_path):
        """Test that the header is written correctly."""
        output = str(tmp_path / "reduced.log")
        _write_reduced_log(output, Counter(), {}, {}, Counter(), {}, {}, 100, 50)
        content = Path(output).read_text()
        assert "# Reduced Locust Log" in content
        assert "100 lines" in content
        assert "50" in content

    def test_writes_errors_section(self, tmp_path):
        """Test that errors section is written."""
        output = str(tmp_path / "reduced.log")
        error_sigs = Counter({"normalized error": 3})
        error_samples = {"normalized error": ["ERROR: the actual error line"]}
        error_apis = {"normalized error": {"GET /users"}}
        _write_reduced_log(
            output,
            error_sigs,
            error_samples,
            error_apis,
            Counter(),
            {},
            {},
            100,
            10,
        )
        content = Path(output).read_text()
        assert "ERRORS" in content
        assert "[3x]" in content
        assert "GET /users" in content
        assert "ERROR: the actual error line" in content

    def test_writes_exceptions_section(self, tmp_path):
        """Test that exceptions section is written."""
        output = str(tmp_path / "reduced.log")
        exc_sigs = Counter({"KeyError: 'id'": 5})
        exc_samples = {
            "KeyError: 'id'": [
                "Traceback (most recent call last):",
                "  File 'test.py', line 10",
                "KeyError: 'id'",
            ]
        }
        _write_reduced_log(
            output,
            Counter(),
            {},
            {},
            exc_sigs,
            exc_samples,
            {},
            200,
            20,
        )
        content = Path(output).read_text()
        assert "EXCEPTIONS" in content
        assert "[5x]" in content
        assert "KeyError: 'id'" in content


class TestAnalyzeLog:
    """Test the full analyze_log function."""

    def test_simple_log_with_errors(self, tmp_path):
        """Test analyzing a log with simple error lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "INFO: starting\n"
            "INFO: processing\n"
            "ERROR: something failed\n"
            "INFO: continuing\n"
            "ERROR: something failed\n"
            "INFO: done\n"
        )
        output = str(tmp_path / "test_reduced.log")
        stats = analyze_log(str(log_file), output)
        assert stats["total_lines"] == 6
        assert stats["unique_errors"] == 1
        assert stats["duplicates_removed"] == 1

    def test_log_with_traceback(self, tmp_path):
        """Test analyzing a log with a traceback block."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "INFO: starting\n"
            "Traceback (most recent call last):\n"
            "  File 'test.py', line 10, in foo\n"
            "    raise ValueError('bad')\n"
            "ValueError: bad\n"
            "INFO: done\n"
        )
        output = str(tmp_path / "test_reduced.log")
        stats = analyze_log(str(log_file), output)
        assert stats["unique_exceptions"] == 1

    def test_default_output_path(self, tmp_path):
        """Test that default output path is generated correctly."""
        log_file = tmp_path / "myrun.log"
        log_file.write_text("INFO: ok\n")
        analyze_log(str(log_file))
        expected_output = tmp_path / "myrun_reduced.log"
        assert expected_output.exists()

    def test_deduplication_across_apis(self, tmp_path):
        """Test that similar errors across different APIs are deduplicated."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "REQUEST: GET /users\n"
            "ERROR: Request failed HTTPError 500\n"
            "REQUEST: GET /orders\n"
            "ERROR: Request failed HTTPError 500\n"
        )
        output = str(tmp_path / "test_reduced.log")
        stats = analyze_log(str(log_file), output)
        # Both errors should normalize to the same signature
        assert stats["unique_errors"] == 1
        assert stats["duplicates_removed"] == 1

    def test_empty_log(self, tmp_path):
        """Test analyzing an empty log file."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        output = str(tmp_path / "empty_reduced.log")
        stats = analyze_log(str(log_file), output)
        assert stats["total_lines"] == 0
        assert stats["unique_errors"] == 0
        assert stats["unique_exceptions"] == 0
        assert stats["duplicates_removed"] == 0
