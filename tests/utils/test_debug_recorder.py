"""
Tests for debug_recorder module
"""

import json
import pytest

from devdox_ai_locust.utils.debug_recorder import (
    DebugRecorder,
    ScenarioDebugInfo,
    EndpointDebugInfo,
)


class TestScenarioDebugInfo:
    """Test ScenarioDebugInfo dataclass."""

    def test_defaults(self):
        """Test default field values."""
        info = ScenarioDebugInfo(scenario_type="happy_path")
        assert info.scenario_type == "happy_path"
        assert info.context == {}
        assert info.prompt == ""
        assert info.retries == []
        assert info.fallback is None


class TestEndpointDebugInfo:
    """Test EndpointDebugInfo dataclass."""

    def test_defaults(self):
        """Test default field values."""
        info = EndpointDebugInfo(
            method="GET", path="/users", operation_id="getUsers", tag="users"
        )
        assert info.scenarios == {}
        assert info.summary == {}


class TestDebugRecorder:
    """Test DebugRecorder."""

    @pytest.fixture
    def recorder_enabled(self, temp_dir):
        return DebugRecorder(output_dir=temp_dir, enabled=True)

    @pytest.fixture
    def recorder_disabled(self, temp_dir):
        return DebugRecorder(output_dir=temp_dir, enabled=False)

    def test_disabled_does_not_create_dirs(self, temp_dir):
        """Test that disabled recorder does not create debug directories."""
        recorder = DebugRecorder(output_dir=temp_dir, enabled=False)
        assert not recorder.debug_root.exists()

    def test_enabled_creates_dirs(self, recorder_enabled):
        """Test that enabled recorder creates debug directory structure."""
        assert recorder_enabled.debug_root.exists()
        assert recorder_enabled.input_dir.exists()
        assert recorder_enabled.static_dir.exists()
        assert recorder_enabled.workflows_dir.exists()

    def test_record_cli_args(self, recorder_enabled):
        """Test recording CLI arguments."""
        recorder_enabled.record_cli_args({"url": "http://example.com", "debug": True})
        path = recorder_enabled.input_dir / "cli_args.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["debug"] is True

    def test_record_cli_args_disabled(self, recorder_disabled):
        """Test that disabled recorder skips writing CLI args."""
        recorder_disabled.record_cli_args({"url": "http://example.com"})
        path = recorder_disabled.input_dir / "cli_args.json"
        assert not path.exists()

    def test_record_resolved_config(self, recorder_enabled):
        """Test recording resolved config."""
        recorder_enabled.record_resolved_config({"model": "test-model"})
        path = recorder_enabled.input_dir / "resolved_config.json"
        assert path.exists()

    def test_record_static_file(self, recorder_enabled):
        """Test recording a static file."""
        recorder_enabled.record_static_file(
            "locustfile.py",
            {"key": "value"},
            "print('hello')",
        )
        rendered = recorder_enabled.static_dir / "locustfile" / "rendered.py"
        context = recorder_enabled.static_dir / "locustfile" / "context.json"
        assert rendered.exists()
        assert context.exists()
        assert rendered.read_text() == "print('hello')"

    def test_record_error(self, recorder_enabled):
        """Test recording errors updates stats."""
        recorder_enabled.record_error("something failed", {"endpoint": "/users"})
        assert len(recorder_enabled.stats["errors"]) == 1
        assert recorder_enabled.stats["errors"][0]["error"] == "something failed"

    def test_record_error_disabled(self, recorder_disabled):
        """Test that disabled recorder skips error recording."""
        recorder_disabled.record_error("something failed")
        assert len(recorder_disabled.stats["errors"]) == 0

    def test_initial_stats(self, recorder_enabled):
        """Test initial statistics are zeroed."""
        assert recorder_enabled.stats["tags"] == 0
        assert recorder_enabled.stats["endpoints"] == 0
        assert recorder_enabled.stats["scenarios"]["total"] == 0
        assert recorder_enabled.stats["llm_calls"]["total"] == 0

    @pytest.mark.asyncio
    async def test_finalize_writes_manifest(self, recorder_enabled):
        """Test finalize writes the manifest file."""
        await recorder_enabled.finalize()
        manifest_path = recorder_enabled.debug_root / "_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["operation"] == "generate"
        assert "duration_seconds" in data
        assert "totals" in data

    @pytest.mark.asyncio
    async def test_finalize_disabled(self, recorder_disabled):
        """Test finalize does nothing when disabled."""
        await recorder_disabled.finalize()
        manifest_path = recorder_disabled.debug_root / "_manifest.json"
        assert not manifest_path.exists()

    @pytest.mark.asyncio
    async def test_record_scenario_summary_updates_stats(self, recorder_enabled):
        """Test that scenario summary updates statistics."""
        await recorder_enabled.record_scenario_summary(
            tag="users",
            endpoint_dir_name="get_users",
            scenario_type="happy_path",
            summary={"success": True, "used_fallback": False},
        )
        assert recorder_enabled.stats["scenarios"]["total"] == 1
        assert recorder_enabled.stats["scenarios"]["succeeded"] == 1
        assert recorder_enabled.stats["scenarios"]["fallback"] == 0
