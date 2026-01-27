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

    @pytest.mark.asyncio
    async def test_record_scenario_summary_fallback(self, recorder_enabled):
        """Test scenario summary with fallback."""
        await recorder_enabled.record_scenario_summary(
            tag="users",
            endpoint_dir_name="get_users",
            scenario_type="positive",
            summary={"success": False, "used_fallback": True},
        )
        assert recorder_enabled.stats["scenarios"]["total"] == 1
        assert recorder_enabled.stats["scenarios"]["succeeded"] == 0
        assert recorder_enabled.stats["scenarios"]["fallback"] == 1

    def test_record_openapi_raw(self, recorder_enabled):
        """Test recording raw OpenAPI spec."""
        recorder_enabled.record_openapi_raw({"openapi": "3.0.0"})
        path = recorder_enabled.input_dir / "openapi_raw.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["openapi"] == "3.0.0"

    def test_record_openapi_raw_disabled(self, recorder_disabled):
        """Test disabled recorder skips raw OpenAPI."""
        recorder_disabled.record_openapi_raw({"openapi": "3.0.0"})
        assert not (recorder_disabled.input_dir / "openapi_raw.json").exists()

    def test_record_openapi_parsed(self, recorder_enabled):
        """Test recording parsed OpenAPI data with endpoints."""
        from unittest.mock import Mock

        ep = Mock()
        ep.method = "GET"
        ep.path = "/users"
        ep.operation_id = "getUsers"
        ep.tags = ["users"]
        ep.summary = "Get users"
        ep.description = "Desc"
        ep.parameters = []
        ep.request_body = None
        ep.responses = {}

        recorder_enabled.record_openapi_parsed([ep], {"title": "API"})
        path = recorder_enabled.input_dir / "openapi_parsed.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["endpoint_count"] == 1
        assert recorder_enabled.stats["endpoints"] == 1

    def test_record_openapi_parsed_with_params_and_body(self, recorder_enabled):
        """Test recording parsed endpoints with parameters and request body."""
        from unittest.mock import Mock

        param = Mock()
        param.name = "id"
        param.location = "path"
        param.required = True
        param.type = "integer"
        param.format = None
        param.enum = None
        param.description = "User ID"

        rb = Mock()
        rb.required = True
        rb.content_type = "application/json"
        rb.schema = {"type": "object"}

        resp = Mock()
        resp.description = "OK"

        ep = Mock()
        ep.method = "POST"
        ep.path = "/users"
        ep.operation_id = "createUser"
        ep.tags = ["users"]
        ep.summary = "Create"
        ep.description = None
        ep.parameters = [param]
        ep.request_body = rb
        ep.responses = {"201": resp}

        recorder_enabled.record_openapi_parsed([ep], {"title": "API"})
        data = json.loads(
            (recorder_enabled.input_dir / "openapi_parsed.json").read_text()
        )
        assert len(data["endpoints"][0]["parameters"]) == 1
        assert data["endpoints"][0]["request_body"]["required"] is True

    def test_record_openapi_parsed_list_responses(self, recorder_enabled):
        """Test recording endpoints with list-type responses."""
        from unittest.mock import Mock

        resp = Mock()
        resp.status_code = "200"
        resp.description = "OK"

        ep = Mock()
        ep.method = "GET"
        ep.path = "/items"
        ep.operation_id = "getItems"
        ep.tags = []
        ep.summary = ""
        ep.description = None
        ep.parameters = []
        ep.request_body = None
        ep.responses = [resp]  # list, not dict

        recorder_enabled.record_openapi_parsed([ep], {})
        data = json.loads(
            (recorder_enabled.input_dir / "openapi_parsed.json").read_text()
        )
        assert "200" in data["endpoints"][0]["responses"]

    def test_record_resolved_config_content(self, recorder_enabled):
        """Test recording resolved config content."""
        recorder_enabled.record_resolved_config({"model": "test"})
        path = recorder_enabled.input_dir / "resolved_config.json"
        assert path.exists()

    def test_record_resolved_config_disabled(self, recorder_disabled):
        """Test disabled recorder skips resolved config."""
        recorder_disabled.record_resolved_config({"model": "test"})
        assert not (recorder_disabled.input_dir / "resolved_config.json").exists()

    def test_record_static_file_txt(self, recorder_enabled):
        """Test recording a .txt static file."""
        recorder_enabled.record_static_file("readme.txt", {"key": "val"}, "Hello")
        rendered = recorder_enabled.static_dir / "readme" / "rendered.txt"
        assert rendered.exists()

    def test_record_static_file_md(self, recorder_enabled):
        """Test recording a .md static file."""
        recorder_enabled.record_static_file("docs.md", {}, "# Title")
        rendered = recorder_enabled.static_dir / "docs" / "rendered.md"
        assert rendered.exists()

    def test_record_static_file_other_ext(self, recorder_enabled):
        """Test recording a file with other extension."""
        recorder_enabled.record_static_file("data.yaml", {}, "key: value")
        # dir_name keeps .yaml since only .py/.txt/.md are stripped
        rendered = recorder_enabled.static_dir / "data.yaml" / "rendered.yaml"
        assert rendered.exists()

    def test_record_static_file_disabled(self, recorder_disabled):
        """Test disabled recorder skips static file."""
        recorder_disabled.record_static_file("f.py", {}, "code")
        assert not (recorder_disabled.static_dir / "f" / "rendered.py").exists()

    def test_record_static_file_filters_context(self, recorder_enabled):
        """Test that base_workflow_content and test_data_content are filtered."""
        recorder_enabled.record_static_file(
            "test.py",
            {"key": "val", "base_workflow_content": "big", "test_data_content": "big"},
            "code",
        )
        ctx = json.loads(
            (recorder_enabled.static_dir / "test" / "context.json").read_text()
        )
        assert "key" in ctx
        assert "base_workflow_content" not in ctx

    def test_get_scenario_dir(self, recorder_enabled):
        """Test _get_scenario_dir returns correct path."""
        path = recorder_enabled._get_scenario_dir("users", "get_users", "positive")
        expected = recorder_enabled.workflows_dir / "users" / "get_users" / "positive"
        assert path == expected

    @pytest.mark.asyncio
    async def test_record_endpoint_details(self, recorder_enabled):
        """Test recording endpoint details."""
        from unittest.mock import Mock

        ep = Mock()
        ep.method = "GET"
        ep.path = "/users"
        ep.operation_id = "getUsers"

        await recorder_enabled.record_endpoint_details(
            "users", "get_users", ep, "formatted"
        )
        path = recorder_enabled.workflows_dir / "users" / "get_users" / "_endpoint.json"
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_endpoint_details_disabled(self, recorder_disabled):
        """Test disabled recorder skips endpoint details."""
        from unittest.mock import Mock

        await recorder_disabled.record_endpoint_details(
            "users", "get_users", Mock(), "formatted"
        )

    @pytest.mark.asyncio
    async def test_record_scenario_context(self, recorder_enabled):
        """Test recording scenario context."""
        await recorder_enabled.record_scenario_context(
            "users", "get_users", "positive", {"key": "val"}
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "context.json"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_scenario_prompt(self, recorder_enabled):
        """Test recording scenario prompt."""
        await recorder_enabled.record_scenario_prompt(
            "users", "get_users", "positive", "Generate tests for..."
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "prompt.txt"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_llm_request(self, recorder_enabled):
        """Test recording LLM request."""
        await recorder_enabled.record_llm_request(
            "users", "get_users", "positive", {"model": "test", "messages": []}
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "llm_request.json"
        )
        assert path.exists()
        assert recorder_enabled.stats["llm_calls"]["total"] == 1

    @pytest.mark.asyncio
    async def test_record_llm_response(self, recorder_enabled):
        """Test recording LLM response."""
        await recorder_enabled.record_llm_response(
            "users", "get_users", "positive", "response text"
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "llm_response.txt"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_extracted_code(self, recorder_enabled):
        """Test recording extracted code."""
        await recorder_enabled.record_extracted_code(
            "users", "get_users", "positive", "print('hello')"
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "extracted.py"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_processed_code(self, recorder_enabled):
        """Test recording processed code."""
        await recorder_enabled.record_processed_code(
            "users", "get_users", "positive", "print('processed')"
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "processed.py"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_validation_result(self, recorder_enabled):
        """Test recording validation result."""
        await recorder_enabled.record_validation_result(
            "users",
            "get_users",
            "positive",
            is_valid=True,
            error=None,
            checks=[{"check": "syntax", "passed": True}],
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "validation.json"
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["valid"] is True

    @pytest.mark.asyncio
    async def test_record_final_code(self, recorder_enabled):
        """Test recording final code."""
        await recorder_enabled.record_final_code(
            "users", "get_users", "positive", "final code"
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "final.py"
        )
        assert path.exists()
        assert recorder_enabled.stats["llm_calls"]["succeeded"] == 1

    @pytest.mark.asyncio
    async def test_record_retry_attempt(self, recorder_enabled):
        """Test recording retry attempt."""
        await recorder_enabled.record_retry_attempt(
            tag="users",
            endpoint_dir_name="get_users",
            scenario_type="positive",
            attempt=1,
            error="syntax error",
            bad_code="bad",
            fix_prompt="fix this",
            llm_response="fixed",
            extracted_code="good",
            validation_result={"valid": True},
        )
        retry_dir = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "_retry_1"
        )
        assert retry_dir.exists()
        assert (retry_dir / "error.txt").exists()
        assert recorder_enabled.stats["llm_calls"]["retries"] == 1
        assert recorder_enabled.stats["llm_calls"]["total"] == 1

    @pytest.mark.asyncio
    async def test_record_fallback(self, recorder_enabled):
        """Test recording fallback."""
        await recorder_enabled.record_fallback(
            tag="users",
            endpoint_dir_name="get_users",
            scenario_type="positive",
            reason="all retries failed",
            all_errors=["err1", "err2"],
            fallback_template="# fallback",
        )
        fb_dir = (
            recorder_enabled.workflows_dir
            / "users"
            / "get_users"
            / "positive"
            / "_fallback"
        )
        assert fb_dir.exists()
        assert (fb_dir / "reason.txt").exists()
        assert (fb_dir / "template.py").exists()

    @pytest.mark.asyncio
    async def test_record_orchestrator_context(self, recorder_enabled):
        """Test recording orchestrator context."""
        await recorder_enabled.record_orchestrator_context(
            "users", {"endpoints": ["GET /users"]}
        )
        path = (
            recorder_enabled.workflows_dir / "users" / "orchestrator" / "context.json"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_orchestrator_prompt(self, recorder_enabled):
        """Test recording orchestrator prompt."""
        await recorder_enabled.record_orchestrator_prompt(
            "users", "Generate orchestrator"
        )
        path = recorder_enabled.workflows_dir / "users" / "orchestrator" / "prompt.txt"
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_orchestrator_llm_request(self, recorder_enabled):
        """Test recording orchestrator LLM request."""
        await recorder_enabled.record_orchestrator_llm_request(
            "users", {"model": "test"}
        )
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "orchestrator"
            / "llm_request.json"
        )
        assert path.exists()
        assert recorder_enabled.stats["llm_calls"]["total"] == 1

    @pytest.mark.asyncio
    async def test_record_orchestrator_llm_response(self, recorder_enabled):
        """Test recording orchestrator LLM response."""
        await recorder_enabled.record_orchestrator_llm_response("users", "response")
        path = (
            recorder_enabled.workflows_dir
            / "users"
            / "orchestrator"
            / "llm_response.txt"
        )
        assert path.exists()

    @pytest.mark.asyncio
    async def test_record_orchestrator_final_success(self, recorder_enabled):
        """Test recording orchestrator final with success."""
        await recorder_enabled.record_orchestrator_final(
            "users", "final code", {"success": True, "used_fallback": False}
        )
        path = recorder_enabled.workflows_dir / "users" / "orchestrator" / "final.py"
        assert path.exists()
        assert recorder_enabled.stats["orchestrators"]["total"] == 1
        assert recorder_enabled.stats["orchestrators"]["succeeded"] == 1
        assert recorder_enabled.stats["llm_calls"]["succeeded"] == 1

    @pytest.mark.asyncio
    async def test_record_orchestrator_final_fallback(self, recorder_enabled):
        """Test recording orchestrator final with fallback."""
        await recorder_enabled.record_orchestrator_final(
            "users", "fallback code", {"success": False, "used_fallback": True}
        )
        assert recorder_enabled.stats["orchestrators"]["fallback"] == 1
        assert recorder_enabled.stats["orchestrators"]["succeeded"] == 0

    @pytest.mark.asyncio
    async def test_record_tag_summary(self, recorder_enabled):
        """Test recording tag summary."""
        await recorder_enabled.record_tag_summary("users", {"endpoints": 3})
        path = recorder_enabled.workflows_dir / "users" / "_summary.json"
        assert path.exists()
        assert recorder_enabled.stats["tags"] == 1

    @pytest.mark.asyncio
    async def test_record_endpoint_summary(self, recorder_enabled):
        """Test recording endpoint summary."""
        await recorder_enabled.record_endpoint_summary(
            "users", "get_users", {"success": True}
        )
        path = recorder_enabled.workflows_dir / "users" / "get_users" / "_summary.json"
        assert path.exists()

    @pytest.mark.asyncio
    async def test_write_json_disabled(self, recorder_disabled):
        """Test _write_json does nothing when disabled."""
        await recorder_disabled._write_json(
            recorder_disabled.debug_root / "test.json", {"key": "val"}
        )
        assert not (recorder_disabled.debug_root / "test.json").exists()

    @pytest.mark.asyncio
    async def test_write_text_disabled(self, recorder_disabled):
        """Test _write_text does nothing when disabled."""
        await recorder_disabled._write_text(
            recorder_disabled.debug_root / "test.txt", "content"
        )
        assert not (recorder_disabled.debug_root / "test.txt").exists()

    def test_write_json_sync_disabled(self, recorder_disabled):
        """Test _write_json_sync does nothing when disabled."""
        recorder_disabled._write_json_sync(
            recorder_disabled.debug_root / "test.json", {"key": "val"}
        )
        assert not (recorder_disabled.debug_root / "test.json").exists()

    def test_write_text_sync_disabled(self, recorder_disabled):
        """Test _write_text_sync does nothing when disabled."""
        recorder_disabled._write_text_sync(
            recorder_disabled.debug_root / "test.txt", "content"
        )
        assert not (recorder_disabled.debug_root / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_disabled_async_methods_are_noop(self, recorder_disabled):
        """Test all async recording methods are no-op when disabled."""
        from unittest.mock import Mock

        await recorder_disabled.record_endpoint_details("t", "e", Mock(), "f")
        await recorder_disabled.record_scenario_context("t", "e", "s", {})
        await recorder_disabled.record_scenario_prompt("t", "e", "s", "p")
        await recorder_disabled.record_llm_request("t", "e", "s", {})
        await recorder_disabled.record_llm_response("t", "e", "s", "r")
        await recorder_disabled.record_extracted_code("t", "e", "s", "c")
        await recorder_disabled.record_processed_code("t", "e", "s", "c")
        await recorder_disabled.record_validation_result("t", "e", "s", True)
        await recorder_disabled.record_final_code("t", "e", "s", "c")
        await recorder_disabled.record_scenario_summary("t", "e", "s", {})
        await recorder_disabled.record_retry_attempt(
            "t", "e", "s", 1, "e", "b", "f", "r", "c", {}
        )
        await recorder_disabled.record_fallback("t", "e", "s", "r", [], "t")
        await recorder_disabled.record_orchestrator_context("t", {})
        await recorder_disabled.record_orchestrator_prompt("t", "p")
        await recorder_disabled.record_orchestrator_llm_request("t", {})
        await recorder_disabled.record_orchestrator_llm_response("t", "r")
        await recorder_disabled.record_orchestrator_final("t", "c", {})
        await recorder_disabled.record_tag_summary("t", {})
        await recorder_disabled.record_endpoint_summary("t", "e", {})
        # All should be no-ops; stats unchanged
        assert recorder_disabled.stats["llm_calls"]["total"] == 0
