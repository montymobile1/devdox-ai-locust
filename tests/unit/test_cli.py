"""Unit tests for CLI helpers."""
import io
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from rich.console import Console

from devdox_ai_locust import cli
from devdox_ai_locust.utils.open_ai_parser import Endpoint


class DummySettings:
    def __init__(self, api_key: str):
        self.API_KEY = api_key


def test_initialize_config_prefers_explicit_key(monkeypatch):
    monkeypatch.setattr(cli, "Settings", lambda: DummySettings("env-key"))
    _, api_key = cli._initialize_config("explicit-key")
    assert api_key == "explicit-key"


def test_initialize_config_requires_key(monkeypatch):
    monkeypatch.setattr(cli, "Settings", lambda: DummySettings(""))
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(cli, "console", console)
    with pytest.raises(SystemExit):
        cli._initialize_config(None)


def test_setup_output_directory(tmp_path):
    output_dir = cli._setup_output_directory(tmp_path / "nested")
    assert output_dir.exists()


def test_show_generated_files_verbose(monkeypatch, tmp_path):
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(cli, "console", console)
    created_files = [
        {"path": str(tmp_path / "output" / "locustfile.py")},
        {"path": str(tmp_path / "output" / "data" / "valid_data.py")},
    ]
    cli._show_generated_files(created_files, verbose=True, output_dir=tmp_path / "output")
    assert "Generated" in console.file.getvalue()


def test_show_run_instructions_uses_locustfile(monkeypatch, tmp_path):
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(cli, "console", console)
    (tmp_path / "locustfile.py").write_text("# locustfile")
    cli._show_run_instructions(tmp_path, 10, 2.0, "5m", "http://localhost")
    output = console.file.getvalue()
    assert "locust -f locustfile.py" in output


@pytest.mark.asyncio
async def test_process_api_schema_url(monkeypatch):
    schema_text = "openapi: 3.0.0"
    monkeypatch.setattr(cli, "get_api_schema", AsyncMock(return_value=schema_text))

    parser = MagicMock()
    parser.parse_schema.return_value = {"paths": {}}
    parser.parse_endpoints.return_value = [
        Endpoint(
            path="/health",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=None,
        )
    ]
    parser.get_schema_info.return_value = {"title": "API"}
    monkeypatch.setattr(cli, "OpenAPIParser", lambda: parser)

    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(cli, "console", console)

    schema_data, endpoints, api_info = await cli._process_api_schema(
        "https://example.com/openapi.yaml", verbose=True
    )
    assert schema_data == {"paths": {}}
    assert endpoints
    assert api_info["title"] == "API"


@pytest.mark.asyncio
async def test_generate_modular_tests_with_patch_tracking(monkeypatch, tmp_path):
    dummy_generator = MagicMock()
    dummy_generator.generate = AsyncMock(return_value={"locustfile.py": "content"})
    monkeypatch.setattr(cli, "ModularGenerator", lambda **kwargs: dummy_generator)

    class DummyPatchTracker:
        def start_session(self):
            return None

        def get_summary(self):
            return {"session_id": "sess", "total_patches": 1}

        def finalize(self):
            return None

    monkeypatch.setattr(cli.PatchTracker, "from_metadata_manager", lambda _: DummyPatchTracker())

    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(cli, "console", console)

    endpoints = [
        Endpoint(
            path="/secure",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[{"bearerAuth": []}],
        ),
        Endpoint(
            path="/public",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[],
        ),
    ]
    api_info = {
        "global_security": [{"bearerAuth": []}],
        "security_schemes": {"bearerAuth": {"type": "http"}},
        "swagger_source": "source",
        "source_type": "url",
    }

    created_files = await cli._generate_modular_tests(
        api_key="key",
        endpoints=endpoints,
        schemas={},
        api_info=api_info,
        output_dir=Path(tmp_path),
        host="http://localhost",
        auth=True,
        db_type="",
        retry_on_invalid=0,
        enable_patch_tracking=True,
        custom_requirement=None,
    )
    assert created_files[0]["path"].endswith("locustfile.py")
