"""Unit tests for ModularGenerator."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from devdox_ai_locust.modular_generator import ModularGenerator
from devdox_ai_locust.utils.open_ai_parser import Endpoint, Parameter, ParameterType


def make_endpoint(method: str, path: str, parameters=None, summary=None):
    """Helper to create Endpoint with default values."""
    return Endpoint(
        path=path,
        method=method,
        operation_id=None,
        summary=summary,
        description=None,
        parameters=parameters or [],
        request_body=None,
        responses=[],
        tags=[],
        security=None,
    )


class TestModularGeneratorInit:
    """Tests for ModularGenerator initialization."""

    def test_init_creates_generator(self, tmp_path):
        """Test generator initializes correctly."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
            target_host="http://localhost:8000",
        )
        assert generator.output_dir == tmp_path
        assert generator.target_host == "http://localhost:8000"
        assert generator.auth_enabled is True

    def test_init_with_custom_options(self, tmp_path):
        """Test generator with custom options."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
            target_host="http://api.example.com",
            auth_enabled=False,
            db_type="mongo",
        )
        assert generator.auth_enabled is False
        assert generator.db_type == "mongo"


class TestModularGeneratorDirectories:
    """Tests for directory creation."""

    def test_create_directories(self, tmp_path):
        """Test directory structure is created."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )
        generator._create_directories()

        assert (tmp_path / "data").exists()
        assert (tmp_path / "scenarios").exists()
        assert (tmp_path / "auth").exists()
        assert (tmp_path / "workflows").exists()


class TestModularGeneratorContext:
    """Tests for context building."""

    def test_build_context_with_endpoints(self, tmp_path):
        """Test context is built correctly from endpoints."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
            target_host="http://localhost",
        )

        endpoints = [
            make_endpoint("POST", "/users", summary="Create user"),
            make_endpoint("GET", "/users/{id}", summary="Get user"),
        ]
        schemas = {"User": {"properties": {"name": {}, "email": {}}, "required": ["name"]}}
        api_info = {"title": "Test API"}

        context = generator._build_context(endpoints, schemas, api_info, ["/auth/login"])

        assert context["target_host"] == "http://localhost"
        assert context["auth_enabled"] is True
        assert context["primary_endpoint"] == "/users"
        assert len(context["endpoints"]) == 2

    def test_build_context_without_auth(self, tmp_path):
        """Test context when auth is disabled."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
            auth_enabled=False,
        )

        endpoints = [make_endpoint("GET", "/items")]
        context = generator._build_context(endpoints, {}, {}, None)

        assert context["auth_enabled"] is False


class TestModularGeneratorFormatting:
    """Tests for formatting helpers."""

    def test_format_endpoints_summary(self, tmp_path):
        """Test endpoint summary formatting."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        endpoints = [
            make_endpoint("POST", "/users", parameters=[
                Parameter(name="data", location=ParameterType.QUERY, required=True, type="string")
            ]),
            make_endpoint("GET", "/users/{id}"),
        ]

        summary = generator._format_endpoints_summary(endpoints)

        assert "POST /users" in summary
        assert "GET /users/{id}" in summary
        assert "(1 params)" in summary

    def test_format_schemas_info(self, tmp_path):
        """Test schema info formatting."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        schemas = {
            "User": {
                "properties": {"name": {}, "email": {}, "age": {}},
                "required": ["name", "email"],
            }
        }

        info = generator._format_schemas_info(schemas)

        assert "User:" in info
        assert "name*" in info  # Required field marked
        assert "email*" in info

    def test_infer_primary_schema_from_post(self, tmp_path):
        """Test primary schema inference from POST endpoint."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        endpoints = [
            make_endpoint("GET", "/health"),
            make_endpoint("POST", "/api/users"),
        ]

        schema = generator._infer_primary_schema(endpoints)
        assert schema == "user"  # Singular form

    def test_infer_primary_schema_fallback(self, tmp_path):
        """Test primary schema fallback when no POST."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        schema = generator._infer_primary_schema([])
        assert schema == "resource"


class TestModularGeneratorTemplates:
    """Tests for template rendering."""

    def test_render_template_success(self, tmp_path):
        """Test successful template rendering."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        content = generator._render_template(
            "data/__init__.py.j2",
            {}
        )

        assert "ValidDataGenerator" in content
        assert "InvalidDataGenerator" in content

    def test_render_template_with_context(self, tmp_path):
        """Test template rendering with context variables."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        content = generator._render_template(
            "workflows/main_workflow.py.j2",
            {"target_host": "http://test.com"}
        )

        assert "http://test.com" in content


class TestModularGeneratorStaticFiles:
    """Tests for static file generation."""

    @pytest.mark.asyncio
    async def test_generate_static_files(self, tmp_path):
        """Test static files are generated."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
            target_host="http://localhost",
        )

        context = {
            "endpoints": [],
            "auth_enabled": False,
            "target_host": "http://localhost",
            "primary_endpoint": "/api/test",
            "primary_schema": "test",
            "api_info": {"title": "Test API"},
            "environment_vars": {
                "API_BASE_URL": "http://localhost",
                "API_VERSION": "v1",
                "API_TITLE": "Test API",
            },
        }

        files = await generator._generate_static_files(context)

        # Essential files
        assert "locustfile.py" in files
        assert "config.py" in files
        assert "utils.py" in files
        assert "requirements.txt" in files
        assert "README.md" in files
        assert ".env.example" in files

        # Modular structure
        assert "data/__init__.py" in files
        assert "data/base_generator.py" in files
        assert "scenarios/__init__.py" in files
        assert "auth/__init__.py" in files
        assert "workflows/__init__.py" in files

    @pytest.mark.asyncio
    async def test_generate_locustfile(self, tmp_path):
        """Test locustfile.py is generated as entry point."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
            target_host="http://localhost:8000",
        )

        context = {
            "endpoints": [],
            "auth_enabled": True,
            "target_host": "http://localhost:8000",
            "primary_endpoint": "/api/test",
            "primary_schema": "test",
            "api_info": {"title": "Test API"},
        }

        files = await generator._generate_static_files(context)

        # Verify locustfile.py is generated
        assert "locustfile.py" in files
        content = files["locustfile.py"]
        assert "APILoadTest" in content
        assert "from locust import HttpUser" in content
        assert "http://localhost:8000" in content
        assert "from workflows.main_workflow import MainWorkflow" in content


class TestModularGeneratorFileWriting:
    """Tests for file writing."""

    def test_write_file_creates_directories(self, tmp_path):
        """Test file writing creates parent directories."""
        generator = ModularGenerator(
            output_dir=str(tmp_path),
            api_key="test-key",
        )

        generator._write_file("nested/path/file.py", "# content")

        assert (tmp_path / "nested/path/file.py").exists()
        assert (tmp_path / "nested/path/file.py").read_text() == "# content"
