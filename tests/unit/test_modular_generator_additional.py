"""Additional unit tests for ModularGenerator coverage."""
from devdox_ai_locust.modular_generator import ModularGenerator
from devdox_ai_locust.utils.open_ai_parser import Endpoint


def make_endpoint(method: str, path: str, tags=None, summary=None):
    return Endpoint(
        path=path,
        method=method,
        operation_id=None,
        summary=summary,
        description=None,
        parameters=[],
        request_body=None,
        responses=[],
        tags=tags or [],
        security=None,
    )


def test_build_context_db_variants(tmp_path):
    generator = ModularGenerator(
        output_dir=str(tmp_path),
        api_key="test-key",
        db_type="mongo",
        custom_requirement="Use custom auth",
    )
    context = generator._build_context(
        endpoints=[make_endpoint("POST", "/items", tags=["Items"])],
        schemas={},
        api_info={"title": "Test API", "version": "v2"},
        auth_endpoints=["/auth/login", "/auth/logout"],
    )

    assert context["environment_vars"]["MONGO_URI"] == "mongodb://localhost:27017"
    assert "MongoDB" in context["db_using"]
    assert context["auth_enabled"] is True
    assert context["auth_login_endpoint"] == "/auth/login"

    generator.db_type = "postgresql"
    context = generator._build_context(
        endpoints=[make_endpoint("GET", "/health")],
        schemas={},
        api_info={},
        auth_endpoints=None,
    )
    assert context["environment_vars"]["POSTGRES_HOST"] == "localhost"
    assert "PostgreSQL" in context["db_using"]
    assert context["auth_enabled"] is False


def test_generate_db_files_mongo(tmp_path):
    generator = ModularGenerator(
        output_dir=str(tmp_path),
        api_key="test-key",
        db_type="mongo",
    )
    files = generator._generate_db_files({"dummy": "context"})
    assert "db_config.py" in files
    assert "data_provider.py" in files


def test_group_workflow_and_common_security(tmp_path):
    generator = ModularGenerator(
        output_dir=str(tmp_path),
        api_key="test-key",
    )
    grouped = generator._group_endpoints_by_tag(
        [make_endpoint("GET", "/users", tags=["Users"], summary="List users")]
    )
    workflows = generator._generate_group_workflows(grouped, {})
    assert "workflows/main_workflow.py" in workflows
    assert "UsersWorkflow" in workflows["workflows/main_workflow.py"]

    security_content = generator._generate_common_security(
        {"auth_enabled": True, "auth_login_endpoint": "/auth/login"}
    )
    assert "CommonSecurityTasks" in security_content


def test_strip_xml_tags_and_normalize_indentation(tmp_path):
    generator = ModularGenerator(
        output_dir=str(tmp_path),
        api_key="test-key",
    )
    raw = "<new_methods>\n@task(1)\n    def foo(self):\n        return 1\n</new_methods>"
    stripped = generator._strip_xml_tags(raw)
    assert "<new_methods>" not in stripped

    normalized = generator._normalize_indentation(stripped)
    assert normalized.startswith("@task")
    assert "def foo" in normalized


def test_generate_group_scenario_file_and_prompt(tmp_path):
    generator = ModularGenerator(
        output_dir=str(tmp_path),
        api_key="test-key",
        custom_requirement="Always include headers",
    )
    endpoints = [
        make_endpoint("GET", "/items", tags=["Items"]),
        make_endpoint("POST", "/items", tags=["Items"]),
    ]
    scenario = generator._generate_group_scenario_file(
        "items",
        "Items",
        "Positive",
        "@task(1)\ndef test_item(self):\n    pass",
        endpoints,
    )
    assert "ItemsPositiveTasks" in scenario
    assert "GET /items" in scenario

    prompt = generator._build_group_prompt("items", "positive", {"endpoints_summary": ""})
    assert "Additional Requirements from Developer" in prompt
