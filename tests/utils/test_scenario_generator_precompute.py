"""
Tests for ScenarioWorkflowGenerator pre-computation methods that do NOT need LLM mocking.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from pathlib import Path

from devdox_ai_locust.utils.scenario_generator import (
    ScenarioWorkflowGenerator,
    ScenarioType,
)
from devdox_ai_locust.utils.open_ai_parser import (
    Endpoint,
    Parameter,
    RequestBody,
    Response,
    ParameterType,
)
from devdox_ai_locust.config import AIEnhancementConfig


@pytest.fixture
def ai_config():
    config = Mock(spec=AIEnhancementConfig)
    config.model = "test-model"
    config.max_tokens = 4000
    config.temperature = 0.3
    config.timeout = 30
    return config


@pytest.fixture
def generator(ai_config):
    prompt_dir = (
        Path(__file__).parent.parent.parent / "src" / "devdox_ai_locust" / "prompt"
    )
    ai_client = AsyncMock()
    gen = ScenarioWorkflowGenerator(
        prompt_dir=prompt_dir,
        ai_client=ai_client,
        ai_config=ai_config,
    )
    return gen


@pytest.fixture
def endpoint_with_body():
    """Endpoint with a request body containing string fields."""
    return Endpoint(
        path="/items",
        method="POST",
        operation_id="createItem",
        summary="Create an item",
        description="Creates a new item",
        parameters=[],
        request_body=RequestBody(
            content_type="application/json",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "count": {"type": "integer"},
                    "price": {"type": "number", "minimum": 0, "maximum": 9999},
                    "category": {"type": "string", "enum": ["A", "B", "C"]},
                    "code": {"type": "string", "pattern": "^[A-Z]{3}-\\d{4}$"},
                },
                "required": ["name", "count"],
            },
            required=True,
        ),
        responses=[
            Response(
                status_code="201",
                description="Created",
                content_type="application/json",
                schema={"type": "object"},
            ),
            Response(status_code="400", description="Bad request"),
        ],
        tags=["items"],
    )


@pytest.fixture
def endpoint_no_body():
    """GET endpoint with no request body."""
    return Endpoint(
        path="/items/{item_id}",
        method="GET",
        operation_id="getItem",
        summary="Get an item",
        description=None,
        parameters=[
            Parameter(
                name="item_id",
                location=ParameterType.PATH,
                required=True,
                type="integer",
                description="Item ID",
            ),
        ],
        request_body=None,
        responses=[
            Response(
                status_code="200",
                description="OK",
                content_type="application/json",
                schema={"type": "object"},
            ),
            Response(status_code="404", description="Not found"),
        ],
        tags=["items"],
    )


@pytest.fixture
def endpoint_with_query_string_params():
    """Endpoint with string query parameters."""
    return Endpoint(
        path="/search",
        method="GET",
        operation_id="searchItems",
        summary="Search items",
        description=None,
        parameters=[
            Parameter(
                name="q",
                location=ParameterType.QUERY,
                required=False,
                type="string",
                description="Search query",
            ),
            Parameter(
                name="filter",
                location=ParameterType.QUERY,
                required=False,
                type="string",
                description="Filter",
            ),
            Parameter(
                name="page",
                location=ParameterType.QUERY,
                required=False,
                type="integer",
                description="Page number",
            ),
        ],
        request_body=None,
        responses=[
            Response(status_code="200", description="OK"),
        ],
        tags=["search"],
    )


@pytest.fixture
def endpoint_no_operation_id():
    """Endpoint without an operation_id."""
    return Endpoint(
        path="/orders/{order_id}/items",
        method="DELETE",
        operation_id=None,
        summary="Delete order items",
        description=None,
        parameters=[
            Parameter(
                name="order_id",
                location=ParameterType.PATH,
                required=True,
                type="string",
                description="Order ID",
            ),
        ],
        request_body=None,
        responses=[],
        tags=["orders"],
    )


@pytest.fixture
def all_endpoints(
    endpoint_with_body, endpoint_no_body, endpoint_with_query_string_params
):
    """Collection of all test endpoints."""
    return [endpoint_with_body, endpoint_no_body, endpoint_with_query_string_params]


# ============================================================
# 1. _operation_to_class_name
# ============================================================


class TestOperationToClassName:
    def test_simple_camel_case(self, generator, sample_endpoints):
        ep = sample_endpoints[0]  # getUsers
        assert generator._operation_to_class_name(ep) == "Getusers"

    def test_create_user(self, generator, sample_endpoints):
        ep = sample_endpoints[1]  # createUser
        assert generator._operation_to_class_name(ep) == "Createuser"

    def test_with_underscores(self, generator):
        ep = Mock()
        ep.operation_id = "get_user_by_id"
        result = generator._operation_to_class_name(ep)
        assert result == "GetUserById"

    def test_no_operation_id_generates_from_path(
        self, generator, endpoint_no_operation_id
    ):
        result = generator._operation_to_class_name(endpoint_no_operation_id)
        assert result[0].isupper()
        assert "Order" in result or "order" in result.lower()

    def test_empty_operation_id(self, generator):
        ep = Mock()
        ep.operation_id = ""
        ep.method = "get"
        ep.path = "/health"
        result = generator._operation_to_class_name(ep)
        assert result[0].isupper()


# ============================================================
# 2. _generate_operation_id
# ============================================================


class TestGenerateOperationId:
    def test_simple_path(self, generator):
        ep = Mock()
        ep.method = "GET"
        ep.path = "/users"
        result = generator._generate_operation_id(ep)
        assert "get" in result
        assert "users" in result

    def test_path_with_param(self, generator):
        ep = Mock()
        ep.method = "PUT"
        ep.path = "/users/{id}"
        result = generator._generate_operation_id(ep)
        assert "put" in result
        assert "id" in result
        assert "{" not in result

    def test_nested_path(self, generator):
        ep = Mock()
        ep.method = "POST"
        ep.path = "/api/v1/users"
        result = generator._generate_operation_id(ep)
        assert "post" in result
        assert "_" in result

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/"),
            ("POST", "/a"),
            ("DELETE", "/users/{user_id}/posts/{post_id}"),
        ],
    )
    def test_various_paths(self, generator, method, path):
        ep = Mock()
        ep.method = method
        ep.path = path
        result = generator._generate_operation_id(ep)
        assert result  # non-empty
        assert result.isidentifier()


# ============================================================
# 3. get_endpoint_dir_name
# ============================================================


class TestGetEndpointDirName:
    def test_uses_operation_id(self, generator, sample_endpoints):
        ep = sample_endpoints[0]  # getUsers
        result = generator.get_endpoint_dir_name(ep)
        assert result == "getusers"

    def test_lowercase(self, generator):
        ep = Mock()
        ep.operation_id = "CreateUser"
        result = generator.get_endpoint_dir_name(ep)
        assert result == result.lower()

    def test_no_operation_id_falls_back(self, generator, endpoint_no_operation_id):
        result = generator.get_endpoint_dir_name(endpoint_no_operation_id)
        assert result
        assert result == result.lower()


# ============================================================
# 4. _sanitize_identifier
# ============================================================


class TestSanitizeIdentifier:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("hello_world", "hello_world"),
            ("hello-world", "hello_world"),
            ("hello world", "hello_world"),
            ("hello.world", "hello_world"),
            ("hello/world", "hello_world"),
            ("123start", "n123start"),
            ("", "unnamed"),
            ("a--b", "a_b"),
            ("__leading__", "leading"),
            ("special!@#chars", "specialchars"),
        ],
    )
    def test_various_inputs(self, generator, input_str, expected):
        assert generator._sanitize_identifier(input_str) == expected


# ============================================================
# 5. _extract_expected_status_codes
# ============================================================


class TestExtractExpectedStatusCodes:
    def test_with_sample_endpoints(self, generator, sample_endpoints):
        # getUsers has 200
        codes = generator._extract_expected_status_codes(sample_endpoints[0])
        assert 200 in codes

    def test_multiple_codes(self, generator, sample_endpoints):
        # getUserById has 200 and 404
        codes = generator._extract_expected_status_codes(sample_endpoints[2])
        assert codes == [200, 404]

    def test_no_responses(self, generator):
        ep = Mock()
        ep.responses = []
        assert generator._extract_expected_status_codes(ep) == []

    def test_no_responses_attr(self, generator):
        ep = Mock(spec=[])
        assert generator._extract_expected_status_codes(ep) == []


# ============================================================
# 6. _precompute_injection_points
# ============================================================


class TestPrecomputeInjectionPoints:
    def test_with_body_string_fields(self, generator, endpoint_with_body):
        result = generator._precompute_injection_points(endpoint_with_body)
        assert result is not None
        assert "name" in result
        assert "description" in result

    def test_no_body_no_query(self, generator, endpoint_no_body):
        result = generator._precompute_injection_points(endpoint_no_body)
        assert result is None

    def test_query_string_params_only(
        self, generator, endpoint_with_query_string_params
    ):
        result = generator._precompute_injection_points(
            endpoint_with_query_string_params
        )
        assert result is not None
        assert "q" in result

    def test_with_sample_login_endpoint(self, generator, sample_endpoints):
        login = sample_endpoints[3]  # login with username/password
        result = generator._precompute_injection_points(login)
        assert result is not None
        assert "username" in result


# ============================================================
# 7. _scan_body_string_fields
# ============================================================


class TestScanBodyStringFields:
    def test_finds_string_fields(self, generator, endpoint_with_body):
        fields = generator._scan_body_string_fields(endpoint_with_body)
        assert "name" in fields
        assert "description" in fields
        # category and code are also strings
        assert "category" in fields

    def test_excludes_non_string(self, generator, endpoint_with_body):
        fields = generator._scan_body_string_fields(endpoint_with_body)
        assert "count" not in fields
        assert "price" not in fields

    def test_no_body(self, generator, endpoint_no_body):
        assert generator._scan_body_string_fields(endpoint_no_body) == []


# ============================================================
# 8. _scan_query_string_params
# ============================================================


class TestScanQueryStringParams:
    def test_finds_string_query_params(
        self, generator, endpoint_with_query_string_params
    ):
        params = generator._scan_query_string_params(endpoint_with_query_string_params)
        assert "q" in params
        assert "filter" in params

    def test_excludes_non_string(self, generator, endpoint_with_query_string_params):
        params = generator._scan_query_string_params(endpoint_with_query_string_params)
        assert "page" not in params

    def test_no_params(self, generator, endpoint_with_body):
        assert generator._scan_query_string_params(endpoint_with_body) == []


# ============================================================
# 9. _precompute_positive_fields
# ============================================================


class TestPrecomputePositiveFields:
    def test_with_body(self, generator, endpoint_with_body):
        result = generator._precompute_positive_fields(endpoint_with_body)
        assert "FIELD GENERATION INSTRUCTIONS" in result
        assert "name" in result
        assert "count" in result

    def test_no_body(self, generator, endpoint_no_body):
        assert generator._precompute_positive_fields(endpoint_no_body) == ""

    def test_required_fields_listed(self, generator, endpoint_with_body):
        result = generator._precompute_positive_fields(endpoint_with_body)
        assert "REQUIRED" in result


# ============================================================
# 10. _precompute_negative_scenarios
# ============================================================


class TestPrecomputeNegativeScenarios:
    def test_with_body_fields(self, generator, endpoint_with_body):
        result = generator._precompute_negative_scenarios(endpoint_with_body)
        assert "TESTABLE NEGATIVE SCENARIOS" in result
        assert "MISSING_REQUIRED" in result

    def test_with_path_params(self, generator, endpoint_no_body):
        result = generator._precompute_negative_scenarios(endpoint_no_body)
        assert "NON_EXISTENT_ID" in result

    def test_no_testable_scenarios(self, generator):
        ep = Endpoint(
            path="/health",
            method="GET",
            operation_id="health",
            summary="Health check",
            description=None,
            parameters=[],
            request_body=None,
            responses=[Response(status_code="200", description="OK")],
            tags=[],
        )
        assert generator._precompute_negative_scenarios(ep) == ""

    def test_enum_scenario(self, generator, endpoint_with_body):
        result = generator._precompute_negative_scenarios(endpoint_with_body)
        assert "INVALID_ENUM" in result

    def test_pattern_scenario(self, generator, endpoint_with_body):
        result = generator._precompute_negative_scenarios(endpoint_with_body)
        assert "INVALID_PATTERN" in result

    def test_boundary_scenario(self, generator, endpoint_with_body):
        result = generator._precompute_negative_scenarios(endpoint_with_body)
        assert "BOUNDARY" in result


# ============================================================
# 11. _extract_body_field_categories
# ============================================================


class TestExtractBodyFieldCategories:
    def test_with_body(self, generator, endpoint_with_body):
        required, typed, enum, pattern, numeric = (
            generator._extract_body_field_categories(endpoint_with_body)
        )
        assert "name" in required
        assert "count" in required
        assert any(n == "count" for n, _ in typed)
        assert any(n == "category" for n, _ in enum)
        assert any(n == "code" for n, _ in pattern)
        assert any(n == "price" for n, _, _ in numeric)

    def test_no_body(self, generator, endpoint_no_body):
        result = generator._extract_body_field_categories(endpoint_no_body)
        assert result == ([], [], [], [], [])


# ============================================================
# 12. _find_setup_endpoints
# ============================================================


class TestFindSetupEndpoints:
    def test_get_endpoint_finds_post(self, generator, endpoint_no_body, all_endpoints):
        section, count = generator._find_setup_endpoints(
            endpoint_no_body, all_endpoints
        )
        # endpoint_no_body is GET /items/{item_id}, should find POST /items
        assert count >= 1

    def test_post_endpoint_returns_empty(
        self, generator, endpoint_with_body, all_endpoints
    ):
        # POST /items should not need setup
        section, count = generator._find_setup_endpoints(
            endpoint_with_body, all_endpoints
        )
        assert count == 0

    def test_no_all_endpoints(self, generator, endpoint_no_body):
        section, count = generator._find_setup_endpoints(endpoint_no_body, None)
        assert count == 0


# ============================================================
# 13. _find_related_create_endpoints
# ============================================================


class TestFindRelatedCreateEndpoints:
    def test_get_finds_post(self, generator, endpoint_no_body, all_endpoints):
        results = generator._find_related_create_endpoints(
            endpoint_no_body, all_endpoints
        )
        assert len(results) >= 1
        # First result should be POST /items
        assert results[0][0].method == "POST"

    def test_post_returns_empty(self, generator, endpoint_with_body, all_endpoints):
        results = generator._find_related_create_endpoints(
            endpoint_with_body, all_endpoints
        )
        assert results == []

    def test_no_match(
        self, generator, endpoint_with_query_string_params, all_endpoints
    ):
        # GET /search has no related POST /search
        results = generator._find_related_create_endpoints(
            endpoint_with_query_string_params, all_endpoints
        )
        # May or may not find related; just check it doesn't crash
        assert isinstance(results, list)


# ============================================================
# 14. _format_endpoint_parameters
# ============================================================


class TestFormatEndpointParameters:
    def test_with_params(self, generator, endpoint_no_body):
        lines, has_cookie, has_header = generator._format_endpoint_parameters(
            endpoint_no_body
        )
        assert len(lines) > 0
        assert "item_id" in "\n".join(lines)
        assert not has_cookie
        assert not has_header

    def test_no_params(self, generator, endpoint_with_body):
        lines, has_cookie, has_header = generator._format_endpoint_parameters(
            endpoint_with_body
        )
        assert lines == []

    def test_query_params(self, generator, endpoint_with_query_string_params):
        lines, _, _ = generator._format_endpoint_parameters(
            endpoint_with_query_string_params
        )
        text = "\n".join(lines)
        assert "q" in text
        assert "page" in text


# ============================================================
# 15. _format_endpoint_request_body
# ============================================================


class TestFormatEndpointRequestBody:
    def test_with_body(self, generator, endpoint_with_body):
        lines = generator._format_endpoint_request_body(endpoint_with_body)
        text = "\n".join(lines)
        assert "Request Body" in text
        assert "application/json" in text

    def test_no_body(self, generator, endpoint_no_body):
        assert generator._format_endpoint_request_body(endpoint_no_body) == []


# ============================================================
# 16. _format_single_endpoint
# ============================================================


class TestFormatSingleEndpoint:
    def test_basic_format(self, generator, sample_endpoints):
        result = generator._format_single_endpoint(sample_endpoints[0])
        assert "GET /users" in result
        assert "getUsers" in result

    def test_includes_parameters(self, generator, endpoint_no_body):
        result = generator._format_single_endpoint(endpoint_no_body)
        assert "item_id" in result

    def test_includes_request_body(self, generator, endpoint_with_body):
        result = generator._format_single_endpoint(endpoint_with_body)
        assert "Request Body" in result


# ============================================================
# 17. estimate_time
# ============================================================


class TestEstimateTime:
    def test_basic(self, generator):
        est = generator.estimate_time(10)
        assert est.total_calls == 30  # 10 * 3 scenarios
        assert est.estimated_seconds > 0

    def test_single_endpoint(self, generator):
        est = generator.estimate_time(1)
        assert est.total_calls == 3

    def test_zero_endpoints(self, generator):
        est = generator.estimate_time(0)
        assert est.total_calls == 0


# ============================================================
# 18. num_scenarios property
# ============================================================


class TestNumScenarios:
    def test_returns_three(self, generator):
        assert generator.num_scenarios == 3


# ============================================================
# 19. _precompute_object_instruction
# ============================================================


class TestPrecomputeObjectInstruction:
    def test_simple_object(self, generator):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            }
        }
        result = generator._precompute_object_instruction(schema)
        assert '"name"' in result
        assert '"age"' in result
        assert result.startswith("{")
        assert result.endswith("}")

    def test_empty_properties(self, generator):
        assert generator._precompute_object_instruction({"properties": {}}) == "{}"

    def test_no_properties(self, generator):
        assert generator._precompute_object_instruction({}) == "{}"

    def test_circular_reference(self, generator):
        schema = {"properties": {"child": {"type": "object", "properties": {}}}}
        # Use the same schema object as a nested property to simulate circularity
        schema["properties"]["child"] = schema
        result = generator._precompute_object_instruction(schema)
        # Should not infinite loop, should return something with {}
        assert "{" in result


# ============================================================
# 20. _get_fallback_status_codes
# ============================================================


class TestGetFallbackStatusCodes:
    @pytest.mark.parametrize(
        "scenario_type,expected_codes",
        [
            (ScenarioType.POSITIVE, [200]),
            (ScenarioType.NEGATIVE, [400, 422]),
            (ScenarioType.SECURITY, [200, 400, 422]),
        ],
    )
    def test_fallback_codes(self, generator, scenario_type, expected_codes):
        assert generator._get_fallback_status_codes(scenario_type) == expected_codes
