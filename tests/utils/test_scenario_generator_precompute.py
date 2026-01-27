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
from devdox_ai_locust.utils.generation_progress import (
    SchemaAnalysis,
    InjectionAnalysis,
    EndpointAnalysis,
    SetupAnalysis,
    OrchestratorAnalysis,
    OrchestratorEndpointInfo,
)


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


# ============================================================
# 21. _analyze_schema_for_verbose
# ============================================================


class TestAnalyzeSchemaForVerbose:
    def test_no_request_body(self, generator, endpoint_no_body):
        result = generator._analyze_schema_for_verbose(endpoint_no_body, SchemaAnalysis)
        assert isinstance(result, SchemaAnalysis)
        assert result.total_fields == 0
        assert result.required_fields == 0

    def test_simple_properties(self, generator, endpoint_with_body):
        result = generator._analyze_schema_for_verbose(
            endpoint_with_body, SchemaAnalysis
        )
        assert result.total_fields == 6
        assert result.required_fields == 2

    def test_oneof_with_discriminator(self, generator):
        ep = Endpoint(
            path="/poly",
            method="POST",
            operation_id="polyCreate",
            summary="",
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/json",
                schema={
                    "oneOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                        {"type": "object", "properties": {"b": {"type": "string"}}},
                    ],
                    "discriminator": {
                        "propertyName": "kind",
                        "mapping": {"typeA": "#/a", "typeB": "#/b"},
                    },
                },
                required=True,
            ),
            responses=[],
            tags=[],
        )
        result = generator._analyze_schema_for_verbose(ep, SchemaAnalysis)
        assert result.schema_type == "discriminated_union"
        assert result.discriminator == "kind"
        assert "typeA" in result.variants
        assert "typeB" in result.variants

    def test_constraints_counted(self, generator):
        ep = Endpoint(
            path="/constrained",
            method="POST",
            operation_id="constrained",
            summary="",
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/json",
                schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "pattern": "^[A-Z]+$"},
                        "status": {"type": "string", "enum": ["a", "b"]},
                        "email": {"type": "string", "format": "email"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 5,
                        },
                    },
                },
                required=True,
            ),
            responses=[],
            tags=[],
        )
        result = generator._analyze_schema_for_verbose(ep, SchemaAnalysis)
        assert result.patterns_found >= 1
        assert result.enums_found >= 1
        assert result.formats_found >= 1
        assert result.arrays_with_constraints >= 1

    def test_empty_schema_dict(self, generator):
        ep = Endpoint(
            path="/empty",
            method="POST",
            operation_id="empty",
            summary="",
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/json",
                schema={},
                required=True,
            ),
            responses=[],
            tags=[],
        )
        result = generator._analyze_schema_for_verbose(ep, SchemaAnalysis)
        assert result.total_fields == 0


# ============================================================
# 22. _analyze_injection_for_verbose
# ============================================================


class TestAnalyzeInjectionForVerbose:
    def test_no_body_no_params(self, generator):
        ep = Endpoint(
            path="/health",
            method="GET",
            operation_id="health",
            summary="",
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._analyze_injection_for_verbose(ep, InjectionAnalysis)
        assert isinstance(result, InjectionAnalysis)
        assert result.injection_locations == []

    def test_body_string_fields(self, generator, endpoint_with_body):
        result = generator._analyze_injection_for_verbose(
            endpoint_with_body, InjectionAnalysis
        )
        assert "body" in result.injection_locations

    def test_query_string_params(self, generator, endpoint_with_query_string_params):
        result = generator._analyze_injection_for_verbose(
            endpoint_with_query_string_params, InjectionAnalysis
        )
        assert "query" in result.injection_locations

    def test_body_and_query(self, generator):
        ep = Endpoint(
            path="/mixed",
            method="POST",
            operation_id="mixed",
            summary="",
            description=None,
            parameters=[
                Parameter(
                    name="q",
                    location=ParameterType.QUERY,
                    required=False,
                    type="string",
                    description="query",
                ),
            ],
            request_body=RequestBody(
                content_type="application/json",
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                required=True,
            ),
            responses=[],
            tags=[],
        )
        result = generator._analyze_injection_for_verbose(ep, InjectionAnalysis)
        assert "body" in result.injection_locations
        assert "query" in result.injection_locations


# ============================================================
# 23. _parse_negative_scenario_types
# ============================================================


class TestParseNegativeScenarioTypes:
    def test_empty_string(self, generator):
        assert generator._parse_negative_scenario_types("") == []

    def test_multiline_entries(self, generator):
        text = "- WRONG_TYPE: send string for int\n- MISSING_REQUIRED: omit name\n- BOUNDARY: exceed max"
        result = generator._parse_negative_scenario_types(text)
        assert "WRONG_TYPE" in result
        assert "MISSING_REQUIRED" in result
        assert "BOUNDARY" in result

    def test_lines_without_dash_ignored(self, generator):
        text = "Some header\n- VALID_ENTRY: something\nAnother line"
        result = generator._parse_negative_scenario_types(text)
        assert len(result) == 1
        assert "VALID_ENTRY" in result

    def test_no_colon(self, generator):
        text = "- JUST_NAME"
        result = generator._parse_negative_scenario_types(text)
        assert result == ["JUST_NAME"]


# ============================================================
# 24. _build_endpoint_analysis
# ============================================================


class TestBuildEndpointAnalysis:
    def test_returns_endpoint_analysis(
        self, generator, endpoint_with_body, all_endpoints
    ):
        result = generator._build_endpoint_analysis(endpoint_with_body, all_endpoints)
        assert isinstance(result, EndpointAnalysis)
        assert result.method == "POST"
        assert result.path == "/items"
        assert result.operation_id == "createItem"

    def test_without_all_endpoints(self, generator, endpoint_no_body):
        result = generator._build_endpoint_analysis(endpoint_no_body, None)
        assert isinstance(result, EndpointAnalysis)
        assert result.method == "GET"

    def test_responses_defined(self, generator, endpoint_with_body):
        result = generator._build_endpoint_analysis(endpoint_with_body)
        assert 201 in result.responses_defined
        assert result.source_of_truth == "spec"

    def test_no_responses_uses_fallback(self, generator):
        ep = Endpoint(
            path="/x",
            method="GET",
            operation_id="x",
            summary="",
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._build_endpoint_analysis(ep)
        assert result.source_of_truth == "fallback"


# ============================================================
# 25. _build_setup_analysis
# ============================================================


class TestBuildSetupAnalysis:
    def test_no_all_endpoints(self, generator, endpoint_no_body):
        result = generator._build_setup_analysis(endpoint_no_body, None)
        assert isinstance(result, SetupAnalysis)
        assert result.needs_setup is False
        assert result.setup_endpoints_found == 0

    def test_with_related_post(self, generator, endpoint_no_body, all_endpoints):
        result = generator._build_setup_analysis(endpoint_no_body, all_endpoints)
        assert isinstance(result, SetupAnalysis)
        assert result.setup_endpoints_found >= 1
        assert result.needs_setup is True

    def test_post_endpoint_no_setup(self, generator, endpoint_with_body, all_endpoints):
        result = generator._build_setup_analysis(endpoint_with_body, all_endpoints)
        assert result.setup_endpoints_found == 0


# ============================================================
# 26. _detect_crud_operations
# ============================================================


class TestDetectCrudOperations:
    def test_mixed_crud(self, generator):
        endpoints = [
            Mock(method="POST"),
            Mock(method="GET"),
            Mock(method="PUT"),
            Mock(method="DELETE"),
        ]
        c, r, u, d = generator._detect_crud_operations(endpoints)
        assert c is True
        assert r is True
        assert u is True
        assert d is True

    def test_only_get(self, generator):
        endpoints = [Mock(method="GET"), Mock(method="GET")]
        c, r, u, d = generator._detect_crud_operations(endpoints)
        assert c is False
        assert r is True
        assert u is False
        assert d is False

    def test_empty_list(self, generator):
        c, r, u, d = generator._detect_crud_operations([])
        assert c is False
        assert r is False
        assert u is False
        assert d is False


# ============================================================
# 27. _detect_state_dependent_tests
# ============================================================


class TestDetectStateDependentTests:
    def test_full_crud(self, generator):
        result = generator._detect_state_dependent_tests(True, True, True, True)
        assert "double_delete" in result
        assert "read_after_delete" in result
        assert "update_after_delete" in result
        assert "409_conflict" in result

    def test_only_create(self, generator):
        result = generator._detect_state_dependent_tests(True, False, False, False)
        assert result == ["409_conflict"]

    def test_no_create(self, generator):
        result = generator._detect_state_dependent_tests(False, True, True, True)
        assert result == []


# ============================================================
# 28. _build_orchestrator_warnings
# ============================================================


class TestBuildOrchestratorWarnings:
    def test_no_create(self, generator):
        warnings = generator._build_orchestrator_warnings(False, True, 3)
        assert any("POST" in w for w in warnings)

    def test_not_crud_lifecycle(self, generator):
        warnings = generator._build_orchestrator_warnings(True, False, 3)
        assert any("CRUD" in w for w in warnings)

    def test_single_endpoint(self, generator):
        warnings = generator._build_orchestrator_warnings(True, True, 1)
        assert any("one endpoint" in w.lower() or "Only one" in w for w in warnings)

    def test_no_warnings(self, generator):
        warnings = generator._build_orchestrator_warnings(True, True, 3)
        assert warnings == []


# ============================================================
# 29. _build_orchestrator_analysis
# ============================================================


class TestBuildOrchestratorAnalysis:
    def test_basic(self, generator, sample_endpoints):
        users = [ep for ep in sample_endpoints if "users" in (ep.tags or [])]
        result = generator._build_orchestrator_analysis(
            "users", "UsersOrchestrator", users
        )
        assert isinstance(result, OrchestratorAnalysis)
        assert result.tag_name == "users"
        assert result.class_name == "UsersOrchestrator"
        assert result.total_endpoints == len(users)
        assert result.has_create is True
        assert result.has_read is True

    def test_with_auth_endpoints(self, generator, sample_endpoints):
        users = [ep for ep in sample_endpoints if "users" in (ep.tags or [])]
        auth = [ep for ep in sample_endpoints if "auth" in (ep.tags or [])]
        result = generator._build_orchestrator_analysis("users", "Users", users, auth)
        assert result.auth_endpoints_found == len(auth)
        assert result.auth_tests_possible is True

    def test_no_auth(self, generator, sample_endpoints):
        users = [ep for ep in sample_endpoints if "users" in (ep.tags or [])]
        result = generator._build_orchestrator_analysis("users", "Users", users, None)
        assert result.auth_endpoints_found == 0
        assert result.auth_tests_possible is False


# ============================================================
# 30. _build_orchestrator_endpoint_info
# ============================================================


class TestBuildOrchestratorEndpointInfo:
    def test_basic(self, generator, sample_endpoints):
        ep = sample_endpoints[0]
        result = generator._build_orchestrator_endpoint_info(
            ep, OrchestratorEndpointInfo
        )
        assert isinstance(result, OrchestratorEndpointInfo)
        assert result.method == "GET"
        assert result.path == "/users"
        assert result.operation_id == "getUsers"
        assert result.has_positive is True
        assert result.has_negative is True
        assert result.has_security is True

    def test_no_operation_id(self, generator, endpoint_no_operation_id):
        result = generator._build_orchestrator_endpoint_info(
            endpoint_no_operation_id, OrchestratorEndpointInfo
        )
        assert result.method == "DELETE"
        assert result.operation_id != ""


# ============================================================
# 31. _process_llm_results
# ============================================================


class TestProcessLlmResults:
    def test_all_successes(self, generator):
        generator.progress = None
        types = [ScenarioType.POSITIVE, ScenarioType.NEGATIVE, ScenarioType.SECURITY]
        llm_results = ["code_pos", "code_neg", "code_sec"]
        results, errors = generator._process_llm_results(types, llm_results, "POST /x")
        assert len(results) == 3
        assert len(errors) == 0
        assert results[ScenarioType.POSITIVE] == "code_pos"

    def test_mixed_results(self, generator):
        generator.progress = None
        types = [ScenarioType.POSITIVE, ScenarioType.NEGATIVE, ScenarioType.SECURITY]
        llm_results = ["code_pos", ValueError("fail"), "code_sec"]
        results, errors = generator._process_llm_results(types, llm_results, "POST /x")
        assert len(results) == 2
        assert len(errors) == 1
        assert errors[0][0] == ScenarioType.NEGATIVE

    def test_all_exceptions(self, generator):
        generator.progress = None
        types = [ScenarioType.POSITIVE, ScenarioType.NEGATIVE]
        llm_results = [RuntimeError("a"), RuntimeError("b")]
        results, errors = generator._process_llm_results(types, llm_results, "POST /x")
        assert len(results) == 0
        assert len(errors) == 2

    def test_none_results_skipped(self, generator):
        generator.progress = None
        types = [ScenarioType.POSITIVE]
        llm_results = [None]
        results, errors = generator._process_llm_results(types, llm_results, "POST /x")
        assert len(results) == 0
        assert len(errors) == 0


# ============================================================
# 32. _record_scenario_verbose_result
# ============================================================


class TestRecordScenarioVerboseResult:
    def test_with_mock_progress(self, generator):
        progress = Mock()
        progress.verbose = True
        generator.progress = progress
        generator._record_scenario_verbose_result(
            "POST /items", ScenarioType.POSITIVE, "success"
        )
        progress.record_scenario_result.assert_called_once()
        call_args = progress.record_scenario_result.call_args
        assert call_args[0][0] == "POST /items"
        assert call_args[0][1] == "positive"

    def test_no_progress(self, generator):
        generator.progress = None
        # Should not raise
        generator._record_scenario_verbose_result(
            "POST /items", ScenarioType.POSITIVE, "success"
        )

    def test_not_verbose(self, generator):
        progress = Mock()
        progress.verbose = False
        generator.progress = progress
        generator._record_scenario_verbose_result(
            "POST /items", ScenarioType.POSITIVE, "success"
        )
        progress.record_scenario_result.assert_not_called()

    def test_with_skip_reason(self, generator):
        progress = Mock()
        progress.verbose = True
        generator.progress = progress
        generator._record_scenario_verbose_result(
            "GET /x", ScenarioType.NEGATIVE, "skipped", skip_reason="no body"
        )
        call_args = progress.record_scenario_result.call_args
        result_obj = call_args[0][2]
        assert result_obj.skip_reason == "no body"


# ============================================================
# 33. generate_all_endpoints (async)
# ============================================================


class TestGenerateAllEndpoints:
    @pytest.mark.asyncio
    async def test_basic(self, generator, sample_endpoints):
        eps = sample_endpoints[:2]

        async def mock_gen(
            endpoint,
            base_workflow_content,
            test_data_content,
            auth_endpoints,
            all_endpoints,
        ):
            op = endpoint.operation_id or "unknown"
            return {ScenarioType.POSITIVE: f"code_{op}"}

        generator.generate_endpoint_workflows = mock_gen
        results = await generator.generate_all_endpoints(
            endpoints=eps,
            base_workflow_content="base",
            test_data_content="data",
        )
        assert len(results) == 2
        assert "getUsers" in results
        assert "createUser" in results

    @pytest.mark.asyncio
    async def test_with_progress_callback(self, generator, sample_endpoints):
        eps = sample_endpoints[:1]
        callback_calls = []

        async def mock_gen(
            endpoint,
            base_workflow_content,
            test_data_content,
            auth_endpoints,
            all_endpoints,
        ):
            return {ScenarioType.POSITIVE: "code"}

        async def callback(endpoint, scenarios):
            callback_calls.append((endpoint, scenarios))

        generator.generate_endpoint_workflows = mock_gen
        await generator.generate_all_endpoints(
            endpoints=eps,
            base_workflow_content="base",
            test_data_content="data",
            progress_callback=callback,
        )
        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_empty_endpoints(self, generator):
        async def mock_gen(**kwargs):
            return {}

        generator.generate_endpoint_workflows = mock_gen
        results = await generator.generate_all_endpoints(
            endpoints=[],
            base_workflow_content="base",
            test_data_content="data",
        )
        assert results == {}
