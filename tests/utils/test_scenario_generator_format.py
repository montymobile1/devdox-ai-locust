"""
Tests for ScenarioWorkflowGenerator formatting and helper methods.
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
    return ScenarioWorkflowGenerator(
        prompt_dir=prompt_dir,
        ai_client=ai_client,
        ai_config=ai_config,
    )


# ---------------------------------------------------------------------------
# 1. _format_single_parameter
# ---------------------------------------------------------------------------


class TestFormatSingleParameter:
    def test_query_param_optional(self, generator):
        param = Parameter(
            name="limit",
            location=ParameterType.QUERY,
            required=False,
            type="integer",
        )
        lines, is_cookie, is_header = generator._format_single_parameter(param)
        assert any("limit" in ln and "(optional)" in ln for ln in lines)
        assert not is_cookie
        assert not is_header

    def test_path_param_required(self, generator):
        param = Parameter(
            name="id",
            location=ParameterType.PATH,
            required=True,
            type="integer",
        )
        lines, is_cookie, is_header = generator._format_single_parameter(param)
        assert any("(required)" in ln for ln in lines)
        assert not is_cookie
        assert not is_header

    def test_param_with_enum(self, generator):
        param = Parameter(
            name="status",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            enum=["active", "inactive"],
        )
        lines, _, _ = generator._format_single_parameter(param)
        assert any("allowed values" in ln for ln in lines)

    def test_param_with_pattern(self, generator):
        param = Parameter(
            name="email",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            pattern="^[a-z]+@",
        )
        lines, _, _ = generator._format_single_parameter(param)
        assert any("pattern" in ln for ln in lines)

    def test_param_with_min_max(self, generator):
        param = Parameter(
            name="age",
            location=ParameterType.QUERY,
            required=False,
            type="integer",
            minimum=0,
            maximum=150,
        )
        lines, _, _ = generator._format_single_parameter(param)
        assert any("min=0" in ln for ln in lines)
        assert any("max=150" in ln for ln in lines)

    def test_param_with_format(self, generator):
        param = Parameter(
            name="created",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            format="date-time",
        )
        lines, _, _ = generator._format_single_parameter(param)
        assert any("[date-time]" in ln for ln in lines)

    def test_cookie_param(self, generator):
        param = Parameter(
            name="session",
            location=ParameterType.COOKIE,
            required=False,
            type="string",
        )
        _, is_cookie, is_header = generator._format_single_parameter(param)
        assert is_cookie
        assert not is_header

    def test_header_param(self, generator):
        param = Parameter(
            name="X-Token",
            location=ParameterType.HEADER,
            required=False,
            type="string",
        )
        _, is_cookie, is_header = generator._format_single_parameter(param)
        assert not is_cookie
        assert is_header


# ---------------------------------------------------------------------------
# 2. _format_param_constraints
# ---------------------------------------------------------------------------


class TestFormatParamConstraints:
    def test_no_constraints(self, generator):
        param = Parameter(
            name="q", location=ParameterType.QUERY, required=False, type="string"
        )
        result = ScenarioWorkflowGenerator._format_param_constraints(param)
        # only description if present
        assert not any("allowed values" in ln for ln in result)

    def test_enum_constraint(self, generator):
        param = Parameter(
            name="s",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            enum=["a", "b"],
        )
        result = ScenarioWorkflowGenerator._format_param_constraints(param)
        assert any("allowed values" in ln for ln in result)

    def test_min_max_length(self, generator):
        param = Parameter(
            name="name",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            min_length=1,
            max_length=50,
        )
        result = ScenarioWorkflowGenerator._format_param_constraints(param)
        assert any("minLength=1" in ln for ln in result)
        assert any("maxLength=50" in ln for ln in result)

    def test_description_truncated(self, generator):
        param = Parameter(
            name="x",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            description="A" * 100,
        )
        result = ScenarioWorkflowGenerator._format_param_constraints(param)
        desc_line = [ln for ln in result if "description:" in ln][0]
        # description truncated to 80 chars
        assert len(desc_line.split("description: ")[1]) == 80


# ---------------------------------------------------------------------------
# 3. _format_endpoint_request_body
# ---------------------------------------------------------------------------


class TestFormatEndpointRequestBody:
    def test_endpoint_with_body(self, generator, sample_endpoints):
        create_user = sample_endpoints[1]  # POST /users
        lines = generator._format_endpoint_request_body(create_user)
        assert any("Request Body:" in ln for ln in lines)
        assert any("application/json" in ln for ln in lines)

    def test_endpoint_without_body(self, generator, sample_endpoints):
        get_users = sample_endpoints[0]  # GET /users
        lines = generator._format_endpoint_request_body(get_users)
        assert lines == []


# ---------------------------------------------------------------------------
# 4. _format_endpoint_responses
# ---------------------------------------------------------------------------


class TestFormatEndpointResponses:
    def test_with_responses(self, generator, sample_endpoints):
        ep = sample_endpoints[2]  # GET /users/{id} has 200 and 404
        lines = generator._format_endpoint_responses(ep)
        assert any("Responses:" in ln for ln in lines)
        assert any("200" in ln for ln in lines)
        assert any("404" in ln for ln in lines)

    def test_exclude_2xx(self, generator, sample_endpoints):
        ep = sample_endpoints[2]
        lines = generator._format_endpoint_responses(ep, exclude_2xx=True)
        assert not any("200" in ln for ln in lines)
        assert any("404" in ln for ln in lines)

    def test_no_responses(self, generator):
        ep = Endpoint(
            path="/empty",
            method="GET",
            operation_id="empty",
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        assert generator._format_endpoint_responses(ep) == []


# ---------------------------------------------------------------------------
# 5. _format_single_response
# ---------------------------------------------------------------------------


class TestFormatSingleResponse:
    def test_basic_response(self, generator):
        resp = Response(status_code="200", description="OK")
        lines = generator._format_single_response("200", resp)
        assert any("200" in ln and "OK" in ln for ln in lines)

    def test_response_with_schema(self, generator):
        resp = Response(
            status_code="200",
            description="OK",
            content_type="application/json",
            schema={"type": "object", "properties": {"id": {"type": "integer"}}},
        )
        lines = generator._format_single_response("200", resp)
        assert any("Response Schema" in ln for ln in lines)


# ---------------------------------------------------------------------------
# 6. _is_2xx_status
# ---------------------------------------------------------------------------


class TestIs2xxStatus:
    @pytest.mark.parametrize(
        "code,expected",
        [
            (200, True),
            (201, True),
            (299, True),
            (404, False),
            (500, False),
            ("200", True),
            ("201", True),
            ("404", False),
            ("default", False),
            (None, False),
        ],
    )
    def test_is_2xx(self, generator, code, expected):
        assert generator._is_2xx_status(code) is expected


# ---------------------------------------------------------------------------
# 7. _format_single_endpoint
# ---------------------------------------------------------------------------


class TestFormatSingleEndpoint:
    def test_full_format(self, generator, sample_endpoints):
        ep = sample_endpoints[0]  # GET /users
        result = generator._format_single_endpoint(ep)
        assert "GET /users" in result
        assert "getUsers" in result
        assert "Get all users" in result

    def test_exclude_2xx(self, generator, sample_endpoints):
        ep = sample_endpoints[2]  # GET /users/{id}
        result = generator._format_single_endpoint(ep, exclude_2xx=True)
        assert "404" in result
        # 200 response should be excluded
        assert "User found" not in result


# ---------------------------------------------------------------------------
# 8. _format_endpoints_for_orchestrator
# ---------------------------------------------------------------------------


class TestFormatEndpointsForOrchestrator:
    def test_formats_multiple(self, generator, sample_endpoints):
        result = generator._format_endpoints_for_orchestrator(sample_endpoints[:2])
        assert "GET endpoints:" in result
        assert "POST endpoints:" in result
        assert "/users" in result

    def test_empty_list(self, generator):
        result = generator._format_endpoints_for_orchestrator([])
        assert result == ""


# ---------------------------------------------------------------------------
# 9. _format_orchestrator_endpoint
# ---------------------------------------------------------------------------


class TestFormatOrchestratorEndpoint:
    def test_basic(self, generator, sample_endpoints):
        ep = sample_endpoints[0]
        lines = generator._format_orchestrator_endpoint(ep, "GET")
        assert any("/users" in ln for ln in lines)
        assert any("Operation ID" in ln for ln in lines)
        assert any("Summary" in ln for ln in lines)

    def test_post_includes_body(self, generator, sample_endpoints):
        ep = sample_endpoints[1]  # POST /users with body
        lines = generator._format_orchestrator_endpoint(ep, "POST")
        assert any("Request Body Schema" in ln for ln in lines)


# ---------------------------------------------------------------------------
# 10. _format_orchestrator_request_body
# ---------------------------------------------------------------------------


class TestFormatOrchestratorRequestBody:
    def test_post_with_body(self, generator, sample_endpoints):
        ep = sample_endpoints[1]
        lines = generator._format_orchestrator_request_body(ep, "POST")
        assert any("Request Body Schema" in ln for ln in lines)

    def test_get_no_body(self, generator, sample_endpoints):
        ep = sample_endpoints[0]
        lines = generator._format_orchestrator_request_body(ep, "GET")
        assert lines == []

    def test_post_without_body(self, generator):
        ep = Endpoint(
            path="/ping",
            method="POST",
            operation_id="ping",
            summary="Ping",
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        lines = generator._format_orchestrator_request_body(ep, "POST")
        assert lines == []


# ---------------------------------------------------------------------------
# 11. _format_orchestrator_response_schema
# ---------------------------------------------------------------------------


class TestFormatOrchestratorResponseSchema:
    def test_with_2xx_schema(self, generator, sample_endpoints):
        ep = sample_endpoints[0]  # has 200 with schema
        lines = generator._format_orchestrator_response_schema(ep)
        assert any("Response" in ln for ln in lines)

    def test_no_responses(self, generator):
        ep = Endpoint(
            path="/x",
            method="GET",
            operation_id="x",
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        assert generator._format_orchestrator_response_schema(ep) == []


# ---------------------------------------------------------------------------
# 12. _group_endpoints_by_method
# ---------------------------------------------------------------------------


class TestGroupEndpointsByMethod:
    def test_groups_correctly(self, generator, sample_endpoints):
        result = generator._group_endpoints_by_method(sample_endpoints)
        assert len(result["GET"]) == 2
        assert len(result["POST"]) == 2
        assert len(result["PUT"]) == 0
        assert len(result["DELETE"]) == 0

    def test_empty(self, generator):
        result = generator._group_endpoints_by_method([])
        assert all(v == [] for v in result.values())


# ---------------------------------------------------------------------------
# 13. _generate_nonexistent_id_scenarios
# ---------------------------------------------------------------------------


class TestGenerateNonexistentIdScenarios:
    def test_integer_param(self, generator):
        result = generator._generate_nonexistent_id_scenarios([("id", "integer")])
        assert len(result) == 1
        assert "999999999" in result[0]

    def test_string_param(self, generator):
        result = generator._generate_nonexistent_id_scenarios([("slug", "string")])
        assert len(result) == 1
        assert "nonexistent-id-12345" in result[0]

    def test_multiple_params(self, generator):
        result = generator._generate_nonexistent_id_scenarios(
            [("id", "integer"), ("name", "string")]
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 14. _generate_field_type_scenarios
# ---------------------------------------------------------------------------


class TestGenerateFieldTypeScenarios:
    def test_all_categories(self, generator):
        result = generator._generate_field_type_scenarios(
            required_fields=["name"],
            typed_fields=[("age", "integer")],
            enum_fields=[("status", ["active", "inactive"])],
            pattern_fields=[("email", "^.+@.+$")],
            numeric_fields=[("age", 0, 150)],
        )
        assert any("MISSING_REQUIRED" in s for s in result)
        assert any("WRONG_TYPE" in s for s in result)
        assert any("INVALID_ENUM" in s for s in result)
        assert any("INVALID_PATTERN" in s for s in result)
        assert any("BOUNDARY" in s for s in result)

    def test_empty_fields(self, generator):
        result = generator._generate_field_type_scenarios([], [], [], [], [])
        assert result == []


# ---------------------------------------------------------------------------
# 15. _build_wrong_type_scenario
# ---------------------------------------------------------------------------


class TestBuildWrongTypeScenario:
    def test_integer_field(self):
        result = ScenarioWorkflowGenerator._build_wrong_type_scenario(
            [("age", "integer")]
        )
        assert "WRONG_TYPE" in result
        assert "not_a_number" in result

    def test_boolean_field(self):
        result = ScenarioWorkflowGenerator._build_wrong_type_scenario(
            [("active", "boolean")]
        )
        assert "not_a_bool" in result

    def test_array_field(self):
        result = ScenarioWorkflowGenerator._build_wrong_type_scenario(
            [("items", "array")]
        )
        assert "not_an_array" in result


# ---------------------------------------------------------------------------
# 16. _build_enum_scenarios
# ---------------------------------------------------------------------------


class TestBuildEnumScenarios:
    def test_single_enum(self):
        result = ScenarioWorkflowGenerator._build_enum_scenarios(
            [("status", ["active", "inactive"])]
        )
        assert len(result) == 1
        assert "INVALID_ENUM" in result[0]
        assert "INVALID_VALUE_XYZ" in result[0]

    def test_multiple_enums(self):
        result = ScenarioWorkflowGenerator._build_enum_scenarios(
            [("status", ["a", "b"]), ("role", ["admin", "user"])]
        )
        assert len(result) == 2

    def test_empty(self):
        assert ScenarioWorkflowGenerator._build_enum_scenarios([]) == []


# ---------------------------------------------------------------------------
# 17. _build_pattern_scenarios
# ---------------------------------------------------------------------------


class TestBuildPatternScenarios:
    def test_single_pattern(self):
        result = ScenarioWorkflowGenerator._build_pattern_scenarios(
            [("email", "^.+@.+$")]
        )
        assert len(result) == 1
        assert "INVALID_PATTERN" in result[0]
        assert "!!!invalid!!!" in result[0]

    def test_empty(self):
        assert ScenarioWorkflowGenerator._build_pattern_scenarios([]) == []


# ---------------------------------------------------------------------------
# 18. _build_boundary_scenarios
# ---------------------------------------------------------------------------


class TestBuildBoundaryScenarios:
    def test_min_and_max(self):
        result = ScenarioWorkflowGenerator._build_boundary_scenarios([("age", 0, 150)])
        assert len(result) == 2
        assert any("min=0" in s and "-1" in s for s in result)
        assert any("max=150" in s and "151" in s for s in result)

    def test_only_min(self):
        result = ScenarioWorkflowGenerator._build_boundary_scenarios(
            [("count", 1, None)]
        )
        assert len(result) == 1
        assert "min=1" in result[0]

    def test_only_max(self):
        result = ScenarioWorkflowGenerator._build_boundary_scenarios(
            [("count", None, 100)]
        )
        assert len(result) == 1
        assert "max=100" in result[0]

    def test_no_constraints(self):
        result = ScenarioWorkflowGenerator._build_boundary_scenarios(
            [("count", None, None)]
        )
        assert result == []


# ---------------------------------------------------------------------------
# 19. _generate_fallback_query_scenarios
# ---------------------------------------------------------------------------


class TestGenerateFallbackQueryScenarios:
    def test_integer_query_param(self, generator):
        result = generator._generate_fallback_query_scenarios([("page", "integer")])
        assert len(result) == 1
        assert "INVALID_QUERY" in result[0]
        assert "not_a_number" in result[0]

    def test_string_query_param_ignored(self, generator):
        result = generator._generate_fallback_query_scenarios([("name", "string")])
        assert result == []

    def test_number_query_param(self, generator):
        result = generator._generate_fallback_query_scenarios([("price", "number")])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 20. _filter_status_codes_for_scenario
# ---------------------------------------------------------------------------


class TestFilterStatusCodesForScenario:
    def test_positive_filters_2xx(self, generator):
        codes = [200, 201, 400, 404, 500]
        result = generator._filter_status_codes_for_scenario(
            codes, ScenarioType.POSITIVE
        )
        assert all(c < 400 for c in result)
        assert 200 in result
        assert 201 in result

    def test_negative_filters_4xx(self, generator):
        codes = [200, 201, 400, 404, 500]
        result = generator._filter_status_codes_for_scenario(
            codes, ScenarioType.NEGATIVE
        )
        assert all(400 <= c < 500 for c in result)
        assert 400 in result
        assert 404 in result

    def test_security_filters_non_5xx(self, generator):
        codes = [200, 401, 403, 500]
        result = generator._filter_status_codes_for_scenario(
            codes, ScenarioType.SECURITY
        )
        assert 500 not in result
        assert 200 in result
        assert 401 in result

    def test_empty_codes_returns_fallback(self, generator):
        result = generator._filter_status_codes_for_scenario(
            [], ScenarioType.POSITIVE, method="GET"
        )
        # Should return fallback codes, all < 400 for positive
        assert all(c < 400 for c in result)


# ---------------------------------------------------------------------------
# 21. _extract_status_codes_with_descriptions
# ---------------------------------------------------------------------------


class TestExtractStatusCodesWithDescriptions:
    def test_from_response_list(self, generator, sample_endpoints):
        ep = sample_endpoints[2]  # GET /users/{id} has 200 and 404
        result = generator._extract_status_codes_with_descriptions(ep)
        codes = [c for c, _ in result]
        assert 200 in codes
        assert 404 in codes

    def test_descriptions_present(self, generator, sample_endpoints):
        ep = sample_endpoints[2]
        result = generator._extract_status_codes_with_descriptions(ep)
        desc_map = {c: d for c, d in result}
        assert "User found" in desc_map[200]
        assert "User not found" in desc_map[404]

    def test_no_responses(self, generator):
        ep = Endpoint(
            path="/x",
            method="GET",
            operation_id="x",
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        assert generator._extract_status_codes_with_descriptions(ep) == []


# ---------------------------------------------------------------------------
# 22. _get_body_properties
# ---------------------------------------------------------------------------


class TestGetBodyProperties:
    def test_with_body(self, generator, sample_endpoints):
        ep = sample_endpoints[1]  # POST /users
        props, required = generator._get_body_properties(ep)
        assert props is not None
        assert "username" in props
        assert "username" in required

    def test_without_body(self, generator, sample_endpoints):
        ep = sample_endpoints[0]  # GET /users
        props, required = generator._get_body_properties(ep)
        assert props is None
        assert required == []


# ---------------------------------------------------------------------------
# 23. _categorize_field
# ---------------------------------------------------------------------------


class TestCategorizeField:
    def _run(self, generator, field_name, field_schema, required_list=None):
        required_fields = []
        typed_fields = []
        enum_fields = []
        pattern_fields = []
        numeric_fields = []
        generator._categorize_field(
            field_name,
            field_schema,
            required_list or [],
            required_fields,
            typed_fields,
            enum_fields,
            pattern_fields,
            numeric_fields,
        )
        return (
            required_fields,
            typed_fields,
            enum_fields,
            pattern_fields,
            numeric_fields,
        )

    def test_enum_field(self, generator):
        _, _, enums, _, _ = self._run(
            generator, "status", {"type": "string", "enum": ["a", "b"]}
        )
        assert len(enums) == 1

    def test_pattern_field(self, generator):
        _, _, _, patterns, _ = self._run(
            generator, "email", {"type": "string", "pattern": "^.+@.+$"}
        )
        assert len(patterns) == 1

    def test_typed_integer(self, generator):
        _, typed, _, _, _ = self._run(generator, "age", {"type": "integer"})
        assert ("age", "integer") in typed

    def test_numeric_with_constraints(self, generator):
        _, _, _, _, numerics = self._run(
            generator, "age", {"type": "integer", "minimum": 0, "maximum": 150}
        )
        assert len(numerics) == 1
        assert numerics[0] == ("age", 0, 150)

    def test_required_field(self, generator):
        req, _, _, _, _ = self._run(
            generator, "name", {"type": "string"}, required_list=["name"]
        )
        assert "name" in req


# ---------------------------------------------------------------------------
# 24. _format_setup_endpoint_schema
# ---------------------------------------------------------------------------


class TestFormatSetupEndpointSchema:
    def test_with_body(self, generator, sample_endpoints):
        ep = sample_endpoints[1]  # POST /users
        lines = generator._format_setup_endpoint_schema(ep)
        assert any("Request Body Schema" in ln for ln in lines)

    def test_without_body(self, generator, sample_endpoints):
        ep = sample_endpoints[0]  # GET /users
        assert generator._format_setup_endpoint_schema(ep) == []


# ---------------------------------------------------------------------------
# 25. _format_setup_call_pattern
# ---------------------------------------------------------------------------


class TestFormatSetupCallPattern:
    def test_output(self, generator, sample_endpoints):
        ep = sample_endpoints[1]  # POST /users
        lines = generator._format_setup_call_pattern(ep)
        assert any("SETUP CALL PATTERN" in ln for ln in lines)
        assert any("make_request" in ln for ln in lines)
        assert any("/users" in ln for ln in lines)
