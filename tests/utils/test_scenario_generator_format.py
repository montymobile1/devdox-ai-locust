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


# ---------------------------------------------------------------------------
# 26. _format_discriminated_union
# ---------------------------------------------------------------------------


class TestFormatDiscriminatedUnion:
    def test_with_discriminator_mapping(self, generator):
        one_of = [
            {
                "$ref": "#/components/schemas/Cat",
                "properties": {
                    "pet_type": {"type": "string", "const": "cat"},
                    "color": {"type": "string"},
                },
                "required": ["pet_type", "color"],
            },
            {
                "$ref": "#/components/schemas/Dog",
                "properties": {
                    "pet_type": {"type": "string", "const": "dog"},
                    "breed": {"type": "string"},
                },
                "required": ["pet_type", "breed"],
            },
        ]
        discriminator = {
            "propertyName": "pet_type",
            "mapping": {
                "cat": "#/components/schemas/Cat",
                "dog": "#/components/schemas/Dog",
            },
        }
        lines = generator._format_discriminated_union(one_of, discriminator, "")
        text = "\n".join(lines)
        assert "DISCRIMINATED UNION" in text
        assert "pet_type" in text
        assert 'pet_type="cat"' in text
        assert 'pet_type="dog"' in text

    def test_empty_mapping(self, generator):
        one_of = [{"properties": {"a": {"type": "string"}}}]
        discriminator = {"propertyName": "type", "mapping": {}}
        lines = generator._format_discriminated_union(one_of, discriminator, "")
        text = "\n".join(lines)
        assert "DISCRIMINATED UNION" in text
        assert "Valid" not in text


# ---------------------------------------------------------------------------
# 27. _format_variant_properties
# ---------------------------------------------------------------------------


class TestFormatVariantProperties:
    def test_with_properties_and_required(self, generator):
        variant_schema = {
            "properties": {
                "type": {"type": "string"},
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        lines = generator._format_variant_properties(variant_schema, "type", "  ")
        text = "\n".join(lines)
        assert "name" in text
        assert "(REQUIRED)" in text
        assert "age" in text
        # discriminator prop should be skipped
        assert text.count("type") == 0 or "type:" not in text.split("\n")[0]

    def test_empty_properties(self, generator):
        variant_schema = {"properties": {}, "required": []}
        lines = generator._format_variant_properties(variant_schema, "type", "  ")
        assert lines == []


# ---------------------------------------------------------------------------
# 28. _format_union_without_discriminator
# ---------------------------------------------------------------------------


class TestFormatUnionWithoutDiscriminator:
    def test_with_ref_variants(self, generator):
        one_of = [
            {"$ref": "#/components/schemas/Cat"},
            {"$ref": "#/components/schemas/Dog"},
        ]
        lines = generator._format_union_without_discriminator(one_of, "")
        text = "\n".join(lines)
        assert "UNION TYPE" in text
        assert "Option 1" in text
        assert "#/components/schemas/Cat" in text
        assert "Option 2" in text

    def test_with_inline_properties(self, generator):
        one_of = [
            {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
            {"properties": {"title": {"type": "string"}}},
        ]
        lines = generator._format_union_without_discriminator(one_of, "")
        text = "\n".join(lines)
        assert "Option 1" in text
        assert "name" in text
        assert "Option 2" in text
        assert "title" in text


# ---------------------------------------------------------------------------
# 29. _format_schema
# ---------------------------------------------------------------------------


class TestFormatSchema:
    def test_simple_schema_with_properties(self, generator):
        schema = {
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["id"],
        }
        lines = generator._format_schema(schema)
        text = "\n".join(lines)
        assert "Schema:" in text
        assert "id" in text
        assert "REQUIRED" in text
        assert "name" in text

    def test_schema_with_discriminated_union(self, generator):
        schema = {
            "oneOf": [
                {
                    "$ref": "#/components/schemas/A",
                    "properties": {"t": {"type": "string"}},
                    "required": ["t"],
                },
            ],
            "discriminator": {
                "propertyName": "t",
                "mapping": {"a": "#/components/schemas/A"},
            },
        }
        lines = generator._format_schema(schema)
        text = "\n".join(lines)
        assert "DISCRIMINATED UNION" in text

    def test_schema_with_union_no_discriminator(self, generator):
        schema = {
            "oneOf": [
                {"$ref": "#/components/schemas/X"},
                {"$ref": "#/components/schemas/Y"},
            ]
        }
        lines = generator._format_schema(schema)
        text = "\n".join(lines)
        assert "UNION TYPE" in text

    def test_empty_schema(self, generator):
        lines = generator._format_schema({})
        assert lines == []


# ---------------------------------------------------------------------------
# 30. _format_nested_type
# ---------------------------------------------------------------------------


class TestFormatNestedType:
    def test_object_type_with_properties(self, generator):
        unwrapped = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        lines = generator._format_nested_type(unwrapped, "", 0)
        assert any("nested object" in ln for ln in lines)

    def test_array_type_with_items(self, generator):
        unwrapped = {
            "type": "array",
            "items": {"type": "string"},
        }
        lines = generator._format_nested_type(unwrapped, "", 0)
        assert any("array items type" in ln for ln in lines)

    def test_other_type(self, generator):
        unwrapped = {"type": "string"}
        lines = generator._format_nested_type(unwrapped, "", 0)
        assert lines == []


# ---------------------------------------------------------------------------
# 31. _format_nested_object
# ---------------------------------------------------------------------------


class TestFormatNestedObject:
    def test_with_properties(self, generator):
        unwrapped = {
            "type": "object",
            "properties": {"foo": {"type": "integer"}},
        }
        lines = generator._format_nested_object(unwrapped, "", 0)
        assert any("nested object" in ln for ln in lines)

    def test_with_additional_properties(self, generator):
        unwrapped = {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }
        lines = generator._format_nested_object(unwrapped, "", 0)
        assert any("map type" in ln for ln in lines)
        assert any("string" in ln for ln in lines)

    def test_empty_object(self, generator):
        unwrapped = {"type": "object"}
        lines = generator._format_nested_object(unwrapped, "", 0)
        assert lines == []


# ---------------------------------------------------------------------------
# 32. _format_array_items
# ---------------------------------------------------------------------------


class TestFormatArrayItems:
    def test_simple_items_type(self, generator):
        unwrapped = {
            "type": "array",
            "items": {"type": "string"},
        }
        lines = generator._format_array_items(unwrapped, "", 0)
        assert any("array items type: string" in ln for ln in lines)

    def test_items_with_properties(self, generator):
        unwrapped = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
            },
        }
        lines = generator._format_array_items(unwrapped, "", 0)
        assert any("array item properties" in ln for ln in lines)

    def test_items_with_one_of(self, generator):
        unwrapped = {
            "type": "array",
            "items": {
                "oneOf": [
                    {"$ref": "#/components/schemas/A"},
                    {"$ref": "#/components/schemas/B"},
                ]
            },
        }
        lines = generator._format_array_items(unwrapped, "", 0)
        assert any("oneOf" in ln for ln in lines)


# ---------------------------------------------------------------------------
# 33. _format_union_array_items
# ---------------------------------------------------------------------------


class TestFormatUnionArrayItems:
    def test_with_ref_variants(self, generator):
        items_one_of = [
            {"$ref": "#/components/schemas/Cat"},
            {"$ref": "#/components/schemas/Dog"},
        ]
        lines = generator._format_union_array_items(items_one_of, "", 0)
        text = "\n".join(lines)
        assert "Cat" in text
        assert "Dog" in text

    def test_unnamed_variants(self, generator):
        items_one_of = [
            {"type": "string"},
            {"type": "integer"},
        ]
        lines = generator._format_union_array_items(items_one_of, "", 0)
        text = "\n".join(lines)
        assert "union of 2 variants" in text


# ---------------------------------------------------------------------------
# 34. _get_variant_name
# ---------------------------------------------------------------------------


class TestGetVariantName:
    def test_with_ref(self, generator):
        variant = {"$ref": "#/components/schemas/MyModel"}
        assert generator._get_variant_name(variant) == "MyModel"

    def test_with_const_property(self, generator):
        variant = {
            "properties": {
                "type": {"type": "string", "const": "cat"},
            }
        }
        assert generator._get_variant_name(variant) == "cat"

    def test_no_match(self, generator):
        variant = {"type": "string"}
        assert generator._get_variant_name(variant) is None


# ---------------------------------------------------------------------------
# 35. _format_array_constraints
# ---------------------------------------------------------------------------


class TestFormatArrayConstraints:
    def test_both_min_and_max(self, generator):
        unwrapped = {"type": "array", "minItems": 1, "maxItems": 10}
        lines = generator._format_array_constraints(unwrapped, "")
        assert len(lines) == 1
        assert "minItems=1" in lines[0]
        assert "maxItems=10" in lines[0]

    def test_only_min(self, generator):
        unwrapped = {"type": "array", "minItems": 1}
        lines = generator._format_array_constraints(unwrapped, "")
        assert len(lines) == 1
        assert "minItems=1" in lines[0]
        assert "maxItems" not in lines[0]

    def test_neither(self, generator):
        unwrapped = {"type": "array"}
        lines = generator._format_array_constraints(unwrapped, "")
        assert lines == []


# ---------------------------------------------------------------------------
# 36. _resolve_ref_in_union
# ---------------------------------------------------------------------------


class TestResolveRefInUnion:
    def test_direct_ref_match(self, generator):
        one_of = [
            {
                "$ref": "#/components/schemas/Cat",
                "properties": {"name": {"type": "string"}},
            },
        ]
        result = generator._resolve_ref_in_union("#/components/schemas/Cat", one_of)
        assert result is not None
        assert "name" in result.get("properties", {})

    def test_no_match(self, generator):
        one_of = [
            {
                "$ref": "#/components/schemas/Dog",
                "properties": {"breed": {"type": "string"}},
            },
        ]
        result = generator._resolve_ref_in_union("#/components/schemas/Cat", one_of)
        assert result is None


# ---------------------------------------------------------------------------
# 37. _try_resolve_variant_by_ref
# ---------------------------------------------------------------------------


class TestTryResolveVariantByRef:
    def test_direct_ref_with_properties(self, generator):
        variant = {
            "$ref": "#/components/schemas/Cat",
            "properties": {"name": {"type": "string"}},
        }
        result = generator._try_resolve_variant_by_ref(
            variant, "#/components/schemas/Cat"
        )
        assert result is not None
        assert "name" in result["properties"]

    def test_ref_with_allof(self, generator):
        variant = {
            "$ref": "#/components/schemas/Cat",
            "allOf": [
                {"properties": {"name": {"type": "string"}}, "required": ["name"]},
            ],
        }
        result = generator._try_resolve_variant_by_ref(
            variant, "#/components/schemas/Cat"
        )
        assert result is not None
        assert "name" in result["properties"]

    def test_allof_sub_matches_ref(self, generator):
        variant = {
            "allOf": [
                {"$ref": "#/components/schemas/Cat"},
                {"properties": {"color": {"type": "string"}}},
            ],
        }
        result = generator._try_resolve_variant_by_ref(
            variant, "#/components/schemas/Cat"
        )
        assert result is not None
        assert "color" in result["properties"]

    def test_no_match(self, generator):
        variant = {
            "$ref": "#/components/schemas/Dog",
            "properties": {"breed": {"type": "string"}},
        }
        result = generator._try_resolve_variant_by_ref(
            variant, "#/components/schemas/Cat"
        )
        assert result is None


# ---------------------------------------------------------------------------
# 38. _format_schema_property
# ---------------------------------------------------------------------------


class TestFormatSchemaProperty:
    def test_required_string_property(self, generator):
        lines = generator._format_schema_property(
            "name", {"type": "string"}, ["name"], "", 0
        )
        text = "\n".join(lines)
        assert "name" in text
        assert "REQUIRED" in text

    def test_optional_integer_property(self, generator):
        lines = generator._format_schema_property("age", {"type": "integer"}, [], "", 0)
        text = "\n".join(lines)
        assert "age" in text
        assert "optional" in text

    def test_property_with_description(self, generator):
        lines = generator._format_schema_property(
            "email",
            {"type": "string", "description": "User email address"},
            [],
            "",
            0,
        )
        text = "\n".join(lines)
        assert "User email address" in text

    def test_nested_object_property(self, generator):
        lines = generator._format_schema_property(
            "address",
            {"type": "object", "properties": {"street": {"type": "string"}}},
            [],
            "",
            0,
        )
        text = "\n".join(lines)
        assert "nested object" in text


# ---------------------------------------------------------------------------
# 39. _format_property_constraints
# ---------------------------------------------------------------------------


class TestFormatPropertyConstraints:
    def test_min_max_length(self, generator):
        result = generator._format_property_constraints(
            {"minLength": 1, "maxLength": 100}
        )
        assert "minLength=1" in result
        assert "maxLength=100" in result

    def test_min_max_numeric(self, generator):
        result = generator._format_property_constraints({"minimum": 0, "maximum": 999})
        assert "min=0" in result
        assert "max=999" in result

    def test_pattern(self, generator):
        result = generator._format_property_constraints({"pattern": "^[a-z]+$"})
        assert any("pattern=" in c for c in result)

    def test_multiple_of(self, generator):
        result = generator._format_property_constraints({"multipleOf": 5})
        assert "multipleOf=5" in result

    def test_no_constraints(self, generator):
        result = generator._format_property_constraints({"type": "string"})
        assert result == []


# ---------------------------------------------------------------------------
# 40. _resolve_variant_properties
# ---------------------------------------------------------------------------


class TestResolveVariantProperties:
    def test_variant_with_properties(self, generator):
        variant = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        props, req = generator._resolve_variant_properties(variant)
        assert "name" in props
        assert "name" in req

    def test_variant_with_allof(self, generator):
        variant = {
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"properties": {"b": {"type": "integer"}}},
            ]
        }
        props, req = generator._resolve_variant_properties(variant)
        assert "a" in props
        assert "b" in props
        assert "a" in req


# ---------------------------------------------------------------------------
# 41. _format_variant_fields
# ---------------------------------------------------------------------------


class TestFormatVariantFields:
    def test_formats_variant_fields(self, generator):
        v_props = {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        }
        v_required = ["name"]
        lines = generator._format_variant_fields(v_props, v_required, "Person")
        text = "\n".join(lines)
        assert "Person" in text
        assert '"name"' in text
        assert '"age"' in text


# ---------------------------------------------------------------------------
# 42. _format_union_field_instructions
# ---------------------------------------------------------------------------


class TestFormatUnionFieldInstructions:
    def test_with_discriminator(self, generator):
        one_of = [
            {
                "title": "Cat",
                "properties": {
                    "type": {"type": "string", "const": "cat"},
                    "color": {"type": "string"},
                },
                "required": ["type", "color"],
            },
        ]
        discriminator = {"propertyName": "type"}
        result = generator._format_union_field_instructions(one_of, discriminator)
        assert "DISCRIMINATED UNION" in result
        assert "type" in result
        assert "Cat" in result


# ---------------------------------------------------------------------------
# 43. _format_field_instruction_line
# ---------------------------------------------------------------------------


class TestFormatFieldInstructionLine:
    def test_required_field(self, generator):
        result = generator._format_field_instruction_line(
            "name", {"type": "string"}, ["name"]
        )
        assert result is not None
        assert "[REQUIRED]" in result
        assert '"name"' in result

    def test_optional_field_with_format(self, generator):
        result = generator._format_field_instruction_line(
            "created_at", {"type": "string", "format": "date-time"}, []
        )
        assert result is not None
        assert "[REQUIRED]" not in result
        assert "format=date-time" in result

    def test_non_dict_schema(self, generator):
        result = generator._format_field_instruction_line("x", "string", [])
        assert result is None


# ---------------------------------------------------------------------------
# 44. _format_endpoints_list
# ---------------------------------------------------------------------------


class TestFormatEndpointsList:
    def test_empty_list(self, generator):
        result = generator._format_endpoints_list([])
        assert result == ""

    def test_multiple_endpoints(self, generator):
        ep1 = Mock()
        ep1.method = "get"
        ep1.path = "/users"
        ep1.summary = "Get users"
        ep2 = Mock()
        ep2.method = "post"
        ep2.path = "/users"
        ep2.summary = "Create user"
        result = generator._format_endpoints_list([ep1, ep2])
        assert "GET /users" in result
        assert "POST /users" in result
        assert "Get users" in result
        assert "Create user" in result

    def test_endpoint_without_summary(self, generator):
        ep = Mock()
        ep.method = "delete"
        ep.path = "/items"
        ep.summary = None
        result = generator._format_endpoints_list([ep])
        assert "DELETE /items" in result
        assert "No summary" in result
