"""
Tests for all 12 issue fixes.

Issue 1: --max-llm-workers CLI flag (default=1, max=10, fail-fast)
Issue 2: Recursive $ref resolution in parser
Issue 3: Dead code (indented_task_methods overwrite) removed
Issue 4: Deterministic fallback templates (no random param skip)
Issue 5: Single _extract_code call (removed from _call_ai_service)
Issue 6: Segment-by-segment path matching in code_validator
Issue 7: Regex captures f-string paths
Issue 8: List[Response] handling in orchestrator formatting
Issue 9: allOf merging in _precompute_positive_fields
Issue 10: Recursive _precompute_object_instruction
Issue 11: TODO comment on auth detection (no test needed)
Issue 12: Enum values not truncated
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from click.testing import CliRunner

from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Response
from devdox_ai_locust.utils.code_validator import CodeValidator


# ============================================================
# Issue 1: --max-llm-workers CLI flag
# ============================================================

class TestMaxLLMWorkers:
    """Tests for --max-llm-workers CLI flag validation."""

    def test_default_value_is_1(self):
        """--max-llm-workers defaults to 1."""
        from devdox_ai_locust.cli import generate
        from click.testing import CliRunner

        runner = CliRunner()
        # We can't fully run generate without a valid swagger URL,
        # but we can check the parameter definition
        params = {p.name: p for p in generate.params}
        assert "max_llm_workers" in params
        assert params["max_llm_workers"].default == 1

    def test_rejects_value_above_10(self):
        """--max-llm-workers > 10 raises BadParameter."""
        from devdox_ai_locust.cli import generate
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(generate, [
            "--max-llm-workers", "11",
            "http://example.com/swagger.json",
        ], catch_exceptions=True)
        assert result.exit_code != 0
        assert "cannot exceed 10" in result.output or "cannot exceed 10" in str(result.exception)

    def test_accepts_value_10(self):
        """--max-llm-workers=10 is accepted (validation passes before other errors)."""
        from devdox_ai_locust.cli import generate
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(generate, [
            "--max-llm-workers", "10",
            "http://example.com/swagger.json",
        ], catch_exceptions=True)
        # Should not fail due to max-llm-workers validation
        assert "cannot exceed 10" not in (result.output or "")


# ============================================================
# Issue 2: Recursive $ref resolution
# ============================================================

class TestRecursiveRefResolution:
    """Tests for _resolve_schema_deep recursive resolution."""

    def _make_parser_with_spec(self, spec_data):
        parser = OpenAPIParser()
        parser.spec_data = spec_data
        parser.components = spec_data.get("components", {})
        return parser

    def test_simple_ref_resolved(self):
        """A direct $ref is resolved to its target."""
        spec = {
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        }
                    }
                }
            }
        }
        parser = self._make_parser_with_spec(spec)
        schema = {"$ref": "#/components/schemas/User"}

        result = parser._resolve_schema_deep(schema)

        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_nested_refs_resolved(self):
        """Nested $refs are recursively resolved."""
        spec = {
            "components": {
                "schemas": {
                    "Address": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        }
                    },
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address": {"$ref": "#/components/schemas/Address"},
                        }
                    }
                }
            }
        }
        parser = self._make_parser_with_spec(spec)
        schema = {"$ref": "#/components/schemas/User"}

        result = parser._resolve_schema_deep(schema)

        assert result["type"] == "object"
        # The nested address ref should be resolved
        assert result["properties"]["address"]["type"] == "object"
        assert "city" in result["properties"]["address"]["properties"]

    def test_circular_ref_detected(self):
        """Circular references are detected and do not cause infinite recursion."""
        spec = {
            "components": {
                "schemas": {
                    "Person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "best_friend": {"$ref": "#/components/schemas/Person"},
                        }
                    }
                }
            }
        }
        parser = self._make_parser_with_spec(spec)
        schema = {"$ref": "#/components/schemas/Person"}

        result = parser._resolve_schema_deep(schema)

        assert result is not None
        assert result["type"] == "object"
        assert "name" in result["properties"]
        # best_friend should be resolved to one level (the Person schema's
        # structure without further recursion into its own best_friend)
        best_friend = result["properties"]["best_friend"]
        assert best_friend["type"] == "object"

    def test_allof_refs_resolved(self):
        """allOf items with $refs are resolved."""
        spec = {
            "components": {
                "schemas": {
                    "Base": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                        }
                    }
                }
            }
        }
        parser = self._make_parser_with_spec(spec)
        schema = {
            "allOf": [
                {"$ref": "#/components/schemas/Base"},
                {"type": "object", "properties": {"name": {"type": "string"}}},
            ]
        }

        result = parser._resolve_schema_deep(schema)

        assert result["allOf"][0]["type"] == "object"
        assert "id" in result["allOf"][0]["properties"]
        assert "name" in result["allOf"][1]["properties"]

    def test_array_items_ref_resolved(self):
        """$ref in array items is resolved."""
        spec = {
            "components": {
                "schemas": {
                    "Tag": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                        }
                    }
                }
            }
        }
        parser = self._make_parser_with_spec(spec)
        schema = {
            "type": "array",
            "items": {"$ref": "#/components/schemas/Tag"},
        }

        result = parser._resolve_schema_deep(schema)

        assert result["items"]["type"] == "object"
        assert "label" in result["items"]["properties"]

    def test_none_schema_returns_none(self):
        """None input returns None."""
        parser = self._make_parser_with_spec({})
        assert parser._resolve_schema_deep(None) is None

    def test_no_ref_returns_same_structure(self):
        """Schema without $ref returns the same structure."""
        parser = self._make_parser_with_spec({})
        schema = {"type": "string", "maxLength": 50}

        result = parser._resolve_schema_deep(schema)

        assert result["type"] == "string"
        assert result["maxLength"] == 50


# ============================================================
# Issue 4: Deterministic fallback templates
# ============================================================

class TestDeterministicFallbackTemplates:
    """Tests that optional params are never randomly skipped."""

    def test_all_optional_params_included(self):
        """All optional parameters are included in generated query params."""
        from devdox_ai_locust.locust_generator import LocustTestGenerator
        from devdox_ai_locust.utils.open_ai_parser import Parameter, ParameterType, Endpoint

        gen = LocustTestGenerator()
        endpoint = Mock()
        endpoint.parameters = [
            Parameter(
                name="required_param",
                location=ParameterType.QUERY,
                required=True,
                type="string",
            ),
            Parameter(
                name="optional_param1",
                location=ParameterType.QUERY,
                required=False,
                type="string",
            ),
            Parameter(
                name="optional_param2",
                location=ParameterType.QUERY,
                required=False,
                type="integer",
            ),
        ]

        # Run 10 times - should always include all params (no random skip)
        for _ in range(10):
            result = gen._generate_query_params_code(endpoint)
            assert "required_param" in result
            assert "optional_param1" in result
            assert "optional_param2" in result


# ============================================================
# Issue 5: Single _extract_code call
# ============================================================

class TestSingleExtractCode:
    """Tests that _call_ai_service returns raw content without extraction."""

    def test_call_ai_service_returns_raw_content(self):
        """_call_ai_service returns raw AI response without _extract_code."""
        import asyncio
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        prompt_dir = Mock()
        mock_client = AsyncMock()
        ai_config = AIEnhancementConfig()

        gen = ScenarioWorkflowGenerator(
            prompt_dir=prompt_dir,
            ai_client=mock_client,
            ai_config=ai_config,
        )

        # Mock the AI response with <analysis> and <code> tags
        raw_response = "<analysis>some analysis</analysis>\n<code>\nprint('hello')\n</code>"
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = raw_response
        mock_response.headers = {}

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = asyncio.run(gen._call_ai_service("test prompt", "positive"))

        # Should return raw content (stripped), NOT extracted code
        assert "<analysis>" in result
        assert "<code>" in result


# ============================================================
# Issue 6: Segment-by-segment path matching
# ============================================================

class TestPathMatching:
    """Tests for _paths_match segment-by-segment comparison."""

    def test_exact_match(self):
        """Identical paths match."""
        assert CodeValidator._paths_match("/items", "/items") is True

    def test_param_wildcard_match(self):
        """Path params act as wildcards."""
        assert CodeValidator._paths_match("/items/{id}", "/items/{item_id}") is True
        assert CodeValidator._paths_match("/items/{id}", "/items/123") is True

    def test_different_segment_count_no_match(self):
        """Different number of segments don't match."""
        assert CodeValidator._paths_match("/items/{id}", "/items/{id}/details") is False
        assert CodeValidator._paths_match("/items", "/items/sub") is False

    def test_different_static_segments_no_match(self):
        """Different static segments don't match."""
        assert CodeValidator._paths_match("/items/{id}", "/users/{id}") is False

    def test_multi_param_paths(self):
        """Multiple path params are all treated as wildcards."""
        assert CodeValidator._paths_match(
            "/users/{user_id}/posts/{post_id}",
            "/users/{uid}/posts/{pid}"
        ) is True

    def test_nested_resource_path(self):
        """Nested resource paths differentiated correctly."""
        assert CodeValidator._paths_match(
            "/items/{id}",
            "/items/{id}/details"
        ) is False
        assert CodeValidator._paths_match(
            "/items/{id}/details",
            "/items/{id}/details"
        ) is True

    def test_trailing_slashes_normalized(self):
        """Trailing slashes are stripped."""
        assert CodeValidator._paths_match("/items/", "/items") is True


# ============================================================
# Issue 7: Regex captures f-string paths
# ============================================================

class TestMakeRequestRegex:
    """Tests for MAKE_REQUEST_CALL_RE capturing f-string paths."""

    def test_captures_simple_path(self):
        """Regex captures a simple quoted path."""
        line = 'self.make_request("GET", "/items")'
        match = CodeValidator.MAKE_REQUEST_CALL_RE.search(line)
        assert match is not None
        assert match.group(1) == "GET"
        assert match.group(2) == "/items"

    def test_captures_fstring_path(self):
        """Regex captures f-string path with variable."""
        line = 'self.make_request("GET", f"/items/{item_id}")'
        match = CodeValidator.MAKE_REQUEST_CALL_RE.search(line)
        assert match is not None
        assert match.group(2) == "/items/{item_id}"

    def test_captures_complex_fstring_path(self):
        """Regex captures f-string with complex expression."""
        line = 'self.make_request("DELETE", f"/users/{self.created_ids[-1]}/posts/{post_id}")'
        match = CodeValidator.MAKE_REQUEST_CALL_RE.search(line)
        assert match is not None
        assert match.group(1) == "DELETE"
        assert "/users/" in match.group(2)
        assert "/posts/" in match.group(2)

    def test_captures_path_with_expected_status(self):
        """Regex works when make_request has additional kwargs."""
        line = 'result = self.make_request("POST", f"/items/{item_id}/verify", expected_status=[200])'
        match = CodeValidator.MAKE_REQUEST_CALL_RE.search(line)
        assert match is not None
        assert match.group(1) == "POST"
        assert "/items/" in match.group(2)
        assert "/verify" in match.group(2)


# ============================================================
# Issue 8: List[Response] handling
# ============================================================

class TestListResponseHandling:
    """Tests that orchestrator formatting handles List[Response] correctly."""

    def test_format_endpoints_with_list_responses(self):
        """_format_endpoints_for_orchestrator handles List[Response]."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Create endpoint with List[Response]
        endpoint = Mock()
        endpoint.method = "GET"
        endpoint.path = "/items"
        endpoint.summary = "Get items"
        endpoint.tags = ["items"]
        endpoint.parameters = []
        endpoint.request_body = None
        endpoint.responses = [
            Response(
                status_code="200",
                description="Success",
                content_type="application/json",
                schema={"type": "object", "properties": {"id": {"type": "integer"}}},
            ),
            Response(
                status_code="404",
                description="Not found",
                content_type=None,
                schema=None,
            ),
        ]

        result = gen._format_endpoints_for_orchestrator([endpoint])

        # Should include the response schema without errors
        assert "GET /items" in result or "/items" in result


# ============================================================
# Issue 9: allOf handling in _precompute_positive_fields
# ============================================================

class TestAllOfHandling:
    """Tests for allOf schema merging in _precompute_positive_fields."""

    def test_allof_properties_merged(self):
        """allOf items' properties are merged together."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        endpoint = Mock()
        endpoint.method = "POST"
        endpoint.path = "/items"
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ]
        }

        result = gen._precompute_positive_fields(endpoint)

        # Both fields from allOf should appear
        assert '"id"' in result
        assert '"name"' in result
        assert "generate_integer" in result
        assert "generate_string" in result

    def test_allof_with_direct_properties(self):
        """allOf + direct properties are all included."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        endpoint = Mock()
        endpoint.method = "POST"
        endpoint.path = "/items"
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {"extra": {"type": "boolean"}},
            "allOf": [
                {
                    "type": "object",
                    "properties": {"base_field": {"type": "string"}},
                },
            ]
        }

        result = gen._precompute_positive_fields(endpoint)

        # Both allOf and direct properties should appear
        assert '"extra"' in result
        assert '"base_field"' in result


# ============================================================
# Issue 10: Recursive _precompute_object_instruction
# ============================================================

class TestRecursiveObjectInstruction:
    """Tests for recursive nested object instruction generation."""

    def test_nested_object_resolved(self):
        """Nested objects generate full dict instructions, not '{}'."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "integer", "minimum": 10000, "maximum": 99999},
                    }
                }
            }
        }

        result = gen._precompute_object_instruction(schema)

        # Should contain nested fields, not just {}
        assert '"address"' in result
        assert '"city"' in result
        assert '"zip"' in result
        assert "generate_string" in result
        assert "generate_integer" in result

    def test_deeply_nested_objects(self):
        """Three levels of nesting are all resolved."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3_val": {"type": "string"},
                            }
                        }
                    }
                }
            }
        }

        result = gen._precompute_object_instruction(schema)

        assert '"level1"' in result
        assert '"level2"' in result
        assert '"level3_val"' in result

    def test_circular_object_does_not_infinite_loop(self):
        """Circular object reference stops recursion via identity tracking."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Create a circular reference (same dict object)
        person_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            }
        }
        # Make best_friend point to the same schema object
        person_schema["properties"]["best_friend"] = person_schema

        # Should not infinite loop
        result = gen._precompute_object_instruction(person_schema)

        assert '"name"' in result
        # best_friend should be {} due to circular detection
        assert result is not None

    def test_array_of_nested_objects(self):
        """Arrays of objects have their items recursively resolved."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "integer"},
                        }
                    }
                }
            }
        }

        result = gen._precompute_object_instruction(schema)

        assert '"items"' in result
        assert '"label"' in result
        assert '"value"' in result


# ============================================================
# Issue 12: Enum values not truncated
# ============================================================

class TestEnumNotTruncated:
    """Tests that enum values are displayed in full."""

    def test_long_enum_not_truncated(self):
        """Enum values longer than 60 chars are not cut off."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.ai_config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Create a schema with a long enum list
        long_enum = ["value_one", "value_two", "value_three", "value_four",
                     "value_five_is_quite_long", "value_six_even_longer_than_before"]
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": long_enum,
                }
            }
        }

        result = gen._format_schema(schema, indent=0)
        result_text = "\n".join(result)

        # All enum values should be present (not truncated)
        for val in long_enum:
            assert val in result_text, f"Enum value '{val}' was truncated"


# ============================================================
# Issue 6 (additional): _path_matches_spec uses _paths_match
# ============================================================

class TestPathMatchesSpec:
    """Tests that _path_matches_spec correctly uses segment matching."""

    def test_matches_exact_path(self):
        """Exact path match works."""
        validator = CodeValidator()
        assert validator._path_matches_spec(
            "/items", "/items", ["/items", "/users"]
        ) is True

    def test_matches_parameterized_path(self):
        """Parameterized paths match with different variable names."""
        validator = CodeValidator()
        assert validator._path_matches_spec(
            "/items/123", "/items/{id}", ["/items/{id}"]
        ) is True

    def test_does_not_match_different_depth(self):
        """Paths with different depths don't match."""
        validator = CodeValidator()
        assert validator._path_matches_spec(
            "/items/123/details", "/items/{id}", ["/items/{id}"]
        ) is False

    def test_does_not_match_different_base(self):
        """Paths with different base segments don't match."""
        validator = CodeValidator()
        assert validator._path_matches_spec(
            "/orders/123", "/items/{id}", ["/items/{id}"]
        ) is False


# ============================================================
# Issue 15: OpenAPI 3.1 anyOf nullable pattern unwrapping
# ============================================================

class TestUnwrapNullableSchema:
    """Tests for _unwrap_nullable_schema handling OpenAPI 3.0 and 3.1 nullable patterns."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def test_unwrap_31_string_nullable(self):
        """OpenAPI 3.1 anyOf with string + null unwraps to string schema."""
        gen = self._get_generator()
        schema = {"anyOf": [{"type": "string", "maxLength": 50}, {"type": "null"}]}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped == {"type": "string", "maxLength": 50}
        assert is_nullable is True

    def test_unwrap_31_integer_nullable(self):
        """OpenAPI 3.1 anyOf with integer + null unwraps to integer schema."""
        gen = self._get_generator()
        schema = {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped == {"type": "integer", "minimum": 0}
        assert is_nullable is True

    def test_unwrap_31_ref_nullable(self):
        """OpenAPI 3.1 anyOf with object ref + null unwraps to the object schema."""
        gen = self._get_generator()
        schema = {"anyOf": [{"type": "object", "properties": {"street": {"type": "string"}}}, {"type": "null"}]}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped["type"] == "object"
        assert "street" in unwrapped["properties"]
        assert is_nullable is True

    def test_unwrap_30_nullable_flag(self):
        """OpenAPI 3.0 nullable: true is detected."""
        gen = self._get_generator()
        schema = {"type": "string", "nullable": True, "format": "email"}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped == schema  # Returns same schema, just flags nullable
        assert is_nullable is True

    def test_no_unwrap_plain_schema(self):
        """Non-nullable schemas are returned unchanged."""
        gen = self._get_generator()
        schema = {"type": "string", "maxLength": 10}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped == schema
        assert is_nullable is False

    def test_no_unwrap_discriminated_union(self):
        """Discriminated unions (multiple non-null variants) are NOT unwrapped."""
        gen = self._get_generator()
        schema = {
            "oneOf": [
                {"type": "object", "properties": {"pet_type": {"const": "dog"}}},
                {"type": "object", "properties": {"pet_type": {"const": "cat"}}},
            ],
            "discriminator": {"propertyName": "pet_type"}
        }
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped == schema  # Not unwrapped
        assert is_nullable is False

    def test_unwrap_oneOf_nullable(self):
        """oneOf with null type also works (some specs use oneOf for nullable)."""
        gen = self._get_generator()
        schema = {"oneOf": [{"type": "number", "minimum": 0.0}, {"type": "null"}]}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped == {"type": "number", "minimum": 0.0}
        assert is_nullable is True

    def test_unwrap_enum_nullable(self):
        """Nullable enum schema is unwrapped to expose enum values."""
        gen = self._get_generator()
        schema = {"anyOf": [{"type": "string", "enum": ["active", "inactive"]}, {"type": "null"}]}
        unwrapped, is_nullable = gen._unwrap_nullable_schema(schema)
        assert unwrapped["enum"] == ["active", "inactive"]
        assert is_nullable is True

    def test_unwrap_non_dict_returns_unchanged(self):
        """Non-dict input is returned unchanged."""
        gen = self._get_generator()
        result, is_nullable = gen._unwrap_nullable_schema("string")
        assert result == "string"
        assert is_nullable is False


# ============================================================
# Issue 15+16: Nullable schema integration in format/precompute
# ============================================================

class TestNullableSchemaIntegration:
    """Tests that nullable unwrapping is applied in formatting and precomputation."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        gen = ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)
        return gen

    def test_format_schema_shows_nullable_type(self):
        """_format_schema shows correct type for nullable fields, not 'any'."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "count": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
            },
            "required": ["count"]
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "notes: string" in output
        assert "count: integer" in output
        assert "nullable" in output
        # Should NOT show "any"
        assert "notes: any" not in output

    def test_format_schema_shows_nullable_format(self):
        """_format_schema shows format from inside anyOf."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "created_at": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
            }
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "date-time" in output

    def test_format_schema_shows_nullable_enum(self):
        """_format_schema shows enum values from inside anyOf."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "status": {"anyOf": [{"type": "string", "enum": ["active", "pending"]}, {"type": "null"}]},
            }
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "active" in output
        assert "pending" in output

    def test_format_schema_shows_nullable_constraints(self):
        """_format_schema shows constraints from inside anyOf."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "age": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 150}, {"type": "null"}]},
            }
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "min=0" in output
        assert "max=150" in output

    def test_precompute_positive_fields_nullable_integer(self):
        """_precompute_positive_fields generates integer instruction for nullable int."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {
                "count": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 100}, {"type": "null"}]},
            },
            "required": ["count"]
        }
        result = gen._precompute_positive_fields(endpoint)
        assert "generate_integer" in result
        assert "min_val=1" in result

    def test_precompute_positive_fields_nullable_email(self):
        """_precompute_positive_fields generates email instruction for nullable email."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {
                "email": {"anyOf": [{"type": "string", "format": "email"}, {"type": "null"}]},
            }
        }
        result = gen._precompute_positive_fields(endpoint)
        assert "generate_email" in result

    def test_precompute_positive_fields_nullable_enum(self):
        """_precompute_positive_fields generates random.choice for nullable enum."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {
                "status": {"anyOf": [{"type": "string", "enum": ["active", "inactive"]}, {"type": "null"}]},
            }
        }
        result = gen._precompute_positive_fields(endpoint)
        assert "random.choice" in result
        assert "active" in result

    def test_precompute_object_instruction_nullable_fields(self):
        """_precompute_object_instruction handles nullable nested fields."""
        gen = self._get_generator()
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "email": {"anyOf": [{"type": "string", "format": "email"}, {"type": "null"}]},
            }
        }
        result = gen._precompute_object_instruction(schema)
        assert "generate_integer" in result
        assert "generate_email" in result

    def test_format_response_schema_nullable(self):
        """_format_response_schema shows correct types for nullable response fields."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "deleted_at": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
            }
        }
        lines = gen._format_response_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "id: integer" in output
        assert "deleted_at: string [date-time]" in output
        assert "nullable" in output


# ============================================================
# Issue 16: Parameter enum extraction for nullable params
# ============================================================

class TestNullableParameterExtraction:
    """Tests that nullable enum parameters are correctly extracted."""

    def _get_parser(self):
        parser = OpenAPIParser()
        parser.spec_data = {"components": {"schemas": {}}}
        parser.components = {}
        return parser

    def test_anyof_enum_param_extracted(self):
        """Parser extracts enum values from anyOf nullable parameter schema."""
        parser = self._get_parser()
        operation = {
            "parameters": [{
                "name": "status",
                "in": "query",
                "schema": {
                    "anyOf": [
                        {"type": "string", "enum": ["active", "inactive", "pending"]},
                        {"type": "null"}
                    ]
                }
            }]
        }
        params = parser._extract_parameters(operation)
        assert params[0].enum == ["active", "inactive", "pending"]
        assert params[0].type == "string"

    def test_anyof_format_param_extracted(self):
        """Parser extracts format from anyOf nullable parameter schema."""
        parser = self._get_parser()
        operation = {
            "parameters": [{
                "name": "created_after",
                "in": "query",
                "schema": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"}
                    ]
                }
            }]
        }
        params = parser._extract_parameters(operation)
        assert params[0].format == "date-time"
        assert params[0].type == "string"

    def test_anyof_constraints_param_extracted(self):
        """Parser extracts constraints from anyOf nullable parameter schema."""
        parser = self._get_parser()
        operation = {
            "parameters": [{
                "name": "limit",
                "in": "query",
                "schema": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": 100},
                        {"type": "null"}
                    ]
                }
            }]
        }
        params = parser._extract_parameters(operation)
        assert params[0].type == "integer"
        assert params[0].minimum == 1
        assert params[0].maximum == 100


# ============================================================
# Issue 17: Discriminator const matching normalization
# ============================================================

class TestDiscriminatorConstMatching:
    """Tests for _resolve_ref_in_union matching multi-word schema names."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        gen = ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)
        return gen

    def test_matches_single_word_const(self):
        """Single-word const values match their ref names."""
        gen = self._get_generator()
        one_of = [
            {"properties": {"pet_type": {"const": "dog"}, "breed": {"type": "string"}}},
            {"properties": {"pet_type": {"const": "cat"}, "indoor": {"type": "boolean"}}},
        ]
        result = gen._resolve_ref_in_union("#/components/schemas/Dog", one_of)
        assert result is not None
        assert "breed" in result["properties"]

    def test_matches_multi_word_snake_case_const(self):
        """Multi-word snake_case const matches PascalCase ref name."""
        gen = self._get_generator()
        one_of = [
            {"properties": {"payment_type": {"const": "credit_card"}, "card_number": {"type": "string"}}},
            {"properties": {"payment_type": {"const": "bank_transfer"}, "account": {"type": "string"}}},
        ]
        result = gen._resolve_ref_in_union("#/components/schemas/CreditCard", one_of)
        assert result is not None
        assert "card_number" in result["properties"]

    def test_matches_kebab_case_const(self):
        """kebab-case const matches PascalCase ref name."""
        gen = self._get_generator()
        one_of = [
            {"properties": {"type": {"const": "bank-transfer"}, "iban": {"type": "string"}}},
        ]
        result = gen._resolve_ref_in_union("#/components/schemas/BankTransfer", one_of)
        assert result is not None
        assert "iban" in result["properties"]

    def test_no_match_returns_none(self):
        """Non-matching ref returns None."""
        gen = self._get_generator()
        one_of = [
            {"properties": {"type": {"const": "dog"}}},
        ]
        result = gen._resolve_ref_in_union("#/components/schemas/Fish", one_of)
        assert result is None


# ============================================================
# Issue 18: additionalProperties (map types) handling
# ============================================================

class TestAdditionalPropertiesHandling:
    """Tests for additionalProperties map type handling in schema formatting."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        gen = ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)
        return gen

    def test_format_schema_shows_map_type(self):
        """_format_schema shows map type info for additionalProperties."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "map type" in output
        assert "string" in output

    def test_format_schema_integer_map(self):
        """_format_schema shows integer value type for integer maps."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"}
                }
            }
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "integer" in output

    def test_format_schema_freeform_map(self):
        """_format_schema handles additionalProperties: true (freeform)."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {
                "extra": {
                    "type": "object",
                    "additionalProperties": True
                }
            }
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "map type" in output
        assert "any" in output

    def test_precompute_positive_fields_string_map(self):
        """_precompute_positive_fields generates dict comprehension for string map."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }
        result = gen._precompute_positive_fields(endpoint)
        assert "key_" in result
        assert "generate_string" in result

    def test_precompute_positive_fields_integer_map(self):
        """_precompute_positive_fields generates integer values for integer map."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"}
                }
            }
        }
        result = gen._precompute_positive_fields(endpoint)
        assert "generate_integer" in result

    def test_precompute_object_instruction_map_type(self):
        """_precompute_object_instruction generates map for additionalProperties."""
        gen = self._get_generator()
        schema = {
            "properties": {
                "labels": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }
        result = gen._precompute_object_instruction(schema)
        assert "key_" in result
        assert "generate_string" in result
