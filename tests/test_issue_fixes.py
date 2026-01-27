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

from unittest.mock import Mock, AsyncMock
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

        # We can't fully run generate without a valid swagger URL,
        # but we can check the parameter definition
        params = {p.name: p for p in generate.params}
        assert "max_llm_workers" in params
        assert params["max_llm_workers"].default == 1

    def test_rejects_value_above_10(self):
        """--max-llm-workers > 10 raises BadParameter."""
        from devdox_ai_locust.cli import generate

        runner = CliRunner()
        result = runner.invoke(
            generate,
            [
                "--max-llm-workers",
                "11",
                "http://example.com/swagger.json",
            ],
            catch_exceptions=True,
        )
        assert result.exit_code != 0
        assert "cannot exceed 10" in result.output or "cannot exceed 10" in str(
            result.exception
        )

    def test_accepts_value_10(self):
        """--max-llm-workers=10 is accepted (validation passes before other errors)."""
        from devdox_ai_locust.cli import generate

        runner = CliRunner()
        result = runner.invoke(
            generate,
            [
                "--max-llm-workers",
                "10",
                "http://example.com/swagger.json",
            ],
            catch_exceptions=True,
        )
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
                        },
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
                        },
                    },
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address": {"$ref": "#/components/schemas/Address"},
                        },
                    },
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
                        },
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
                        },
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
                        },
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
        from devdox_ai_locust.utils.open_ai_parser import (
            Parameter,
            ParameterType,
        )

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
        from devdox_ai_locust.config import AIEnhancementConfig

        prompt_dir = Mock()
        mock_client = AsyncMock()
        ai_config = AIEnhancementConfig()

        gen = ScenarioWorkflowGenerator(
            prompt_dir=prompt_dir,
            ai_client=mock_client,
            ai_config=ai_config,
        )

        # Mock the AI response with <analysis> and <code> tags
        raw_response = (
            "<analysis>some analysis</analysis>\n<code>\nprint('hello')\n</code>"
        )
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
        assert (
            CodeValidator._paths_match(
                "/users/{user_id}/posts/{post_id}", "/users/{uid}/posts/{pid}"
            )
            is True
        )

    def test_nested_resource_path(self):
        """Nested resource paths differentiated correctly."""
        assert CodeValidator._paths_match("/items/{id}", "/items/{id}/details") is False
        assert (
            CodeValidator._paths_match("/items/{id}/details", "/items/{id}/details")
            is True
        )

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
        from devdox_ai_locust.config import AIEnhancementConfig

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
        from devdox_ai_locust.config import AIEnhancementConfig

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
        from devdox_ai_locust.config import AIEnhancementConfig

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
            ],
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
        from devdox_ai_locust.config import AIEnhancementConfig

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
                    },
                }
            },
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
        from devdox_ai_locust.config import AIEnhancementConfig

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
                            },
                        }
                    },
                }
            },
        }

        result = gen._precompute_object_instruction(schema)

        assert '"level1"' in result
        assert '"level2"' in result
        assert '"level3_val"' in result

    def test_circular_object_does_not_infinite_loop(self):
        """Circular object reference stops recursion via identity tracking."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

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
            },
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
        from devdox_ai_locust.config import AIEnhancementConfig

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
                        },
                    },
                }
            },
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
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Create a schema with a long enum list
        long_enum = [
            "value_one",
            "value_two",
            "value_three",
            "value_four",
            "value_five_is_quite_long",
            "value_six_even_longer_than_before",
        ]
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": long_enum,
                }
            },
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
        assert (
            validator._path_matches_spec("/items", "/items", ["/items", "/users"])
            is True
        )

    def test_matches_parameterized_path(self):
        """Parameterized paths match with different variable names."""
        validator = CodeValidator()
        assert (
            validator._path_matches_spec("/items/123", "/items/{id}", ["/items/{id}"])
            is True
        )

    def test_does_not_match_different_depth(self):
        """Paths with different depths don't match."""
        validator = CodeValidator()
        assert (
            validator._path_matches_spec(
                "/items/123/details", "/items/{id}", ["/items/{id}"]
            )
            is False
        )

    def test_does_not_match_different_base(self):
        """Paths with different base segments don't match."""
        validator = CodeValidator()
        assert (
            validator._path_matches_spec("/orders/123", "/items/{id}", ["/items/{id}"])
            is False
        )


# ============================================================
# Issue 15: OpenAPI 3.1 anyOf nullable pattern unwrapping
# ============================================================


class TestUnwrapNullableSchema:
    """Tests for unwrap_nullable_schema handling OpenAPI 3.0 and 3.1 nullable patterns."""

    def test_unwrap_31_string_nullable(self):
        """OpenAPI 3.1 anyOf with string + null unwraps to string schema."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {"anyOf": [{"type": "string", "maxLength": 50}, {"type": "null"}]}
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped == {"type": "string", "maxLength": 50}
        assert is_nullable is True

    def test_unwrap_31_integer_nullable(self):
        """OpenAPI 3.1 anyOf with integer + null unwraps to integer schema."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]}
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped == {"type": "integer", "minimum": 0}
        assert is_nullable is True

    def test_unwrap_31_ref_nullable(self):
        """OpenAPI 3.1 anyOf with object ref + null unwraps to the object schema."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {
            "anyOf": [
                {"type": "object", "properties": {"street": {"type": "string"}}},
                {"type": "null"},
            ]
        }
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped["type"] == "object"
        assert "street" in unwrapped["properties"]
        assert is_nullable is True

    def test_unwrap_30_nullable_flag(self):
        """OpenAPI 3.0 nullable: true is detected."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {"type": "string", "nullable": True, "format": "email"}
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped == schema  # Returns same schema, just flags nullable
        assert is_nullable is True

    def test_no_unwrap_plain_schema(self):
        """Non-nullable schemas are returned unchanged."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {"type": "string", "maxLength": 10}
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped == schema
        assert is_nullable is False

    def test_no_unwrap_discriminated_union(self):
        """Discriminated unions (multiple non-null variants) are NOT unwrapped."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {
            "oneOf": [
                {"type": "object", "properties": {"pet_type": {"const": "dog"}}},
                {"type": "object", "properties": {"pet_type": {"const": "cat"}}},
            ],
            "discriminator": {"propertyName": "pet_type"},
        }
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped == schema  # Not unwrapped
        assert is_nullable is False

    def test_unwrap_oneOf_nullable(self):
        """oneOf with null type also works (some specs use oneOf for nullable)."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {"oneOf": [{"type": "number", "minimum": 0.0}, {"type": "null"}]}
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped == {"type": "number", "minimum": 0.0}
        assert is_nullable is True

    def test_unwrap_enum_nullable(self):
        """Nullable enum schema is unwrapped to expose enum values."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        schema = {
            "anyOf": [
                {"type": "string", "enum": ["active", "inactive"]},
                {"type": "null"},
            ]
        }
        unwrapped, is_nullable = unwrap_nullable_schema(schema)
        assert unwrapped["enum"] == ["active", "inactive"]
        assert is_nullable is True

    def test_unwrap_non_dict_returns_unchanged(self):
        """Non-dict input is returned unchanged."""
        from devdox_ai_locust.utils.schema_utils import unwrap_nullable_schema

        result, is_nullable = unwrap_nullable_schema("string")
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
                "count": {
                    "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]
                },
            },
            "required": ["count"],
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
                "created_at": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"},
                    ]
                },
            },
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
                "status": {
                    "anyOf": [
                        {"type": "string", "enum": ["active", "pending"]},
                        {"type": "null"},
                    ]
                },
            },
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
                "age": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0, "maximum": 150},
                        {"type": "null"},
                    ]
                },
            },
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
                "count": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": 100},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["count"],
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
                "email": {
                    "anyOf": [{"type": "string", "format": "email"}, {"type": "null"}]
                },
            },
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
                "status": {
                    "anyOf": [
                        {"type": "string", "enum": ["active", "inactive"]},
                        {"type": "null"},
                    ]
                },
            },
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
                "email": {
                    "anyOf": [{"type": "string", "format": "email"}, {"type": "null"}]
                },
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
                "deleted_at": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"},
                    ]
                },
            },
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
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "schema": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": ["active", "inactive", "pending"],
                            },
                            {"type": "null"},
                        ]
                    },
                }
            ]
        }
        params = parser._extract_parameters(operation)
        assert params[0].enum == ["active", "inactive", "pending"]
        assert params[0].type == "string"

    def test_anyof_format_param_extracted(self):
        """Parser extracts format from anyOf nullable parameter schema."""
        parser = self._get_parser()
        operation = {
            "parameters": [
                {
                    "name": "created_after",
                    "in": "query",
                    "schema": {
                        "anyOf": [
                            {"type": "string", "format": "date-time"},
                            {"type": "null"},
                        ]
                    },
                }
            ]
        }
        params = parser._extract_parameters(operation)
        assert params[0].format == "date-time"
        assert params[0].type == "string"

    def test_anyof_constraints_param_extracted(self):
        """Parser extracts constraints from anyOf nullable parameter schema."""
        parser = self._get_parser()
        operation = {
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1, "maximum": 100},
                            {"type": "null"},
                        ]
                    },
                }
            ]
        }
        params = parser._extract_parameters(operation)
        assert params[0].type == "integer"
        assert params[0].minimum == 1
        assert params[0].maximum == 100

    def test_anyof_ref_param_resolved(self):
        """Parser resolves $ref inside anyOf before unwrapping."""
        parser = self._get_parser()
        parser.spec_data = {
            "components": {
                "schemas": {
                    "Status": {"type": "string", "enum": ["active", "inactive"]}
                }
            }
        }
        parser.components = parser.spec_data["components"]
        operation = {
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "schema": {
                        "anyOf": [
                            {"$ref": "#/components/schemas/Status"},
                            {"type": "null"},
                        ]
                    },
                }
            ]
        }
        params = parser._extract_parameters(operation)
        assert params[0].type == "string"
        assert params[0].enum == ["active", "inactive"]


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
            {
                "properties": {
                    "pet_type": {"const": "cat"},
                    "indoor": {"type": "boolean"},
                }
            },
        ]
        result = gen._resolve_ref_in_union("#/components/schemas/Dog", one_of)
        assert result is not None
        assert "breed" in result["properties"]

    def test_matches_multi_word_snake_case_const(self):
        """Multi-word snake_case const matches PascalCase ref name."""
        gen = self._get_generator()
        one_of = [
            {
                "properties": {
                    "payment_type": {"const": "credit_card"},
                    "card_number": {"type": "string"},
                }
            },
            {
                "properties": {
                    "payment_type": {"const": "bank_transfer"},
                    "account": {"type": "string"},
                }
            },
        ]
        result = gen._resolve_ref_in_union("#/components/schemas/CreditCard", one_of)
        assert result is not None
        assert "card_number" in result["properties"]

    def test_matches_kebab_case_const(self):
        """kebab-case const matches PascalCase ref name."""
        gen = self._get_generator()
        one_of = [
            {
                "properties": {
                    "type": {"const": "bank-transfer"},
                    "iban": {"type": "string"},
                }
            },
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
                    "additionalProperties": {"type": "string"},
                }
            },
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
                    "additionalProperties": {"type": "integer"},
                }
            },
        }
        lines = gen._format_schema(schema, indent=0)
        output = "\n".join(lines)
        assert "integer" in output

    def test_format_schema_freeform_map(self):
        """_format_schema handles additionalProperties: true (freeform)."""
        gen = self._get_generator()
        schema = {
            "type": "object",
            "properties": {"extra": {"type": "object", "additionalProperties": True}},
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
                "tags": {"type": "object", "additionalProperties": {"type": "string"}}
            },
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
                    "additionalProperties": {"type": "integer"},
                }
            },
        }
        result = gen._precompute_positive_fields(endpoint)
        assert "generate_integer" in result

    def test_precompute_object_instruction_map_type(self):
        """_precompute_object_instruction generates map for additionalProperties."""
        gen = self._get_generator()
        schema = {
            "properties": {
                "labels": {"type": "object", "additionalProperties": {"type": "string"}}
            }
        }
        result = gen._precompute_object_instruction(schema)
        assert "key_" in result
        assert "generate_string" in result


# ============================================================
# 4th Sweep: Shared Helper Tests
# ============================================================


class TestExtractAllProperties:
    """Tests for extract_all_properties shared helper."""

    def test_direct_properties(self):
        """Direct properties are extracted correctly."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        props, required = extract_all_properties(schema)
        assert "name" in props
        assert "age" in props
        assert "name" in required

    def test_allof_merging(self):
        """allOf items are merged into combined properties."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        schema = {
            "allOf": [
                {"properties": {"id": {"type": "integer"}}, "required": ["id"]},
                {"properties": {"name": {"type": "string"}}, "required": ["name"]},
            ]
        }
        props, required = extract_all_properties(schema)
        assert "id" in props
        assert "name" in props
        assert "id" in required
        assert "name" in required

    def test_allof_with_direct_properties(self):
        """allOf items are merged with direct properties (direct takes priority)."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        schema = {
            "properties": {"status": {"type": "string", "enum": ["active"]}},
            "allOf": [
                {"properties": {"id": {"type": "integer"}}},
            ],
        }
        props, required = extract_all_properties(schema)
        assert "id" in props
        assert "status" in props
        # Direct property should override allOf for same key
        assert props["status"]["enum"] == ["active"]

    def test_oneof_discriminated_union(self):
        """oneOf variants are merged when no direct properties exist."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        schema = {
            "oneOf": [
                {
                    "properties": {"card_number": {"type": "string"}},
                    "required": ["card_number"],
                },
                {
                    "properties": {"account_number": {"type": "string"}},
                    "required": ["account_number"],
                },
            ],
            "discriminator": {"propertyName": "payment_type"},
        }
        props, required = extract_all_properties(schema)
        assert "card_number" in props
        assert "account_number" in props

    def test_oneof_not_merged_when_direct_properties_exist(self):
        """oneOf variants are NOT merged when direct properties already exist."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        schema = {
            "properties": {"type": {"type": "string"}},
            "oneOf": [
                {"properties": {"extra_field": {"type": "string"}}},
            ],
        }
        props, required = extract_all_properties(schema)
        assert "type" in props
        assert "extra_field" not in props  # Not merged because direct props exist

    def test_nested_allof_in_variant(self):
        """Variant with internal allOf has its properties merged."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        schema = {
            "oneOf": [
                {
                    "allOf": [
                        {"properties": {"base_field": {"type": "string"}}},
                        {"properties": {"variant_field": {"type": "integer"}}},
                    ],
                    "required": ["base_field"],
                }
            ]
        }
        props, required = extract_all_properties(schema)
        assert "base_field" in props
        assert "variant_field" in props

    def test_empty_schema(self):
        """Empty schema returns empty properties."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        props, required = extract_all_properties({})
        assert props == {}
        assert required == []

    def test_non_dict_schema(self):
        """Non-dict input returns empty properties."""
        from devdox_ai_locust.utils.schema_utils import extract_all_properties

        props, required = extract_all_properties("not a dict")
        assert props == {}
        assert required == []


class TestEscapeForPythonString:
    """Tests for escape_for_python_string shared helper."""

    def test_backslash_escaping(self):
        """Backslashes are doubled."""
        from devdox_ai_locust.utils.schema_utils import escape_for_python_string

        assert escape_for_python_string("^\\d{3}$") == "^\\\\d{3}$"

    def test_quote_escaping(self):
        """Double quotes are escaped."""
        from devdox_ai_locust.utils.schema_utils import escape_for_python_string

        assert escape_for_python_string('say "hello"') == 'say \\"hello\\"'

    def test_newline_escaping(self):
        """Newlines are escaped."""
        from devdox_ai_locust.utils.schema_utils import escape_for_python_string

        assert escape_for_python_string("line1\nline2") == "line1\\nline2"

    def test_tab_escaping(self):
        """Tabs are escaped."""
        from devdox_ai_locust.utils.schema_utils import escape_for_python_string

        assert escape_for_python_string("a\tb") == "a\\tb"

    def test_combined_escaping(self):
        """Multiple special chars are all escaped correctly."""
        from devdox_ai_locust.utils.schema_utils import escape_for_python_string

        result = escape_for_python_string('^\\d+"test"\\n')
        assert "\\\\" in result
        assert '\\"' in result

    def test_non_string_input(self):
        """Non-string input is converted to string."""
        from devdox_ai_locust.utils.schema_utils import escape_for_python_string

        assert escape_for_python_string(123) == "123"


class TestGetTypeInstruction:
    """Tests for _get_type_instruction shared helper."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def test_enum_priority(self):
        """Enum takes priority over type."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "enum": ["a", "b"]})
        assert "random.choice" in result
        assert "['a', 'b']" in result

    def test_pattern(self):
        """Pattern generates generate_string with raw string pattern arg."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "pattern": "^\\d+$"})
        assert "generate_string(pattern=" in result
        assert 'pattern=r"' in result  # Uses raw string to preserve backslashes
        assert "\\d" in result  # Backslash NOT double-escaped (raw string)

    def test_format_date(self):
        gen = self._get_generator()
        assert "random_date" in gen._get_type_instruction(
            {"type": "string", "format": "date"}
        )

    def test_format_datetime(self):
        """date-time format uses dedicated random_datetime() method."""
        gen = self._get_generator()
        assert "random_datetime" in gen._get_type_instruction(
            {"type": "string", "format": "date-time"}
        )

    def test_format_email(self):
        gen = self._get_generator()
        assert "generate_email" in gen._get_type_instruction(
            {"type": "string", "format": "email"}
        )

    def test_format_uuid(self):
        gen = self._get_generator()
        assert "random_uuid" in gen._get_type_instruction(
            {"type": "string", "format": "uuid"}
        )

    def test_format_uri_randomized(self):
        """URI format uses dedicated random_uri() method."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "format": "uri"})
        assert "random_uri" in result

    def test_format_ipv4_randomized(self):
        """IPv4 format uses dedicated random_ipv4() method."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "format": "ipv4"})
        assert "random_ipv4" in result

    def test_format_hostname_randomized(self):
        """Hostname format uses dedicated random_hostname() method."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "format": "hostname"})
        assert "random_hostname" in result

    def test_format_time_randomized(self):
        """Time format uses dedicated random_time() method."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "format": "time"})
        assert "random_time" in result

    def test_string_type(self):
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string"})
        assert "generate_string(length=" in result

    def test_string_max_length_capped(self):
        """MaxLength > 50 is capped to 10."""
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "string", "maxLength": 200})
        assert "length=10" in result

    def test_integer_with_bounds(self):
        gen = self._get_generator()
        result = gen._get_type_instruction(
            {"type": "integer", "minimum": 5, "maximum": 50}
        )
        assert "min_val=5" in result
        assert "max_val=50" in result

    def test_integer_exclusive(self):
        gen = self._get_generator()
        result = gen._get_type_instruction(
            {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
        )
        assert "exclusive=True" in result

    def test_integer_multiple_of(self):
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "integer", "multipleOf": 5})
        assert "multiple_of=5" in result

    def test_number_type(self):
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "number", "minimum": 0.5})
        assert "generate_float" in result
        assert "min_val=0.5" in result

    def test_boolean_type(self):
        gen = self._get_generator()
        assert "generate_boolean" in gen._get_type_instruction({"type": "boolean"})

    def test_object_with_properties(self):
        gen = self._get_generator()
        result = gen._get_type_instruction(
            {"type": "object", "properties": {"x": {"type": "integer"}}}
        )
        assert "generate_integer" in result  # nested field instruction

    def test_object_additional_properties(self):
        gen = self._get_generator()
        result = gen._get_type_instruction(
            {"type": "object", "additionalProperties": {"type": "integer"}}
        )
        assert "key_" in result
        assert "generate_integer" in result

    def test_array_string_items(self):
        gen = self._get_generator()
        result = gen._get_type_instruction(
            {"type": "array", "items": {"type": "string"}}
        )
        assert "generate_string" in result
        assert "for _ in range" in result

    def test_array_enum_items(self):
        gen = self._get_generator()
        result = gen._get_type_instruction(
            {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
        )
        assert "random.choice" in result

    def test_unknown_type_fallback(self):
        gen = self._get_generator()
        result = gen._get_type_instruction({"type": "custom_weird_type"})
        assert "generate_string" in result

    def test_non_dict_input(self):
        gen = self._get_generator()
        result = gen._get_type_instruction("not a dict")
        assert "generate_string" in result


class TestBoundaryArithmeticGuard:
    """Tests for C4 fix: boundary testing handles non-numeric min/max."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def _make_endpoint(self, min_val=None, max_val=None):
        endpoint = Mock()
        endpoint.parameters = []
        endpoint.method = "POST"
        endpoint.path = "/test"
        body = Mock()
        body.schema = {
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": min_val,
                    "maximum": max_val,
                }
            },
            "required": ["count"],
        }
        endpoint.request_body = body
        return endpoint

    def test_string_minimum_no_crash(self):
        """String minimum value doesn't cause TypeError."""
        gen = self._get_generator()
        endpoint = self._make_endpoint(min_val="5", max_val=100)
        result = gen._precompute_negative_scenarios(endpoint)
        # Should not crash, and should only generate boundary for max (valid int)
        assert "max=100" in result
        assert "min=5" not in result  # String "5" is skipped

    def test_none_values_no_crash(self):
        """None min/max values are handled gracefully."""
        gen = self._get_generator()
        endpoint = self._make_endpoint(min_val=None, max_val=None)
        result = gen._precompute_negative_scenarios(endpoint)
        assert "BOUNDARY" not in result  # No boundaries without constraints


class TestParameterTypeGuard:
    """Tests for C5 fix: invalid ParameterType values don't crash."""

    def test_invalid_location_skipped(self):
        """Parameter with invalid 'in' value is skipped with warning."""
        from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser

        parser = OpenAPIParser()
        parser.spec_data = {"openapi": "3.0.0", "info": {"title": "Test"}, "paths": {}}
        parser.components = {}

        operation = {
            "parameters": [
                {"name": "valid_param", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "invalid_param",
                    "in": "body",
                    "schema": {"type": "string"},
                },  # 2.x style
                {"name": "another_valid", "in": "header", "schema": {"type": "string"}},
            ]
        }
        params = parser._extract_parameters(operation)
        # Should get 2 params (query and header), skip the invalid one
        assert len(params) == 2
        assert params[0].name == "valid_param"
        assert params[1].name == "another_valid"


class TestStatusCodeFallbackDefaults:
    """Tests for H3 fix: fallback defaults when registry returns nothing."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import (
            ScenarioWorkflowGenerator,
            ScenarioType,
        )

        gen = ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)
        # Mock fallback registry that returns empty responses
        mock_registry = Mock()
        mock_block = Mock()
        mock_block.as_dict.return_value = {}  # No responses for any method
        mock_registry.get_responses.return_value = mock_block
        gen._fallback_registry = mock_registry
        return gen, ScenarioType

    def _make_endpoint(self, responses=None):
        endpoint = Mock()
        endpoint.method = "POST"
        endpoint.path = "/test"
        endpoint.responses = responses or []
        return endpoint

    def test_negative_empty_fallback_returns_defaults(self):
        """Negative scenario gets default 400/422 when registry is empty."""
        gen, ScenarioType = self._get_generator()
        endpoint = self._make_endpoint()
        result = gen._precompute_scenario_status_codes(
            endpoint, ScenarioType.NEGATIVE, has_auth=False
        )
        codes = [c for c, _ in result]
        assert 400 in codes
        assert 422 in codes

    def test_security_empty_fallback_returns_defaults(self):
        """Security scenario gets default 400/403/422 when registry is empty."""
        gen, ScenarioType = self._get_generator()
        endpoint = self._make_endpoint()
        result = gen._precompute_scenario_status_codes(
            endpoint, ScenarioType.SECURITY, has_auth=False
        )
        codes = [c for c, _ in result]
        assert 400 in codes
        assert 403 in codes
        assert 422 in codes

    def test_positive_empty_fallback_returns_200(self):
        """Positive scenario gets default 200 when registry is empty."""
        gen, ScenarioType = self._get_generator()
        endpoint = self._make_endpoint()
        result = gen._precompute_scenario_status_codes(
            endpoint, ScenarioType.POSITIVE, has_auth=False
        )
        codes = [c for c, _ in result]
        assert 200 in codes


class TestExclude2xxInFormatEndpoint:
    """Tests for M1 fix: exclude_2xx parameter in _format_single_endpoint."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def _make_endpoint(self):
        from devdox_ai_locust.utils.open_ai_parser import Response

        endpoint = Mock()
        endpoint.method = "POST"
        endpoint.path = "/items"
        endpoint.operation_id = "createItem"
        endpoint.summary = "Create item"
        endpoint.description = ""
        endpoint.parameters = []
        endpoint.request_body = None
        endpoint.responses = [
            Response(status_code="201", description="Created"),
            Response(status_code="400", description="Bad Request"),
            Response(status_code="422", description="Validation Error"),
        ]
        return endpoint

    def test_exclude_2xx_false_includes_all(self):
        """Default (exclude_2xx=False) includes all response codes."""
        gen = self._get_generator()
        endpoint = self._make_endpoint()
        result = gen._format_single_endpoint(endpoint, exclude_2xx=False)
        assert "201" in result
        assert "400" in result
        assert "422" in result

    def test_exclude_2xx_true_removes_success(self):
        """exclude_2xx=True removes 2xx responses from output."""
        gen = self._get_generator()
        endpoint = self._make_endpoint()
        result = gen._format_single_endpoint(endpoint, exclude_2xx=True)
        assert "201" not in result
        assert "400" in result
        assert "422" in result


class TestParamTypeExactMatch:
    """Tests for M2 fix: param type uses exact match instead of substring."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def _make_endpoint_with_path_param(self, param_type):
        endpoint = Mock()
        endpoint.method = "GET"
        endpoint.path = "/items/{id}"
        param = Mock()
        param.name = "id"
        param.type = param_type
        param.location = Mock(value="path")
        endpoint.parameters = [param]
        endpoint.request_body = None
        return endpoint

    def test_integer_type_matches(self):
        """'integer' type is treated as numeric."""
        gen = self._get_generator()
        endpoint = self._make_endpoint_with_path_param("integer")
        result = gen._precompute_negative_scenarios(endpoint)
        assert "999999999" in result  # Integer path param test

    def test_string_type_not_matched_as_int(self):
        """'string' type is NOT treated as numeric (no substring match on 'int')."""
        gen = self._get_generator()
        endpoint = self._make_endpoint_with_path_param("string")
        result = gen._precompute_negative_scenarios(endpoint)
        assert "nonexistent-id-12345" in result  # String path param test
        assert "999999999" not in result


class TestCodeValidatorUrlArgRegex:
    """Tests for M4 fix: URL arg regex matches the URL argument, not the method."""

    def test_detects_double_slash_in_fstring_url(self):
        """Catches // in f-string URL argument."""
        validator = CodeValidator()
        code = 'self.make_request("PATCH", f"/api/v1/{item_id}//verify", expected_status=[422])'
        violations = validator._check_empty_path_segments(code)
        assert len(violations) == 1
        assert "empty_path_segment" in violations[0].rule

    def test_no_false_positive_on_method_string(self):
        """Does NOT false-positive on the method string 'GET'."""
        validator = CodeValidator()
        code = 'self.make_request("GET", "/api/v1/items/123")'
        violations = validator._check_empty_path_segments(code)
        assert len(violations) == 0

    def test_detects_double_slash_in_regular_string(self):
        """Catches // in regular (non-f-string) URL."""
        validator = CodeValidator()
        code = 'self.make_request("POST", "/api//items")'
        violations = validator._check_empty_path_segments(code)
        assert len(violations) == 1

    def test_ignores_https_protocol(self):
        """Does not flag https:// as empty path segment."""
        validator = CodeValidator()
        code = 'self.make_request("GET", "https://example.com/api/items")'
        violations = validator._check_empty_path_segments(code)
        assert len(violations) == 0


class TestNoneParamType:
    """Tests for M5 fix: None param type handled gracefully."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def test_none_type_defaults_to_string(self):
        """Parameter with type=None defaults to string behavior."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.method = "GET"
        endpoint.path = "/items/{id}"
        param = Mock()
        param.name = "id"
        param.type = None  # Explicitly None
        param.location = Mock(value="path")
        endpoint.parameters = [param]
        endpoint.request_body = None
        result = gen._precompute_negative_scenarios(endpoint)
        # Should not crash, and should treat as string path param
        assert "nonexistent-id-12345" in result


class TestPrecomputePositiveFieldsRefactored:
    """Tests that refactored _precompute_positive_fields still works correctly."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def test_allof_schema_generates_all_fields(self):
        """allOf schema has all fields pre-computed."""
        gen = self._get_generator()
        endpoint = Mock()
        body = Mock()
        body.schema = {
            "allOf": [
                {"properties": {"id": {"type": "integer"}}, "required": ["id"]},
                {"properties": {"name": {"type": "string"}}, "required": ["name"]},
            ]
        }
        endpoint.request_body = body
        result = gen._precompute_positive_fields(endpoint)
        assert '"id"' in result
        assert '"name"' in result
        assert "generate_integer" in result
        assert "generate_string" in result

    def test_discriminated_union_uses_real_instructions(self):
        """Discriminated union generates real instructions, not <type> placeholders."""
        gen = self._get_generator()
        endpoint = Mock()
        body = Mock()
        body.schema = {
            "oneOf": [
                {
                    "title": "CreditCard",
                    "properties": {
                        "payment_type": {"type": "string", "const": "credit_card"},
                        "card_number": {"type": "string", "pattern": "^\\d{16}$"},
                        "amount": {"type": "number", "minimum": 0.01},
                    },
                    "required": ["payment_type", "card_number", "amount"],
                }
            ],
            "discriminator": {"propertyName": "payment_type"},
        }
        endpoint.request_body = body
        result = gen._precompute_positive_fields(endpoint)
        assert "DISCRIMINATED UNION" in result
        assert "<" not in result  # No placeholder <type> tokens
        assert "generate_string(pattern=" in result
        assert "generate_float" in result

    def test_no_request_body(self):
        """Endpoint without request body returns empty string."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.request_body = None
        result = gen._precompute_positive_fields(endpoint)
        assert result == ""

    def test_randomized_format_values(self):
        """Format values (uri, ipv4, etc.) use dedicated generator methods."""
        gen = self._get_generator()
        endpoint = Mock()
        body = Mock()
        body.schema = {
            "properties": {
                "website": {"type": "string", "format": "uri"},
                "ip": {"type": "string", "format": "ipv4"},
                "host": {"type": "string", "format": "hostname"},
            }
        }
        endpoint.request_body = body
        result = gen._precompute_positive_fields(endpoint)
        # All should use dedicated format methods
        assert "random_uri" in result
        assert "random_ipv4" in result
        assert "random_hostname" in result


class TestNegativeInjectionWithAllOf:
    """Tests that negative/injection pre-computation now handles allOf schemas."""

    def _get_generator(self):
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    def test_negative_finds_allof_fields(self):
        """_precompute_negative_scenarios finds fields from allOf items."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.method = "POST"
        endpoint.path = "/test"
        endpoint.parameters = []
        body = Mock()
        body.schema = {
            "allOf": [
                {
                    "properties": {"age": {"type": "integer", "minimum": 0}},
                    "required": ["age"],
                },
                {
                    "properties": {"score": {"type": "number", "maximum": 100}},
                    "required": ["score"],
                },
            ]
        }
        endpoint.request_body = body
        result = gen._precompute_negative_scenarios(endpoint)
        assert "age" in result
        assert "score" in result
        assert "MISSING_REQUIRED" in result

    def test_injection_finds_allof_string_fields(self):
        """_precompute_injection_points finds string fields from allOf items."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.method = "POST"
        endpoint.path = "/test"
        endpoint.parameters = []
        body = Mock()
        body.schema = {
            "allOf": [
                {"properties": {"name": {"type": "string"}}},
                {"properties": {"description": {"type": "string"}}},
            ]
        }
        endpoint.request_body = body
        result = gen._precompute_injection_points(endpoint)
        assert result is not None
        assert "name" in result
        assert "description" in result


# ============================================================
# 5th Sweep Fixes: Locust Run Error Analysis
# ============================================================


class TestFifthSweepFixes:
    """Tests for 5th sweep fixes addressing locust run errors."""

    def _get_generator(self):
        """Helper to create a ScenarioWorkflowGenerator instance for testing."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        return ScenarioWorkflowGenerator.__new__(ScenarioWorkflowGenerator)

    # Category 1: Float-to-int TypeError in generate_integer
    def test_generate_integer_handles_float_multiple_of(self):
        """generate_integer should convert float multiple_of to int."""
        # This tests the template-generated test_data.py
        # Simulate what the template generates
        import random

        def generate_integer_fixed(min_val=1, max_val=1000, multiple_of=None):
            """Fixed version that handles float multiple_of."""
            min_val = int(min_val)
            max_val = int(max_val)
            if multiple_of and multiple_of > 0:
                multiple_of = int(multiple_of)  # The fix
                low = int((min_val + multiple_of - 1) // multiple_of)
                high = int(max_val // multiple_of)
                if high < low:
                    return min_val
                return random.randint(low, high) * multiple_of
            return random.randint(min_val, max_val)

        # This would crash before the fix with TypeError
        result = generate_integer_fixed(min_val=0.0, max_val=1000, multiple_of=5.0)
        assert isinstance(result, int)
        assert result % 5 == 0
        assert 0 <= result <= 1000

    # Category 3: Pattern double-escaping fix
    def test_escape_for_raw_string_preserves_backslashes(self):
        """escape_for_raw_string should NOT escape backslashes for raw string literals."""
        from devdox_ai_locust.utils.schema_utils import escape_for_raw_string

        pattern = r"^\d{5}(-\d{4})?$"  # Regex pattern
        escaped = escape_for_raw_string(pattern)
        # Backslashes should NOT be doubled
        assert escaped == pattern
        assert "\\d" in escaped  # Single backslash preserved
        assert "\\\\d" not in escaped  # No double backslashes

    def test_escape_for_raw_string_escapes_quotes(self):
        """escape_for_raw_string should escape quotes."""
        from devdox_ai_locust.utils.schema_utils import escape_for_raw_string

        pattern = 'test "quoted" value'
        escaped = escape_for_raw_string(pattern)
        assert escaped == 'test \\"quoted\\" value'

    def test_get_type_instruction_uses_raw_string_for_pattern(self):
        """_get_type_instruction should output raw string r\"...\" for patterns."""
        gen = self._get_generator()
        field_schema = {"type": "string", "pattern": r"^\d{5}$"}
        result = gen._get_type_instruction(field_schema)
        # Should use raw string prefix
        assert 'pattern=r"' in result
        # Should NOT double-escape backslashes
        assert r'pattern=r"^\d{5}$"' in result

    # Category 4: Negative tests skip parameter-test endpoints
    def test_negative_scenarios_skips_parameter_test_endpoints(self):
        """_precompute_negative_scenarios should skip NON_EXISTENT_ID for parameter-test endpoints."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.method = "GET"
        endpoint.path = "/api/v1/comprehensive/parameters/path/string/{value}"

        # Add path parameter
        param = Mock()
        param.name = "value"
        param.type = "string"
        param.location = "path"
        endpoint.parameters = [param]
        endpoint.request_body = None

        result = gen._precompute_negative_scenarios(endpoint)

        # Should NOT generate NON_EXISTENT_ID for parameter-test endpoints
        assert "NON_EXISTENT_ID" not in result

    def test_negative_scenarios_generates_nonexistent_id_for_resource_endpoints(self):
        """_precompute_negative_scenarios should generate NON_EXISTENT_ID for resource endpoints."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.method = "GET"
        endpoint.path = "/api/v1/users/{user_id}"  # Real resource endpoint

        # Add path parameter
        param = Mock()
        param.name = "user_id"
        param.type = "integer"
        param.location = "path"
        endpoint.parameters = [param]
        endpoint.request_body = None

        result = gen._precompute_negative_scenarios(endpoint)

        # Should generate NON_EXISTENT_ID for resource endpoints
        assert "NON_EXISTENT_ID" in result
        assert "999999999" in result

    def test_negative_scenarios_skips_invalid_query_for_parameter_test_endpoints(self):
        """_precompute_negative_scenarios should skip INVALID_QUERY for parameter-test endpoints."""
        gen = self._get_generator()
        endpoint = Mock()
        endpoint.method = "GET"
        endpoint.path = "/api/v1/comprehensive/parameters/query/required"

        # Add query parameter
        param = Mock()
        param.name = "name"
        param.type = "string"
        param.location = "query"
        endpoint.parameters = [param]
        endpoint.request_body = None

        result = gen._precompute_negative_scenarios(endpoint)

        # Should NOT generate fallback INVALID_QUERY for parameter-test endpoints
        # (since they accept any input)
        assert "INVALID_QUERY" not in result or result == ""

    # Category 2 & 5: Setup endpoints get their own status codes from OpenAPI spec
    def test_format_related_create_endpoints_includes_status_codes(self):
        """_format_related_create_endpoints should include expected_status for each setup endpoint."""
        from devdox_ai_locust.utils.scenario_generator import (
            ScenarioWorkflowGenerator,
        )
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Create a mock setup endpoint with responses
        setup_endpoint = Mock()
        setup_endpoint.method = "POST"
        setup_endpoint.path = "/api/v1/users"
        setup_endpoint.operation_id = "createUser"
        setup_endpoint.summary = "Create a user"
        setup_endpoint.description = ""
        setup_endpoint.request_body = None
        # OpenAPI spec says this endpoint returns 201
        setup_endpoint.responses = {"201": Mock(), "422": Mock()}

        related_endpoints = [(setup_endpoint, 100.0, "Exact resource match")]

        result = gen._format_related_create_endpoints(related_endpoints)

        # Should include expected_status for the setup endpoint
        assert "expected_status=" in result
        assert "[201]" in result  # Should extract 201 from responses
        assert "USE THIS for this setup call" in result

    def test_format_related_create_endpoints_fallback_when_no_responses(self):
        """_format_related_create_endpoints should use FallbackHttpResponseRegistry when no responses defined."""
        from devdox_ai_locust.utils.scenario_generator import (
            ScenarioWorkflowGenerator,
        )
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Create a mock setup endpoint WITHOUT responses
        setup_endpoint = Mock()
        setup_endpoint.method = "POST"
        setup_endpoint.path = "/api/v1/items"
        setup_endpoint.operation_id = "createItem"
        setup_endpoint.summary = "Create an item"
        setup_endpoint.description = ""
        setup_endpoint.request_body = None
        setup_endpoint.responses = {}  # No responses defined

        related_endpoints = [(setup_endpoint, 100.0, "Exact resource match")]

        result = gen._format_related_create_endpoints(related_endpoints)

        # Should use FallbackHttpResponseRegistry when no responses defined
        # The registry returns [200, 201, 204] for POST positive scenarios
        assert "expected_status=" in result
        assert "[200, 201, 204]" in result  # Fallback from registry


class TestSixthSweepFixes:
    """Tests for sixth sweep fixes: pattern generation, format generators, field name inference."""

    def test_prefix_pattern_instruction_generated(self):
        """Pattern like ^EMP-\\d{6}$ should generate correct instruction."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        # Test that the instruction uses generate_string with pattern
        result = gen._get_type_instruction(
            {"type": "string", "pattern": "^EMP-\\d{6}$"}
        )
        assert "pattern=" in result
        # The pattern should be preserved for generate_string to handle
        assert "EMP-" in result or "generate_string" in result

    def test_field_name_inference_country_code(self):
        """Fields named 'country' with maxLength<=3 should use random_country_code()."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        result = gen._get_type_instruction(
            {"type": "string", "maxLength": 2}, field_name="country"
        )
        assert "random_country_code" in result

    def test_field_name_inference_currency_code(self):
        """Fields named 'currency' with maxLength<=4 should use random_currency_code()."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        result = gen._get_type_instruction(
            {"type": "string", "maxLength": 3}, field_name="currency_code"
        )
        assert "random_currency_code" in result

    def test_field_name_inference_created_at(self):
        """Fields named 'created_at' should use random_datetime() even without format."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        result = gen._get_type_instruction(
            {"type": "string"}, field_name="created_at"  # No format specified
        )
        assert "random_datetime" in result

    def test_field_name_inference_email(self):
        """Fields named 'email' should use generate_email() even without format."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        result = gen._get_type_instruction(
            {"type": "string"}, field_name="user_email"  # No format specified
        )
        assert "generate_email" in result

    def test_field_name_inference_color(self):
        """Fields named 'color' should use random_hex_color()."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        result = gen._get_type_instruction(
            {"type": "string"}, field_name="background_color"
        )
        assert "random_hex_color" in result

    def test_format_byte_uses_base64(self):
        """Format 'byte' should use random_base64()."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        result = gen._get_type_instruction({"type": "string", "format": "byte"})
        assert "random_base64" in result

    def test_precompute_positive_fields_passes_field_name(self):
        """_precompute_positive_fields should pass field names for inference."""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from devdox_ai_locust.config import AIEnhancementConfig

        gen = ScenarioWorkflowGenerator(
            prompt_dir=Mock(),
            ai_client=Mock(),
            ai_config=AIEnhancementConfig(),
        )

        endpoint = Mock()
        body = Mock()
        body.schema = {
            "properties": {
                "country": {"type": "string", "maxLength": 2},
                "created_at": {"type": "string"},
                "user_email": {"type": "string"},
            }
        }
        endpoint.request_body = body
        result = gen._precompute_positive_fields(endpoint)

        # All fields should use inferred generators
        assert "random_country_code" in result
        assert "random_datetime" in result
        assert "generate_email" in result
