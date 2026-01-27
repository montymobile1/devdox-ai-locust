"""
Tests for schema_utils module
"""

import pytest

from devdox_ai_locust.utils.schema_utils import (
    extract_all_properties,
    unwrap_nullable_schema,
    escape_for_raw_string,
    escape_for_python_string,
    resolve_ref_in_union,
    get_schema_type,
    is_required_field,
    get_field_constraints,
)


class TestUnwrapNullableSchema:
    """Test unwrap_nullable_schema function."""

    def test_non_dict_returns_unchanged(self):
        """Non-dict input is returned as-is with False."""
        result, nullable = unwrap_nullable_schema("not a dict")
        assert result == "not a dict"
        assert nullable is False

    def test_openapi_30_nullable(self):
        """OpenAPI 3.0 nullable: true pattern."""
        schema = {"type": "string", "nullable": True}
        result, nullable = unwrap_nullable_schema(schema)
        assert result == schema
        assert nullable is True

    def test_openapi_31_anyof_nullable(self):
        """OpenAPI 3.1 anyOf with null type."""
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result, nullable = unwrap_nullable_schema(schema)
        assert result == {"type": "string"}
        assert nullable is True

    def test_openapi_31_oneof_nullable(self):
        """OpenAPI 3.1 oneOf with null type."""
        schema = {"oneOf": [{"type": "integer"}, {"type": "null"}]}
        result, nullable = unwrap_nullable_schema(schema)
        assert result == {"type": "integer"}
        assert nullable is True

    def test_non_nullable_schema(self):
        """Plain schema without nullable markers."""
        schema = {"type": "string"}
        result, nullable = unwrap_nullable_schema(schema)
        assert result == schema
        assert nullable is False

    def test_anyof_multiple_real_variants_not_unwrapped(self):
        """anyOf with multiple non-null variants is not unwrapped."""
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
        result, nullable = unwrap_nullable_schema(schema)
        # Two real variants, so not unwrapped
        assert result == schema
        assert nullable is False

    def test_empty_dict(self):
        """Empty dict returns unchanged."""
        result, nullable = unwrap_nullable_schema({})
        assert result == {}
        assert nullable is False


class TestExtractAllProperties:
    """Test extract_all_properties function."""

    def test_non_dict_returns_empty(self):
        result, required = extract_all_properties("bad")
        assert result == {}
        assert required == []

    def test_simple_properties(self):
        schema = {
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        props, req = extract_all_properties(schema)
        assert "name" in props
        assert "age" in props
        assert "name" in req

    def test_allof_merges_properties(self):
        schema = {
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"properties": {"b": {"type": "integer"}}},
            ]
        }
        props, req = extract_all_properties(schema)
        assert "a" in props
        assert "b" in props
        assert "a" in req

    def test_nested_allof(self):
        schema = {
            "allOf": [
                {
                    "allOf": [
                        {
                            "properties": {"deep": {"type": "boolean"}},
                            "required": ["deep"],
                        }
                    ]
                }
            ]
        }
        props, req = extract_all_properties(schema)
        assert "deep" in props
        assert "deep" in req

    def test_oneof_union_when_no_direct_properties(self):
        schema = {
            "oneOf": [
                {"properties": {"x": {"type": "string"}}, "required": ["x"]},
                {"properties": {"y": {"type": "integer"}}},
            ]
        }
        props, req = extract_all_properties(schema)
        assert "x" in props
        assert "y" in props
        assert "x" in req

    def test_oneof_skipped_when_properties_exist(self):
        """oneOf is not merged if schema already has direct properties."""
        schema = {
            "properties": {"direct": {"type": "string"}},
            "oneOf": [
                {"properties": {"variant": {"type": "integer"}}},
            ],
        }
        props, req = extract_all_properties(schema)
        assert "direct" in props
        assert "variant" not in props

    def test_required_deduplication(self):
        schema = {
            "required": ["a"],
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            ],
        }
        props, req = extract_all_properties(schema)
        assert req.count("a") == 1


class TestEscapeForRawString:
    """Test escape_for_raw_string function."""

    def test_no_escaping_needed(self):
        assert escape_for_raw_string("hello") == "hello"

    def test_quotes_escaped(self):
        assert escape_for_raw_string('say "hi"') == 'say \\"hi\\"'

    def test_backslashes_not_escaped(self):
        assert escape_for_raw_string("a\\b") == "a\\b"

    def test_non_string_input(self):
        assert escape_for_raw_string(42) == "42"


class TestEscapeForPythonString:
    """Test escape_for_python_string function."""

    def test_backslash_escaped(self):
        assert escape_for_python_string("a\\b") == "a\\\\b"

    def test_quotes_escaped(self):
        assert escape_for_python_string('"hi"') == '\\"hi\\"'

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("\n", "\\n"),
            ("\r", "\\r"),
            ("\t", "\\t"),
        ],
    )
    def test_control_chars(self, input_str, expected):
        assert escape_for_python_string(input_str) == expected

    def test_non_string_input(self):
        assert escape_for_python_string(123) == "123"


class TestResolveRefInUnion:
    """Test resolve_ref_in_union function."""

    def test_matches_by_ref(self):
        one_of = [
            {"$ref": "#/components/schemas/Dog"},
            {"$ref": "#/components/schemas/Cat"},
        ]
        result = resolve_ref_in_union("#/components/schemas/Dog", one_of)
        assert result == {"$ref": "#/components/schemas/Dog"}

    def test_matches_by_title(self):
        one_of = [{"title": "Dog", "type": "object"}]
        result = resolve_ref_in_union("#/components/schemas/Dog", one_of)
        assert result == {"title": "Dog", "type": "object"}

    def test_no_match_returns_none(self):
        one_of = [{"$ref": "#/components/schemas/Cat"}]
        result = resolve_ref_in_union("#/components/schemas/Dog", one_of)
        assert result is None

    def test_empty_list(self):
        assert resolve_ref_in_union("ref", []) is None

    def test_none_list(self):
        assert resolve_ref_in_union("ref", None) is None


class TestGetSchemaType:
    """Test get_schema_type function."""

    def test_simple_type(self):
        assert get_schema_type({"type": "string"}) == "string"

    def test_nullable_unwrapped(self):
        schema = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        assert get_schema_type(schema) == "integer"

    def test_non_dict(self):
        assert get_schema_type("bad") == "unknown"

    def test_missing_type_defaults_to_object(self):
        assert get_schema_type({}) == "object"


class TestIsRequiredField:
    """Test is_required_field function."""

    def test_field_is_required(self):
        assert is_required_field("name", ["name", "age"]) is True

    def test_field_not_required(self):
        assert is_required_field("email", ["name"]) is False


class TestGetFieldConstraints:
    """Test get_field_constraints function."""

    def test_extracts_known_constraints(self):
        schema = {
            "minLength": 1,
            "maxLength": 100,
            "pattern": "^[a-z]+$",
            "type": "string",
        }
        result = get_field_constraints(schema)
        assert result == {"minLength": 1, "maxLength": 100, "pattern": "^[a-z]+$"}

    def test_empty_schema(self):
        assert get_field_constraints({}) == {}

    def test_enum_constraint(self):
        schema = {"enum": ["a", "b"], "type": "string"}
        result = get_field_constraints(schema)
        assert result == {"enum": ["a", "b"]}

    def test_numeric_constraints(self):
        schema = {"minimum": 0, "maximum": 100, "type": "integer"}
        result = get_field_constraints(schema)
        assert result == {"minimum": 0, "maximum": 100}
