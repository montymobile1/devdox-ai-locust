"""
Tests for type_instruction module
"""

import pytest

from devdox_ai_locust.utils.type_instruction import (
    get_format_instruction,
    get_string_instruction_by_name,
    get_string_instruction,
    get_integer_instruction,
    get_number_instruction,
    get_object_instruction,
    get_array_instruction,
    _compute_array_length,
    _get_array_items_info,
    _get_nested_array_instruction,
)


class TestGetFormatInstruction:
    """Test get_format_instruction function."""

    @pytest.mark.parametrize(
        "fmt, expected_fragment",
        [
            ("date", "random_date()"),
            ("date-time", "random_datetime()"),
            ("time", "random_time()"),
            ("email", "generate_email()"),
            ("uuid", "random_uuid()"),
            ("uri", "random_uri()"),
            ("url", "random_uri()"),
            ("ipv4", "random_ipv4()"),
            ("ipv6", "random_ipv6()"),
            ("hostname", "random_hostname()"),
            ("byte", "random_base64()"),
        ],
    )
    def test_known_formats(self, fmt, expected_fragment):
        """Test all known format strings return the correct generator."""
        result = get_format_instruction(fmt)
        assert result is not None
        assert expected_fragment in result

    def test_unknown_format_returns_none(self):
        """Test that an unrecognized format returns None."""
        assert get_format_instruction("binary") is None

    def test_empty_string_returns_none(self):
        """Test that an empty string returns None."""
        assert get_format_instruction("") is None


class TestGetStringInstructionByName:
    """Test get_string_instruction_by_name function."""

    def test_empty_name_returns_none(self):
        assert get_string_instruction_by_name("", 50) is None

    def test_country_code_short(self):
        result = get_string_instruction_by_name("country_code", 2)
        assert "random_country_code()" in result

    def test_country_code_long_max_length(self):
        """country with maxLength > 3 should not match."""
        result = get_string_instruction_by_name("country_code", 50)
        assert result is None

    def test_currency_code(self):
        result = get_string_instruction_by_name("currency", 3)
        assert "random_currency_code()" in result

    def test_locale_field(self):
        result = get_string_instruction_by_name("user_locale", 50)
        assert "random_locale()" in result

    def test_color_field(self):
        result = get_string_instruction_by_name("background_color", 50)
        assert "random_hex_color()" in result

    def test_colour_field(self):
        result = get_string_instruction_by_name("bg_colour", 50)
        assert "random_hex_color()" in result

    def test_timestamp_field(self):
        result = get_string_instruction_by_name("created_at", 50)
        assert "random_datetime()" in result

    def test_date_field(self):
        result = get_string_instruction_by_name("birth_date", 50)
        assert "random_date()" in result

    def test_date_field_with_time_excluded(self):
        """Fields containing both 'date' and 'time' should not match date keywords."""
        result = get_string_instruction_by_name("datetime_value", 50)
        # "datetime" contains "time", so date branch is skipped
        assert result is None or "random_date()" not in result

    def test_email_field(self):
        result = get_string_instruction_by_name("user_email", 50)
        assert "generate_email()" in result

    def test_url_field(self):
        result = get_string_instruction_by_name("website_url", 50)
        assert "random_uri()" in result

    def test_phone_field(self):
        result = get_string_instruction_by_name("phone_number", 50)
        assert "random_ipv4()" in result

    def test_ipv4_field(self):
        result = get_string_instruction_by_name("ip_address", 50)
        assert "random_ipv4()" in result

    def test_ipv6_field(self):
        result = get_string_instruction_by_name("ipv6_address", 50)
        assert "random_ipv6()" in result

    def test_hostname_field(self):
        result = get_string_instruction_by_name("hostname", 50)
        assert "random_hostname()" in result

    def test_no_match_returns_none(self):
        result = get_string_instruction_by_name("foobar", 50)
        assert result is None


class TestGetStringInstruction:
    """Test get_string_instruction function."""

    def test_name_inference_takes_priority(self):
        result = get_string_instruction({"maxLength": 50}, field_name="user_email")
        assert "generate_email()" in result

    def test_default_with_max_length(self):
        result = get_string_instruction({"maxLength": 20})
        assert "generate_string(length=20)" in result

    def test_default_large_max_length_capped(self):
        """maxLength > 50 should fall back to length=10."""
        result = get_string_instruction({"maxLength": 200})
        assert "generate_string(length=10)" in result

    def test_default_no_max_length(self):
        result = get_string_instruction({})
        assert "generate_string(length=50)" in result

    def test_empty_field_name(self):
        result = get_string_instruction({"maxLength": 5}, field_name="")
        assert "generate_string(length=5)" in result


class TestGetIntegerInstruction:
    """Test get_integer_instruction function."""

    def test_defaults(self):
        result = get_integer_instruction({})
        assert "min_val=1" in result
        assert "max_val=1000" in result
        assert "exclusive" not in result

    def test_min_max(self):
        result = get_integer_instruction({"minimum": 5, "maximum": 100})
        assert "min_val=5" in result
        assert "max_val=100" in result

    def test_exclusive_minimum(self):
        result = get_integer_instruction({"exclusiveMinimum": 0})
        assert "min_val=0" in result
        assert "exclusive=True" in result

    def test_exclusive_maximum(self):
        result = get_integer_instruction({"exclusiveMaximum": 50})
        assert "max_val=50" in result
        assert "exclusive=True" in result

    def test_multiple_of(self):
        result = get_integer_instruction({"multipleOf": 5})
        assert "multiple_of=5" in result

    def test_rgb_field_name(self):
        result = get_integer_instruction({}, field_name="rgb_red")
        assert "min_val=0" in result
        assert "max_val=255" in result

    def test_color_field_name(self):
        result = get_integer_instruction({}, field_name="color_value")
        assert "max_val=255" in result


class TestGetNumberInstruction:
    """Test get_number_instruction function."""

    def test_defaults(self):
        result = get_number_instruction({})
        assert "min_val=0.0" in result
        assert "max_val=1000.0" in result
        assert "exclusive" not in result

    def test_min_max(self):
        result = get_number_instruction({"minimum": 1.5, "maximum": 99.9})
        assert "min_val=1.5" in result
        assert "max_val=99.9" in result

    def test_exclusive(self):
        result = get_number_instruction({"exclusiveMinimum": 0, "exclusiveMaximum": 1})
        assert "exclusive=True" in result


class TestGetObjectInstruction:
    """Test get_object_instruction function."""

    def test_with_properties_calls_callback(self):
        schema = {"properties": {"name": {"type": "string"}}}

        def callback(s, a):
            return '{"name": "test"}'

        result = get_object_instruction(schema, callback)
        assert '"name"' in result

    def test_additional_properties_integer(self):
        schema = {"additionalProperties": {"type": "integer"}}

        def callback(s, a):
            return "{}"

        result = get_object_instruction(schema, callback)
        assert "generate_integer()" in result

    def test_additional_properties_boolean(self):
        schema = {"additionalProperties": {"type": "boolean"}}

        def callback(s, a):
            return "{}"

        result = get_object_instruction(schema, callback)
        assert "generate_boolean()" in result

    def test_additional_properties_true(self):
        schema = {"additionalProperties": True}

        def callback(s, a):
            return "{}"

        result = get_object_instruction(schema, callback)
        assert result == '{"key1": "value1", "key2": "value2"}'

    def test_empty_schema(self):
        def callback(s, a):
            return "{}"

        result = get_object_instruction({}, callback)
        assert result == "{}"


class TestComputeArrayLength:
    """Test _compute_array_length helper."""

    def test_default(self):
        assert _compute_array_length({}) == 3

    def test_min_items_greater_than_one(self):
        assert _compute_array_length({"minItems": 5}) == 5

    def test_max_items_caps_length(self):
        assert _compute_array_length({"minItems": 5, "maxItems": 4}) == 4

    def test_max_items_alone(self):
        assert _compute_array_length({"maxItems": 2}) == 2


class TestGetArrayItemsInfo:
    """Test _get_array_items_info helper."""

    def test_non_dict_returns_defaults(self):
        info = _get_array_items_info("string")
        assert info["type"] == "string"
        assert info["enum"] is None

    def test_dict_extracts_fields(self):
        info = _get_array_items_info({"type": "integer", "enum": [1, 2, 3]})
        assert info["type"] == "integer"
        assert info["enum"] == [1, 2, 3]


class TestGetArrayInstruction:
    """Test get_array_instruction function."""

    def _noop_callback(self, schema, ancestors=None):
        return '{"key": "val"}'

    def test_enum_items(self):
        schema = {"items": {"type": "string", "enum": ["a", "b", "c"]}}
        result = get_array_instruction(schema, self._noop_callback)
        assert "random.choice" in result

    def test_object_items_with_properties(self):
        schema = {
            "items": {"type": "object", "properties": {"id": {"type": "integer"}}}
        }
        result = get_array_instruction(schema, self._noop_callback)
        assert '{"key": "val"}' in result

    def test_object_items_without_properties(self):
        schema = {"items": {"type": "object"}}
        result = get_array_instruction(schema, self._noop_callback)
        assert result == "[{}]"

    def test_string_items(self):
        schema = {"items": {"type": "string"}}
        result = get_array_instruction(schema, self._noop_callback)
        assert "generate_string()" in result

    def test_integer_items(self):
        schema = {"items": {"type": "integer"}}
        result = get_array_instruction(schema, self._noop_callback)
        assert "generate_integer()" in result

    def test_integer_items_rgb_field(self):
        schema = {"items": {"type": "integer"}}
        result = get_array_instruction(
            schema, self._noop_callback, field_name="rgb_values"
        )
        assert "min_val=0, max_val=255" in result

    def test_boolean_items(self):
        schema = {"items": {"type": "boolean"}}
        result = get_array_instruction(schema, self._noop_callback)
        assert "generate_boolean()" in result

    def test_nested_array_items(self):
        schema = {"items": {"type": "array", "items": {"type": "integer"}}}
        result = get_array_instruction(schema, self._noop_callback)
        assert "generate_integer()" in result
        assert "range(2)" in result

    def test_one_of_items(self):
        schema = {
            "items": {
                "oneOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]
            }
        }
        result = get_array_instruction(schema, self._noop_callback)
        assert "for _ in range" in result


class TestGetNestedArrayInstruction:
    """Test _get_nested_array_instruction helper."""

    def test_inner_string(self):
        result = _get_nested_array_instruction({"items": {"type": "string"}}, 3)
        assert "generate_string()" in result
        assert "range(3)" in result

    def test_inner_number(self):
        result = _get_nested_array_instruction({"items": {"type": "number"}}, 2)
        assert "generate_float()" in result
