"""
Type Instruction Module

Maps OpenAPI schema types to Python code instructions for test data generation.
Used by ScenarioGenerator to produce valid test values.
"""

from typing import Any, Callable, Dict, Optional

from devdox_ai_locust.utils.constants import (
    TYPE_ARRAY,
    TYPE_BOOLEAN,
    TYPE_INTEGER,
    TYPE_NUMBER,
    TYPE_OBJECT,
    TYPE_STRING,
)
from devdox_ai_locust.utils.schema_utils import (
    extract_all_properties,
)

# Generator instruction constants for commonly used generators
_GEN_RANDOM_URI = "test_data_generator.random_uri()"
_GEN_RANDOM_IPV4 = "test_data_generator.random_ipv4()"


def get_format_instruction(field_format: str) -> Optional[str]:
    """Get instruction for format-specific fields.

    Args:
        field_format: OpenAPI format string (date, email, uuid, etc.)

    Returns:
        Python code instruction or None if format not recognized
    """
    format_map = {
        "date": "test_data_generator.random_date()",
        "date-time": "test_data_generator.random_datetime()",
        "time": "test_data_generator.random_time()",
        "email": "test_data_generator.generate_email()",
        "uuid": "test_data_generator.random_uuid()",
        "uri": _GEN_RANDOM_URI,
        "url": _GEN_RANDOM_URI,
        "ipv4": _GEN_RANDOM_IPV4,
        "ipv6": "test_data_generator.random_ipv6()",
        "hostname": "test_data_generator.random_hostname()",
        "byte": "test_data_generator.random_base64()",
    }
    return format_map.get(field_format)


# Field name patterns and their corresponding generators
_TIMESTAMP_KEYWORDS = ("created_at", "updated_at", "modified_at", "timestamp")
_DATE_KEYWORDS = ("birth_date", "start_date", "end_date", "date")
_URL_KEYWORDS = ("url", "uri", "website", "link")


def _check_code_field(field_name_lower: str, max_length: int) -> Optional[str]:
    """Check for country/currency code fields."""
    if "country" in field_name_lower and max_length <= 3:
        return "test_data_generator.random_country_code()"
    if "currency" in field_name_lower and max_length <= 4:
        return "test_data_generator.random_currency_code()"
    return None


def _check_datetime_field(field_name_lower: str) -> Optional[str]:
    """Check for date/time related fields."""
    if any(kw in field_name_lower for kw in _TIMESTAMP_KEYWORDS):
        return "test_data_generator.random_datetime()"
    if (
        any(kw in field_name_lower for kw in _DATE_KEYWORDS)
        and "time" not in field_name_lower
    ):
        return "test_data_generator.random_date()"
    return None


def _check_network_field(field_name_lower: str) -> Optional[str]:
    """Check for network-related fields (IP, hostname, URL)."""
    if any(kw in field_name_lower for kw in _URL_KEYWORDS):
        return _GEN_RANDOM_URI
    if "ipv4" in field_name_lower or "ip_address" in field_name_lower:
        return _GEN_RANDOM_IPV4
    if "ipv6" in field_name_lower:
        return "test_data_generator.random_ipv6()"
    if "hostname" in field_name_lower or "host" in field_name_lower:
        return "test_data_generator.random_hostname()"
    return None


def get_string_instruction_by_name(
    field_name_lower: str, max_length: int
) -> Optional[str]:
    """Infer string generator from field name.

    Args:
        field_name_lower: Lowercase field name
        max_length: Maximum length constraint

    Returns:
        Python code instruction or None if no inference possible
    """
    if not field_name_lower:
        return None

    # Check code fields (country, currency)
    result = _check_code_field(field_name_lower, max_length)
    if result:
        return result

    # Simple keyword matches
    if "locale" in field_name_lower:
        return "test_data_generator.random_locale()"
    if "color" in field_name_lower or "colour" in field_name_lower:
        return "test_data_generator.random_hex_color()"

    # Date/time fields
    result = _check_datetime_field(field_name_lower)
    if result:
        return result

    # Email and phone
    if "email" in field_name_lower:
        return "test_data_generator.generate_email()"
    if "phone" in field_name_lower:
        return _GEN_RANDOM_IPV4

    # Network fields
    return _check_network_field(field_name_lower)


def get_string_instruction(field_schema: dict, field_name: str = "") -> str:
    """Get instruction for string type fields.

    Args:
        field_schema: Field schema dict
        field_name: Optional field name for inference

    Returns:
        Python code instruction for generating string value
    """
    max_length = field_schema.get("maxLength", 50)
    field_name_lower = field_name.lower() if field_name else ""

    # Try name-based inference first
    inferred = get_string_instruction_by_name(field_name_lower, max_length)
    if inferred:
        return inferred

    # Default string generation with length constraints
    length = max_length if isinstance(max_length, int) and max_length <= 50 else 10
    return f"test_data_generator.generate_string(length={length})"


def get_integer_instruction(field_schema: dict, field_name: str = "") -> str:
    """Get instruction for integer type fields.

    Args:
        field_schema: Field schema dict
        field_name: Optional field name for inference

    Returns:
        Python code instruction for generating integer value
    """
    field_name_lower = field_name.lower() if field_name else ""

    # RGB/color detection
    color_keywords = ["rgb", "color", "colour", "red", "green", "blue", "alpha"]
    if field_name_lower and any(kw in field_name_lower for kw in color_keywords):
        return "test_data_generator.generate_integer(min_val=0, max_val=255)"

    exclusive_min = field_schema.get("exclusiveMinimum")
    exclusive_max = field_schema.get("exclusiveMaximum")
    min_val = (
        exclusive_min if exclusive_min is not None else field_schema.get("minimum", 1)
    )
    max_val = (
        exclusive_max
        if exclusive_max is not None
        else field_schema.get("maximum", 1000)
    )
    exclusive = exclusive_min is not None or exclusive_max is not None
    multiple_of = field_schema.get("multipleOf")

    parts = [f"min_val={min_val}", f"max_val={max_val}"]
    if exclusive:
        parts.append("exclusive=True")
    if multiple_of:
        parts.append(f"multiple_of={multiple_of}")

    return f"test_data_generator.generate_integer({', '.join(parts)})"


def get_number_instruction(field_schema: dict) -> str:
    """Get instruction for number (float) type fields.

    Args:
        field_schema: Field schema dict

    Returns:
        Python code instruction for generating float value
    """
    exclusive_min = field_schema.get("exclusiveMinimum")
    exclusive_max = field_schema.get("exclusiveMaximum")
    min_val = (
        exclusive_min if exclusive_min is not None else field_schema.get("minimum", 0.0)
    )
    max_val = (
        exclusive_max
        if exclusive_max is not None
        else field_schema.get("maximum", 1000.0)
    )
    exclusive = exclusive_min is not None or exclusive_max is not None

    if exclusive:
        return f"test_data_generator.generate_float(min_val={min_val}, max_val={max_val}, exclusive=True)"
    return f"test_data_generator.generate_float(min_val={min_val}, max_val={max_val})"


def get_object_instruction(
    field_schema: dict,
    object_instruction_fn: Callable[[dict, Optional[frozenset]], str],
    ancestors: Optional[frozenset] = None,
) -> str:
    """Get instruction for object type fields.

    Args:
        field_schema: Field schema dict
        object_instruction_fn: Callback to generate nested object instructions
        ancestors: Frozenset of ancestor schema IDs for cycle detection

    Returns:
        Python code instruction for generating object value
    """
    sub_props = field_schema.get("properties", {})
    if sub_props:
        return object_instruction_fn(field_schema, ancestors)

    add_props = field_schema.get("additionalProperties")
    if add_props and isinstance(add_props, dict):
        val_type = add_props.get("type", TYPE_STRING)
        type_generators = {
            TYPE_INTEGER: "test_data_generator.generate_integer()",
            TYPE_NUMBER: "test_data_generator.generate_float()",
            TYPE_BOOLEAN: "test_data_generator.generate_boolean()",
        }
        val_gen = type_generators.get(val_type, "test_data_generator.generate_string()")
        return '{f"key_{i}": ' + val_gen + " for i in range(3)}"

    if add_props:
        return '{"key1": "value1", "key2": "value2"}'

    return "{}"


def _compute_array_length(field_schema: dict) -> int:
    """Compute appropriate array length from schema constraints."""
    min_items = field_schema.get("minItems", 1)
    max_items = field_schema.get("maxItems")
    array_len = max(min_items, 2) if min_items > 1 else 3
    if max_items and array_len > max_items:
        array_len = max_items
    return array_len


def _get_array_items_info(field_items: Any) -> Dict[str, Any]:
    """Extract type information from array items schema."""
    if not isinstance(field_items, dict):
        return {"type": TYPE_STRING, "enum": None, "ref": None, "one_of": None}

    return {
        "type": field_items.get("type", TYPE_STRING),
        "enum": field_items.get("enum"),
        "ref": field_items.get("$ref"),
        "one_of": field_items.get("oneOf") or field_items.get("anyOf"),
        "properties": field_items.get("properties", {}),
    }


def get_array_instruction(
    field_schema: dict,
    object_instruction_fn: Callable[[dict, Optional[frozenset]], str],
    ancestors: Optional[frozenset] = None,
    field_name: str = "",
) -> str:
    """Get instruction for array type fields.

    Args:
        field_schema: Field schema dict
        object_instruction_fn: Callback to generate nested object instructions
        ancestors: Frozenset of ancestor schema IDs for cycle detection
        field_name: Optional field name for inference

    Returns:
        Python code instruction for generating array value
    """
    field_items = field_schema.get("items", {})
    array_len = _compute_array_length(field_schema)
    items_info = _get_array_items_info(field_items)
    field_name_lower = field_name.lower() if field_name else ""

    # Enum items
    if items_info["enum"]:
        return f"[random.choice({items_info['enum']}) for _ in range({array_len})]"

    # Discriminated union items
    if (
        items_info["one_of"]
        and isinstance(items_info["one_of"], list)
        and len(items_info["one_of"]) > 0
    ):
        first_variant = items_info["one_of"][0]
        if isinstance(first_variant, dict):
            variant_props, _ = extract_all_properties(first_variant)
            if variant_props:
                obj_instr = object_instruction_fn(first_variant, ancestors)
                return f"[{obj_instr} for _ in range({array_len})]"
        return "[{}]"

    # Object items
    if items_info["type"] == TYPE_OBJECT or items_info["ref"]:
        if items_info.get("properties"):
            obj_instr = object_instruction_fn(field_items, ancestors)
            return f"[{obj_instr} for _ in range({array_len})]"
        return "[{}]"

    # Primitive type items
    return _get_primitive_array_instruction(
        items_info["type"], array_len, field_items, field_name_lower
    )


def _get_primitive_array_instruction(
    items_type: str,
    array_len: int,
    field_items: dict,
    field_name_lower: str,
) -> str:
    """Get instruction for arrays of primitive types."""
    if items_type == TYPE_STRING:
        return f"[test_data_generator.generate_string() for _ in range({array_len})]"

    if items_type == TYPE_INTEGER:
        color_keywords = ["rgb", "color", "colour"]
        if field_name_lower and any(kw in field_name_lower for kw in color_keywords):
            return f"[test_data_generator.generate_integer(min_val=0, max_val=255) for _ in range({array_len})]"
        return f"[test_data_generator.generate_integer() for _ in range({array_len})]"

    if items_type == TYPE_NUMBER:
        return f"[test_data_generator.generate_float() for _ in range({array_len})]"

    if items_type == TYPE_BOOLEAN:
        return f"[test_data_generator.generate_boolean() for _ in range({array_len})]"

    if items_type == TYPE_ARRAY:
        return _get_nested_array_instruction(field_items, array_len)

    # Default to string array
    return f"[test_data_generator.generate_string() for _ in range({array_len})]"


def _get_nested_array_instruction(field_items: dict, array_len: int) -> str:
    """Get instruction for nested arrays (array of arrays)."""
    inner_items = field_items.get("items", {}) if isinstance(field_items, dict) else {}
    inner_type = (
        inner_items.get("type", TYPE_STRING)
        if isinstance(inner_items, dict)
        else TYPE_STRING
    )

    type_generators = {
        TYPE_INTEGER: "test_data_generator.generate_integer()",
        TYPE_NUMBER: "test_data_generator.generate_float()",
        TYPE_BOOLEAN: "test_data_generator.generate_boolean()",
    }
    inner_gen = type_generators.get(inner_type, "test_data_generator.generate_string()")

    return f"[[{inner_gen} for _ in range(2)] for _ in range({array_len})]"
