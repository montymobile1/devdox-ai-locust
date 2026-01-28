"""
Schema Utilities Module

Provides utilities for parsing and processing OpenAPI schemas.
Handles schema unwrapping, property extraction, and string escaping.
"""

from typing import Any, Dict, List, Optional, Tuple


def unwrap_nullable_schema(schema: Any) -> Tuple[Any, bool]:
    """
    Unwrap OpenAPI nullable patterns.

    Handles:
    - OpenAPI 3.1: anyOf: [{actual_type_schema}, {type: null}]
    - OpenAPI 3.0: {type: string, nullable: true}

    Args:
        schema: OpenAPI schema dict

    Returns:
        Tuple of (unwrapped_schema, is_nullable)
    """
    if not isinstance(schema, dict):
        return schema, False

    if schema.get("nullable") is True and schema.get("type"):
        return schema, True

    any_of = schema.get("anyOf") or schema.get("oneOf")
    if any_of and isinstance(any_of, list):
        null_variants = [
            v for v in any_of if isinstance(v, dict) and v.get("type") == "null"
        ]
        real_variants = [
            v for v in any_of if isinstance(v, dict) and v.get("type") != "null"
        ]
        if len(null_variants) >= 1 and len(real_variants) == 1:
            return real_variants[0], True

    return schema, False


def extract_all_properties(schema: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract all properties from a schema, handling allOf and discriminated unions.

    Args:
        schema: Request body schema dict

    Returns:
        Tuple of (properties_dict, required_list)
    """
    if not isinstance(schema, dict):
        return {}, []

    properties = dict(schema.get("properties", {}))
    required_list = list(schema.get("required", []))

    properties, required_list = _merge_all_of_properties(
        schema, properties, required_list
    )
    properties, required_list = _merge_union_properties(
        schema, properties, required_list
    )

    return properties, list(set(required_list))


def _merge_all_of_properties(
    schema: dict,
    properties: Dict[str, Any],
    required_list: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge properties from allOf composition."""
    all_of = schema.get("allOf")
    if not all_of or not isinstance(all_of, list):
        return properties, required_list

    for item in all_of:
        if not isinstance(item, dict):
            continue
        item_props = item.get("properties", {})
        if item_props:
            properties.update(item_props)
            required_list.extend(item.get("required", []))

        nested_all_of = item.get("allOf")
        if nested_all_of and isinstance(nested_all_of, list):
            for sub in nested_all_of:
                if isinstance(sub, dict) and sub.get("properties"):
                    properties.update(sub["properties"])
                    required_list.extend(sub.get("required", []))

    return properties, required_list


def _merge_union_properties(
    schema: dict,
    properties: Dict[str, Any],
    required_list: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge properties from oneOf/anyOf discriminated unions."""
    one_of = schema.get("oneOf") or schema.get("anyOf")
    if not one_of or not isinstance(one_of, list) or properties:
        return properties, required_list

    for variant in one_of:
        if not isinstance(variant, dict):
            continue
        v_props = variant.get("properties", {})
        if v_props:
            properties.update(v_props)
            required_list.extend(variant.get("required", []))

        v_all_of = variant.get("allOf")
        if v_all_of and isinstance(v_all_of, list):
            for sub in v_all_of:
                if isinstance(sub, dict) and sub.get("properties"):
                    properties.update(sub["properties"])
                    required_list.extend(sub.get("required", []))

    return properties, required_list


def escape_for_python_string(value: Any) -> str:
    """
    Escape a string for safe embedding in a Python double-quoted string.

    Args:
        value: String to escape

    Returns:
        Escaped string safe for Python literals
    """
    if not isinstance(value, str):
        return str(value)
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def escape_for_raw_string(value: Any) -> str:
    """
    Escape a string for safe embedding in a Python raw string (r"...").

    For raw strings, backslashes are NOT escaped. Only quotes need escaping.
    Used for regex patterns.

    Args:
        value: String to escape

    Returns:
        Escaped string safe for raw Python literals
    """
    if not isinstance(value, str):
        return str(value)
    return value.replace('"', '\\"')


def resolve_ref_in_union(
    ref: str,
    one_of: List[Any],
) -> Optional[dict]:
    """
    Resolve a $ref within the context of a discriminated union.

    Finds the variant in oneOf/anyOf that matches the reference.

    Args:
        ref: Reference string (e.g., "#/components/schemas/Dog")
        one_of: List of oneOf/anyOf variants

    Returns:
        Resolved schema dict or None if not found
    """
    if not one_of or not isinstance(one_of, list):
        return None

    ref_name = ref.split("/")[-1] if ref else ""

    for variant in one_of:
        if not isinstance(variant, dict):
            continue

        variant_ref = variant.get("$ref", "")
        if variant_ref and variant_ref.split("/")[-1] == ref_name:
            return variant

        if variant.get("title") == ref_name:
            return variant

    return None


def get_schema_type(schema: Any) -> str:
    """
    Get the type of a schema, handling nullable wrappers.

    Args:
        schema: OpenAPI schema dict

    Returns:
        Schema type string (e.g., "string", "integer", "object")
    """
    if not isinstance(schema, dict):
        return "unknown"

    unwrapped, _ = unwrap_nullable_schema(schema)
    return str(unwrapped.get("type", "object"))


def is_required_field(field_name: str, required_list: List[str]) -> bool:
    """
    Check if a field is in the required list.

    Args:
        field_name: Name of the field
        required_list: List of required field names

    Returns:
        True if field is required
    """
    return field_name in required_list


def get_field_constraints(schema: dict) -> Dict[str, Any]:
    """
    Extract validation constraints from a field schema.

    Args:
        schema: Field schema dict

    Returns:
        Dict of constraints (minLength, maxLength, pattern, enum, etc.)
    """
    constraints = {}

    for key in [
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "pattern",
        "enum",
        "format",
        "minItems",
        "maxItems",
    ]:
        if key in schema:
            constraints[key] = schema[key]

    return constraints
