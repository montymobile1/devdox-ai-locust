"""
Post-Generation Code Validator

Validates LLM-generated workflow code for semantic correctness patterns
that cannot be caught by syntax validation alone. Catches common LLM
hallucination patterns including:
- Template boilerplate copying
- Placeholder comments
- Security payloads in path parameters
- Empty path segments
- Hallucinated endpoints
- Success codes in negative workflows
- Schema compliance (mixed array types, wrong formats, ignored enums)
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from devdox_ai_locust.utils.constants import (
    LITERAL_PATH_PARAM_RE,
    MAKE_REQUEST_CALL_RE,
    PLACEHOLDER_PATTERNS,
    SUCCESS_IN_EXPECTED_STATUS_RE,
    TEMPLATE_BOILERPLATE_PATTERNS,
    URL_ARG_RE,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationViolation:
    """A single validation violation found in generated code."""

    rule: str
    message: str
    line_number: Optional[int] = None
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationResult:
    """Result of validating generated code."""

    is_valid: bool
    violations: List[ValidationViolation] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        """Format violations into a single error message for the fix prompt."""
        if not self.violations:
            return ""
        lines = ["The generated code has the following semantic issues:"]
        for v in self.violations:
            loc = f" (line {v.line_number})" if v.line_number else ""
            lines.append(f"- [{v.rule}]{loc}: {v.message}")
        return "\n".join(lines)


class CodeValidator:
    """
    Validates LLM-generated workflow code for semantic correctness.

    Checks are organized by classification:
    - C: Template boilerplate / placeholder comments
    - D: Security payloads in path parameters
    - E: Empty path segments
    - F: Hallucinated endpoints
    - G: Success codes in negative workflows
    """

    # Patterns imported from constants module for centralized maintenance

    # Maps OpenAPI string formats to their correct generator functions
    FORMAT_GENERATOR_MAP: Dict[str, Set[str]] = {
        "date": {"random_date"},
        "date-time": {"random_date", "isoformat", "now"},
        "email": {"generate_email"},
        "uuid": {"random_uuid"},
        "uri": set(),  # Literal string is acceptable
        "url": set(),  # Literal string is acceptable
        "ipv4": set(),  # Literal string is acceptable
        "ipv6": set(),  # Literal string is acceptable
        "hostname": set(),  # Literal string is acceptable
        "time": set(),  # Literal string is acceptable
    }

    def validate(
        self,
        code: str,
        scenario_type: str,
        endpoint_path: str,
        all_endpoint_paths: Optional[List[str]] = None,
        request_body_schema: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate generated code for semantic correctness.

        Args:
            code: The generated Python code
            scenario_type: "positive", "negative", or "security"
            endpoint_path: The endpoint path being tested (e.g., "/api/v1/items")
            all_endpoint_paths: List of all valid endpoint paths from the OpenAPI spec
            request_body_schema: JSON Schema dict for the request body (if endpoint has one)

        Returns:
            ValidationResult with violations list
        """
        violations: List[ValidationViolation] = []

        # Run all checks
        violations.extend(self._check_template_boilerplate(code))
        violations.extend(self._check_placeholder_comments(code))
        violations.extend(self._check_empty_setup_dicts(code))
        violations.extend(self._check_empty_path_segments(code))
        violations.extend(self._check_literal_path_params(code))

        if scenario_type == "security":
            violations.extend(self._check_security_path_injection(code))

        if scenario_type == "negative":
            violations.extend(
                self._check_success_codes_in_negative(code, endpoint_path)
            )

        if all_endpoint_paths:
            violations.extend(
                self._check_hallucinated_endpoints(
                    code, endpoint_path, all_endpoint_paths
                )
            )

        # Schema compliance check (only for positive workflows with a schema)
        if request_body_schema and scenario_type == "positive":
            violations.extend(self._check_schema_compliance(code, request_body_schema))

        # Only errors (not warnings) make validation fail
        has_errors = any(v.severity == "error" for v in violations)
        return ValidationResult(
            is_valid=not has_errors,
            violations=violations,
        )

    def _check_template_boilerplate(self, code: str) -> List[ValidationViolation]:
        """Check for verbatim template comment copying (Classification C)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in TEMPLATE_BOILERPLATE_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(
                        ValidationViolation(
                            rule="template_boilerplate",
                            message=f"Template comment copied verbatim: {line.strip()}",
                            line_number=i,
                            severity="error",
                        )
                    )
                    break  # One violation per line

        return violations

    def _check_placeholder_comments(self, code: str) -> List[ValidationViolation]:
        """Check for placeholder comments instead of real code (Classification C)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in PLACEHOLDER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(
                        ValidationViolation(
                            rule="placeholder_comment",
                            message=f"Placeholder comment instead of code: {line.strip()}. "
                            f"You MUST generate ALL required fields.",
                            line_number=i,
                            severity="error",
                        )
                    )
                    break

        return violations

    # Suffixes that indicate setup data variables
    _SETUP_VAR_SUFFIXES = ["_data", "_payload", "_body", "_request"]

    def _check_empty_setup_dicts(self, code: str) -> List[ValidationViolation]:
        """Check for empty setup data dicts that should have generated fields."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            violations.extend(self._check_inline_empty_dict(line, i))
            violations.extend(self._check_multiline_empty_dict(lines, line, i))

        return violations

    def _check_inline_empty_dict(
        self, line: str, line_num: int
    ) -> List[ValidationViolation]:
        """Check for single-line empty dict assignments like `user_data = {}`."""
        for pattern in self._SETUP_VAR_SUFFIXES:
            if pattern in line and "=" in line:
                match = re.search(
                    r"(\w+" + re.escape(pattern) + r")\s*=\s*\{\s*(?:#[^\n]*)?\s*\}",
                    line,
                )
                if match:
                    return [
                        ValidationViolation(
                            rule="empty_setup_dict",
                            message=f"Empty setup data dict '{match.group(1)}'. "
                            f"You MUST generate actual field values for setup API calls. "
                            f"Do NOT leave empty dicts with placeholder comments.",
                            line_number=line_num,
                            severity="error",
                        )
                    ]
        return []

    def _check_multiline_empty_dict(
        self, lines: List[str], line: str, line_num: int
    ) -> List[ValidationViolation]:
        """Check for multi-line empty dict assignments."""
        if not re.match(r"\s*(\w+_data|\w+_payload|\w+_body)\s*=\s*\{\s*$", line):
            return []
        if line_num < len(lines):
            next_line = lines[line_num].strip() if line_num < len(lines) else ""
            if next_line == "}" or (
                next_line.startswith("#")
                and line_num + 1 < len(lines)
                and lines[line_num + 1].strip() == "}"
            ):
                var_match = re.match(r"\s*(\w+_data|\w+_payload|\w+_body)", line)
                if var_match:
                    return [
                        ValidationViolation(
                            rule="empty_setup_dict",
                            message=f"Empty setup data dict '{var_match.group(1)}'. "
                            f"You MUST generate actual field values for setup API calls.",
                            line_number=line_num,
                            severity="error",
                        )
                    ]
        return []

    def _check_security_path_injection(self, code: str) -> List[ValidationViolation]:
        """Check for security payloads injected into URL path parameters (Classification D)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for f-string paths with payload variables in path segments
            if "make_request" in line and re.search(
                r'f"[^"]*\/\{(?:payload|random\.choice)',
                line,
                re.IGNORECASE,
            ):
                violations.append(
                    ValidationViolation(
                        rule="path_param_injection",
                        message="Security payload injected into URL path parameter. "
                        "Path params are URL routing - inject into body/query/headers instead.",
                        line_number=i,
                        severity="error",
                    )
                )

        return violations

    def _check_empty_path_segments(self, code: str) -> List[ValidationViolation]:
        """Check for empty path segments (double slashes) in URLs (Classification E)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            if "make_request" in line:
                # Match the URL argument specifically (second string arg after method)
                url_match = URL_ARG_RE.search(line)
                if url_match:
                    url = url_match.group(1)
                    # Remove protocol prefix before checking
                    url_no_protocol = re.sub(r"^https?://", "", url)
                    if "//" in url_no_protocol:
                        violations.append(
                            ValidationViolation(
                                rule="empty_path_segment",
                                message=f"Empty path segment (double slash) in URL: {url}. "
                                f"Use a present but invalid value instead of empty segment.",
                                line_number=i,
                                severity="error",
                            )
                        )

        return violations

    def _check_literal_path_params(self, code: str) -> List[ValidationViolation]:
        """Check for literal {param} in non-f-string paths.

        Catches cases where the LLM forgot to use an f-string, leaving
        literal text like '/items/{item_id}' instead of f'/items/{item_id}'.
        """
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            if "make_request" in line:
                # Check for non-f-string with curly braces (literal path param)
                match = LITERAL_PATH_PARAM_RE.search(line)
                if match:
                    url = match.group(1)
                    violations.append(
                        ValidationViolation(
                            rule="literal_path_param",
                            message=f"Path contains literal '{{param}}' without f-string: \"{url}\". "
                            f'Use f-string like f"{url}" to substitute variables.',
                            line_number=i,
                            severity="error",
                        )
                    )

        return violations

    @staticmethod
    def _paths_match(spec_path: str, call_path: str) -> bool:
        """Compare an OpenAPI spec path with a code call path segment-by-segment.

        Treats {param} segments in the spec path and {expression} segments in
        the call path (f-string variables) as wildcards that match any segment.

        Examples:
            _paths_match("/items/{id}", "/items/{item_id}") -> True
            _paths_match("/items/{id}", "/items/{item_id}/details") -> False
            _paths_match("/items/{id}/details", "/items/{x}/details") -> True
        """
        spec_segments = [s for s in spec_path.strip("/").split("/") if s]
        call_segments = [s for s in call_path.strip("/").split("/") if s]

        if len(spec_segments) != len(call_segments):
            return False

        for spec_seg, call_seg in zip(spec_segments, call_segments):
            # If either segment is a parameter placeholder, it matches anything
            if spec_seg.startswith("{") or call_seg.startswith("{"):
                continue
            if spec_seg != call_seg:
                return False

        return True

    def _check_success_codes_in_negative(
        self, code: str, endpoint_path: str = ""
    ) -> List[ValidationViolation]:
        """Check for 2xx status codes in negative workflow expected_status (Classification G).

        Only flags 2xx codes on make_request calls that target the endpoint under test.
        Setup calls (POST/PUT to different endpoints to create test data) are exempt.
        """
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            match = SUCCESS_IN_EXPECTED_STATUS_RE.search(line)
            if match:
                codes_str = match.group(1)
                try:
                    codes = [int(c.strip()) for c in codes_str.split(",") if c.strip()]
                    success_codes = [c for c in codes if 200 <= c < 300]
                    if not success_codes:
                        continue

                    # Check if this call targets the endpoint under test or a different endpoint
                    if endpoint_path:
                        call_match = MAKE_REQUEST_CALL_RE.search(line)
                        if call_match:
                            call_path = call_match.group(2)
                            # If calling a DIFFERENT endpoint (setup call), allow 2xx
                            if call_path and not self._paths_match(
                                endpoint_path, call_path
                            ):
                                continue

                    violations.append(
                        ValidationViolation(
                            rule="success_in_negative",
                            message=f"Negative workflow has success codes {success_codes} in expected_status. "
                            f"Negative tests must ONLY expect 4xx error codes.",
                            line_number=i,
                            severity="error",
                        )
                    )
                except (ValueError, TypeError):
                    pass

        return violations

    def _check_hallucinated_endpoints(
        self,
        code: str,
        endpoint_path: str,
        all_endpoint_paths: List[str],
    ) -> List[ValidationViolation]:
        """Check for invented endpoints not in the OpenAPI spec (Classification F)."""
        violations = []
        lines = code.split("\n")

        # Extract all URL paths used in make_request calls
        for i, line in enumerate(lines, 1):
            if "make_request" not in line:
                continue

            # Extract URL from make_request call - handle both regular and f-strings
            url_match = re.search(r'make_request\([^,]+,\s*(?:f)?"([^"]+)"', line)
            if not url_match:
                continue

            used_path = url_match.group(1)

            # Skip if it's the endpoint under test or its base path
            if self._path_matches_spec(used_path, endpoint_path, all_endpoint_paths):
                continue

            violations.append(
                ValidationViolation(
                    rule="hallucinated_endpoint",
                    message=f"Endpoint '{used_path}' not found in OpenAPI spec. "
                    f"Use ONLY endpoints from ENDPOINT TO TEST or SETUP ENDPOINTS sections.",
                    line_number=i,
                    severity="error",
                )
            )

        return violations

    def _path_matches_spec(
        self,
        used_path: str,
        endpoint_path: str,
        all_paths: List[str],
    ) -> bool:
        """Check if a used path matches any known endpoint path."""
        # Direct match
        if used_path == endpoint_path:
            return True

        # Check against all known paths using segment-by-segment comparison
        for spec_path in all_paths:
            if self._paths_match(spec_path, used_path):
                return True

        return False

    # --- Schema Compliance Checks (Classification B) ---

    def _check_schema_compliance(
        self,
        code: str,
        schema: Dict[str, Any],
    ) -> List[ValidationViolation]:
        """
        Check generated code against the request body JSON Schema.

        Detects:
        - Mixed types in typed arrays (e.g., ["str", 123] for string array)
        - Wrong generators for format fields (generate_string for date fields)
        - Ignored enum constraints (generator call instead of random.choice)
        """
        violations: List[ValidationViolation] = []

        # Parse schema properties
        properties = schema.get("properties", {})
        if not properties:
            return violations

        # Parse the code AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return violations  # Syntax errors handled elsewhere

        # Find all dict literals that could be request bodies
        request_dicts = self._extract_request_body_dicts(tree)

        for dict_node in request_dicts:
            violations.extend(
                self._validate_dict_against_schema(
                    dict_node, properties, schema.get("required", [])
                )
            )

        return violations

    def _extract_request_body_dicts(self, tree: ast.AST) -> List[ast.Dict]:
        """
        Extract Dict AST nodes that are likely request body data.

        Finds dicts in:
        - Variable assignments (data = {...}, json_data = {...}, payload = {...})
        - json= keyword argument in function calls
        """
        dicts: List[ast.Dict] = []
        body_var_names = {
            "data",
            "json_data",
            "payload",
            "body",
            "request_data",
            "request_body",
        }

        for node in ast.walk(tree):
            # Case 1: Assignment like `data = {...}`
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in body_var_names:
                        if isinstance(node.value, ast.Dict):
                            dicts.append(node.value)

            # Case 2: json={...} keyword in make_request call
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "json" and isinstance(kw.value, ast.Dict):
                        dicts.append(kw.value)

        return dicts

    def _validate_dict_against_schema(
        self,
        dict_node: ast.Dict,
        properties: Dict[str, Any],
        required_fields: List[str],
    ) -> List[ValidationViolation]:
        """Validate a Dict AST node against schema properties."""
        violations: List[ValidationViolation] = []

        for key_node, value_node in zip(dict_node.keys, dict_node.values):
            if key_node is None or value_node is None:
                continue

            # Get the field name from the key
            field_name = self._get_constant_value(key_node)
            if not isinstance(field_name, str):
                continue

            # Look up field in schema
            field_schema = properties.get(field_name)
            if not field_schema:
                continue

            field_type = field_schema.get("type", "")
            field_format = field_schema.get("format")
            field_enum = field_schema.get("enum")
            line_num = getattr(value_node, "lineno", None)

            # Check 1: Enum constraint ignored
            if field_enum:
                violation = self._check_enum_usage(
                    field_name, value_node, field_enum, line_num
                )
                if violation:
                    violations.append(violation)

            # Check 2: String format with wrong generator
            elif field_type == "string" and field_format:
                violation = self._check_format_usage(
                    field_name, value_node, field_format, line_num
                )
                if violation:
                    violations.append(violation)

            # Check 3: Array with mixed types
            if field_type == "array":
                items_schema = field_schema.get("items", {})
                violation = self._check_array_types(
                    field_name, value_node, items_schema, line_num
                )
                if violation:
                    violations.append(violation)

        return violations

    def _check_enum_usage(
        self,
        field_name: str,
        value_node: ast.AST,
        enum_values: List[Any],
        line_num: Optional[int],
    ) -> Optional[ValidationViolation]:
        """Check if an enum field uses random.choice() with correct values."""
        # Accept: random.choice([...]), Constant that's in enum, variable that was set from choice
        if isinstance(value_node, ast.Constant):
            if value_node.value in enum_values:
                return None  # Hardcoded valid enum value is fine

        if isinstance(value_node, ast.Call):
            func_name = self._get_call_name(value_node)
            if func_name in ("random.choice", "choice"):
                return None  # Using random.choice is correct

            # Using a generator (generate_string, generate_integer, etc.) on an enum field
            if "generate_" in func_name or func_name in (
                "random_uuid",
                "random_date",
                "generate_email",
            ):
                return ValidationViolation(
                    rule="enum_ignored",
                    message=f"Field '{field_name}' has enum constraint {enum_values} but uses "
                    f"{func_name}() instead of random.choice({enum_values}).",
                    line_number=line_num,
                    severity="error",
                )

        return None

    def _check_format_usage(
        self,
        field_name: str,
        value_node: ast.AST,
        field_format: str,
        line_num: Optional[int],
    ) -> Optional[ValidationViolation]:
        """Check if a formatted string field uses the correct generator."""
        expected_generators = self.FORMAT_GENERATOR_MAP.get(field_format)
        if expected_generators is None:
            return None  # Unknown format, skip

        # If format has no specific generator requirement (uri, ipv4, etc.),
        # just ensure it's NOT using generate_string()
        if not expected_generators:
            # These formats accept literal strings, just reject generate_string
            if isinstance(value_node, ast.Call):
                func_name = self._get_call_name(value_node)
                if func_name in (
                    "generate_string",
                    "test_data_generator.generate_string",
                ):
                    return ValidationViolation(
                        rule="wrong_format_generator",
                        message=f"Field '{field_name}' has format '{field_format}' but uses "
                        f"generate_string(). Use an appropriate literal value or generator "
                        f"for '{field_format}' format.",
                        line_number=line_num,
                        severity="error",
                    )
            return None

        # Format has specific required generators
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            # Literal string - check if it looks valid for the format
            if self._literal_matches_format(value_node.value, field_format):
                return None  # Valid literal

        if isinstance(value_node, ast.Call):
            func_name = self._get_call_name(value_node)
            # Check if any expected generator is in the function name
            for gen in expected_generators:
                if gen in func_name:
                    return None  # Correct generator used

            # Wrong generator used
            if "generate_string" in func_name:
                expected_list = " or ".join(expected_generators)
                return ValidationViolation(
                    rule="wrong_format_generator",
                    message=f"Field '{field_name}' has format '{field_format}' but uses "
                    f"generate_string(). Use {expected_list}() instead.",
                    line_number=line_num,
                    severity="error",
                )

        return None

    def _check_array_types(
        self,
        field_name: str,
        value_node: ast.AST,
        items_schema: Dict[str, Any],
        line_num: Optional[int],
    ) -> Optional[ValidationViolation]:
        """Check if array elements are all the same type as defined in items schema."""
        items_type = items_schema.get("type", "")
        if not items_type:
            return None  # No items type defined, skip

        # Only check literal List nodes
        if not isinstance(value_node, ast.List):
            return None

        if not value_node.elts:
            return None  # Empty list is fine

        # Map schema types to Python constant types
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
        }

        expected_python_type = type_map.get(items_type)
        if not expected_python_type:
            return None  # Complex type (object, array), skip

        # Check each element
        wrong_elements = []
        for i, elt in enumerate(value_node.elts):
            if isinstance(elt, ast.Constant):
                # Bool is a subclass of int in Python, handle explicitly
                if items_type == "integer" and isinstance(elt.value, bool):
                    wrong_elements.append((i, type(elt.value).__name__))
                elif (
                    items_type == "boolean"
                    and isinstance(elt.value, int)
                    and not isinstance(elt.value, bool)
                ):
                    wrong_elements.append((i, type(elt.value).__name__))
                elif not isinstance(elt.value, expected_python_type):  # type: ignore[arg-type]
                    wrong_elements.append((i, type(elt.value).__name__))

        if wrong_elements:
            wrong_types = set(t for _, t in wrong_elements)
            return ValidationViolation(
                rule="mixed_array_types",
                message=f"Field '{field_name}' is a {items_type} array but contains mixed types: "
                f"{wrong_types}. ALL elements must be {items_type}.",
                line_number=line_num,
                severity="error",
            )

        return None

    def _get_constant_value(self, node: ast.AST) -> Any:
        """Extract a constant value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        return None

    def _get_call_name(self, call_node: ast.Call) -> str:
        """Extract the full function name from a Call node (e.g., 'test_data_generator.generate_string')."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            current: ast.expr = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _literal_matches_format(self, value: str, field_format: str) -> bool:
        """Check if a literal string value looks valid for a given format."""
        if field_format == "date":
            return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", value))
        if field_format == "date-time":
            return bool(re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value))
        if field_format == "email":
            return "@" in value and "." in value
        if field_format == "uuid":
            return bool(
                re.match(
                    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                    value,
                    re.I,
                )
            )
        if field_format in ("uri", "url"):
            from devdox_ai_locust.utils.constants import is_url

            return is_url(value)
        if field_format == "ipv4":
            return bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", value))
        if field_format == "hostname":
            return "." in value and " " not in value
        if field_format == "time":
            return bool(re.match(r"^\d{2}:\d{2}", value))
        return True  # Unknown format, accept any literal
