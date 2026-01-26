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
- Mixed array types
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

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

    # Known template boilerplate patterns that should never appear in generated code
    TEMPLATE_BOILERPLATE_PATTERNS = [
        r"#\s*Check if request succeeded \(result is dict or None\)",
        r"#\s*Success - result contains JSON response data",
        r"#\s*Example: item_id = result\.get",
        r"#\s*If result is None, make_request\(\) already logged the failure",
        r"#\s*Use make_request\(\) with SUCCESS codes only",
        r"#\s*NEVER include 4xx codes here",
        r"#\s*make_request\(\) returns dict \(JSON data\) or None",
        r"#\s*Build request data with VALID values",
    ]

    # Placeholder comment patterns
    PLACEHOLDER_PATTERNS = [
        r"#\s*Add other required .* fields",
        r"#\s*TODO:?\s",
        r"#\s*Fill in remaining",
        r"#\s*Complete this",
        r"#\s*Add remaining",
        r"#\s*Add more fields",
    ]

    # Security payload patterns that should NOT appear in URL path segments
    SECURITY_PAYLOAD_IN_PATH_RE = re.compile(
        r'make_request\([^)]*f"[^"]*\{(?:payload|random\.choice\([^)]*PAYLOAD|XSS_PAYLOAD|SQL_PAYLOAD|PATH_TRAVERSAL)[^}]*\}[^"]*"',
        re.IGNORECASE,
    )

    # Empty path segment pattern (double slashes in URLs, excluding https://)
    EMPTY_PATH_SEGMENT_RE = re.compile(
        r'make_request\([^)]*"[^"]*(?<!https:)(?<!http:)//[^"]*"',
    )

    # Expected status with 2xx in make_request calls
    SUCCESS_IN_EXPECTED_STATUS_RE = re.compile(
        r'expected_status=\[([^\]]*)\]',
    )

    def validate(
        self,
        code: str,
        scenario_type: str,
        endpoint_path: str,
        all_endpoint_paths: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate generated code for semantic correctness.

        Args:
            code: The generated Python code
            scenario_type: "positive", "negative", or "security"
            endpoint_path: The endpoint path being tested (e.g., "/api/v1/items")
            all_endpoint_paths: List of all valid endpoint paths from the OpenAPI spec

        Returns:
            ValidationResult with violations list
        """
        violations: List[ValidationViolation] = []

        # Run all checks
        violations.extend(self._check_template_boilerplate(code))
        violations.extend(self._check_placeholder_comments(code))
        violations.extend(self._check_empty_path_segments(code))

        if scenario_type == "security":
            violations.extend(self._check_security_path_injection(code))

        if scenario_type == "negative":
            violations.extend(self._check_success_codes_in_negative(code))

        if all_endpoint_paths:
            violations.extend(self._check_hallucinated_endpoints(code, endpoint_path, all_endpoint_paths))

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
            for pattern in self.TEMPLATE_BOILERPLATE_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(ValidationViolation(
                        rule="template_boilerplate",
                        message=f"Template comment copied verbatim: {line.strip()}",
                        line_number=i,
                        severity="error",
                    ))
                    break  # One violation per line

        return violations

    def _check_placeholder_comments(self, code: str) -> List[ValidationViolation]:
        """Check for placeholder comments instead of real code (Classification C)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in self.PLACEHOLDER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(ValidationViolation(
                        rule="placeholder_comment",
                        message=f"Placeholder comment instead of code: {line.strip()}. "
                                f"You MUST generate ALL required fields.",
                        line_number=i,
                        severity="error",
                    ))
                    break

        return violations

    def _check_security_path_injection(self, code: str) -> List[ValidationViolation]:
        """Check for security payloads injected into URL path parameters (Classification D)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for f-string paths with payload variables in path segments
            if "make_request" in line and re.search(
                r'f"[^"]*\/\{(?:payload|random\.choice)',
                line, re.IGNORECASE,
            ):
                violations.append(ValidationViolation(
                    rule="path_param_injection",
                    message="Security payload injected into URL path parameter. "
                            "Path params are URL routing - inject into body/query/headers instead.",
                    line_number=i,
                    severity="error",
                ))

        return violations

    def _check_empty_path_segments(self, code: str) -> List[ValidationViolation]:
        """Check for empty path segments (double slashes) in URLs (Classification E)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            if "make_request" in line:
                # Look for // in URL strings (but not in https:// or http://)
                url_match = re.search(r'"([^"]*)"', line)
                if url_match:
                    url = url_match.group(1)
                    # Remove protocol prefix before checking
                    url_no_protocol = re.sub(r'^https?://', '', url)
                    if "//" in url_no_protocol:
                        violations.append(ValidationViolation(
                            rule="empty_path_segment",
                            message=f"Empty path segment (double slash) in URL: {url}. "
                                    f"Use a present but invalid value instead of empty segment.",
                            line_number=i,
                            severity="error",
                        ))

        return violations

    def _check_success_codes_in_negative(self, code: str) -> List[ValidationViolation]:
        """Check for 2xx status codes in negative workflow expected_status (Classification G)."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            match = self.SUCCESS_IN_EXPECTED_STATUS_RE.search(line)
            if match:
                codes_str = match.group(1)
                # Parse the codes
                try:
                    codes = [int(c.strip()) for c in codes_str.split(",") if c.strip()]
                    success_codes = [c for c in codes if 200 <= c < 300]
                    if success_codes:
                        violations.append(ValidationViolation(
                            rule="success_in_negative",
                            message=f"Negative workflow has success codes {success_codes} in expected_status. "
                                    f"Negative tests must ONLY expect 4xx error codes.",
                            line_number=i,
                            severity="error",
                        ))
                except (ValueError, TypeError):
                    pass

        return violations

    def _check_hallucinated_endpoints(
        self, code: str, endpoint_path: str, all_endpoint_paths: List[str],
    ) -> List[ValidationViolation]:
        """Check for invented endpoints not in the OpenAPI spec (Classification F)."""
        violations = []
        lines = code.split("\n")

        # Extract all URL paths used in make_request calls
        for i, line in enumerate(lines, 1):
            if "make_request" not in line:
                continue

            # Extract URL from make_request call - handle both regular and f-strings
            url_match = re.search(r'make_request\([^,]+,\s*(?:f)?"([^"{}]+)', line)
            if not url_match:
                continue

            used_path = url_match.group(1)

            # Skip if it's the endpoint under test or its base path
            if self._path_matches_spec(used_path, endpoint_path, all_endpoint_paths):
                continue

            violations.append(ValidationViolation(
                rule="hallucinated_endpoint",
                message=f"Endpoint '{used_path}' not found in OpenAPI spec. "
                        f"Use ONLY endpoints from ENDPOINT TO TEST or SETUP ENDPOINTS sections.",
                line_number=i,
                severity="error",
            ))

        return violations

    def _path_matches_spec(
        self, used_path: str, endpoint_path: str, all_paths: List[str],
    ) -> bool:
        """Check if a used path matches any known endpoint path."""
        # Direct match
        if used_path == endpoint_path:
            return True

        # Check against all known paths (resolve path params)
        for spec_path in all_paths:
            # Convert spec path params to regex: /items/{id} -> /items/[^/]+
            pattern = re.sub(r'\{[^}]+\}', r'[^/]+', spec_path)
            if re.fullmatch(pattern, used_path):
                return True

            # Also check base path match (for endpoints with path params)
            base_spec = spec_path.split("{")[0].rstrip("/")
            base_used = used_path.split("{")[0].rstrip("/")
            if base_spec and base_used.startswith(base_spec):
                return True

        return False
