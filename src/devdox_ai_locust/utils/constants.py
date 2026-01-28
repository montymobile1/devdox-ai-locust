"""
Centralized constants, compiled regex patterns, and shared utility functions.

All regex patterns used across the codebase are defined here to:
- Eliminate duplication
- Apply ReDoS-safe bounded quantifiers where needed
- Provide a single place to maintain and test patterns
"""

import re
from typing import Pattern

# ---------------------------------------------------------------------------
# Shared helper regex (used by sanitize_identifier / to_class_name)
# ---------------------------------------------------------------------------
NON_ALNUM_RE: Pattern[str] = re.compile(r"[^a-zA-Z0-9_]")
MULTI_UNDERSCORE_RE: Pattern[str] = re.compile(r"_+")

# ---------------------------------------------------------------------------
# log_analyzer.py patterns
# ---------------------------------------------------------------------------
ERROR_PATTERN: Pattern[str] = re.compile(
    r"(ERROR|CRITICAL|Request failed|HTTPError|Exception|Error:)",
    re.IGNORECASE,
)
TRACEBACK_START: Pattern[str] = re.compile(r"^Traceback \(most recent call last\):")
UUID_RE: Pattern[str] = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
TIMESTAMP_RE: Pattern[str] = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*")
NUMERIC_PATH_RE: Pattern[str] = re.compile(r"/\d{2,}")
ADDR_RE: Pattern[str] = re.compile(r"0x[0-9a-f]+", re.I)
LONG_VALUE_RE: Pattern[str] = re.compile(r"'[^']{50,}'")
REQUEST_LINE: Pattern[str] = re.compile(
    r"REQUEST:\s*(GET|POST|PUT|PATCH|DELETE)\s+(\S+)", re.I
)
API_PATH_RE: Pattern[str] = re.compile(r"(GET|POST|PUT|PATCH|DELETE)\s+(/\S+)", re.I)
CONTEXT_PATTERN: Pattern[str] = re.compile(
    r"(REQUEST:|Response|status|body|json|failed|validation)",
    re.IGNORECASE,
)
CONTEXT_BUFFER_SIZE: int = 8

# ---------------------------------------------------------------------------
# code_validator.py patterns
# ---------------------------------------------------------------------------

# Template boilerplate patterns that should never appear in generated code
TEMPLATE_BOILERPLATE_PATTERNS: list[str] = [
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
PLACEHOLDER_PATTERNS: list[str] = [
    r"#\s*Add other required .* fields",
    r"#\s*TODO:?\s",
    r"#\s*Fill in remaining",
    r"#\s*Complete this",
    r"#\s*Add remaining",
    r"#\s*Add more fields",
    r"#\s*Generate .* here if needed",
    r"#\s*Generate .* data here",
    r"#\s*Add .* data here",
    r"#\s*Fill in .* data",
    r"#\s*Populate .* fields",
    r"#\s*Add .* fields here",
    r"#\s*Insert .* data",
    r"#\s*\.\.\.",  # Catch "# ..." ellipsis placeholder
]

# Pattern to detect empty setup data dicts (dict with only whitespace/comment)
# ReDoS-safe: bounded \s quantifiers
EMPTY_SETUP_DICT_RE: Pattern[str] = re.compile(
    r"(\w+_data)\s{0,20}=\s{0,20}\{\s{0,20}(?:#[^\n]*)?\s{0,20}\}",
    re.MULTILINE,
)

# Security payload patterns that should NOT appear in URL path segments
# ReDoS-safe: bounded [^)] and [^}] quantifiers
SECURITY_PAYLOAD_IN_PATH_RE: Pattern[str] = re.compile(
    r'make_request\([^)\n]{0,200}f"[^"\n]{0,500}\{'
    r"(?:payload|random\.choice\([^)\n]{0,200}PAYLOAD|XSS_PAYLOAD|SQL_PAYLOAD|PATH_TRAVERSAL)"
    r'[^}\n]{0,200}\}[^"\n]{0,500}"',
    re.IGNORECASE,
)

# Empty path segment pattern (double slashes in URLs, excluding https://)
EMPTY_PATH_SEGMENT_RE: Pattern[str] = re.compile(
    r'make_request\([^)\n]{0,200}"[^"\n]{0,500}(?<!https:)(?<!http:)//[^"\n]{0,500}"',
)

# Expected status with 2xx in make_request calls
SUCCESS_IN_EXPECTED_STATUS_RE: Pattern[str] = re.compile(
    r"expected_status=\[([^\]]*)\]",
)

# Regex to extract URL argument from make_request calls
# ReDoS-safe: bounded quantifiers
URL_ARG_RE: Pattern[str] = re.compile(
    r'make_request\(\s{0,20}"[^"\n]{0,500}"\s{0,20},\s{0,20}f?"([^"\n]{0,500})"'
)

# Regex to detect non-f-string with literal {param} placeholders
# ReDoS-safe: bounded quantifiers
LITERAL_PATH_PARAM_RE: Pattern[str] = re.compile(
    r'make_request\(\s{0,20}"[^"\n]{0,500}"\s{0,20},'
    r'\s{0,20}"([^"\n]{0,500}\{[^}\n]{0,200}\}[^"\n]{0,500})"'
)

# Regex to extract method and path from make_request calls
# ReDoS-safe: bounded quantifiers
MAKE_REQUEST_CALL_RE: Pattern[str] = re.compile(
    r'make_request\(\s{0,20}"(\w+)"\s{0,20},\s{0,20}(?:f)?"([^"\n]{0,500})"',
)

# ---------------------------------------------------------------------------
# code_processor.py patterns
# ---------------------------------------------------------------------------

# Match class inheriting from BaseWorkflow
# ReDoS-safe: bounded [^)] quantifiers
WORKFLOW_CLASS_RE: Pattern[str] = re.compile(
    r"class\s+(\w+)\s*\([^)\n]{0,200}BaseWorkflow[^)\n]{0,200}\)\s*:"
)

# Problematic regex escape sequences in generated code
PROBLEMATIC_ESCAPES_RE: Pattern[str] = re.compile(r"\\[dDwWsS+*?^$.|()\\[\]{}]")

# ---------------------------------------------------------------------------
# scenario_generator.py patterns
# ---------------------------------------------------------------------------

# Pattern to find ALLOWED IMPORTS section in templates
ALLOWED_IMPORTS_RE: Pattern[str] = re.compile(
    r"===\s{0,20}ALLOWED IMPORTS[^=\n]{0,100}===.*?```python\s{0,20}(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Jinja2 block/variable syntax patterns (for cleaning template code)
JINJA_BLOCK_RE: Pattern[str] = re.compile(r"\{%.*?%\}")
JINJA_VAR_RE: Pattern[str] = re.compile(r"\{\{.*?\}\}")


# ---------------------------------------------------------------------------
# Shared utility functions (de-duplicated from locust_generator + scenario_generator)
# ---------------------------------------------------------------------------


def sanitize_identifier(name: str) -> str:
    """Sanitize string to be a valid Python identifier.

    Replaces common separators with underscores, removes non-alphanumeric
    characters, collapses consecutive underscores, and ensures the result
    doesn't start with a digit.
    """
    # Replace common separators with underscores
    name = name.replace("-", "_").replace(" ", "_").replace(".", "_").replace("/", "_")
    # Remove any remaining non-alphanumeric chars (except underscore)
    name = NON_ALNUM_RE.sub("", name)
    # Remove consecutive underscores
    name = MULTI_UNDERSCORE_RE.sub("_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    # Ensure doesn't start with a number
    if name and name[0].isdigit():
        name = f"n{name}"
    return name or "unnamed"


def to_class_name(name: str) -> str:
    """Convert name to PascalCase class name.

    First sanitizes the name, then capitalizes each word segment.
    """
    sanitized = sanitize_identifier(name)
    words = sanitized.replace("_", " ").split()
    return "".join(word.capitalize() for word in words) or "Unnamed"


# ---------------------------------------------------------------------------
# URL detection helpers — these check protocol prefixes, NOT make connections.
# SonarCloud S5332 (clear-text protocol) does not apply here.
# ---------------------------------------------------------------------------
URL_PREFIXES: tuple[str, ...] = ("http://", "https://")  # NOSONAR


def is_url(value: str) -> bool:
    """Check if a string is a URL (starts with http:// or https://).

    Used for distinguishing URLs from file paths, not for making connections.
    """
    return value.startswith(URL_PREFIXES)  # NOSONAR
