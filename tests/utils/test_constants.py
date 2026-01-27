"""Tests for devdox_ai_locust.utils.constants module."""

import re
import time

from devdox_ai_locust.utils.constants import (
    ADDR_RE,
    ALLOWED_IMPORTS_RE,
    API_PATH_RE,
    CONTEXT_BUFFER_SIZE,
    CONTEXT_PATTERN,
    EMPTY_PATH_SEGMENT_RE,
    EMPTY_SETUP_DICT_RE,
    ERROR_PATTERN,
    JINJA_BLOCK_RE,
    JINJA_VAR_RE,
    LITERAL_PATH_PARAM_RE,
    LONG_VALUE_RE,
    MAKE_REQUEST_CALL_RE,
    MULTI_UNDERSCORE_RE,
    NON_ALNUM_RE,
    NUMERIC_PATH_RE,
    PLACEHOLDER_PATTERNS,
    PROBLEMATIC_ESCAPES_RE,
    REQUEST_LINE,
    SECURITY_PAYLOAD_IN_PATH_RE,
    SUCCESS_IN_EXPECTED_STATUS_RE,
    TEMPLATE_BOILERPLATE_PATTERNS,
    TIMESTAMP_RE,
    TRACEBACK_START,
    URL_ARG_RE,
    UUID_RE,
    WORKFLOW_CLASS_RE,
    sanitize_identifier,
    to_class_name,
)


# ---- sanitize_identifier ----


def test_sanitize_identifier_normal():
    assert sanitize_identifier("hello_world") == "hello_world"


def test_sanitize_identifier_special_chars():
    assert sanitize_identifier("hello-world.foo/bar baz") == "hello_world_foo_bar_baz"


def test_sanitize_identifier_numeric_leading():
    assert sanitize_identifier("123abc") == "n123abc"


def test_sanitize_identifier_empty():
    assert sanitize_identifier("") == "unnamed"


def test_sanitize_identifier_only_special():
    assert sanitize_identifier("@#$%") == "unnamed"


def test_sanitize_identifier_unicode():
    assert sanitize_identifier("café") == "caf"


def test_sanitize_identifier_very_long():
    result = sanitize_identifier("a" * 5000)
    assert result == "a" * 5000


def test_sanitize_identifier_leading_trailing_underscores():
    assert sanitize_identifier("__foo__") == "foo"


def test_sanitize_identifier_consecutive_separators():
    assert sanitize_identifier("a--b..c//d  e") == "a_b_c_d_e"


# ---- to_class_name ----


def test_to_class_name_snake_case():
    assert to_class_name("hello_world") == "HelloWorld"


def test_to_class_name_kebab_case():
    assert to_class_name("hello-world") == "HelloWorld"


def test_to_class_name_pascal_passthrough():
    assert to_class_name("HelloWorld") == "Helloworld"


def test_to_class_name_empty():
    assert to_class_name("") == "Unnamed"


# ---- Regex: ERROR_PATTERN ----


def test_error_pattern_match():
    assert ERROR_PATTERN.search("some ERROR happened")
    assert ERROR_PATTERN.search("critical failure")


def test_error_pattern_no_match():
    assert not ERROR_PATTERN.search("everything is fine")


# ---- Regex: TRACEBACK_START ----


def test_traceback_start_match():
    assert TRACEBACK_START.search("Traceback (most recent call last):")


def test_traceback_start_no_match():
    assert not TRACEBACK_START.search("no traceback here")


# ---- Regex: UUID_RE ----


def test_uuid_re_match():
    assert UUID_RE.search("id=550e8400-e29b-41d4-a716-446655440000")


def test_uuid_re_no_match():
    assert not UUID_RE.search("not-a-uuid-value")


# ---- Regex: TIMESTAMP_RE ----


def test_timestamp_re_match():
    assert TIMESTAMP_RE.search("2024-01-15T12:30:45.123")
    assert TIMESTAMP_RE.search("2024-01-15 12:30:45")


def test_timestamp_re_no_match():
    assert not TIMESTAMP_RE.search("not a timestamp")


# ---- Regex: NUMERIC_PATH_RE ----


def test_numeric_path_re_match():
    assert NUMERIC_PATH_RE.search("/api/items/123")


def test_numeric_path_re_no_match():
    assert not NUMERIC_PATH_RE.search("/api/items/a")


# ---- Regex: ADDR_RE ----


def test_addr_re_match():
    assert ADDR_RE.search("object at 0x7f1234abcdef")


def test_addr_re_no_match():
    assert not ADDR_RE.search("no address here")


# ---- Regex: LONG_VALUE_RE ----


def test_long_value_re_match():
    assert LONG_VALUE_RE.search("'" + "x" * 60 + "'")


def test_long_value_re_no_match():
    assert not LONG_VALUE_RE.search("'short'")


# ---- Regex: REQUEST_LINE ----


def test_request_line_match():
    m = REQUEST_LINE.search("REQUEST: GET /api/users")
    assert m and m.group(1) == "GET" and m.group(2) == "/api/users"


def test_request_line_no_match():
    assert not REQUEST_LINE.search("just a normal line")


# ---- Regex: API_PATH_RE ----


def test_api_path_re_match():
    m = API_PATH_RE.search("POST /api/items")
    assert m and m.group(1) == "POST"


def test_api_path_re_no_match():
    assert not API_PATH_RE.search("no method here")


# ---- Regex: CONTEXT_PATTERN ----


def test_context_pattern_match():
    assert CONTEXT_PATTERN.search("Response body is json")


def test_context_pattern_no_match():
    assert not CONTEXT_PATTERN.search("nothing relevant")


# ---- Regex: EMPTY_SETUP_DICT_RE ----


def test_empty_setup_dict_re_match():
    assert EMPTY_SETUP_DICT_RE.search("login_data = {}")
    assert EMPTY_SETUP_DICT_RE.search("create_data = { # empty\n}")


def test_empty_setup_dict_re_no_match():
    assert not EMPTY_SETUP_DICT_RE.search('login_data = {"key": "val"}')


# ---- Regex: SECURITY_PAYLOAD_IN_PATH_RE ----


def test_security_payload_in_path_re_match():
    text = 'make_request("GET", f"/api/{payload}/items"'
    assert SECURITY_PAYLOAD_IN_PATH_RE.search(text)


def test_security_payload_in_path_re_no_match():
    text = 'make_request("GET", f"/api/items"'
    assert not SECURITY_PAYLOAD_IN_PATH_RE.search(text)


# ---- Regex: EMPTY_PATH_SEGMENT_RE ----


def test_empty_path_segment_re_match():
    text = 'make_request("GET", "/api//items"'
    assert EMPTY_PATH_SEGMENT_RE.search(text)


def test_empty_path_segment_re_no_match():
    text = 'make_request("GET", "https://example.com/api"'
    assert not EMPTY_PATH_SEGMENT_RE.search(text)


# ---- Regex: SUCCESS_IN_EXPECTED_STATUS_RE ----


def test_success_in_expected_status_re_match():
    m = SUCCESS_IN_EXPECTED_STATUS_RE.search("expected_status=[200, 201]")
    assert m and "200" in m.group(1)


def test_success_in_expected_status_re_no_match():
    assert not SUCCESS_IN_EXPECTED_STATUS_RE.search("status=200")


# ---- Regex: URL_ARG_RE ----


def test_url_arg_re_match():
    m = URL_ARG_RE.search('make_request("GET", "/api/items"')
    assert m and m.group(1) == "/api/items"


def test_url_arg_re_no_match():
    assert not URL_ARG_RE.search("not_a_call()")


# ---- Regex: LITERAL_PATH_PARAM_RE ----


def test_literal_path_param_re_match():
    text = 'make_request("GET", "/api/{item_id}/details"'
    assert LITERAL_PATH_PARAM_RE.search(text)


def test_literal_path_param_re_no_match():
    text = 'make_request("GET", f"/api/{item_id}/details"'
    assert not LITERAL_PATH_PARAM_RE.search(text)


# ---- Regex: MAKE_REQUEST_CALL_RE ----


def test_make_request_call_re_match():
    m = MAKE_REQUEST_CALL_RE.search('make_request("POST", "/api/users"')
    assert m and m.group(1) == "POST" and m.group(2) == "/api/users"


def test_make_request_call_re_no_match():
    assert not MAKE_REQUEST_CALL_RE.search("other_func()")


# ---- Regex: WORKFLOW_CLASS_RE ----


def test_workflow_class_re_match():
    m = WORKFLOW_CLASS_RE.search("class MyFlow(BaseWorkflow):")
    assert m and m.group(1) == "MyFlow"


def test_workflow_class_re_no_match():
    assert not WORKFLOW_CLASS_RE.search("class MyFlow(OtherBase):")


# ---- Regex: PROBLEMATIC_ESCAPES_RE ----


def test_problematic_escapes_re_match():
    assert PROBLEMATIC_ESCAPES_RE.search(r"pattern = \d+")


def test_problematic_escapes_re_no_match():
    assert not PROBLEMATIC_ESCAPES_RE.search("no escapes here")


# ---- Regex: ALLOWED_IMPORTS_RE ----


def test_allowed_imports_re_match():
    text = "=== ALLOWED IMPORTS ===\n```python\nimport os\n```"
    m = ALLOWED_IMPORTS_RE.search(text)
    assert m and "import os" in m.group(1)


def test_allowed_imports_re_no_match():
    assert not ALLOWED_IMPORTS_RE.search("no imports section here")


# ---- Regex: JINJA_BLOCK_RE ----


def test_jinja_block_re_match():
    assert JINJA_BLOCK_RE.search("{% if x %}")


def test_jinja_block_re_no_match():
    assert not JINJA_BLOCK_RE.search("no jinja here")


# ---- Regex: JINJA_VAR_RE ----


def test_jinja_var_re_match():
    assert JINJA_VAR_RE.search("{{ variable }}")


def test_jinja_var_re_no_match():
    assert not JINJA_VAR_RE.search("no jinja vars")


# ---- Regex: NON_ALNUM_RE ----


def test_non_alnum_re_match():
    assert NON_ALNUM_RE.search("hello@world")


def test_non_alnum_re_no_match():
    assert not NON_ALNUM_RE.search("hello_world123")


# ---- Regex: MULTI_UNDERSCORE_RE ----


def test_multi_underscore_re_match():
    assert MULTI_UNDERSCORE_RE.search("a___b")


def test_multi_underscore_re_no_match():
    assert not MULTI_UNDERSCORE_RE.search("no underscores here")


# ---- Pattern lists ----


def test_template_boilerplate_patterns_compile_and_match():
    for pat_str in TEMPLATE_BOILERPLATE_PATTERNS:
        compiled = re.compile(pat_str)
        assert compiled.pattern  # compiles without error

    # Spot-check a known match
    p = re.compile(TEMPLATE_BOILERPLATE_PATTERNS[0])
    assert p.search("# Check if request succeeded (result is dict or None)")


def test_placeholder_patterns_compile_and_match():
    for pat_str in PLACEHOLDER_PATTERNS:
        compiled = re.compile(pat_str)
        assert compiled.pattern

    p = re.compile(PLACEHOLDER_PATTERNS[1])
    assert p.search("# TODO: fix this")


# ---- CONTEXT_BUFFER_SIZE ----


def test_context_buffer_size():
    assert CONTEXT_BUFFER_SIZE == 8


# ---- ReDoS adversarial tests ----


def _assert_fast(pattern, text, max_seconds=1.0):
    """Assert that pattern matching completes within max_seconds."""
    start = time.time()
    pattern.search(text)
    elapsed = time.time() - start
    assert (
        elapsed < max_seconds
    ), f"Pattern {pattern.pattern!r:.60} took {elapsed:.2f}s on input of length {len(text)}"


def test_redos_empty_setup_dict_re():
    adversarial = "login_data" + " " * 1000 + "= {}"
    _assert_fast(EMPTY_SETUP_DICT_RE, adversarial)


def test_redos_security_payload_in_path_re():
    adversarial = "make_request(" + "x" * 1000 + '"'
    _assert_fast(SECURITY_PAYLOAD_IN_PATH_RE, adversarial)


def test_redos_url_arg_re():
    adversarial = "make_request(" + " " * 1000 + '"GET"'
    _assert_fast(URL_ARG_RE, adversarial)


def test_redos_literal_path_param_re():
    adversarial = "make_request(" + " " * 1000 + '"GET"'
    _assert_fast(LITERAL_PATH_PARAM_RE, adversarial)


def test_redos_make_request_call_re():
    adversarial = "make_request(" + " " * 1000 + '"GET"'
    _assert_fast(MAKE_REQUEST_CALL_RE, adversarial)


def test_redos_workflow_class_re():
    adversarial = "class " + "A" * 1000 + "(BaseWorkflow):"
    _assert_fast(WORKFLOW_CLASS_RE, adversarial)


def test_redos_allowed_imports_re():
    adversarial = "=== ALLOWED IMPORTS ===" + "x" * 1000 + "```python\nimport os\n```"
    _assert_fast(ALLOWED_IMPORTS_RE, adversarial)
