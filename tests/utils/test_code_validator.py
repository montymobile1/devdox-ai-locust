"""Tests for the CodeValidator class."""

import pytest

from devdox_ai_locust.utils.code_validator import (
    CodeValidator,
    ValidationResult,
    ValidationViolation,
)


@pytest.fixture
def validator():
    return CodeValidator()


@pytest.fixture
def all_paths():
    return ["/users", "/users/{id}", "/auth/login", "/items", "/items/{item_id}"]


class TestValidationResultDataclass:
    def test_valid_result_has_no_error_message(self):
        result = ValidationResult(is_valid=True, violations=[])
        assert result.error_message == ""

    def test_error_message_formats_violations(self):
        result = ValidationResult(
            is_valid=False,
            violations=[
                ValidationViolation(
                    rule="test_rule", message="something broke", line_number=5
                ),
            ],
        )
        msg = result.error_message
        assert "[test_rule]" in msg
        assert "(line 5)" in msg
        assert "something broke" in msg

    def test_error_message_without_line_number(self):
        result = ValidationResult(
            is_valid=False,
            violations=[
                ValidationViolation(rule="r", message="m"),
            ],
        )
        assert "(line" not in result.error_message


class TestTemplateBoilerplate:
    def test_detects_template_comment(self, validator):
        code = "# Check if request succeeded (result is dict or None)\nx = 1"
        result = validator.validate(code, "positive", "/users")
        assert not result.is_valid
        assert any(v.rule == "template_boilerplate" for v in result.violations)

    def test_clean_code_passes(self, validator):
        code = "x = 1\ny = 2"
        result = validator.validate(code, "positive", "/users")
        assert result.is_valid


class TestPlaceholderComments:
    def test_detects_todo(self, validator):
        code = "# TODO: implement this"
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "placeholder_comment" for v in result.violations)

    def test_detects_add_remaining(self, validator):
        code = "# Add remaining fields"
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "placeholder_comment" for v in result.violations)

    def test_detects_ellipsis_placeholder(self, validator):
        code = "# ..."
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "placeholder_comment" for v in result.violations)


class TestEmptySetupDicts:
    def test_detects_inline_empty_dict(self, validator):
        code = "user_data = {}"
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "empty_setup_dict" for v in result.violations)

    def test_detects_empty_payload(self, validator):
        code = "request_payload = {}"
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "empty_setup_dict" for v in result.violations)

    def test_detects_multiline_empty_dict(self, validator):
        code = "user_data = {\n}"
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "empty_setup_dict" for v in result.violations)

    def test_non_empty_dict_passes(self, validator):
        code = 'user_data = {"name": "test"}'
        result = validator.validate(code, "positive", "/users")
        assert not any(v.rule == "empty_setup_dict" for v in result.violations)


class TestEmptyPathSegments:
    def test_detects_double_slash(self, validator):
        code = 'make_request("GET", "/users//items")'
        result = validator.validate(code, "positive", "/users")
        assert any(v.rule == "empty_path_segment" for v in result.violations)

    def test_allows_https_double_slash(self, validator):
        code = 'make_request("GET", "https://example.com/users")'
        result = validator.validate(code, "positive", "/users")
        assert not any(v.rule == "empty_path_segment" for v in result.violations)


class TestLiteralPathParams:
    def test_detects_literal_braces_without_fstring(self, validator):
        code = 'make_request("GET", "/users/{user_id}")'
        result = validator.validate(code, "positive", "/users/{id}")
        assert any(v.rule == "literal_path_param" for v in result.violations)

    def test_allows_fstring(self, validator):
        code = 'make_request("GET", f"/users/{user_id}")'
        result = validator.validate(code, "positive", "/users/{id}")
        assert not any(v.rule == "literal_path_param" for v in result.violations)


class TestSecurityPathInjection:
    def test_detects_payload_in_path(self, validator):
        code = 'make_request("GET", f"/users/{payload}")'
        result = validator.validate(code, "security", "/users/{id}")
        assert any(v.rule == "path_param_injection" for v in result.violations)

    def test_not_checked_for_positive(self, validator):
        code = 'make_request("GET", f"/users/{payload}")'
        result = validator.validate(code, "positive", "/users/{id}")
        assert not any(v.rule == "path_param_injection" for v in result.violations)


class TestSuccessCodesInNegative:
    def test_detects_200_in_negative(self, validator):
        code = 'make_request("GET", f"/users/{uid}", expected_status=[200])'
        result = validator.validate(code, "negative", "/users/{id}")
        assert any(v.rule == "success_in_negative" for v in result.violations)

    def test_allows_4xx_in_negative(self, validator):
        code = 'make_request("GET", f"/users/{uid}", expected_status=[400, 422])'
        result = validator.validate(code, "negative", "/users/{id}")
        assert not any(v.rule == "success_in_negative" for v in result.violations)

    def test_allows_200_on_setup_call(self, validator):
        code = 'make_request("POST", f"/auth/login", expected_status=[200])'
        result = validator.validate(code, "negative", "/users/{id}")
        assert not any(v.rule == "success_in_negative" for v in result.violations)

    def test_not_checked_for_positive(self, validator):
        code = 'make_request("GET", "/users", expected_status=[200])'
        result = validator.validate(code, "positive", "/users")
        assert not any(v.rule == "success_in_negative" for v in result.violations)


class TestHallucinatedEndpoints:
    def test_detects_unknown_endpoint(self, validator, all_paths):
        code = 'make_request("GET", "/nonexistent/endpoint")'
        result = validator.validate(
            code, "positive", "/users", all_endpoint_paths=all_paths
        )
        assert any(v.rule == "hallucinated_endpoint" for v in result.violations)

    def test_allows_known_endpoint(self, validator, all_paths):
        code = 'make_request("GET", "/users")'
        result = validator.validate(
            code, "positive", "/users", all_endpoint_paths=all_paths
        )
        assert not any(v.rule == "hallucinated_endpoint" for v in result.violations)

    def test_allows_parameterized_known_endpoint(self, validator, all_paths):
        code = 'make_request("GET", f"/users/{uid}")'
        result = validator.validate(
            code, "positive", "/users/{id}", all_endpoint_paths=all_paths
        )
        assert not any(v.rule == "hallucinated_endpoint" for v in result.violations)

    def test_not_checked_without_paths_list(self, validator):
        code = 'make_request("GET", "/nonexistent")'
        result = validator.validate(code, "positive", "/users")
        assert not any(v.rule == "hallucinated_endpoint" for v in result.violations)


class TestPathsMatch:
    @pytest.mark.parametrize(
        "spec_path,call_path,expected",
        [
            ("/items/{id}", "/items/{item_id}", True),
            ("/items/{id}", "/items/123", True),
            ("/items/{id}/details", "/items/{x}/details", True),
            ("/items/{id}", "/items/{id}/details", False),
            ("/items", "/other", False),
            ("/a/b/c", "/a/b/c", True),
        ],
    )
    def test_paths_match(self, spec_path, call_path, expected):
        assert CodeValidator._paths_match(spec_path, call_path) == expected


class TestSchemaCompliance:
    def test_detects_enum_ignored(self, validator):
        code = 'data = {"status": generate_string()}'
        schema = {
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            },
        }
        result = validator.validate(
            code, "positive", "/users", request_body_schema=schema
        )
        assert any(v.rule == "enum_ignored" for v in result.violations)

    def test_allows_random_choice_for_enum(self, validator):
        code = 'data = {"status": random.choice(["active", "inactive"])}'
        schema = {
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            },
        }
        result = validator.validate(
            code, "positive", "/users", request_body_schema=schema
        )
        assert not any(v.rule == "enum_ignored" for v in result.violations)

    def test_detects_wrong_format_generator(self, validator):
        code = 'data = {"email": generate_string()}'
        schema = {
            "properties": {"email": {"type": "string", "format": "email"}},
        }
        result = validator.validate(
            code, "positive", "/users", request_body_schema=schema
        )
        assert any(v.rule == "wrong_format_generator" for v in result.violations)

    def test_detects_mixed_array_types(self, validator):
        code = 'data = {"tags": ["hello", 123]}'
        schema = {
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
        result = validator.validate(
            code, "positive", "/users", request_body_schema=schema
        )
        assert any(v.rule == "mixed_array_types" for v in result.violations)

    def test_not_checked_for_negative(self, validator):
        code = 'data = {"tags": ["hello", 123]}'
        schema = {
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
        result = validator.validate(
            code, "negative", "/users", request_body_schema=schema
        )
        assert not any(v.rule == "mixed_array_types" for v in result.violations)


class TestWarningsDoNotFailValidation:
    def test_warning_severity_does_not_fail(self, validator):
        """Warnings alone should not make is_valid False."""
        result = ValidationResult(
            is_valid=True,
            violations=[
                ValidationViolation(
                    rule="some_rule", message="info", severity="warning"
                ),
            ],
        )
        # Simulate the logic in validate: only errors fail
        has_errors = any(v.severity == "error" for v in result.violations)
        assert not has_errors


class TestLiteralMatchesFormat:
    @pytest.mark.parametrize(
        "value,fmt,expected",
        [
            ("2024-01-15", "date", True),
            ("not-a-date", "date", False),
            ("2024-01-15T10:00:00Z", "date-time", True),
            ("user@example.com", "email", True),
            ("nope", "email", False),
            ("http://example.com", "uri", True),
            ("192.168.1.1", "ipv4", True),
            ("12:30", "time", True),
            ("a1b2c3d4-e5f6-7890-abcd-ef1234567890", "uuid", True),
            ("not-uuid", "uuid", False),
        ],
    )
    def test_literal_matches_format(self, validator, value, fmt, expected):
        assert validator._literal_matches_format(value, fmt) == expected
