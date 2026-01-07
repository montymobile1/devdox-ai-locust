"""Unit tests for :mod:`devdox_ai_locust.validation.code_validator`."""

import textwrap

import pytest

from devdox_ai_locust.validation.code_validator import CodeValidator


class TestCodeValidator:
    def test_is_valid_python_reports_line_number(self):
        code = "def broken(:\n    pass"

        valid, message = CodeValidator.is_valid_python(code)

        assert not valid
        assert "Line 1" in message

    def test_hardcoded_secrets_detected_and_placeholders_ignored(self):
        real_secret = "token = 'abcdefghijklmnopqrstuvwxyz1234567890'"
        placeholder_secret = "api_key = 'example_api_key'"

        has_secrets, issues = CodeValidator.has_hardcoded_secrets(real_secret)
        placeholder_has_secrets, placeholder_issues = CodeValidator.has_hardcoded_secrets(
            placeholder_secret
        )

        assert has_secrets
        assert any("Possible hardcoded token" in issue for issue in issues)
        assert not placeholder_has_secrets
        assert placeholder_issues == []

    def test_find_missing_catch_response_flags_contexts(self):
        code = textwrap.dedent(
            """
            class Example(HttpUser):
                @task
                def missing_catch(self):
                    with self.client.get("/missing") as response:
                        response.success()

                @task
                def has_catch(self):
                    with self.client.post("/ok", catch_response=True) as response:
                        response.success()
            """
        )

        issues = CodeValidator.find_missing_catch_response(code)

        assert issues == [
            "Missing catch_response=True in with self.client.get() context manager"
        ]

    def test_validate_combines_issues_and_warnings(self):
        code = textwrap.dedent(
            """
            from something import UserTask

            class UserTask:
                pass

            # Mention HttpUser to trigger structure check
            class Another:
                def on_start(self):
                    with self.client.get("/one") as response:
                        response.failure("oops")
            """
        )

        result = CodeValidator().validate(code)

        assert not result.is_valid
        assert any("Classes both imported and defined" in issue for issue in result.issues)
        assert any("HttpUser should be used as base class" in warn for warn in result.warnings)

    def test_validate_method_code_wraps_and_validates(self):
        method_body = "return value"

        result = CodeValidator().validate_method_code(method_body)

        assert result.is_valid

