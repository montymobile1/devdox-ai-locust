"""Unit tests for HybridLocustGenerator helpers."""
import io
import re
from unittest.mock import AsyncMock

from devdox_ai_locust.hybrid_loctus_generator import (
    HybridLocustGenerator,
    SafeCodeMerger,
    format_auth_endpoints_for_prompt,
    CodebaseAwareness,
    METHOD_PATTERN,
)
from devdox_ai_locust.utils.open_ai_parser import Endpoint


def make_endpoint(path: str, method: str = "GET", summary: str | None = None):
    return Endpoint(
        path=path,
        method=method,
        operation_id=None,
        summary=summary,
        description=None,
        parameters=[],
        request_body=None,
        responses=[],
        tags=[],
        security=None,
    )


def test_safe_code_merger_extracts_new_methods():
    original = "class Demo:\n    def existing(self):\n        return 1\n"
    ai_code = "class Demo:\n    def existing(self):\n        return 2\n\n    def new_method(self):\n        return 3\n"
    new_methods = SafeCodeMerger.extract_new_methods_only(original, ai_code)
    assert "new_method" in new_methods
    assert "existing" not in new_methods


def test_imported_classes_fallback_parsing():
    code = "from utils import Helper\nclass Demo\n    pass"
    imported = SafeCodeMerger.get_imported_classes(code)
    assert "Helper" in imported


def test_safe_merge_inserts_methods():
    original = "class Demo:\n    def existing(self):\n        return 1\n"
    ai_code = "def new_method(self):\n    return 3\n"
    merged = SafeCodeMerger.safe_merge(original, ai_code, target_class="Demo")
    assert "new_method" in merged
    assert "existing" in merged


def test_format_auth_endpoints():
    endpoints = [make_endpoint("/auth/login", "POST", "Login")]
    formatted = format_auth_endpoints_for_prompt(endpoints)
    assert "Available authentication endpoints" in formatted
    assert "POST /auth/login" in formatted


def test_codebase_awareness_constraints():
    awareness = CodebaseAwareness()
    base_files = {
        "utils.py": "class Helper:\n    def run(self):\n        pass\n",
        "locustfile.py": "from utils import Helper\n",
    }
    awareness.analyze_codebase(base_files, [])
    constraints = awareness.get_constraints_for_file("utils.py")
    assert "PROTECTED SYMBOLS" in constraints
    assert "Helper" in constraints


def test_hybrid_error_classification_and_domains():
    generator = HybridLocustGenerator(ai_client=AsyncMock())
    auth_error = generator._classify_error(Exception("401 unauthorized"), 0)
    assert auth_error.is_retryable is False

    rate_error = generator._classify_error(Exception("429 rate limit"), 1)
    assert rate_error.is_retryable is True
    assert rate_error.backoff_seconds == generator.RATE_LIMIT_BACKOFF

    endpoints = [make_endpoint("/cart/items", "GET")]
    assert generator._detect_domain_patterns(endpoints, {"title": "Shop"}) is True


def test_extract_code_from_response_and_clean():
    generator = HybridLocustGenerator(ai_client=AsyncMock())
    response = "<new_methods>\n@task(1)\ndef foo(self):\n    pass\n</new_methods>"
    extracted = generator.extract_code_from_response(response, require_tags=True)
    assert "def foo" in extracted

    raw = "```python\nclass TestDataGenerator:\n    def foo(self):\n        pass\n```"
    cleaned = generator._clean_ai_response(raw)
    assert "class TestDataGenerator" not in cleaned


def test_validation_and_path_helpers():
    generator = HybridLocustGenerator(ai_client=AsyncMock())
    enhanced = (
        "class ResponseValidator:\n    pass\n"
        "class RequestLogger:\n    pass\n"
        "class PerformanceMonitor:\n    pass\n"
        "class DataManager:\n    pass\n"
    )
    valid, _, missing = generator._validate_critical_elements(
        "utils.py", enhanced, enhanced
    )
    assert valid is True
    assert missing == []

    patterns = generator._extract_path_patterns(["/api/v1/users/{id}", "/api/v1/orders"])
    assert "/api/v1" in patterns

    resources = generator._extract_resources_from_paths(["/api/v1/users/{id}", "/api/v1/orders"])
    assert "users" in resources


def test_regex_consumes_body_not_zero_with_dotall():
    ai_code = (
        "def foo():\n"
        "    x = 1\n"
        "    return x\n"
        "\n"
        "def bar():\n"
        "    return 2\n"
    )

    matches = re.findall(METHOD_PATTERN, ai_code, re.DOTALL)
    full_match, name = matches[0]

    assert name == "foo"
    assert "return x" in full_match
    assert len(full_match) > len("def foo():")


def test_regex_can_match_zero_between_def_and_boundary():
    ai_code = "def foo():\ndef bar():\n"

    matches = re.findall(METHOD_PATTERN, ai_code, re.DOTALL)
    full_match, name = matches[0]

    assert name == "foo"
    assert full_match == "def foo():"
