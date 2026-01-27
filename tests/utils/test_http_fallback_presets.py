"""
Tests for http_fallback_presets module
"""

import json
import pytest

from devdox_ai_locust.utils.http_fallback_presets import (
    FallbackHttpResponseRegistry,
    ResponseBlock,
)


class TestResponseBlock:
    """Test ResponseBlock model."""

    def test_to_json(self):
        """Test JSON export."""
        block = ResponseBlock(responses={"GET": {"200": {"description": "OK"}}})
        result = json.loads(block.to_json())
        assert result == {"GET": {"200": {"description": "OK"}}}

    def test_to_yaml(self):
        """Test YAML export."""
        block = ResponseBlock(responses={"GET": {"200": {"description": "OK"}}})
        result = block.to_yaml()
        assert "200" in result
        assert "OK" in result

    def test_as_dict(self):
        """Test dictionary export."""
        data = {"GET": {"200": {"description": "OK"}}}
        block = ResponseBlock(responses=data)
        assert block.as_dict() == data


class TestFallbackHttpResponseRegistry:
    """Test FallbackHttpResponseRegistry."""

    @pytest.fixture
    def registry(self):
        return FallbackHttpResponseRegistry()

    def test_get_responses_single_method(self, registry):
        """Test getting responses for a single HTTP method."""
        result = registry.get_responses("GET")
        responses = result.as_dict()
        assert "GET" in responses
        assert "200" in responses["GET"]

    def test_get_responses_multiple_methods(self, registry):
        """Test getting responses for multiple HTTP methods."""
        result = registry.get_responses(["GET", "POST"])
        responses = result.as_dict()
        assert "GET" in responses
        assert "POST" in responses

    def test_get_responses_includes_5xx_common(self, registry):
        """Test that 5xx common codes are merged into each method."""
        result = registry.get_responses("GET")
        responses = result.as_dict()["GET"]
        assert "500" in responses
        assert "502" in responses
        assert "503" in responses
        assert "504" in responses

    def test_get_responses_filter_by_status(self, registry):
        """Test filtering by specific status codes."""
        result = registry.get_responses("GET", status=["200", "404"])
        responses = result.as_dict()["GET"]
        assert set(responses.keys()) == {"200", "404"}

    def test_get_responses_filter_by_class(self, registry):
        """Test filtering by status class like '2xx'."""
        result = registry.get_responses("GET", status=["2xx"])
        responses = result.as_dict()["GET"]
        for code in responses:
            assert code.startswith("2")

    def test_get_responses_exclude_status(self, registry):
        """Test excluding specific status codes."""
        result = registry.get_responses("GET", exclude_status=["500"])
        responses = result.as_dict()["GET"]
        assert "500" not in responses
        assert "200" in responses

    def test_get_responses_exclude_class(self, registry):
        """Test excluding a status class like '5xx'."""
        result = registry.get_responses("GET", exclude_status=["5xx"])
        responses = result.as_dict()["GET"]
        for code in responses:
            assert not code.startswith("5")

    def test_get_responses_exclude_auth(self, registry):
        """Test excluding auth codes 401 and 403."""
        result = registry.get_responses("GET", exclude_auth=True)
        responses = result.as_dict()["GET"]
        assert "401" not in responses
        assert "403" not in responses

    def test_get_responses_unknown_method(self, registry):
        """Test unknown method returns empty."""
        result = registry.get_responses("OPTIONS")
        assert result.as_dict() == {}

    def test_get_responses_case_insensitive(self, registry):
        """Test that method names are uppercased."""
        result = registry.get_responses("get")
        assert "GET" in result.as_dict()

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
    def test_all_methods_have_responses(self, registry, method):
        """Test that all standard methods return responses."""
        result = registry.get_responses(method)
        assert method in result.as_dict()
        assert len(result.as_dict()[method]) > 0

    def test_get_responses_status_as_int(self, registry):
        """Test passing status codes as integers."""
        result = registry.get_responses("GET", status=[200, 404])
        responses = result.as_dict()["GET"]
        assert "200" in responses
        assert "404" in responses
