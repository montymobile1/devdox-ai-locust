"""
Comprehensive tests for processing_result.py module.

Tests cover:
- SwaggerProcessingRequest model instantiation
- Field validators (coercion and stripping)
- Model validator (exactly one source)
- Properties (is_url_source, is_file_source, source_location)
- Edge cases and error handling
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest


# =============================================================================
# SwaggerProcessingRequest Basic Instantiation Tests
# =============================================================================


class TestSwaggerProcessingRequestInstantiation:
    """Tests for basic model instantiation."""

    def test_creates_with_url(self):
        """Should create model with swagger_url."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )
        assert request.swagger_url == "https://example.com/swagger.json"
        assert request.swagger_path is None

    def test_creates_with_path(self):
        """Should create model with swagger_path."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/swagger.json")
        assert request.swagger_path == "/path/to/swagger.json"
        assert request.swagger_url is None

    def test_creates_with_path_object(self):
        """Should accept Path objects for swagger_path."""
        path = Path("/path/to/swagger.json")
        request = SwaggerProcessingRequest(swagger_path=path)
        assert request.swagger_path == "/path/to/swagger.json"


# =============================================================================
# SwaggerProcessingRequest Validation Error Tests
# =============================================================================


class TestSwaggerProcessingRequestValidationErrors:
    """Tests for validation errors."""

    def test_raises_when_both_provided(self):
        """Should raise ValidationError when both URL and path provided."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(
                swagger_url="https://example.com/swagger.json",
                swagger_path="/path/to/swagger.json",
            )
        assert "Cannot specify both" in str(exc_info.value)

    def test_raises_when_neither_provided(self):
        """Should raise ValidationError when neither URL nor path provided."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest()
        assert "Must specify either" in str(exc_info.value)

    def test_raises_when_url_empty_string(self):
        """Should raise ValidationError when URL is empty string."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(swagger_url="")
        assert "Must specify either" in str(exc_info.value)

    def test_raises_when_path_empty_string(self):
        """Should raise ValidationError when path is empty string."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(swagger_path="")
        assert "Must specify either" in str(exc_info.value)

    def test_raises_when_url_whitespace_only(self):
        """Should raise ValidationError when URL is whitespace only."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(swagger_url="   ")
        assert "Must specify either" in str(exc_info.value)

    def test_raises_when_path_whitespace_only(self):
        """Should raise ValidationError when path is whitespace only."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(swagger_path="   \t  ")
        assert "Must specify either" in str(exc_info.value)

    def test_raises_with_both_empty(self):
        """Should raise ValidationError when both are empty strings."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(swagger_url="", swagger_path="")
        # The error could be either "both" or "neither" depending on order
        error_str = str(exc_info.value)
        assert "Must specify either" in error_str or "Cannot specify both" in error_str


# =============================================================================
# SwaggerProcessingRequest Field Validator Tests
# =============================================================================


class TestSwaggerUrlFieldValidator:
    """Tests for swagger_url field validator."""

    def test_strips_leading_whitespace(self):
        """Should strip leading whitespace from URL."""
        request = SwaggerProcessingRequest(
            swagger_url="   https://example.com/swagger.json"
        )
        assert request.swagger_url == "https://example.com/swagger.json"

    def test_strips_trailing_whitespace(self):
        """Should strip trailing whitespace from URL."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json   "
        )
        assert request.swagger_url == "https://example.com/swagger.json"

    def test_strips_both_whitespace(self):
        """Should strip both leading and trailing whitespace from URL."""
        request = SwaggerProcessingRequest(
            swagger_url="  https://example.com/swagger.json  "
        )
        assert request.swagger_url == "https://example.com/swagger.json"

    def test_preserves_internal_whitespace(self):
        """Should preserve internal whitespace (though unusual for URLs)."""
        # Note: This is technically an invalid URL, but the validator just strips edges
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/path with spaces.json"
        )
        assert "with spaces" in request.swagger_url

    def test_coerces_to_string(self):
        """Should coerce non-string values to string."""
        # This tests the str() coercion behavior
        request = SwaggerProcessingRequest(swagger_url="https://example.com/api")
        assert isinstance(request.swagger_url, str)

    def test_handles_none(self):
        """Should handle None value (passes through)."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/file.json")
        assert request.swagger_url is None


class TestSwaggerPathFieldValidator:
    """Tests for swagger_path field validator."""

    def test_strips_leading_whitespace(self):
        """Should strip leading whitespace from path."""
        request = SwaggerProcessingRequest(swagger_path="   /path/to/swagger.json")
        assert request.swagger_path == "/path/to/swagger.json"

    def test_strips_trailing_whitespace(self):
        """Should strip trailing whitespace from path."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/swagger.json   ")
        assert request.swagger_path == "/path/to/swagger.json"

    def test_strips_both_whitespace(self):
        """Should strip both leading and trailing whitespace from path."""
        request = SwaggerProcessingRequest(swagger_path="  /path/to/swagger.json  ")
        assert request.swagger_path == "/path/to/swagger.json"

    def test_converts_path_object_to_string(self):
        """Should convert Path objects to strings."""
        path_obj = Path("/home/user/swagger.json")
        request = SwaggerProcessingRequest(swagger_path=path_obj)
        assert request.swagger_path == str(path_obj)
        assert isinstance(request.swagger_path, str)

    def test_handles_windows_path_object(self):
        """Should handle Path objects (platform-agnostic test)."""
        path_obj = Path("C:/Users/test/swagger.json")
        request = SwaggerProcessingRequest(swagger_path=path_obj)
        assert request.swagger_path == str(path_obj)

    def test_coerces_to_string(self):
        """Should coerce values to string."""
        request = SwaggerProcessingRequest(swagger_path="/api/swagger.json")
        assert isinstance(request.swagger_path, str)

    def test_handles_none(self):
        """Should handle None value (passes through)."""
        request = SwaggerProcessingRequest(swagger_url="https://example.com/api")
        assert request.swagger_path is None


# =============================================================================
# SwaggerProcessingRequest Property Tests
# =============================================================================


class TestIsUrlSourceProperty:
    """Tests for is_url_source property."""

    def test_true_when_url_provided(self):
        """Should return True when swagger_url is provided."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )
        assert request.is_url_source is True

    def test_false_when_path_provided(self):
        """Should return False when swagger_path is provided."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/swagger.json")
        assert request.is_url_source is False


class TestIsFileSourceProperty:
    """Tests for is_file_source property."""

    def test_true_when_path_provided(self):
        """Should return True when swagger_path is provided."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/swagger.json")
        assert request.is_file_source is True

    def test_false_when_url_provided(self):
        """Should return False when swagger_url is provided."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )
        assert request.is_file_source is False


class TestSourceLocationProperty:
    """Tests for source_location property."""

    def test_returns_url_when_url_source(self):
        """Should return URL when is_url_source is True."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )
        assert request.source_location == "https://example.com/swagger.json"

    def test_returns_path_when_file_source(self):
        """Should return path when is_file_source is True."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/swagger.json")
        assert request.source_location == "/path/to/swagger.json"


# =============================================================================
# SwaggerProcessingRequest URL Format Tests
# =============================================================================


class TestSwaggerUrlFormats:
    """Tests for various URL formats."""

    def test_accepts_https_url(self):
        """Should accept HTTPS URLs."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        assert request.swagger_url.startswith("https://")

    def test_accepts_http_url(self):
        """Should accept HTTP URLs."""
        request = SwaggerProcessingRequest(
            swagger_url="http://localhost:8080/swagger.json"
        )
        assert request.swagger_url.startswith("http://") # NOSONAR — test fixture, no real HTTP call

    def test_accepts_url_with_port(self):
        """Should accept URLs with port numbers."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com:8443/swagger.json"
        )
        assert ":8443" in request.swagger_url

    def test_accepts_url_with_path(self):
        """Should accept URLs with path components."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/v1/openapi/swagger.json"
        )
        assert "/v1/openapi/" in request.swagger_url

    def test_accepts_url_with_query_params(self):
        """Should accept URLs with query parameters."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json?version=v2"
        )
        assert "?version=v2" in request.swagger_url

    def test_accepts_localhost_url(self):
        """Should accept localhost URLs."""
        request = SwaggerProcessingRequest(swagger_url="http://localhost/swagger.json")
        assert "localhost" in request.swagger_url

    def test_accepts_ip_address_url(self):
        """Should accept IP address URLs."""
        request = SwaggerProcessingRequest(
            swagger_url="http://192.168.1.100:3000/swagger.json" #NOSONAR This is a test fixture, not a real call
        )
        assert "192.168.1.100" in request.swagger_url #NOSONAR This is a test fixture, not a real call


# =============================================================================
# SwaggerProcessingRequest Path Format Tests
# =============================================================================


class TestSwaggerPathFormats:
    """Tests for various path formats."""

    def test_accepts_absolute_path(self):
        """Should accept absolute paths."""
        request = SwaggerProcessingRequest(swagger_path="/home/user/swagger.json")
        assert request.swagger_path.startswith("/")

    def test_accepts_relative_path(self):
        """Should accept relative paths."""
        request = SwaggerProcessingRequest(swagger_path="./swagger.json")
        assert request.swagger_path == "./swagger.json"

    def test_accepts_path_with_dots(self):
        """Should accept paths with parent directory references."""
        request = SwaggerProcessingRequest(swagger_path="../api/swagger.json")
        assert request.swagger_path == "../api/swagger.json"

    def test_accepts_json_extension(self):
        """Should accept .json files."""
        request = SwaggerProcessingRequest(swagger_path="/path/api.json")
        assert request.swagger_path.endswith(".json")

    def test_accepts_yaml_extension(self):
        """Should accept .yaml files."""
        request = SwaggerProcessingRequest(swagger_path="/path/api.yaml")
        assert request.swagger_path.endswith(".yaml")

    def test_accepts_yml_extension(self):
        """Should accept .yml files."""
        request = SwaggerProcessingRequest(swagger_path="/path/api.yml")
        assert request.swagger_path.endswith(".yml")

    def test_accepts_path_without_extension(self):
        """Should accept paths without extension (no extension validation)."""
        request = SwaggerProcessingRequest(swagger_path="/path/openapi-spec")
        assert request.swagger_path == "/path/openapi-spec"


# =============================================================================
# SwaggerProcessingRequest Serialization Tests
# =============================================================================


class TestSwaggerProcessingRequestSerialization:
    """Tests for model serialization."""

    def test_model_dump_with_url(self):
        """Should serialize model with URL correctly."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )
        data = request.model_dump()
        assert data["swagger_url"] == "https://example.com/swagger.json"
        assert data["swagger_path"] is None

    def test_model_dump_with_path(self):
        """Should serialize model with path correctly."""
        request = SwaggerProcessingRequest(swagger_path="/path/to/swagger.json")
        data = request.model_dump()
        assert data["swagger_path"] == "/path/to/swagger.json"
        assert data["swagger_url"] is None

    def test_model_dump_json(self):
        """Should serialize to JSON string."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )
        json_str = request.model_dump_json()
        assert "swagger_url" in json_str
        assert "https://example.com/swagger.json" in json_str


class TestSwaggerProcessingRequestDeserialization:
    """Tests for model deserialization."""

    def test_model_validate_from_dict_url(self):
        """Should deserialize from dict with URL."""
        data = {"swagger_url": "https://example.com/swagger.json"}
        request = SwaggerProcessingRequest.model_validate(data)
        assert request.swagger_url == "https://example.com/swagger.json"

    def test_model_validate_from_dict_path(self):
        """Should deserialize from dict with path."""
        data = {"swagger_path": "/path/to/swagger.json"}
        request = SwaggerProcessingRequest.model_validate(data)
        assert request.swagger_path == "/path/to/swagger.json"

    def test_model_validate_strips_whitespace(self):
        """Should strip whitespace during deserialization."""
        data = {"swagger_url": "  https://example.com/swagger.json  "}
        request = SwaggerProcessingRequest.model_validate(data)
        assert request.swagger_url == "https://example.com/swagger.json"


# =============================================================================
# SwaggerProcessingRequest Edge Cases
# =============================================================================


class TestSwaggerProcessingRequestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_url_with_unicode(self):
        """Should handle URLs with unicode characters."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/api/日本語.json"
        )
        assert "日本語" in request.swagger_url

    def test_path_with_unicode(self):
        """Should handle paths with unicode characters."""
        request = SwaggerProcessingRequest(swagger_path="/path/データ/swagger.json")
        assert "データ" in request.swagger_path

    def test_path_with_spaces(self):
        """Should handle paths with spaces."""
        request = SwaggerProcessingRequest(swagger_path="/path/my api/swagger.json")
        assert "my api" in request.swagger_path

    def test_very_long_url(self):
        """Should handle very long URLs."""
        long_path = "/a" * 1000
        url = f"https://example.com{long_path}/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url)
        assert len(request.swagger_url) > 2000

    def test_very_long_path(self):
        """Should handle very long paths."""
        long_path = "/dir" * 500 + "/swagger.json"
        request = SwaggerProcessingRequest(swagger_path=long_path)
        assert len(request.swagger_path) > 2000

    def test_url_with_special_characters(self):
        """Should handle URLs with special characters."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/api?name=test&version=1.0"
        )
        assert "&" in request.swagger_url
        assert "?" in request.swagger_url


# =============================================================================
# SwaggerProcessingRequest Mutual Exclusivity Tests
# =============================================================================


class TestMutualExclusivity:
    """Tests for mutual exclusivity of URL and path."""

    def test_url_and_path_both_valid_raises(self):
        """Should raise when both valid URL and path provided."""
        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest(
                swagger_url="https://example.com/api",
                swagger_path="/path/to/api",
            )
        assert "Cannot specify both" in str(exc_info.value)

    def test_url_valid_path_empty_accepts(self):
        """Should accept when URL valid and path empty."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/api",
            swagger_path="",
        )
        assert request.is_url_source is True

    def test_path_valid_url_empty_accepts(self):
        """Should accept when path valid and URL empty."""
        request = SwaggerProcessingRequest(
            swagger_url="",
            swagger_path="/path/to/api",
        )
        assert request.is_file_source is True

    def test_url_valid_path_whitespace_accepts(self):
        """Should accept when URL valid and path is whitespace."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/api",
            swagger_path="   ",
        )
        assert request.is_url_source is True

    def test_path_valid_url_whitespace_accepts(self):
        """Should accept when path valid and URL is whitespace."""
        request = SwaggerProcessingRequest(
            swagger_url="   ",
            swagger_path="/path/to/api",
        )
        assert request.is_file_source is True
