"""
Tests for processing_result schema module
"""

import pytest
from pydantic import ValidationError

from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest


class TestSwaggerProcessingRequestWithUrl:
    """Test SwaggerProcessingRequest when providing swagger_url only."""

    def test_valid_swagger_url(self):
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        assert request.swagger_url == "https://api.example.com/swagger.json"
        assert request.swagger_file_path is None

    def test_http_url(self):
        request = SwaggerProcessingRequest(
            swagger_url="http://api.example.com/swagger.json"
        )
        assert request.swagger_url == "http://api.example.com/swagger.json"

    def test_localhost_url(self):
        request = SwaggerProcessingRequest(
            swagger_url="http://localhost:8080/swagger.json"
        )
        assert request.swagger_url == "http://localhost:8080/swagger.json"

    def test_ip_address_url(self):
        request = SwaggerProcessingRequest(
            swagger_url="http://192.168.1.100:8080/swagger.json"
        )
        assert request.swagger_url == "http://192.168.1.100:8080/swagger.json"

    def test_yaml_file_url(self):
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.yaml"
        )
        assert request.swagger_url == "https://api.example.com/swagger.yaml"

    def test_url_with_query_params(self):
        url = "https://api.example.com/swagger.json?version=v1&format=json"
        request = SwaggerProcessingRequest(swagger_url=url)
        assert request.swagger_url == url

    def test_url_with_fragment(self):
        url = "https://api.example.com/swagger.json#section"
        request = SwaggerProcessingRequest(swagger_url=url)
        assert request.swagger_url == url

    def test_very_long_url(self):
        long_path = "x" * 1000
        url = f"https://api.example.com/{long_path}/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url)
        assert request.swagger_url == url

    def test_unicode_url(self):
        url = (
            "https://api.example.com/\u30c9\u30ad\u30e5\u30e1\u30f3\u30c8/swagger.json"
        )
        request = SwaggerProcessingRequest(swagger_url=url)
        assert request.swagger_url == url

    def test_url_with_credentials(self):
        url = "https://username:password@api.example.com/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url)
        assert request.swagger_url == url

    def test_url_with_port(self):
        url = "https://api.example.com:8443/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url)
        assert request.swagger_url == url


class TestSwaggerProcessingRequestWithFilePath:
    """Test SwaggerProcessingRequest when providing swagger_file_path only."""

    def test_valid_file_path(self):
        request = SwaggerProcessingRequest(swagger_file_path="/path/to/swagger.json")
        assert request.swagger_file_path == "/path/to/swagger.json"
        assert request.swagger_url is None

    def test_relative_file_path(self):
        request = SwaggerProcessingRequest(swagger_file_path="./swagger.json")
        assert request.swagger_file_path == "./swagger.json"

    def test_windows_file_path(self):
        request = SwaggerProcessingRequest(
            swagger_file_path="C:\\Users\\User\\Documents\\swagger.json"
        )
        assert request.swagger_file_path == "C:\\Users\\User\\Documents\\swagger.json"

    def test_network_path(self):
        request = SwaggerProcessingRequest(
            swagger_file_path="\\\\server\\share\\swagger.json"
        )
        assert request.swagger_file_path == "\\\\server\\share\\swagger.json"


class TestSwaggerProcessingRequestValidation:
    """Test the model validator that enforces exactly one source."""

    def test_both_provided_raises_error(self):
        with pytest.raises(ValidationError, match="not both"):
            SwaggerProcessingRequest(
                swagger_url="https://api.example.com/swagger.json",
                swagger_file_path="/path/to/swagger.json",
            )

    def test_neither_provided_raises_error(self):
        with pytest.raises(ValidationError, match="required"):
            SwaggerProcessingRequest()

    def test_both_none_raises_error(self):
        with pytest.raises(ValidationError, match="required"):
            SwaggerProcessingRequest(swagger_url=None, swagger_file_path=None)

    def test_empty_string_url_raises_error(self):
        """Empty string is falsy, so treated as not provided."""
        with pytest.raises(ValidationError, match="required"):
            SwaggerProcessingRequest(swagger_url="")

    def test_empty_string_file_path_raises_error(self):
        """Empty string is falsy, so treated as not provided."""
        with pytest.raises(ValidationError, match="required"):
            SwaggerProcessingRequest(swagger_file_path="")

    def test_both_empty_strings_raises_error(self):
        with pytest.raises(ValidationError, match="required"):
            SwaggerProcessingRequest(swagger_url="", swagger_file_path="")


class TestSwaggerProcessingRequestCoercion:
    """Test the coerce_to_string field validator."""

    def test_coerce_integer_to_string_url(self):
        request = SwaggerProcessingRequest(swagger_url=123)
        assert request.swagger_url == "123"

    def test_coerce_boolean_to_string_url(self):
        request = SwaggerProcessingRequest(swagger_url=True)
        assert request.swagger_url == "True"

    def test_coerce_integer_to_string_file_path(self):
        request = SwaggerProcessingRequest(swagger_file_path=123)
        assert request.swagger_file_path == "123"

    def test_coerce_boolean_to_string_file_path(self):
        request = SwaggerProcessingRequest(swagger_file_path=True)
        assert request.swagger_file_path == "True"

    def test_none_stays_none(self):
        """None values are not coerced; they remain None."""
        request = SwaggerProcessingRequest(swagger_url="test")
        assert request.swagger_file_path is None


class TestSwaggerProcessingRequestSerialization:
    """Test serialization and deserialization."""

    def test_model_dump_with_url(self):
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        data = request.model_dump()
        assert data == {
            "swagger_url": "https://api.example.com/swagger.json",
            "swagger_file_path": None,
        }

    def test_model_dump_with_file_path(self):
        request = SwaggerProcessingRequest(swagger_file_path="/path/to/file.json")
        data = request.model_dump()
        assert data == {
            "swagger_url": None,
            "swagger_file_path": "/path/to/file.json",
        }

    def test_model_dump_json(self):
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        json_str = request.model_dump_json()
        assert '"swagger_url":"https://api.example.com/swagger.json"' in json_str

    def test_model_deserialization_from_dict(self):
        data = {"swagger_url": "https://api.example.com/swagger.json"}
        request = SwaggerProcessingRequest(**data)
        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_model_deserialization_from_json(self):
        json_str = '{"swagger_url": "https://api.example.com/swagger.json"}'
        request = SwaggerProcessingRequest.model_validate_json(json_str)
        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_model_deserialization_file_path_from_json(self):
        json_str = '{"swagger_file_path": "/path/to/file.json"}'
        request = SwaggerProcessingRequest.model_validate_json(json_str)
        assert request.swagger_file_path == "/path/to/file.json"


class TestSwaggerProcessingRequestMisc:
    """Test miscellaneous model behavior."""

    def test_model_repr(self):
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        repr_str = repr(request)
        assert "SwaggerProcessingRequest" in repr_str
        assert "swagger_url=" in repr_str

    def test_model_equality(self):
        r1 = SwaggerProcessingRequest(swagger_url="https://a.com/swagger.json")
        r2 = SwaggerProcessingRequest(swagger_url="https://a.com/swagger.json")
        r3 = SwaggerProcessingRequest(swagger_url="https://b.com/swagger.json")
        assert r1 == r2
        assert r1 != r3

    def test_model_copy(self):
        original = SwaggerProcessingRequest(swagger_url="https://a.com/swagger.json")
        copied = original.model_copy()
        assert copied == original
        assert copied is not original

    def test_model_copy_with_update(self):
        original = SwaggerProcessingRequest(swagger_url="https://a.com/swagger.json")
        updated = original.model_copy(
            update={"swagger_url": "https://b.com/swagger.json"}
        )
        assert updated.swagger_url == "https://b.com/swagger.json"
        assert original.swagger_url == "https://a.com/swagger.json"

    def test_model_fields(self):
        fields = SwaggerProcessingRequest.model_fields
        assert "swagger_url" in fields
        assert "swagger_file_path" in fields
        assert fields["swagger_url"].default is None
        assert fields["swagger_file_path"].default is None

    def test_extra_fields_ignored(self):
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json",
            extra_field="should_be_ignored",
        )
        assert request.swagger_url == "https://api.example.com/swagger.json"
        assert not hasattr(request, "extra_field")
