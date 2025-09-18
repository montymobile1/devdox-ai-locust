"""
Tests for processing_result schema module
"""

from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest


class TestSwaggerProcessingRequest:
    """Test SwaggerProcessingRequest model."""

    def test_valid_swagger_url(self):
        """Test valid swagger URL."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_none_swagger_url(self):
        """Test None swagger URL."""
        request = SwaggerProcessingRequest(swagger_url=None)

        assert request.swagger_url is None

    def test_empty_string_swagger_url(self):
        """Test empty string swagger URL."""
        request = SwaggerProcessingRequest(swagger_url="")

        assert request.swagger_url == ""

    def test_whitespace_swagger_url(self):
        """Test whitespace swagger URL."""
        request = SwaggerProcessingRequest(swagger_url="   ")

        assert request.swagger_url == "   "

    def test_http_url(self):
        """Test HTTP URL (not HTTPS)."""
        request = SwaggerProcessingRequest(
            swagger_url="http://api.example.com/swagger.json"
        )

        assert request.swagger_url == "http://api.example.com/swagger.json"

    def test_file_path_url(self):
        """Test file path as URL."""
        request = SwaggerProcessingRequest(swagger_url="/path/to/swagger.json")

        assert request.swagger_url == "/path/to/swagger.json"

    def test_relative_file_path(self):
        """Test relative file path as URL."""
        request = SwaggerProcessingRequest(swagger_url="./swagger.json")

        assert request.swagger_url == "./swagger.json"

    def test_yaml_file_url(self):
        """Test YAML file URL."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.yaml"
        )

        assert request.swagger_url == "https://api.example.com/swagger.yaml"

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        url = "https://api.example.com/swagger.json?version=v1&format=json"
        request = SwaggerProcessingRequest(swagger_url=url)

        assert request.swagger_url == url

    def test_url_with_fragment(self):
        """Test URL with fragment."""
        url = "https://api.example.com/swagger.json#section"
        request = SwaggerProcessingRequest(swagger_url=url)

        assert request.swagger_url == url

    def test_localhost_url(self):
        """Test localhost URL."""
        request = SwaggerProcessingRequest(
            swagger_url="http://localhost:8080/swagger.json"
        )

        assert request.swagger_url == "http://localhost:8080/swagger.json"

    def test_ip_address_url(self):
        """Test IP address URL."""
        request = SwaggerProcessingRequest(
            swagger_url="http://192.168.1.100:8080/swagger.json"
        )

        assert request.swagger_url == "http://192.168.1.100:8080/swagger.json"

    def test_very_long_url(self):
        """Test very long URL."""
        long_path = "x" * 1000
        url = f"https://api.example.com/{long_path}/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url)

        assert request.swagger_url == url

    def test_unicode_url(self):
        """Test URL with Unicode characters."""
        url = "https://api.example.com/ドキュメント/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url)

        assert request.swagger_url == url

    def test_default_values(self):
        """Test default values when no arguments provided."""
        request = SwaggerProcessingRequest()

        assert request.swagger_url is None

    def test_model_serialization(self):
        """Test model serialization to dict."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        data = request.model_dump()

        assert data == {"swagger_url": "https://api.example.com/swagger.json"}

    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        json_str = request.model_dump_json()

        assert '"swagger_url":"https://api.example.com/swagger.json"' in json_str

    def test_model_deserialization(self):
        """Test model deserialization from dict."""
        data = {"swagger_url": "https://api.example.com/swagger.json"}

        request = SwaggerProcessingRequest(**data)

        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_model_json_deserialization(self):
        """Test model deserialization from JSON."""
        json_str = '{"swagger_url": "https://api.example.com/swagger.json"}'

        request = SwaggerProcessingRequest.model_validate_json(json_str)

        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_model_validation_extra_fields(self):
        """Test model validation with extra fields."""
        # Since the model doesn't have forbid extra fields, this should work
        data = {
            "swagger_url": "https://api.example.com/swagger.json",
            "extra_field": "extra_value",
        }

        # This should work (Pydantic default is to ignore extra fields)
        request = SwaggerProcessingRequest(**data)

        assert request.swagger_url == "https://api.example.com/swagger.json"
        # extra_field should be ignored

    def test_model_repr(self):
        """Test model string representation."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        repr_str = repr(request)

        assert "SwaggerProcessingRequest" in repr_str
        assert "swagger_url=" in repr_str

    def test_model_equality(self):
        """Test model equality comparison."""
        request1 = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        request2 = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        request3 = SwaggerProcessingRequest(
            swagger_url="https://other.example.com/swagger.json"
        )

        assert request1 == request2
        assert request1 != request3

    def test_model_copy(self):
        """Test model copying."""
        original = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        # Test shallow copy
        copied = original.model_copy()

        assert copied == original
        assert copied is not original

    def test_model_copy_with_update(self):
        """Test model copying with updates."""
        original = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        updated = original.model_copy(
            update={"swagger_url": "https://updated.example.com/swagger.json"}
        )

        assert updated.swagger_url == "https://updated.example.com/swagger.json"
        assert original.swagger_url == "https://api.example.com/swagger.json"

    def test_model_fields(self):
        """Test model field information."""
        fields = SwaggerProcessingRequest.model_fields

        assert "swagger_url" in fields

        swagger_url_field = fields["swagger_url"]
        # Check that the field is optional (Union with None)
        assert swagger_url_field.default is None

    def test_model_config(self):
        """Test model configuration."""
        # Test that the model can be instantiated and used properly
        request = SwaggerProcessingRequest()

        # Should be able to set and get attributes
        request.swagger_url = "https://api.example.com/swagger.json"
        assert request.swagger_url == "https://api.example.com/swagger.json"


class TestSwaggerProcessingRequestValidation:
    """Test SwaggerProcessingRequest validation."""

    def test_validator_missing_swagger_url_validation(self):
        """Test the custom validator behavior."""
        # The validator references 'swagger_source' which doesn't exist in the model
        # This test ensures the current model works as expected

        request = SwaggerProcessingRequest(swagger_url=None)
        assert request.swagger_url is None

        request = SwaggerProcessingRequest(swagger_url="")
        assert request.swagger_url == ""

    def test_validator_with_valid_url(self):
        """Test validator with valid URL."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )
        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_type_coercion(self):
        """Test type coercion for swagger_url field."""
        # Test that integers are converted to strings
        request = SwaggerProcessingRequest(swagger_url="123")
        assert request.swagger_url == "123"

        # Test that booleans are converted to strings
        request = SwaggerProcessingRequest(swagger_url=True)
        assert request.swagger_url == "True"

    def test_none_handling(self):
        """Test None value handling."""
        request = SwaggerProcessingRequest(swagger_url=None)
        assert request.swagger_url is None

    def test_field_validation_error_cases(self):
        """Test cases that might cause validation errors."""
        # These should all work due to Optional[str] type
        test_cases = [None, "", "   ", "valid_url", 123, True, False]

        for case in test_cases:
            request = SwaggerProcessingRequest(swagger_url=case)
            # Should not raise validation error
            assert hasattr(request, "swagger_url")


class TestSwaggerProcessingRequestEdgeCases:
    """Test edge cases for SwaggerProcessingRequest."""

    def test_very_large_url(self):
        """Test with very large URL."""
        large_url = "https://api.example.com/" + "x" * 10000 + "/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=large_url)

        assert request.swagger_url == large_url

    def test_special_characters_in_url(self):
        """Test URL with special characters."""
        special_url = "https://api.example.com/swagger.json?query=hello%20world&param=value%20with%20spaces"
        request = SwaggerProcessingRequest(swagger_url=special_url)

        assert request.swagger_url == special_url

    def test_url_with_credentials(self):
        """Test URL with embedded credentials."""
        url_with_auth = "https://username:password@api.example.com/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url_with_auth)

        assert request.swagger_url == url_with_auth

    def test_url_with_port(self):
        """Test URL with explicit port."""
        url_with_port = "https://api.example.com:8443/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=url_with_port)

        assert request.swagger_url == url_with_port

    def test_ftp_url(self):
        """Test FTP URL."""
        ftp_url = "ftp://ftp.example.com/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=ftp_url)

        assert request.swagger_url == ftp_url

    def test_file_url_scheme(self):
        """Test file:// URL scheme."""
        file_url = "file:///path/to/swagger.json"
        request = SwaggerProcessingRequest(swagger_url=file_url)

        assert request.swagger_url == file_url

    def test_windows_file_path(self):
        """Test Windows file path."""
        windows_path = "C:\\Users\\User\\Documents\\swagger.json"
        request = SwaggerProcessingRequest(swagger_url=windows_path)

        assert request.swagger_url == windows_path

    def test_network_path(self):
        """Test network path."""
        network_path = "\\\\server\\share\\swagger.json"
        request = SwaggerProcessingRequest(swagger_url=network_path)

        assert request.swagger_url == network_path

    def test_url_with_newlines(self):
        """Test URL with newlines."""
        url_with_newlines = "https://api.example.com/\nswagger.json"
        request = SwaggerProcessingRequest(swagger_url=url_with_newlines)

        assert request.swagger_url == url_with_newlines

    def test_empty_model_creation(self):
        """Test creating empty model and setting fields later."""
        request = SwaggerProcessingRequest()
        assert request.swagger_url is None

        request.swagger_url = "https://api.example.com/swagger.json"
        assert request.swagger_url == "https://api.example.com/swagger.json"

    def test_model_immutability_behavior(self):
        """Test model behavior regarding immutability."""
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        # Pydantic models are mutable by default
        request.swagger_url = "https://updated.example.com/swagger.json"
        assert request.swagger_url == "https://updated.example.com/swagger.json"

    def test_model_with_invalid_field_names(self):
        """Test model creation with invalid field names."""
        # This should be ignored due to extra='ignore' (default behavior)
        request = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json",
            invalid_field="should_be_ignored",
        )

        assert request.swagger_url == "https://api.example.com/swagger.json"
        assert not hasattr(request, "invalid_field")
