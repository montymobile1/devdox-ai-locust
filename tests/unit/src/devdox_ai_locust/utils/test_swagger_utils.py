"""
Comprehensive tests for swagger_utils.py module.

Tests cover:
- get_api_schema function (URL and file sources)
- _read_from_file helper function
- _fetch_from_url helper function
- _sanitize_filename helper function
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from devdox_ai_locust.utils.swagger_utils import (
    get_api_schema,
    _read_from_file,
    _fetch_from_url,
    _sanitize_filename,
)
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest


# =============================================================================
# _sanitize_filename Tests
# =============================================================================


class TestSanitizeFilenameBasic:
    """Basic tests for _sanitize_filename function."""

    def test_simple_filename_preserved(self):
        """Simple valid filenames should be preserved."""
        assert _sanitize_filename("test.py") == "test.py"

    def test_preserves_underscores(self):
        """Underscores should be preserved."""
        assert _sanitize_filename("my_file.py") == "my_file.py"

    def test_preserves_dashes(self):
        """Dashes should be preserved."""
        assert _sanitize_filename("my-file.py") == "my-file.py"

    def test_preserves_dots(self):
        """Dots should be preserved."""
        assert _sanitize_filename("file.test.py") == "file.test.py"


class TestSanitizeFilenamePathTraversal:
    """Tests for path traversal prevention in _sanitize_filename."""

    def test_removes_directory_components(self):
        """Should remove directory path components."""
        assert _sanitize_filename("/etc/passwd") == "passwd"
        assert _sanitize_filename("path/to/file.py") == "file.py"

    def test_handles_parent_directory_traversal(self):
        """Should handle parent directory traversal attempts."""
        result = _sanitize_filename("../../secret.py")
        assert "/" not in result
        # Path traversal characters are stripped, leaving just the filename
        assert result == "secret.py"

    def test_removes_windows_path_separators(self):
        """Should handle Windows-style path separators."""
        result = _sanitize_filename("C:\\Users\\file.py")
        # After basename, it's still 'C:\\Users\\file.py' on Linux
        # The regex removes backslashes and colons
        assert "\\" not in result
        assert ":" not in result


class TestSanitizeFilenameDangerousChars:
    """Tests for dangerous character removal in _sanitize_filename."""

    def test_removes_special_characters(self):
        """Should remove special characters that aren't alphanumeric, underscore, dash, or dot."""
        result = _sanitize_filename("file@#$%.py")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result

    def test_removes_spaces(self):
        """Should remove spaces."""
        result = _sanitize_filename("my file.py")
        assert " " not in result
        assert result == "myfile.py"

    def test_removes_angle_brackets(self):
        """Should remove angle brackets."""
        result = _sanitize_filename("file<test>.py")
        assert "<" not in result
        assert ">" not in result


class TestSanitizeFilenameHiddenFiles:
    """Tests for hidden file handling in _sanitize_filename."""

    def test_hidden_file_gets_generated_name(self):
        """Hidden files (starting with dot) should get generated name."""
        result = _sanitize_filename(".htaccess")
        assert result.startswith("generated_")
        assert result.endswith(".py")

    def test_hidden_file_with_path_gets_generated_name(self):
        """Hidden files from path should get generated name."""
        result = _sanitize_filename("/path/to/.secret")
        assert result.startswith("generated_")


class TestSanitizeFilenameEmptyInput:
    """Tests for empty/invalid input handling in _sanitize_filename."""

    def test_empty_string_gets_generated_name(self):
        """Empty string should get a generated name."""
        result = _sanitize_filename("")
        assert result.startswith("generated_")
        assert result.endswith(".py")

    def test_only_special_chars_gets_generated_name(self):
        """Filename with only special chars should get generated name."""
        result = _sanitize_filename("@#$%")
        assert result.startswith("generated_")

    def test_generated_names_are_unique(self):
        """Generated names should be unique."""
        results = [_sanitize_filename("") for _ in range(10)]
        assert len(set(results)) == 10


# =============================================================================
# _read_from_file Tests
# =============================================================================


class TestReadFromFileBasic:
    """Basic tests for _read_from_file function."""

    @pytest.mark.asyncio
    async def test_reads_json_file(self, temp_dir):
        """Should read content from JSON file."""
        file_path = temp_dir / "schema.json"
        content = '{"openapi": "3.0.0"}'
        file_path.write_text(content)

        result = await _read_from_file(str(file_path))
        assert result == content

    @pytest.mark.asyncio
    async def test_reads_yaml_file(self, temp_dir):
        """Should read content from YAML file."""
        file_path = temp_dir / "schema.yaml"
        content = "openapi: '3.0.0'"
        file_path.write_text(content)

        result = await _read_from_file(str(file_path))
        assert result == content

    @pytest.mark.asyncio
    async def test_reads_yml_file(self, temp_dir):
        """Should read content from .yml file."""
        file_path = temp_dir / "schema.yml"
        content = "openapi: '3.0.0'"
        file_path.write_text(content)

        result = await _read_from_file(str(file_path))
        assert result == content

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, temp_dir):
        """Should strip leading/trailing whitespace."""
        file_path = temp_dir / "schema.json"
        file_path.write_text("  \n  content  \n  ")

        result = await _read_from_file(str(file_path))
        assert result == "content"


class TestReadFromFileErrors:
    """Tests for error handling in _read_from_file."""

    @pytest.mark.asyncio
    async def test_raises_for_nonexistent_file(self, temp_dir):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            await _read_from_file(str(temp_dir / "missing.json"))
        assert "Schema file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_for_directory(self, temp_dir):
        """Should raise ValueError when path is a directory."""
        with pytest.raises(ValueError) as exc_info:
            await _read_from_file(str(temp_dir))
        assert "Path is not a file" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_for_empty_file(self, temp_dir):
        """Should raise ValueError for empty file."""
        file_path = temp_dir / "empty.json"
        file_path.write_text("")

        with pytest.raises(ValueError) as exc_info:
            await _read_from_file(str(file_path))
        assert "Empty file" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_for_whitespace_only_file(self, temp_dir):
        """Should raise ValueError for file with only whitespace."""
        file_path = temp_dir / "whitespace.json"
        file_path.write_text("   \n   \t   ")

        with pytest.raises(ValueError) as exc_info:
            await _read_from_file(str(file_path))
        assert "Empty file" in str(exc_info.value)


class TestReadFromFileExtensionWarning:
    """Tests for file extension warning in _read_from_file."""

    @pytest.mark.asyncio
    async def test_warns_for_non_standard_extension(self, temp_dir):
        """Should log warning for non-standard file extensions."""
        file_path = temp_dir / "schema.txt"
        file_path.write_text('{"openapi": "3.0.0"}')

        with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
            await _read_from_file(str(file_path))
            mock_logger.warning.assert_called_once()
            assert ".txt" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_no_warning_for_json(self, temp_dir):
        """Should not warn for .json extension."""
        file_path = temp_dir / "schema.json"
        file_path.write_text('{"openapi": "3.0.0"}')

        with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
            await _read_from_file(str(file_path))
            mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_warning_for_yaml(self, temp_dir):
        """Should not warn for .yaml extension."""
        file_path = temp_dir / "schema.yaml"
        file_path.write_text("openapi: '3.0.0'")

        with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
            await _read_from_file(str(file_path))
            mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_warning_for_yml(self, temp_dir):
        """Should not warn for .yml extension."""
        file_path = temp_dir / "schema.yml"
        file_path.write_text("openapi: '3.0.0'")

        with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
            await _read_from_file(str(file_path))
            mock_logger.warning.assert_not_called()


class TestReadFromFileEncoding:
    """Tests for file encoding in _read_from_file."""

    @pytest.mark.asyncio
    async def test_reads_utf8_content(self, temp_dir):
        """Should read UTF-8 encoded content correctly."""
        file_path = temp_dir / "schema.json"
        content = '{"description": "日本語テスト"}'
        file_path.write_text(content, encoding="utf-8")

        result = await _read_from_file(str(file_path))
        assert result == content

    @pytest.mark.asyncio
    async def test_reads_unicode_characters(self, temp_dir):
        """Should handle various Unicode characters."""
        file_path = temp_dir / "schema.json"
        content = '{"emoji": "🚀", "chinese": "你好", "arabic": "مرحبا"}'
        file_path.write_text(content, encoding="utf-8")

        result = await _read_from_file(str(file_path))
        assert "🚀" in result
        assert "你好" in result
        assert "مرحبا" in result


class TestReadFromFileLogging:
    """Tests for logging in _read_from_file."""

    @pytest.mark.asyncio
    async def test_logs_success_info(self, temp_dir):
        """Should log success message with file size."""
        file_path = temp_dir / "schema.json"
        content = '{"openapi": "3.0.0"}'
        file_path.write_text(content)

        with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
            await _read_from_file(str(file_path))
            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args[0][0]
            assert "Successfully read" in log_message
            assert str(len(content)) in log_message


# =============================================================================
# _fetch_from_url Tests
# =============================================================================


class TestFetchFromUrlBasic:
    """Basic tests for _fetch_from_url function."""

    @pytest.mark.asyncio
    async def test_fetches_content_from_url(self):
        """Should fetch content from URL successfully."""
        mock_response = MagicMock()
        mock_response.text = '{"openapi": "3.0.0"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await _fetch_from_url("https://example.com/swagger.json")
            assert result == '{"openapi": "3.0.0"}'

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self):
        """Should strip whitespace from response."""
        mock_response = MagicMock()
        mock_response.text = "  \n content \n  "
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await _fetch_from_url("https://example.com/swagger.json")
            assert result == "content"


class TestFetchFromUrlHeaders:
    """Tests for request headers in _fetch_from_url."""

    @pytest.mark.asyncio
    async def test_sends_correct_user_agent(self):
        """Should send proper User-Agent header."""
        mock_response = MagicMock()
        mock_response.text = "content"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            await _fetch_from_url("https://example.com/swagger.json")

            call_kwargs = mock_instance.get.call_args
            headers = call_kwargs.kwargs["headers"]
            assert "User-Agent" in headers
            assert "API-Schema-Fetcher" in headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_sends_accept_header(self):
        """Should send proper Accept header for JSON/YAML."""
        mock_response = MagicMock()
        mock_response.text = "content"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            await _fetch_from_url("https://example.com/swagger.json")

            call_kwargs = mock_instance.get.call_args
            headers = call_kwargs.kwargs["headers"]
            assert "Accept" in headers
            assert "application/json" in headers["Accept"]
            assert "application/yaml" in headers["Accept"]


class TestFetchFromUrlErrors:
    """Tests for error handling in _fetch_from_url."""

    @pytest.mark.asyncio
    async def test_raises_for_http_error(self):
        """Should raise httpx.HTTPError for HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(httpx.HTTPError) as exc_info:
                await _fetch_from_url("https://example.com/missing.json")
            assert "HTTP 404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_for_timeout(self):
        """Should raise httpx.HTTPError for timeouts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(httpx.HTTPError) as exc_info:
                await _fetch_from_url("https://example.com/slow.json")
            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raises_for_request_error(self):
        """Should raise httpx.HTTPError for request errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.RequestError("Connection failed")
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(httpx.HTTPError) as exc_info:
                await _fetch_from_url("https://example.com/api.json")
            assert "Request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_for_empty_response(self):
        """Should raise ValueError for empty response."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ValueError) as exc_info:
                await _fetch_from_url("https://example.com/empty.json")
            assert "Empty response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_for_whitespace_only_response(self):
        """Should raise ValueError for whitespace-only response."""
        mock_response = MagicMock()
        mock_response.text = "   \n   "
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ValueError) as exc_info:
                await _fetch_from_url("https://example.com/whitespace.json")
            assert "Empty response" in str(exc_info.value)


class TestFetchFromUrlTimeout:
    """Tests for timeout configuration in _fetch_from_url."""

    @pytest.mark.asyncio
    async def test_uses_30_second_timeout(self):
        """Should use 30 second timeout."""
        mock_response = MagicMock()
        mock_response.text = "content"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            await _fetch_from_url("https://example.com/api.json")

            # Check that AsyncClient was called with timeout=30
            mock_client.assert_called_once_with(timeout=30)


class TestFetchFromUrlLogging:
    """Tests for logging in _fetch_from_url."""

    @pytest.mark.asyncio
    async def test_logs_content_type(self):
        """Should log the content type from response."""
        mock_response = MagicMock()
        mock_response.text = "content"
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
                await _fetch_from_url("https://example.com/api.json")
                mock_logger.info.assert_called()
                log_message = mock_logger.info.call_args[0][0]
                assert "Content-Type" in log_message


# =============================================================================
# get_api_schema Tests
# =============================================================================


class TestGetApiSchemaUrl:
    """Tests for get_api_schema with URL sources."""

    @pytest.mark.asyncio
    async def test_fetches_from_url(self):
        """Should fetch schema from URL when swagger_url is provided."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )

        with patch(
            "devdox_ai_locust.utils.swagger_utils._fetch_from_url"
        ) as mock_fetch:
            mock_fetch.return_value = '{"openapi": "3.0.0"}'
            result = await get_api_schema(request)

            mock_fetch.assert_called_once_with("https://example.com/swagger.json")
            assert result == '{"openapi": "3.0.0"}'

    @pytest.mark.asyncio
    async def test_strips_url_whitespace(self):
        """Should strip whitespace from URL before fetching."""
        request = SwaggerProcessingRequest(
            swagger_url="  https://example.com/swagger.json  "
        )

        with patch(
            "devdox_ai_locust.utils.swagger_utils._fetch_from_url"
        ) as mock_fetch:
            mock_fetch.return_value = "content"
            await get_api_schema(request)

            mock_fetch.assert_called_once_with("https://example.com/swagger.json")


class TestGetApiSchemaFile:
    """Tests for get_api_schema with file sources."""

    @pytest.mark.asyncio
    async def test_reads_from_file(self, temp_dir):
        """Should read schema from file when swagger_path is provided."""
        file_path = temp_dir / "schema.json"
        file_path.write_text('{"openapi": "3.0.0"}')

        request = SwaggerProcessingRequest(swagger_path=str(file_path))

        result = await get_api_schema(request)
        assert result == '{"openapi": "3.0.0"}'

    @pytest.mark.asyncio
    async def test_strips_path_whitespace(self, temp_dir):
        """Should strip whitespace from path before reading."""
        file_path = temp_dir / "schema.json"
        file_path.write_text('{"openapi": "3.0.0"}')

        request = SwaggerProcessingRequest(swagger_path=f"  {file_path}  ")

        with patch("devdox_ai_locust.utils.swagger_utils._read_from_file") as mock_read:
            mock_read.return_value = "content"
            await get_api_schema(request)

            mock_read.assert_called_once_with(str(file_path))


class TestGetApiSchemaNoSource:
    """Tests for get_api_schema with missing sources."""

    def test_raises_validation_error_for_no_source(self):
        """Should raise ValidationError when neither URL nor path provided.

        The SwaggerProcessingRequest model validates at construction time.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SwaggerProcessingRequest()
        assert "Must specify either swagger_url or swagger_path" in str(exc_info.value)


class TestGetApiSchemaLogging:
    """Tests for logging in get_api_schema."""

    @pytest.mark.asyncio
    async def test_logs_info_for_url_fetch(self):
        """Should log info when fetching from URL."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )

        with patch(
            "devdox_ai_locust.utils.swagger_utils._fetch_from_url"
        ) as mock_fetch:
            mock_fetch.return_value = "content"

            with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
                await get_api_schema(request)
                mock_logger.info.assert_called()
                assert "URL" in mock_logger.info.call_args[0][0]

    @pytest.mark.asyncio
    async def test_logs_info_for_file_read(self, temp_dir):
        """Should log info when reading from file."""
        file_path = temp_dir / "schema.json"
        file_path.write_text('{"openapi": "3.0.0"}')

        request = SwaggerProcessingRequest(swagger_path=str(file_path))

        with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
            await get_api_schema(request)
            # Should have at least one info call about reading from file
            assert any(
                "file" in str(call).lower() for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_logs_error_on_failure(self):
        """Should log error when fetching fails."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )

        with patch(
            "devdox_ai_locust.utils.swagger_utils._fetch_from_url"
        ) as mock_fetch:
            mock_fetch.side_effect = httpx.HTTPError("Connection failed")

            with patch("devdox_ai_locust.utils.swagger_utils.logger") as mock_logger:
                with pytest.raises(httpx.HTTPError):
                    await get_api_schema(request)
                mock_logger.error.assert_called_once()


class TestGetApiSchemaErrorPropagation:
    """Tests for error propagation in get_api_schema."""

    @pytest.mark.asyncio
    async def test_propagates_http_error(self):
        """Should propagate httpx.HTTPError from URL fetch."""
        request = SwaggerProcessingRequest(
            swagger_url="https://example.com/swagger.json"
        )

        with patch(
            "devdox_ai_locust.utils.swagger_utils._fetch_from_url"
        ) as mock_fetch:
            mock_fetch.side_effect = httpx.HTTPError("HTTP 404")

            with pytest.raises(httpx.HTTPError):
                await get_api_schema(request)

    @pytest.mark.asyncio
    async def test_propagates_file_not_found(self):
        """Should propagate FileNotFoundError from file read."""
        request = SwaggerProcessingRequest(swagger_path="/nonexistent/path.json")

        with pytest.raises(FileNotFoundError):
            await get_api_schema(request)

    def test_validation_error_at_construction(self):
        """Should raise ValidationError at construction when no source provided."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SwaggerProcessingRequest()


# =============================================================================
# Integration Tests
# =============================================================================


class TestSwaggerUtilsIntegration:
    """Integration tests for swagger_utils module."""

    @pytest.mark.asyncio
    async def test_full_file_workflow(self, temp_dir):
        """Test complete workflow reading from file."""
        # Create a realistic OpenAPI schema file
        schema_content = """{
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "paths": {
                "/users": {
                    "get": {
                        "summary": "Get users"
                    }
                }
            }
        }"""
        file_path = temp_dir / "api-spec.json"
        file_path.write_text(schema_content)

        request = SwaggerProcessingRequest(swagger_path=str(file_path))
        result = await get_api_schema(request)

        assert "openapi" in result
        assert "3.0.0" in result
        assert "Test API" in result

    @pytest.mark.asyncio
    async def test_yaml_file_workflow(self, temp_dir):
        """Test workflow with YAML file."""
        schema_content = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
paths:
  /users:
    get:
      summary: Get users
"""
        file_path = temp_dir / "api-spec.yaml"
        file_path.write_text(schema_content)

        request = SwaggerProcessingRequest(swagger_path=str(file_path))
        result = await get_api_schema(request)

        assert "openapi" in result
        assert "Test API" in result
