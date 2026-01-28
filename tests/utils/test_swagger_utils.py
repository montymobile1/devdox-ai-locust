"""
Tests for swagger_utils module
"""

import pytest
import asyncio
import httpx
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from devdox_ai_locust.utils.swagger_utils import (
    get_api_schema,
    _fetch_from_url,
    _read_from_file,
)
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest


class TestGetApiSchema:
    """Test get_api_schema function."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.utils.swagger_utils._fetch_from_url")
    async def test_get_api_schema_from_url(self, mock_fetch_url):
        """Test fetching API schema from URL."""
        mock_fetch_url.return_value = '{"openapi": "3.0.0"}'

        source = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        result = await get_api_schema(source)

        assert result == '{"openapi": "3.0.0"}'
        mock_fetch_url.assert_called_once_with("https://api.example.com/swagger.json")

    def test_model_rejects_empty_url(self):
        """Test that SwaggerProcessingRequest rejects empty URL (neither field truthy)."""
        with pytest.raises(ValidationError):
            SwaggerProcessingRequest(swagger_url="")

    @pytest.mark.asyncio
    async def test_get_api_schema_whitespace_only_url(self):
        """Test get_api_schema with whitespace-only URL raises ValueError."""
        source = SwaggerProcessingRequest(swagger_url="   ")

        with pytest.raises(ValueError, match="swagger_url is empty after stripping"):
            await get_api_schema(source)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.utils.swagger_utils._fetch_from_url")
    async def test_get_api_schema_url_error(self, mock_fetch_url):
        """Test get_api_schema with URL fetch error."""
        mock_fetch_url.side_effect = httpx.HTTPError("Connection failed")

        source = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        with pytest.raises(httpx.HTTPError):
            await get_api_schema(source)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.utils.swagger_utils._fetch_from_url")
    async def test_get_api_schema_strips_url(self, mock_fetch_url):
        """Test that URL is stripped of whitespace."""
        mock_fetch_url.return_value = '{"openapi": "3.0.0"}'

        source = SwaggerProcessingRequest(
            swagger_url="  https://api.example.com/swagger.json  "
        )

        await get_api_schema(source)

        mock_fetch_url.assert_called_once_with("https://api.example.com/swagger.json")

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.utils.swagger_utils._read_from_file")
    async def test_get_api_schema_from_file(self, mock_read_file):
        """Test fetching API schema from a local file path."""
        mock_read_file.return_value = '{"openapi": "3.0.0"}'

        source = SwaggerProcessingRequest(swagger_file_path="path/to/swagger.json")

        result = await get_api_schema(source)

        assert result == '{"openapi": "3.0.0"}'
        mock_read_file.assert_called_once_with("path/to/swagger.json")

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.utils.swagger_utils._read_from_file")
    async def test_get_api_schema_file_not_found(self, mock_read_file):
        """Test get_api_schema with non-existent file."""
        mock_read_file.side_effect = FileNotFoundError(
            "Schema file not found: missing.json"
        )

        source = SwaggerProcessingRequest(swagger_file_path="missing.json")

        with pytest.raises(FileNotFoundError):
            await get_api_schema(source)


class TestSwaggerProcessingRequestModel:
    """Test SwaggerProcessingRequest model validation."""

    def test_model_rejects_no_args(self):
        """Test that model rejects construction with no arguments."""
        with pytest.raises(ValidationError):
            SwaggerProcessingRequest()

    def test_model_rejects_both_fields(self):
        """Test that model rejects both swagger_url and swagger_file_path."""
        with pytest.raises(ValidationError):
            SwaggerProcessingRequest(
                swagger_url="https://example.com/swagger.json",
                swagger_file_path="path/to/file.json",
            )

    def test_model_rejects_none_url_none_file(self):
        """Test that model rejects None for both fields."""
        with pytest.raises(ValidationError):
            SwaggerProcessingRequest(swagger_url=None, swagger_file_path=None)

    def test_model_accepts_url_only(self):
        """Test model accepts swagger_url only."""
        req = SwaggerProcessingRequest(swagger_url="https://example.com/swagger.json")
        assert req.swagger_url == "https://example.com/swagger.json"
        assert req.swagger_file_path is None

    def test_model_accepts_file_path_only(self):
        """Test model accepts swagger_file_path only."""
        req = SwaggerProcessingRequest(swagger_file_path="path/to/file.json")
        assert req.swagger_file_path == "path/to/file.json"
        assert req.swagger_url is None


class TestReadFromFile:
    """Test _read_from_file function."""

    def test_read_from_file_success(self, tmp_path):
        """Test successful file read."""
        schema_file = tmp_path / "swagger.json"
        schema_file.write_text('{"openapi": "3.0.0"}', encoding="utf-8")

        result = _read_from_file(str(schema_file))

        assert result == '{"openapi": "3.0.0"}'

    def test_read_from_file_not_found(self):
        """Test file not found."""
        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            _read_from_file("nonexistent/path/swagger.json")

    def test_read_from_file_empty(self, tmp_path):
        """Test empty file."""
        schema_file = tmp_path / "empty.json"
        schema_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Empty schema file"):
            _read_from_file(str(schema_file))

    def test_read_from_file_whitespace_only(self, tmp_path):
        """Test whitespace-only file."""
        schema_file = tmp_path / "whitespace.json"
        schema_file.write_text("   \n\t   ", encoding="utf-8")

        with pytest.raises(ValueError, match="Empty schema file"):
            _read_from_file(str(schema_file))

    def test_read_from_file_directory(self, tmp_path):
        """Test that passing a directory raises ValueError."""
        with pytest.raises(ValueError, match="Path is not a file"):
            _read_from_file(str(tmp_path))

    def test_read_from_file_strips_content(self, tmp_path):
        """Test that content is stripped."""
        schema_file = tmp_path / "swagger.json"
        schema_file.write_text('  {"openapi": "3.0.0"}  \n', encoding="utf-8")

        result = _read_from_file(str(schema_file))

        assert result == '{"openapi": "3.0.0"}'


class TestFetchFromUrl:
    """Test _fetch_from_url function."""

    @pytest.mark.asyncio
    async def test_fetch_from_url_success(self, mock_httpx_client):
        """Test successful URL fetch."""
        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_httpx_client

            result = await _fetch_from_url("https://api.example.com/swagger.json")

            assert result == '{"openapi": "3.0.0", "info": {"title": "Test API"}}'
            mock_httpx_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_from_url_with_custom_headers(self):
        """Test URL fetch with custom headers."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.text = '{"openapi": "3.0.0"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            await _fetch_from_url("https://api.example.com/swagger.json")

            # Check that headers were set correctly
            call_args = mock_client.get.call_args
            headers = call_args[1]["headers"]
            assert "User-Agent" in headers
            assert "Accept" in headers
            assert "API-Schema-Fetcher/1.0" in headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_fetch_from_url_http_error(self):
        """Test URL fetch with HTTP error."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"
        mock_error = httpx.HTTPStatusError(
            "404", request=Mock(), response=mock_response
        )
        mock_client.get.side_effect = mock_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPError, match="HTTP 404"):
                await _fetch_from_url("https://api.example.com/swagger.json")

    @pytest.mark.asyncio
    async def test_fetch_from_url_request_error(self):
        """Test URL fetch with request error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPError, match="Request failed"):
                await _fetch_from_url("https://api.example.com/swagger.json")

    @pytest.mark.asyncio
    async def test_fetch_from_url_empty_response(self):
        """Test URL fetch with empty response."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.text = ""
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            with pytest.raises(ValueError, match="Empty response"):
                await _fetch_from_url("https://api.example.com/swagger.json")

    @pytest.mark.asyncio
    async def test_fetch_from_url_whitespace_response(self):
        """Test URL fetch with whitespace-only response."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.text = "   \n\t   "
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            with pytest.raises(ValueError, match="Empty response"):
                await _fetch_from_url("https://api.example.com/swagger.json")

    @pytest.mark.asyncio
    async def test_fetch_from_url_yaml_content_type(self):
        """Test URL fetch with YAML content type."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.text = "openapi: 3.0.0\ninfo:\n  title: Test API"
        mock_response.headers = {"content-type": "application/yaml"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            result = await _fetch_from_url("https://api.example.com/swagger.yaml")

            assert result == "openapi: 3.0.0\ninfo:\n  title: Test API"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_model_rejects_none_url_no_file(self):
        """Test that model rejects None URL with no file path."""
        with pytest.raises(ValidationError):
            SwaggerProcessingRequest(swagger_url=None)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.utils.swagger_utils._fetch_from_url")
    async def test_get_api_schema_unexpected_error(self, mock_fetch_url):
        """Test get_api_schema with unexpected error."""
        mock_fetch_url.side_effect = RuntimeError("Unexpected error")

        source = SwaggerProcessingRequest(
            swagger_url="https://api.example.com/swagger.json"
        )

        with pytest.raises(RuntimeError):
            await get_api_schema(source)

    @pytest.mark.asyncio
    async def test_fetch_from_url_malformed_url(self):
        """Test _fetch_from_url with malformed URL."""
        with pytest.raises(httpx.HTTPError):
            await _fetch_from_url("not-a-valid-url")

    @pytest.mark.asyncio
    async def test_fetch_from_url_very_long_url(self):
        """Test _fetch_from_url with very long URL."""
        long_url = "https://api.example.com/" + "x" * 2000 + "/swagger.json"

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("URL too long")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "devdox_ai_locust.utils.swagger_utils.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPError):
                await _fetch_from_url(long_url)


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_url_fetches(self):
        """Test multiple concurrent URL fetches."""
        urls = [
            "https://api1.example.com/swagger.json",
            "https://api2.example.com/swagger.json",
            "https://api3.example.com/swagger.json",
        ]

        async def mock_fetch(url):
            await asyncio.sleep(0.1)  # Simulate network delay
            return f'{{"url": "{url}", "openapi": "3.0.0"}}'

        with patch(
            "devdox_ai_locust.utils.swagger_utils._fetch_from_url",
            side_effect=mock_fetch,
        ):
            tasks = [
                get_api_schema(SwaggerProcessingRequest(swagger_url=url))
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert urls[i] in result
