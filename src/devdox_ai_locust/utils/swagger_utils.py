import httpx
import os
import re
import uuid
import aiofiles
from pathlib import Path
from typing import Optional
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest
import logging

logger = logging.getLogger(__name__)


async def get_api_schema(source: SwaggerProcessingRequest) -> Optional[str]:
    """
    Get API schema content from URL or file path.

    Args:
        source: SwaggerProcessingRequest containing either swagger_url or swagger_path

    Returns:
        Optional[str]: Schema content as string, or None if failed

    Raises:
        ValueError: If source is invalid or missing required fields
        FileNotFoundError: If file path doesn't exist
        httpx.HTTPError: If URL request fails
        Exception: For other unexpected errors
    """
    try:
        if source.is_url_source:
            logger.info(f"Fetching schema from URL: {source.swagger_url}")
            return await _fetch_from_url(source.swagger_url.strip())
        elif source.is_file_source:
            logger.info(f"Reading schema from file: {source.swagger_path}")
            return await _read_from_file(source.swagger_path.strip())
        else:
            raise ValueError("No valid source provided (neither URL nor file path)")

    except Exception as e:
        source_info = source.source_location if hasattr(source, 'source_location') else "unknown"
        logger.error(f"Failed to get API schema from source '{source_info}': {str(e)}")
        raise


async def _read_from_file(file_path: str) -> str:
    """Read schema content from a local file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file extension for content type hints
    suffix = path.suffix.lower()
    if suffix not in {'.json', '.yaml', '.yml'}:
        logger.warning(
            f"File extension '{suffix}' is not a standard OpenAPI format. "
            "Expected .json, .yaml, or .yml"
        )

    try:
        async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
            content = await f.read()

        if not content or not content.strip():
            raise ValueError(f"Empty file: {file_path}")

        logger.info(f"Successfully read schema file: {file_path} ({len(content)} bytes)")
        return content.strip()

    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error (expected UTF-8): {file_path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading file: {file_path}") from e


async def _fetch_from_url(url: str) -> str:
    """Fetch schema content from URL."""
    headers = {
        "User-Agent": "API-Schema-Fetcher/1.0",
        "Accept": "application/json, application/yaml, text/yaml, text/plain, */*",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            logger.info(
                f"Fetching schema from URL: {url}, Content-Type: {content_type}"
            )

            # Read content as text
            content = response.text

            if not content or not content.strip():
                raise ValueError(f"Empty response from URL: {url}")

            return content.strip()

        except httpx.HTTPStatusError as e:
            raise httpx.HTTPError(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase} for URL: {url}"
            )
        except httpx.TimeoutException:
            raise httpx.HTTPError(f"Request timeout after 30s for URL: {url}")
        except httpx.RequestError as e:
            raise httpx.HTTPError(f"Request failed for URL {url}: {str(e)}")


def _sanitize_filename(filename: str) -> str:
    # Remove directory components and sanitize
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^\w\-\.]", "", clean_name)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"generated_{uuid.uuid4().hex[:8]}.py"
    return clean_name
