import httpx
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest
import logging

logger = logging.getLogger(__name__)


async def get_api_schema(source: SwaggerProcessingRequest) -> Optional[str]:
    """
    Get API schema content from a URL or local file path.

    Args:
        source: A ``SwaggerProcessingRequest`` with either ``swagger_url``
                or ``swagger_file_path`` set (never both).

    Returns:
        Schema content as a string, or ``None`` if retrieval failed.

    Raises:
        ValueError: If the source fields are invalid or empty.
        FileNotFoundError: If the file path doesn't exist.
        httpx.HTTPError: If the URL request fails.
    """
    try:
        if source.swagger_url:
            swagger_url = source.swagger_url.strip()
            if not swagger_url:
                raise ValueError("swagger_url is empty after stripping whitespace")
            return await _fetch_from_url(swagger_url)

        if source.swagger_file_path:
            file_path = source.swagger_file_path.strip()
            if not file_path:
                raise ValueError(
                    "swagger_file_path is empty after stripping whitespace"
                )
            return _read_from_file(file_path)

        raise ValueError("No swagger_url or swagger_file_path provided")

    except Exception as e:
        source_info = source.swagger_url or source.swagger_file_path or "unknown"
        logger.error(f"Failed to get API schema from source '{source_info}': {e}")
        raise


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


def _read_from_file(file_path: str) -> str:
    """Read schema content from a local file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    content = path.read_text(encoding="utf-8")
    if not content or not content.strip():
        raise ValueError(f"Empty schema file: {file_path}")

    return content.strip()


def _sanitize_filename(filename: str) -> str:
    # Remove directory components and sanitize
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^\w\-\.]", "", clean_name)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"generated_{uuid.uuid4().hex[:8]}.py"
    return clean_name
