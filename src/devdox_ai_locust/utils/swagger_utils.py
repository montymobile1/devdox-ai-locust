import httpx
import os
import re
import uuid
from typing import Optional
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest
import logging

logger = logging.getLogger(__name__)


async def get_api_schema(source: SwaggerProcessingRequest) -> Optional[str]:
    """
    Get API schema content from URL or file path based on source dictionary.

    Args:
        source (dict): Dictionary containing swagger_source ("url" or "file") and swagger_url/swagger_path


    Expected source structure:
        {
            "swagger_source": "url",  # or "file"
            "swagger_url": "https://api.example.com/swagger.json",  # if source is "url"
            "swagger_path": "/path/to/swagger.json"  # if source is "file"
        }

    Returns:
        Optional[str]: Schema content as string, or None if failed

    Raises:
        ValueError: If source is invalid or missing required fields
        FileNotFoundError: If file path doesn't exist
        httpx.HTTPError: If URL request fails
        Exception: For other unexpected errors
    """

    try:
        if not source.swagger_url:
            raise ValueError("Missing or empty 'swagger_url'")
        swagger_url = source.swagger_url.strip()
        if not swagger_url:
            raise ValueError("Missing 'swagger_url' for url source")
        return await _fetch_from_url(swagger_url)

    except Exception as e:
        source_info = getattr(source, "swagger_url", "unknown")

        logger.error(f"Failed to get API schema from  source '{source_info}': {str(e)}")
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


def _sanitize_filename(filename: str) -> str:
    # Remove directory components and sanitize
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^\w\-\.]", "", clean_name)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"generated_{uuid.uuid4().hex[:8]}.py"
    return clean_name
