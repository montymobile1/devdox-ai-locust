"""
Shared Together AI client for reusable async API calls.

Provides a centralized, reusable client for making calls to the Together AI API
with retry logic, error classification, concurrency control, and response parsing.
"""

import re
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from together import AsyncTogether

from devdox_ai_locust.utils.response_parser import (
    clean_response as _clean_response,
    extract_code_from_response as _extract_code_from_response,
    validate_python_code as _validate_python_code,
)

logger = logging.getLogger(__name__)


@dataclass
class AIEnhancementConfig:
    """Configuration for AI enhancement."""

    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 60
    enhance_workflows: bool = True
    enhance_test_data: bool = True
    enhance_validation: bool = True
    create_domain_flows: bool = True
    update_main_locust: bool = True


@dataclass
class ErrorClassification:
    """Classification of an error for retry logic."""

    is_retryable: bool
    backoff_seconds: float
    error_type: str


class TogetherAIClient:
    """Reusable async client for Together AI API calls.

    Encapsulates API communication, retry logic with exponential backoff,
    error classification, semaphore-based concurrency control, and
    response extraction/validation utilities.

    Usage::

        async with TogetherAIClient(api_key="...") as client:
            response = await client.call(system_prompt, user_prompt)
            code = client.extract_code_from_response(response)
    """

    MAX_RETRIES = 3
    RATE_LIMIT_BACKOFF = 10
    NON_RETRYABLE_CODES = [
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "authentication",
        "unauthorized",
        "invalid token",
    ]
    RATE_LIMIT_INDICATORS = ["429", "rate limit"]

    def __init__(
        self,
        api_key: str,
        config: Optional[AIEnhancementConfig] = None,
    ):
        self.api_key = api_key
        self.config = config or AIEnhancementConfig()
        self._semaphore = asyncio.Semaphore(5)
        self._client: Optional[AsyncTogether] = None

    async def __aenter__(self) -> "TogetherAIClient":
        """Async context manager entry."""
        self._client = AsyncTogether(api_key=self.api_key)
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit -- cleanup."""
        self._client = None

    @property
    def client(self) -> AsyncTogether:
        """Return the underlying AsyncTogether client.

        Raises ``RuntimeError`` when accessed outside a context manager or
        before explicit initialisation.
        """
        if self._client is None:
            raise RuntimeError(
                "TogetherAIClient must be used as an async context manager "
                "or _client must be set explicitly."
            )
        return self._client

    @client.setter
    def client(self, value: AsyncTogether) -> None:
        self._client = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call(
        self, system_prompt: str, user_prompt: str, *, raw: bool = False
    ) -> str:
        """Call the AI with retry logic.

        Args:
            system_prompt: The system prompt for the AI.
            user_prompt: The user prompt for the AI.
            raw: If True, return the raw response without cleaning or
                extracting code from tags. Use this when expecting
                structured XML-tagged responses (e.g., enhance prompts).

        Returns the cleaned response text, or an empty string on failure.
        """
        messages = self._build_messages(system_prompt, user_prompt)

        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._semaphore:
                    content = await self._make_api_call(messages, raw=raw)
                    if content:
                        return content

            except asyncio.TimeoutError:
                logger.warning(
                    f"AI service timeout on attempt {attempt + 1}"
                )

            except Exception as e:
                classification = self.classify_error(e, attempt)

                if not classification.is_retryable:
                    return ""

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(classification.backoff_seconds)
                    continue

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)

        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
        """Build the messages payload for the API call."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _make_api_call(
        self, messages: list[dict], *, raw: bool = False
    ) -> Optional[str]:
        """Make a single API call with timeout.

        Args:
            messages: The messages payload for the API call.
            raw: If True, return the raw response without cleaning.
        """
        api_call = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
        )

        response = await asyncio.wait_for(
            api_call,
            timeout=self.config.timeout,
        )

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content.strip()
            if not raw:
                content = self.clean_response(
                    self.extract_code_from_response(content)
                )
            return content

        return None

    def classify_error(
        self, error: Exception, attempt: int
    ) -> ErrorClassification:
        """Classify an error to determine retry behaviour.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (0-indexed).

        Returns:
            ``ErrorClassification`` with retry decision and backoff time.
        """
        error_str = str(error).lower()

        # Non-retryable errors (auth / permission)
        if any(code in error_str for code in self.NON_RETRYABLE_CODES):
            logger.error(f"Authentication error, not retrying: {error}")
            return ErrorClassification(
                is_retryable=False, backoff_seconds=0, error_type="auth"
            )

        # Rate-limit errors (retryable with longer backoff)
        if any(
            indicator in error_str
            for indicator in self.RATE_LIMIT_INDICATORS
        ):
            logger.warning(f"Rate limit hit on attempt {attempt + 1}")
            return ErrorClassification(
                is_retryable=True,
                backoff_seconds=self.RATE_LIMIT_BACKOFF,
                error_type="rate_limit",
            )

        # Other retryable errors (exponential backoff)
        logger.warning(
            f"Retryable error on attempt {attempt + 1}: {type(error).__name__}"
        )
        return ErrorClassification(
            is_retryable=True,
            backoff_seconds=2**attempt,
            error_type="retryable",
        )

    # ------------------------------------------------------------------
    # Static / utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code_from_response(response_text: str) -> str:
        """Extract code from ``<code>...</code>`` tags in an AI response.

        If no tags are found, or the content inside tags is too short
        (<= 10 chars), the full response is returned instead.
        """
        return _extract_code_from_response(response_text)

    @staticmethod
    def clean_response(content: str) -> str:
        """Clean and normalise an AI response.

        Strips markdown code fences and removes leading/trailing
        explanatory text that is not valid Python.
        """
        return _clean_response(content)

    @staticmethod
    def validate_python_code(code: str) -> bool:
        """Check if *code* is syntactically valid Python."""
        return _validate_python_code(code)

    @staticmethod
    def extract_tagged_sections(response: str) -> dict[str, str]:
        """Extract sections from a structured AI response with XML-like tags.

        Parses sections like ``<new_imports>...</new_imports>``,
        ``<new_tasks>...</new_tasks>``, etc.

        Returns:
            A dict mapping tag names to their content.
        """
        sections: dict[str, str] = {}
        pattern = r"<(\w+)>(.*?)</\1>"
        for match in re.finditer(pattern, response, re.DOTALL):
            sections[match.group(1)] = match.group(2).strip()
        return sections
