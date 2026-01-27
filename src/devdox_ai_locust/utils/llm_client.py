"""
LLM Client Module

Handles AI service calls, rate limiting, and retry logic.
Provides a clean interface for interacting with LLM APIs.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Raised when AI service fails after all retries"""

    pass


@dataclass
class RateLimitInfo:
    """Rate limit information from API response"""

    requests_per_second: int
    requests_per_minute: int
    remaining: int
    reset_seconds: float

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "RateLimitInfo":
        """Parse rate limit info from response headers."""
        try:
            rps = int(headers.get("x-ratelimit-limit", "1"))
        except (ValueError, TypeError):
            rps = 1
        try:
            remaining = int(headers.get("x-ratelimit-remaining", "0"))
        except (ValueError, TypeError):
            remaining = 0
        try:
            reset = float(headers.get("x-ratelimit-reset", "1"))
        except (ValueError, TypeError):
            reset = 1.0

        return cls(
            requests_per_second=rps,
            requests_per_minute=rps * 60,
            remaining=remaining,
            reset_seconds=reset,
        )

    @classmethod
    def default(cls) -> "RateLimitInfo":
        """Default rate limit (conservative estimate)."""
        return cls(
            requests_per_second=1,
            requests_per_minute=60,
            remaining=60,
            reset_seconds=1,
        )


@dataclass
class TimeEstimate:
    """Time estimate for generation."""

    total_calls: int
    rpm: int
    estimated_minutes: float
    estimated_seconds: float

    def __str__(self) -> str:
        if self.estimated_minutes < 1:
            return f"~{self.estimated_seconds:.0f} seconds"
        return f"~{self.estimated_minutes:.1f} minutes"


class LLMClient:
    """
    Client for making LLM API calls with rate limiting and retry logic.

    Handles:
    - Concurrent request management via semaphore
    - Rate limit tracking and adjustment
    - Automatic retries with exponential backoff
    """

    MAX_CONCURRENCY = 50
    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]

    def __init__(
        self,
        ai_client: Any,
        ai_config: Any,
        max_concurrency: int = MAX_CONCURRENCY,
    ):
        """
        Initialize the LLM client.

        Args:
            ai_client: The underlying AI client (e.g., Together AI client)
            ai_config: AI configuration (model, timeout, etc.)
            max_concurrency: Maximum concurrent API calls
        """
        self.ai_client = ai_client
        self.ai_config = ai_config
        self._rate_limit_info: Optional[RateLimitInfo] = None
        self._max_concurrency = max_concurrency
        self._current_concurrency = max_concurrency
        self._api_semaphore = asyncio.Semaphore(self._current_concurrency)

    @property
    def current_concurrency(self) -> int:
        """Current concurrency level."""
        return self._current_concurrency

    @property
    def rate_limit_info(self) -> Optional[RateLimitInfo]:
        """Current rate limit information."""
        return self._rate_limit_info

    def update_concurrency(self, rpm: int) -> None:
        """
        Update concurrency based on rate limit.

        Args:
            rpm: Requests per minute from rate limit headers
        """
        optimal = min(int(rpm * 0.8 / 20), self._max_concurrency)
        new_concurrency = max(2, optimal)

        if new_concurrency != self._current_concurrency:
            logger.info(
                f"Adjusting concurrency: {self._current_concurrency} → "
                f"{new_concurrency} (based on {rpm} RPM)"
            )
            self._current_concurrency = new_concurrency
            self._api_semaphore = asyncio.Semaphore(new_concurrency)

    def update_rate_limit(self, headers: Dict[str, str]) -> RateLimitInfo:
        """
        Update rate limit info from API response headers.

        Args:
            headers: Response headers from AI API

        Returns:
            Updated RateLimitInfo
        """
        self._rate_limit_info = RateLimitInfo.from_headers(headers)
        self.update_concurrency(self._rate_limit_info.requests_per_minute)
        return self._rate_limit_info

    def estimate_time(self, num_calls: int) -> TimeEstimate:
        """
        Estimate time for a given number of API calls.

        Args:
            num_calls: Number of API calls to make

        Returns:
            TimeEstimate with calculated values
        """
        rpm = self._rate_limit_info.requests_per_minute if self._rate_limit_info else 60
        effective_rpm = min(rpm, self._current_concurrency * 20)
        estimated_minutes = num_calls / effective_rpm
        estimated_seconds = estimated_minutes * 60

        return TimeEstimate(
            total_calls=num_calls,
            rpm=effective_rpm,
            estimated_minutes=estimated_minutes,
            estimated_seconds=estimated_seconds,
        )

    async def call(
        self,
        messages: List[Dict[str, str]],
        endpoint_info: str = "",
    ) -> str:
        """
        Call the AI service with rate limiting and retry logic.

        Args:
            messages: Chat messages to send to the AI
            endpoint_info: Info string for logging

        Returns:
            Generated text from the AI

        Raises:
            AIServiceError: If all retries fail
        """
        async with self._api_semaphore:
            last_error: Optional[Exception] = None

            for attempt in range(self.MAX_RETRIES):
                try:
                    response = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.ai_client.chat.completions.create(
                                model=self.ai_config.model,
                                messages=messages,
                                max_tokens=self.ai_config.max_tokens,
                                temperature=self.ai_config.temperature,
                            ),
                        ),
                        timeout=self.ai_config.timeout,
                    )

                    # Update rate limits from response headers if available
                    if hasattr(response, "_headers"):
                        self.update_rate_limit(dict(response._headers))

                    return response.choices[0].message.content

                except asyncio.TimeoutError:
                    last_error = asyncio.TimeoutError(
                        f"API call timed out after {self.ai_config.timeout}s"
                    )
                    logger.warning(
                        f"[{endpoint_info}] Timeout on attempt {attempt + 1}, "
                        f"retrying in {self.RETRY_DELAYS[attempt]}s..."
                    )
                except Exception as e:
                    last_error = e
                    error_msg = str(e)

                    # Check for rate limit error
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        delay = self.RETRY_DELAYS[attempt] * 2
                        logger.warning(
                            f"[{endpoint_info}] Rate limited, waiting {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.warning(
                        f"[{endpoint_info}] API error on attempt {attempt + 1}: "
                        f"{error_msg}"
                    )

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAYS[attempt])

            raise AIServiceError(
                f"AI service failed after {self.MAX_RETRIES} attempts: {last_error}"
            )
