"""
AI Client Implementations

Production and mock implementations of the AIClient protocol.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List

from devdox_ai_locust.abstractions.ai_client import (
    AIClient,
    AICompletionRequest,
    AICompletionResponse,
    AIClientError,
    RateLimitError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)


class TogetherAIClient:
    """
    Production AI client using Together AI API.

    Wraps the AsyncTogether client to conform to the AIClient protocol.
    """

    def __init__(self, together_client: Any):
        """
        Initialize with Together AI client.

        Args:
            together_client: AsyncTogether instance
        """
        self._client = together_client
        self._is_available = together_client is not None

    async def complete(self, request: AICompletionRequest) -> Optional[AICompletionResponse]:
        """Generate completion from Together AI."""
        if not self._is_available:
            logger.warning("AI client not available")
            return None

        try:
            api_call = self._client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop_sequences,
                **request.extra_params,
            )

            response = await asyncio.wait_for(api_call, timeout=request.timeout)

            if not response.choices:
                logger.warning("Empty response from AI")
                return None

            content = response.choices[0].message.content
            if not content:
                return None

            return AICompletionResponse(
                content=content,
                model=request.model,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                } if response.usage else None,
            )

        except asyncio.TimeoutError:
            raise AIClientError(
                f"Request timed out after {request.timeout}s",
                is_retryable=True,
            )
        except Exception as e:
            error_str = str(e).lower()

            if "429" in error_str or "rate limit" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")

            if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")

            raise AIClientError(
                str(e),
                is_retryable=True,
                original_error=e,
            )

    def is_available(self) -> bool:
        """Check if client is available."""
        return self._is_available


class MockAIClient:
    """
    Mock AI client for testing.

    Returns predefined responses or generates simple outputs.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "# AI generated code\npass",
        should_fail: bool = False,
        fail_after: int = 0,
    ):
        """
        Initialize mock client.

        Args:
            responses: Dict mapping message content to responses
            default_response: Default response if no match found
            should_fail: If True, all requests fail
            fail_after: Fail after this many successful requests
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.should_fail = should_fail
        self.fail_after = fail_after
        self.call_count = 0
        self.call_history: List[AICompletionRequest] = []

    async def complete(self, request: AICompletionRequest) -> Optional[AICompletionResponse]:
        """Return mock response."""
        self.call_count += 1
        self.call_history.append(request)

        if self.should_fail:
            raise AIClientError("Mock failure", is_retryable=False)

        if self.fail_after > 0 and self.call_count > self.fail_after:
            raise AIClientError("Mock failure after limit", is_retryable=True)

        # Look for matching response
        last_message = request.messages[-1].get("content", "") if request.messages else ""
        response_content = self.default_response

        for key, value in self.responses.items():
            if key in last_message:
                response_content = value
                break

        return AICompletionResponse(
            content=response_content,
            model=request.model,
            finish_reason="stop",
        )

    def is_available(self) -> bool:
        """Always available in tests."""
        return not self.should_fail

    def reset(self) -> None:
        """Reset call history."""
        self.call_count = 0
        self.call_history.clear()


class RecordingAIClient:
    """
    AI client that records all requests and responses.

    Useful for capturing real responses for replay in tests.
    """

    def __init__(self, wrapped_client: AIClient):
        """
        Initialize with wrapped client.

        Args:
            wrapped_client: The real AI client to wrap
        """
        self._client = wrapped_client
        self.recordings: List[tuple[AICompletionRequest, Optional[AICompletionResponse]]] = []

    async def complete(self, request: AICompletionRequest) -> Optional[AICompletionResponse]:
        """Record request and response."""
        response = await self._client.complete(request)
        self.recordings.append((request, response))
        return response

    def is_available(self) -> bool:
        """Delegate to wrapped client."""
        return self._client.is_available()

    def get_recordings(self) -> List[tuple[AICompletionRequest, Optional[AICompletionResponse]]]:
        """Get all recorded request/response pairs."""
        return self.recordings.copy()

    def clear_recordings(self) -> None:
        """Clear all recordings."""
        self.recordings.clear()
