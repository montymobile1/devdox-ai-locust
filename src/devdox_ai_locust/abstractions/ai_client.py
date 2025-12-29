"""
AI Client Protocol

Defines the contract for AI/LLM client implementations.
Allows swapping between different AI providers (Together, OpenAI, Anthropic, etc.)
and enables testing with mock clients.
"""

from typing import Protocol, Optional, List, Dict, Any, runtime_checkable
from pydantic import BaseModel, Field


class AICompletionRequest(BaseModel):
    """Request model for AI completion"""
    messages: List[Dict[str, str]]
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 60
    stop_sequences: Optional[List[str]] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class AICompletionResponse(BaseModel):
    """Response model from AI completion"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@runtime_checkable
class AIClient(Protocol):
    """
    Protocol for AI/LLM client implementations.

    Implementations:
        - TogetherAIClient: Uses Together AI API
        - OpenAIClient: Uses OpenAI API
        - MockAIClient: Returns predefined responses for testing
        - RecordingAIClient: Records and replays responses

    Example:
        class TogetherAIClient:
            async def complete(self, request: AICompletionRequest) -> AICompletionResponse:
                response = await self._client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                return AICompletionResponse(
                    content=response.choices[0].message.content,
                    model=request.model,
                )
    """

    async def complete(self, request: AICompletionRequest) -> Optional[AICompletionResponse]:
        """
        Generate a completion from the AI model.

        Args:
            request: The completion request with messages and parameters

        Returns:
            AICompletionResponse if successful, None if failed

        Raises:
            AIClientError: If the request fails and cannot be retried
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the AI client is available and configured.

        Returns:
            True if the client can make requests
        """
        ...


class AIClientError(Exception):
    """Base exception for AI client errors"""

    def __init__(
        self,
        message: str,
        is_retryable: bool = False,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.is_retryable = is_retryable
        self.status_code = status_code
        self.original_error = original_error


class RateLimitError(AIClientError):
    """Rate limit exceeded"""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, is_retryable=True, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(AIClientError):
    """Authentication failed"""

    def __init__(self, message: str):
        super().__init__(message, is_retryable=False, status_code=401)
