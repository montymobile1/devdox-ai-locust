"""
Retry Policy Protocol

Defines the contract for retry logic implementations.
Separates retry decisions from business logic.
"""

from typing import Protocol, Optional, runtime_checkable
from pydantic import BaseModel


class RetryDecision(BaseModel):
    """Decision about whether to retry an operation"""
    should_retry: bool
    backoff_seconds: float = 0.0
    reason: str = ""
    attempt: int = 0
    max_attempts: int = 3


@runtime_checkable
class RetryPolicy(Protocol):
    """
    Protocol for retry policy implementations.

    Implementations:
        - ExponentialBackoffPolicy: Exponential backoff (1s, 2s, 4s...)
        - LinearBackoffPolicy: Linear backoff (1s, 2s, 3s...)
        - RateLimitAwarePolicy: Respects rate limit headers
        - NoRetryPolicy: Never retries (for testing)

    Example:
        class ExponentialBackoffPolicy:
            def __init__(self, max_attempts: int = 3, base_backoff: float = 1.0):
                self.max_attempts = max_attempts
                self.base_backoff = base_backoff

            def should_retry(self, error: Exception, attempt: int) -> RetryDecision:
                if attempt >= self.max_attempts:
                    return RetryDecision(should_retry=False, reason="max attempts reached")

                if isinstance(error, AuthenticationError):
                    return RetryDecision(should_retry=False, reason="auth error")

                backoff = self.base_backoff * (2 ** attempt)
                return RetryDecision(
                    should_retry=True,
                    backoff_seconds=backoff,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                )
    """

    def should_retry(self, error: Exception, attempt: int) -> RetryDecision:
        """
        Determine if an operation should be retried.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            RetryDecision with retry recommendation
        """
        ...

    def get_max_attempts(self) -> int:
        """
        Get the maximum number of attempts.

        Returns:
            Maximum attempts before giving up
        """
        ...

    def reset(self) -> None:
        """
        Reset the policy state.

        Called when starting a new operation.
        """
        ...
