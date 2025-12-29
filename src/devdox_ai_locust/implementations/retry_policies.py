"""
Retry Policy Implementations

Various retry strategies for handling transient failures.
"""

import logging
from typing import Set, Optional

from devdox_ai_locust.abstractions.retry_policy import RetryPolicy, RetryDecision
from devdox_ai_locust.abstractions.ai_client import (
    AIClientError,
    RateLimitError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)


class ExponentialBackoffPolicy:
    """
    Retry policy with exponential backoff.

    Backoff times: 1s, 2s, 4s, 8s, ...
    """

    # Errors that should never be retried
    NON_RETRYABLE_ERRORS = (
        AuthenticationError,
        PermissionError,
    )

    # Error message patterns that indicate non-retryable errors
    NON_RETRYABLE_PATTERNS = {
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "invalid token",
        "invalid api key",
    }

    def __init__(
        self,
        max_attempts: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
    ):
        """
        Initialize policy.

        Args:
            max_attempts: Maximum number of attempts
            base_backoff: Base backoff time in seconds
            max_backoff: Maximum backoff time in seconds
        """
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._attempt_count = 0

    def should_retry(self, error: Exception, attempt: int) -> RetryDecision:
        """Determine if operation should be retried."""
        # Check max attempts
        if attempt >= self.max_attempts:
            return RetryDecision(
                should_retry=False,
                reason=f"Max attempts ({self.max_attempts}) reached",
                attempt=attempt,
                max_attempts=self.max_attempts,
            )

        # Check non-retryable error types
        if isinstance(error, self.NON_RETRYABLE_ERRORS):
            return RetryDecision(
                should_retry=False,
                reason=f"Non-retryable error type: {type(error).__name__}",
                attempt=attempt,
                max_attempts=self.max_attempts,
            )

        # Check non-retryable patterns in error message
        error_str = str(error).lower()
        for pattern in self.NON_RETRYABLE_PATTERNS:
            if pattern in error_str:
                return RetryDecision(
                    should_retry=False,
                    reason=f"Non-retryable error pattern: {pattern}",
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                )

        # Check if error indicates retryable
        if isinstance(error, AIClientError) and not error.is_retryable:
            return RetryDecision(
                should_retry=False,
                reason="Error marked as non-retryable",
                attempt=attempt,
                max_attempts=self.max_attempts,
            )

        # Calculate exponential backoff
        backoff = min(
            self.base_backoff * (2 ** attempt),
            self.max_backoff,
        )

        return RetryDecision(
            should_retry=True,
            backoff_seconds=backoff,
            reason=f"Retrying with {backoff}s backoff",
            attempt=attempt,
            max_attempts=self.max_attempts,
        )

    def get_max_attempts(self) -> int:
        """Get max attempts."""
        return self.max_attempts

    def reset(self) -> None:
        """Reset policy state."""
        self._attempt_count = 0


class RateLimitAwarePolicy:
    """
    Retry policy that respects rate limit information.

    Uses longer backoff for rate limit errors.
    """

    def __init__(
        self,
        base_policy: Optional[RetryPolicy] = None,
        rate_limit_backoff: float = 10.0,
        rate_limit_patterns: Optional[Set[str]] = None,
    ):
        """
        Initialize policy.

        Args:
            base_policy: Base retry policy
            rate_limit_backoff: Backoff time for rate limits
            rate_limit_patterns: Patterns indicating rate limiting
        """
        self.base_policy = base_policy or ExponentialBackoffPolicy()
        self.rate_limit_backoff = rate_limit_backoff
        self.rate_limit_patterns = rate_limit_patterns or {"429", "rate limit", "too many requests"}

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        if isinstance(error, RateLimitError):
            return True

        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.rate_limit_patterns)

    def should_retry(self, error: Exception, attempt: int) -> RetryDecision:
        """Determine if operation should be retried."""
        # Check base policy first
        base_decision = self.base_policy.should_retry(error, attempt)

        if not base_decision.should_retry:
            return base_decision

        # If rate limit error, use longer backoff
        if self._is_rate_limit_error(error):
            # Use retry_after if available
            backoff = self.rate_limit_backoff
            if isinstance(error, RateLimitError) and error.retry_after:
                backoff = error.retry_after

            return RetryDecision(
                should_retry=True,
                backoff_seconds=backoff,
                reason=f"Rate limit hit, waiting {backoff}s",
                attempt=attempt,
                max_attempts=base_decision.max_attempts,
            )

        return base_decision

    def get_max_attempts(self) -> int:
        """Get max attempts."""
        return self.base_policy.get_max_attempts()

    def reset(self) -> None:
        """Reset policy state."""
        self.base_policy.reset()


class NoRetryPolicy:
    """
    Policy that never retries.

    Useful for testing or when retries are not desired.
    """

    def should_retry(self, error: Exception, attempt: int) -> RetryDecision:
        """Never retry."""
        return RetryDecision(
            should_retry=False,
            reason="Retries disabled",
            attempt=attempt,
            max_attempts=1,
        )

    def get_max_attempts(self) -> int:
        """Max attempts is 1 (no retries)."""
        return 1

    def reset(self) -> None:
        """Nothing to reset."""
        pass


class LinearBackoffPolicy:
    """
    Retry policy with linear backoff.

    Backoff times: 1s, 2s, 3s, 4s, ...
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_increment: float = 1.0,
        initial_backoff: float = 1.0,
    ):
        """
        Initialize policy.

        Args:
            max_attempts: Maximum number of attempts
            backoff_increment: Increment per attempt
            initial_backoff: Initial backoff time
        """
        self.max_attempts = max_attempts
        self.backoff_increment = backoff_increment
        self.initial_backoff = initial_backoff

    def should_retry(self, error: Exception, attempt: int) -> RetryDecision:
        """Determine if operation should be retried."""
        if attempt >= self.max_attempts:
            return RetryDecision(
                should_retry=False,
                reason=f"Max attempts ({self.max_attempts}) reached",
                attempt=attempt,
                max_attempts=self.max_attempts,
            )

        backoff = self.initial_backoff + (attempt * self.backoff_increment)

        return RetryDecision(
            should_retry=True,
            backoff_seconds=backoff,
            reason=f"Retrying with {backoff}s backoff",
            attempt=attempt,
            max_attempts=self.max_attempts,
        )

    def get_max_attempts(self) -> int:
        """Get max attempts."""
        return self.max_attempts

    def reset(self) -> None:
        """Nothing to reset."""
        pass
