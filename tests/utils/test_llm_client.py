"""
Tests for llm_client module
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from devdox_ai_locust.utils.llm_client import (
    LLMClient,
    RateLimitInfo,
    TimeEstimate,
    AIServiceError,
)


class TestRateLimitInfo:
    """Test RateLimitInfo dataclass."""

    def test_from_headers_valid(self):
        """Test parsing valid rate limit headers."""
        headers = {
            "x-ratelimit-limit": "10",
            "x-ratelimit-remaining": "5",
            "x-ratelimit-reset": "2.5",
        }
        info = RateLimitInfo.from_headers(headers)
        assert info.requests_per_second == 10
        assert info.requests_per_minute == 600
        assert info.remaining == 5
        assert info.reset_seconds == 2.5

    def test_from_headers_missing(self):
        """Test parsing with missing headers uses defaults."""
        info = RateLimitInfo.from_headers({})
        assert info.requests_per_second == 1
        assert info.remaining == 0
        assert info.reset_seconds == 1.0

    def test_from_headers_invalid_values(self):
        """Test parsing with invalid header values falls back to defaults."""
        headers = {
            "x-ratelimit-limit": "not_a_number",
            "x-ratelimit-remaining": "bad",
            "x-ratelimit-reset": "nope",
        }
        info = RateLimitInfo.from_headers(headers)
        assert info.requests_per_second == 1
        assert info.remaining == 0
        assert info.reset_seconds == 1.0

    def test_default(self):
        """Test default rate limit info."""
        info = RateLimitInfo.default()
        assert info.requests_per_second == 1
        assert info.requests_per_minute == 60
        assert info.remaining == 60
        assert info.reset_seconds == 1


class TestTimeEstimate:
    """Test TimeEstimate dataclass."""

    def test_str_seconds(self):
        """Test string representation for short estimates."""
        est = TimeEstimate(
            total_calls=5, rpm=60, estimated_minutes=0.5, estimated_seconds=30
        )
        assert "30 seconds" in str(est)

    def test_str_minutes(self):
        """Test string representation for longer estimates."""
        est = TimeEstimate(
            total_calls=100, rpm=60, estimated_minutes=1.7, estimated_seconds=100
        )
        assert "1.7 minutes" in str(est)


class TestLLMClient:
    """Test LLMClient wrapper."""

    @pytest.fixture
    def mock_ai_client(self):
        return Mock()

    @pytest.fixture
    def mock_ai_config(self):
        config = Mock()
        config.model = "test-model"
        config.max_tokens = 4000
        config.temperature = 0.3
        config.timeout = 30
        return config

    @pytest.fixture
    def client(self, mock_ai_client, mock_ai_config):
        return LLMClient(mock_ai_client, mock_ai_config)

    def test_initial_concurrency(self, client):
        """Test initial concurrency is set to max."""
        assert client.current_concurrency == LLMClient.MAX_CONCURRENCY

    def test_custom_concurrency(self, mock_ai_client, mock_ai_config):
        """Test custom max concurrency."""
        c = LLMClient(mock_ai_client, mock_ai_config, max_concurrency=10)
        assert c.current_concurrency == 10

    def test_rate_limit_info_initially_none(self, client):
        """Test rate limit info starts as None."""
        assert client.rate_limit_info is None

    def test_update_rate_limit(self, client):
        """Test updating rate limit from headers."""
        headers = {
            "x-ratelimit-limit": "5",
            "x-ratelimit-remaining": "3",
            "x-ratelimit-reset": "1.0",
        }
        info = client.update_rate_limit(headers)
        assert info.requests_per_second == 5
        assert client.rate_limit_info is not None

    def test_update_concurrency_adjusts(self, client):
        """Test concurrency adjustment based on RPM."""
        client.update_concurrency(1000)
        # optimal = min(int(1000 * 0.8 / 20), 50) = min(40, 50) = 40
        assert client.current_concurrency == 40

    def test_update_concurrency_minimum(self, client):
        """Test concurrency never goes below 2."""
        client.update_concurrency(1)
        assert client.current_concurrency >= 2

    def test_estimate_time_no_rate_limit(self, client):
        """Test time estimation without rate limit info (defaults to 60 RPM)."""
        est = client.estimate_time(60)
        assert est.total_calls == 60
        assert est.estimated_minutes > 0

    def test_estimate_time_with_rate_limit(self, client):
        """Test time estimation with rate limit info set."""
        client.update_rate_limit(
            {
                "x-ratelimit-limit": "10",
                "x-ratelimit-remaining": "10",
                "x-ratelimit-reset": "1.0",
            }
        )
        est = client.estimate_time(100)
        assert est.total_calls == 100
        assert est.rpm > 0

    @pytest.mark.asyncio
    async def test_call_success(self, mock_ai_client, mock_ai_config):
        """Test successful API call."""
        mock_response = Mock(spec=[])
        mock_choice = Mock()
        mock_choice.message.content = "generated text"
        mock_response.choices = [mock_choice]

        mock_ai_client.chat.completions.create = Mock(return_value=mock_response)

        client = LLMClient(mock_ai_client, mock_ai_config)

        with patch("asyncio.get_event_loop") as mock_loop:
            future = asyncio.Future()
            future.set_result(mock_response)
            mock_loop.return_value.run_in_executor = Mock(return_value=future)

            result = await client.call([{"role": "user", "content": "hello"}])
            assert result == "generated text"

    @pytest.mark.asyncio
    async def test_call_raises_after_retries(self, mock_ai_client, mock_ai_config):
        """Test that AIServiceError is raised after all retries fail."""
        mock_ai_client.chat.completions.create = Mock(side_effect=RuntimeError("fail"))

        client = LLMClient(mock_ai_client, mock_ai_config)

        with patch("asyncio.get_event_loop") as mock_loop:
            future = asyncio.Future()
            future.set_exception(RuntimeError("fail"))

            def make_future(*args, **kwargs):
                f = asyncio.Future()
                f.set_exception(RuntimeError("fail"))
                return f

            mock_loop.return_value.run_in_executor = make_future

            with pytest.raises(AIServiceError, match="AI service failed after"):
                await client.call([{"role": "user", "content": "hello"}])
