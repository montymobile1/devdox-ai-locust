"""
Tests for TogetherAIClient module.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import asyncio

from devdox_ai_locust.utils.ai_client import (
    TogetherAIClient,
    AIEnhancementConfig,
    ErrorClassification,
)


@pytest.fixture
def config():
    """Create default config."""
    return AIEnhancementConfig()


@pytest.fixture
def client(config):
    """Create client instance."""
    return TogetherAIClient(api_key="test-key", config=config)


class TestAIEnhancementConfig:
    """Tests for AIEnhancementConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AIEnhancementConfig()

        assert config.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert config.max_tokens == 8000
        assert config.temperature == 0.3
        assert config.timeout == 60

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AIEnhancementConfig(
            model="custom-model",
            max_tokens=4000,
            temperature=0.5,
            timeout=120,
        )

        assert config.model == "custom-model"
        assert config.max_tokens == 4000
        assert config.temperature == 0.5
        assert config.timeout == 120


class TestTogetherAIClientInit:
    """Tests for client initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        client = TogetherAIClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.config.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def test_init_with_custom_config(self, config):
        """Test initialization with custom config."""
        config.model = "custom-model"
        client = TogetherAIClient(api_key="test-key", config=config)

        assert client.config.model == "custom-model"

    def test_client_not_set_outside_context(self, client):
        """Test that client raises error when not in context."""
        with pytest.raises(RuntimeError, match="must be used as an async context manager"):
            _ = client.client


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, client):
        """Test entering context manager."""
        async with client as c:
            assert c._client is not None
            assert c is client

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, client):
        """Test exiting context manager."""
        async with client:
            pass

        assert client._client is None


class TestExtractCodeFromResponse:
    """Tests for extract_code_from_response static method."""

    def test_extract_code_tags(self):
        """Test extracting code from <code> tags."""
        response = """
Here is the code:
<code>
def hello():
    print("Hello, World!")
</code>
That's it!
"""
        result = TogetherAIClient.extract_code_from_response(response)

        assert "def hello():" in result
        assert "print" in result

    def test_no_code_tags(self):
        """Test when no <code> tags are present."""
        response = """
def hello():
    print("Hello")
"""
        result = TogetherAIClient.extract_code_from_response(response)

        assert result == response.strip()

    def test_multiple_code_tags(self):
        """Test with multiple <code> tags - uses longest."""
        response = """
<code>short</code>
<code>
This is the longer code block
with multiple lines
</code>
"""
        result = TogetherAIClient.extract_code_from_response(response)

        assert "longer code block" in result

    def test_empty_code_tags(self):
        """Test with empty <code> tags."""
        response = """
<code></code>
Full response here
"""
        result = TogetherAIClient.extract_code_from_response(response)

        assert "Full response" in result

    def test_short_code_content(self):
        """Test when code content is too short."""
        response = """
<code>x=1</code>
Full response is longer
"""
        result = TogetherAIClient.extract_code_from_response(response)

        # Should use full response since code is <= 10 chars
        assert "Full response" in result


class TestCleanResponse:
    """Tests for clean_response static method."""

    def test_remove_markdown_python_block(self):
        """Test removing ```python code blocks."""
        content = """```python
def hello():
    pass
```"""
        result = TogetherAIClient.clean_response(content)

        assert "```" not in result
        assert "def hello():" in result

    def test_remove_generic_markdown_block(self):
        """Test removing generic ``` code blocks."""
        content = """```
def hello():
    pass
```"""
        result = TogetherAIClient.clean_response(content)

        assert "```" not in result

    def test_remove_leading_explanation(self):
        """Test removing leading explanation text."""
        content = """Here is the implementation:
Some more explanation text.

import locust
from locust import HttpUser
"""
        result = TogetherAIClient.clean_response(content)

        assert result.startswith("import locust")

    def test_remove_trailing_explanation(self):
        """Test removing trailing explanation text."""
        content = """import locust

class User:
    pass

Note: This is an explanation.
The above code does xyz.
"""
        result = TogetherAIClient.clean_response(content)

        assert not result.strip().endswith("xyz.")

    def test_preserve_valid_code(self):
        """Test that valid code is preserved."""
        content = """from locust import HttpUser, task

class APIUser(HttpUser):
    @task
    def my_task(self):
        self.client.get("/api")
"""
        result = TogetherAIClient.clean_response(content)

        assert "from locust import" in result
        assert "class APIUser" in result
        assert "@task" in result


class TestValidatePythonCode:
    """Tests for validate_python_code static method."""

    def test_valid_code(self):
        """Test with valid Python code."""
        code = """
def hello():
    print("Hello")

class MyClass:
    pass
"""
        assert TogetherAIClient.validate_python_code(code) is True

    def test_invalid_code(self):
        """Test with invalid Python code."""
        code = """
def broken(
    invalid syntax
"""
        assert TogetherAIClient.validate_python_code(code) is False

    def test_empty_code(self):
        """Test with empty code."""
        assert TogetherAIClient.validate_python_code("") is True

    def test_code_with_syntax_error(self):
        """Test code with syntax error."""
        code = "if True print('x')"
        assert TogetherAIClient.validate_python_code(code) is False


class TestExtractTaggedSections:
    """Tests for extract_tagged_sections static method."""

    def test_extract_single_section(self):
        """Test extracting a single tagged section."""
        response = """
<new_imports>
import json
from datetime import datetime
</new_imports>
"""
        result = TogetherAIClient.extract_tagged_sections(response)

        assert "new_imports" in result
        assert "import json" in result["new_imports"]

    def test_extract_multiple_sections(self):
        """Test extracting multiple tagged sections."""
        response = """
<new_imports>
import json
</new_imports>

<new_tasks>
@task
def my_task(self):
    pass
</new_tasks>

<new_helpers>
def helper():
    pass
</new_helpers>
"""
        result = TogetherAIClient.extract_tagged_sections(response)

        assert len(result) == 3
        assert "new_imports" in result
        assert "new_tasks" in result
        assert "new_helpers" in result

    def test_empty_sections(self):
        """Test with empty sections."""
        response = """
<new_imports></new_imports>
<new_tasks>
@task
def task(self):
    pass
</new_tasks>
"""
        result = TogetherAIClient.extract_tagged_sections(response)

        assert result["new_imports"] == ""
        assert "@task" in result["new_tasks"]

    def test_no_sections(self):
        """Test when no sections are present."""
        response = "Just plain text without any tags"
        result = TogetherAIClient.extract_tagged_sections(response)

        assert result == {}

    def test_nested_content(self):
        """Test sections with nested content."""
        response = """
<new_tasks>
@task
def my_task(self):
    data = {"key": "value"}
    if True:
        self.client.post("/api", json=data)
</new_tasks>
"""
        result = TogetherAIClient.extract_tagged_sections(response)

        assert "new_tasks" in result
        assert '{"key": "value"}' in result["new_tasks"]


class TestErrorClassification:
    """Tests for error classification."""

    def test_auth_error(self, client):
        """Test authentication error classification."""
        error = Exception("401 Unauthorized")
        result = client.classify_error(error, 0)

        assert result.is_retryable is False
        assert result.error_type == "auth"

    def test_forbidden_error(self, client):
        """Test forbidden error classification."""
        error = Exception("403 Forbidden - Access denied")
        result = client.classify_error(error, 0)

        assert result.is_retryable is False
        assert result.error_type == "auth"

    def test_rate_limit_error(self, client):
        """Test rate limit error classification."""
        error = Exception("429 Too Many Requests - rate limit exceeded")
        result = client.classify_error(error, 0)

        assert result.is_retryable is True
        assert result.error_type == "rate_limit"
        assert result.backoff_seconds == TogetherAIClient.RATE_LIMIT_BACKOFF

    def test_generic_retryable_error(self, client):
        """Test generic retryable error classification."""
        error = Exception("Connection timeout")
        result = client.classify_error(error, 1)

        assert result.is_retryable is True
        assert result.error_type == "retryable"
        assert result.backoff_seconds == 2  # 2^1

    def test_exponential_backoff(self, client):
        """Test exponential backoff calculation."""
        error = Exception("Some error")

        for attempt in range(3):
            result = client.classify_error(error, attempt)
            assert result.backoff_seconds == 2**attempt


class TestAPICall:
    """Tests for API call functionality."""

    @pytest.mark.asyncio
    async def test_successful_call(self, client):
        """Test successful API call."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "<code>print('hello')</code>"

        mock_together = AsyncMock()
        mock_together.chat.completions.create = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_together
            result = await client.call("system", "user")

        assert "print" in result

    @pytest.mark.asyncio
    async def test_empty_response(self, client):
        """Test handling of empty response."""
        mock_response = Mock()
        mock_response.choices = []

        mock_together = AsyncMock()
        mock_together.chat.completions.create = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_together
            result = await client.call("system", "user")

        assert result == ""

    @pytest.mark.asyncio
    async def test_timeout_retry(self, client):
        """Test retry on timeout."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "success"
            return mock_response

        mock_together = AsyncMock()
        mock_together.chat.completions.create = mock_create

        async with client:
            client._client = mock_together
            result = await client.call("system", "user")

        assert call_count == 3
        assert "success" in result

    @pytest.mark.asyncio
    async def test_non_retryable_error(self, client):
        """Test non-retryable error stops retries."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized")

        mock_together = AsyncMock()
        mock_together.chat.completions.create = mock_create

        async with client:
            client._client = mock_together
            result = await client.call("system", "user")

        assert call_count == 1  # No retries
        assert result == ""

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client):
        """Test that max retries is respected."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Server error")

        mock_together = AsyncMock()
        mock_together.chat.completions.create = mock_create

        async with client:
            client._client = mock_together
            result = await client.call("system", "user")

        assert call_count == TogetherAIClient.MAX_RETRIES
        assert result == ""


class TestBuildMessages:
    """Tests for message building."""

    def test_build_messages(self):
        """Test building messages payload."""
        messages = TogetherAIClient._build_messages("system prompt", "user prompt")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user prompt"
