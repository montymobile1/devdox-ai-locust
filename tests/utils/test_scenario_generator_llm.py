"""
Tests for LLM-dependent methods in ScenarioWorkflowGenerator.

All tests use heavy mocking to avoid real LLM calls.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from devdox_ai_locust.utils.scenario_generator import (
    ScenarioWorkflowGenerator,
    ScenarioType,
    CodeValidationError,
)
from devdox_ai_locust.utils.open_ai_parser import (
    Endpoint,
    Parameter,
    RequestBody,
    Response,
    ParameterType,
)
from devdox_ai_locust.config import AIEnhancementConfig
from devdox_ai_locust.utils.llm_client import AIServiceError, RateLimitInfo


# ---------------------------------------------------------------------------
# Sample valid code blocks
# ---------------------------------------------------------------------------

VALID_POSITIVE_CODE = '''
import random
from locust import HttpUser, task, between, tag

class GetUsersPositiveWorkflow(HttpUser):
    """Positive workflow tests for GET /users"""
    wait_time = between(1, 3)

    @task
    @tag("positive")
    def test_get_users_success(self):
        with self.client.get("/users", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Expected 200, got {response.status_code}")
'''

VALID_NEGATIVE_CODE = '''
import random
from locust import HttpUser, task, between, tag

class GetUsersNegativeWorkflow(HttpUser):
    """Negative workflow tests for GET /users"""
    wait_time = between(1, 3)

    @task
    @tag("negative")
    def test_get_users_invalid_limit(self):
        with self.client.get("/users?limit=-1", catch_response=True) as response:
            if response.status_code in [400, 422]:
                response.success()
            else:
                response.failure(f"Expected 4xx, got {response.status_code}")
'''

VALID_SECURITY_CODE = '''
import random
from locust import HttpUser, task, between, tag

class GetUsersSecurityWorkflow(HttpUser):
    """Security workflow tests for GET /users"""
    wait_time = between(1, 3)

    @task
    @tag("security")
    def test_sql_injection(self):
        with self.client.get("/users?limit=1%27%20OR%201%3D1", catch_response=True) as response:
            if response.status_code in [400, 422]:
                response.success()
            else:
                response.failure(f"Expected 4xx, got {response.status_code}")
'''

VALID_ORCHESTRATOR_CODE = '''
import random
from locust import HttpUser, task, between, tag, SequentialTaskSet

class UsersOrchestrator(SequentialTaskSet):
    """Orchestrator for users tag"""

    @task
    def test_crud_lifecycle(self):
        pass
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content):
    """Helper to create a mock LLM response without headers attr."""
    mock_response = Mock(spec=["choices"])
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def _make_endpoint(method="GET", path="/users", operation_id="getUsers"):
    """Create a simple Endpoint for testing."""
    return Endpoint(
        path=path,
        method=method,
        operation_id=operation_id,
        summary=f"{method} {path}",
        description=None,
        parameters=[
            Parameter(
                name="limit",
                location=ParameterType.QUERY,
                required=False,
                type="integer",
                description="limit",
            )
        ],
        request_body=None,
        responses=[
            Response(
                status_code="200",
                description="OK",
                content_type="application/json",
                schema={"type": "object"},
            )
        ],
        tags=["users"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ai_config():
    config = Mock(spec=AIEnhancementConfig)
    config.model = "test-model"
    config.max_tokens = 4000
    config.temperature = 0.3
    config.timeout = 30
    return config


@pytest.fixture
def mock_ai_client():
    client = AsyncMock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(f"<code>{VALID_POSITIVE_CODE}</code>")
    )
    return client


@pytest.fixture
def generator(ai_config, mock_ai_client):
    prompt_dir = (
        Path(__file__).parent.parent.parent / "src" / "devdox_ai_locust" / "prompt"
    )
    gen = ScenarioWorkflowGenerator(
        prompt_dir=prompt_dir,
        ai_client=mock_ai_client,
        ai_config=ai_config,
    )
    return gen


@pytest.fixture
def endpoint():
    return _make_endpoint()


@pytest.fixture
def post_endpoint():
    return Endpoint(
        path="/users",
        method="POST",
        operation_id="createUser",
        summary="Create user",
        description=None,
        parameters=[],
        request_body=RequestBody(
            content_type="application/json",
            schema={
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["username", "email"],
            },
            required=True,
        ),
        responses=[
            Response(
                status_code="201",
                description="Created",
                content_type="application/json",
                schema={"type": "object"},
            )
        ],
        tags=["users"],
    )


# ===========================================================================
# 1. _try_ai_call tests
# ===========================================================================


@pytest.mark.asyncio
async def test_try_ai_call_success(generator, mock_ai_client):
    """Successful call returns content string."""
    messages = [{"role": "user", "content": "test"}]
    result = await generator._try_ai_call(messages, "positive", 0)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_try_ai_call_returns_exception_on_timeout(generator, mock_ai_client):
    """Returns exception on timeout."""
    mock_ai_client.chat.completions.create = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )
    messages = [{"role": "user", "content": "test"}]
    result = await generator._try_ai_call(messages, "positive", 0)
    assert isinstance(result, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_try_ai_call_returns_exception_on_generic_error(
    generator, mock_ai_client
):
    """Returns exception on generic error."""
    mock_ai_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("API down")
    )
    messages = [{"role": "user", "content": "test"}]
    result = await generator._try_ai_call(messages, "positive", 0)
    assert isinstance(result, RuntimeError)


@pytest.mark.asyncio
async def test_try_ai_call_empty_response(generator, mock_ai_client):
    """Returns content even when response has empty message content."""
    empty_resp = _make_llm_response("")
    mock_ai_client.chat.completions.create = AsyncMock(return_value=empty_resp)
    messages = [{"role": "user", "content": "test"}]
    result = await generator._try_ai_call(messages, "positive", 0)
    assert isinstance(result, str)


# ===========================================================================
# 2. _call_ai_service tests
# ===========================================================================


@pytest.mark.asyncio
async def test_call_ai_service_success(generator, mock_ai_client):
    """Successful call extracts content from response."""
    result = await generator._call_ai_service("test prompt", "positive")
    assert isinstance(result, str)
    assert "GetUsersPositiveWorkflow" in result


@pytest.mark.asyncio
async def test_call_ai_service_raises_after_retries(generator, mock_ai_client):
    """Raises AIServiceError after max retries."""
    mock_ai_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail"))
    with pytest.raises(AIServiceError, match="AI service failed after 3 attempts"):
        await generator._call_ai_service("test prompt", "positive")


@pytest.mark.asyncio
async def test_call_ai_service_retries_then_succeeds(generator, mock_ai_client):
    """Retries on exception and succeeds on later attempt."""
    mock_ai_client.chat.completions.create = AsyncMock(
        side_effect=[
            RuntimeError("rate limit"),
            _make_llm_response(f"<code>{VALID_POSITIVE_CODE}</code>"),
        ]
    )
    result = await generator._call_ai_service("test prompt", "positive")
    assert isinstance(result, str)


# ===========================================================================
# 3. _validate_llm_response tests
# ===========================================================================


def test_validate_llm_response_passes_with_valid(generator, endpoint):
    """Passes with valid content."""
    generator._validate_llm_response(
        "some valid content", ScenarioType.POSITIVE, endpoint
    )


def test_validate_llm_response_raises_on_none(generator, endpoint):
    """Raises on None content."""
    with pytest.raises(AIServiceError, match="empty response"):
        generator._validate_llm_response(None, ScenarioType.POSITIVE, endpoint)


def test_validate_llm_response_raises_on_empty(generator, endpoint):
    """Raises on empty string content."""
    with pytest.raises(AIServiceError, match="empty response"):
        generator._validate_llm_response("", ScenarioType.POSITIVE, endpoint)


def test_validate_llm_response_raises_on_html(generator, endpoint):
    """Raises on HTML error page."""
    html = "<html><body>Error 503</body></html>"
    with pytest.raises(AIServiceError, match="HTML error page"):
        generator._validate_llm_response(html, ScenarioType.POSITIVE, endpoint)


# ===========================================================================
# 4. _extract_and_validate_code tests
# ===========================================================================


@pytest.mark.asyncio
async def test_extract_and_validate_code_valid(generator, endpoint):
    """Extracts code from <code> tags and returns valid."""
    content = f"<code>{VALID_POSITIVE_CODE}</code>"
    fixed, is_valid, error = await generator._extract_and_validate_code(
        content,
        "GetUsersPositiveWorkflow",
        ScenarioType.POSITIVE,
        "default",
        "get_users",
        "positive",
    )
    assert is_valid
    assert not error  # empty string or None
    assert "GetUsersPositiveWorkflow" in fixed


@pytest.mark.asyncio
async def test_extract_and_validate_code_invalid_syntax(generator, endpoint):
    """Handles invalid Python syntax."""
    bad_code = "<code>def foo(\n    pass</code>"
    fixed, is_valid, error = await generator._extract_and_validate_code(
        bad_code, "Test", ScenarioType.POSITIVE, "default", "test_ep", "positive"
    )
    assert not is_valid
    assert error is not None


@pytest.mark.asyncio
async def test_extract_and_validate_code_no_tags(generator, endpoint):
    """Extracts code even without explicit <code> tags."""
    fixed, is_valid, error = await generator._extract_and_validate_code(
        VALID_POSITIVE_CODE,
        "GetUsersPositiveWorkflow",
        ScenarioType.POSITIVE,
        "default",
        "get_users",
        "positive",
    )
    assert is_valid


# ===========================================================================
# 5. _apply_code_fixes tests
# ===========================================================================


def test_apply_code_fixes_class_name(generator):
    """Fixes class name in generated code."""
    code = """
import random
from locust import HttpUser, task, between

class WrongName(HttpUser):
    wait_time = between(1, 3)
    @task
    def test_it(self):
        pass
"""
    result = generator._apply_code_fixes(code, "CorrectName", "positive")
    assert isinstance(result, str)


def test_apply_code_fixes_unchanged(generator):
    """Returns code when no fixes needed (does not error)."""
    result = generator._apply_code_fixes(VALID_POSITIVE_CODE, "GetUsers", "positive")
    assert isinstance(result, str)
    assert len(result) > 0


def test_apply_code_fixes_handles_unicode(generator):
    """Sanitizes unicode characters."""
    code_with_unicode = VALID_POSITIVE_CODE.replace("Positive", "Pos\u200bitive")
    result = generator._apply_code_fixes(code_with_unicode, "GetUsers", "positive")
    assert "\u200b" not in result


# ===========================================================================
# 6. _run_semantic_check tests
# ===========================================================================


def test_run_semantic_check_valid_code(generator, endpoint):
    """Returns None for valid code."""
    with patch.object(generator._code_validator, "validate") as mock_validate:
        mock_result = Mock()
        mock_result.is_valid = True
        mock_validate.return_value = mock_result
        result = generator._run_semantic_check(
            VALID_POSITIVE_CODE, ScenarioType.POSITIVE, endpoint, [endpoint]
        )
        assert result is None


def test_run_semantic_check_invalid_code(generator, endpoint):
    """Returns result for code with semantic issues."""
    with patch.object(generator._code_validator, "validate") as mock_validate:
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.violations = [Mock(rule="F", message="hallucinated", line_number=5)]
        mock_result.error_message = "semantic error"
        mock_validate.return_value = mock_result
        result = generator._run_semantic_check(
            "bad code", ScenarioType.POSITIVE, endpoint, [endpoint]
        )
        assert result is not None
        assert result.is_valid is False


# ===========================================================================
# 7. _check_and_finalize_scenario tests
# ===========================================================================


@pytest.mark.asyncio
async def test_check_and_finalize_success(generator, endpoint):
    """Returns code string on success."""
    with (
        patch.object(generator, "_run_semantic_check", return_value=None),
        patch.object(
            generator,
            "_finalize_scenario_success",
            new_callable=AsyncMock,
            return_value=VALID_POSITIVE_CODE,
        ),
    ):
        result = await generator._check_and_finalize_scenario(
            VALID_POSITIVE_CODE,
            ScenarioType.POSITIVE,
            endpoint,
            [endpoint],
            "default",
            "get_users",
            "positive",
            "GET /users",
            0,
            2,
        )
        assert isinstance(result, str)
        assert "GetUsersPositiveWorkflow" in result


@pytest.mark.asyncio
async def test_check_and_finalize_semantic_failure(generator, endpoint):
    """Returns (error, True) on semantic failure."""
    mock_result = Mock()
    mock_result.is_valid = False
    mock_result.violations = [Mock(rule="F", message="bad", line_number=1)]
    mock_result.error_message = "semantic issue"

    with (
        patch.object(generator, "_run_semantic_check", return_value=mock_result),
        patch.object(generator, "_log_semantic_failure"),
    ):
        result = await generator._check_and_finalize_scenario(
            "code",
            ScenarioType.POSITIVE,
            endpoint,
            [endpoint],
            "default",
            "get_users",
            "positive",
            "GET /users",
            0,
            2,
        )
        assert isinstance(result, tuple)
        assert result[1] is True
        assert "semantic issue" in result[0]


# ===========================================================================
# 8. generate_endpoint_workflows tests
# ===========================================================================


@pytest.mark.asyncio
async def test_generate_endpoint_workflows_all_types(
    generator, mock_ai_client, endpoint
):
    """Generates all 3 scenario types."""
    responses = {
        ScenarioType.POSITIVE: VALID_POSITIVE_CODE,
        ScenarioType.NEGATIVE: VALID_NEGATIVE_CODE,
        ScenarioType.SECURITY: VALID_SECURITY_CODE,
    }

    async def mock_generate(scenario_type, ep, *args, **kwargs):
        return responses[scenario_type]

    with patch.object(generator, "_generate_llm_scenario", side_effect=mock_generate):
        result = await generator.generate_endpoint_workflows(
            endpoint=endpoint,
            base_workflow_content="# base",
            test_data_content="# test data",
        )
        assert isinstance(result, dict)
        assert ScenarioType.POSITIVE in result
        assert ScenarioType.NEGATIVE in result
        assert ScenarioType.SECURITY in result


@pytest.mark.asyncio
async def test_generate_endpoint_workflows_partial_failure(generator, endpoint):
    """Returns partial results when some scenarios fail."""

    async def mock_generate(scenario_type, ep, *args, **kwargs):
        if scenario_type == ScenarioType.SECURITY:
            raise CodeValidationError("security", "syntax error", "bad code")
        return VALID_POSITIVE_CODE

    with patch.object(generator, "_generate_llm_scenario", side_effect=mock_generate):
        result = await generator.generate_endpoint_workflows(
            endpoint=endpoint,
            base_workflow_content="# base",
            test_data_content="# test data",
        )
        assert ScenarioType.POSITIVE in result
        assert ScenarioType.NEGATIVE in result
        assert ScenarioType.SECURITY not in result


@pytest.mark.asyncio
async def test_generate_endpoint_workflows_all_fail(generator, endpoint):
    """Raises when all scenarios fail."""

    async def mock_generate(scenario_type, ep, *args, **kwargs):
        raise CodeValidationError(scenario_type.value, "err", "code")

    with patch.object(generator, "_generate_llm_scenario", side_effect=mock_generate):
        with pytest.raises(CodeValidationError):
            await generator.generate_endpoint_workflows(
                endpoint=endpoint,
                base_workflow_content="# base",
                test_data_content="# test data",
            )


# ===========================================================================
# 9. _generate_llm_scenario tests
# ===========================================================================


@pytest.mark.asyncio
async def test_generate_llm_scenario_success(generator, mock_ai_client, endpoint):
    """Returns generated code on success."""
    mock_ai_client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(f"<code>{VALID_POSITIVE_CODE}</code>")
    )
    with patch.object(generator, "_run_semantic_check", return_value=None):
        result = await generator._generate_llm_scenario(
            ScenarioType.POSITIVE,
            endpoint,
            "# base",
            "# test_data",
        )
        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_generate_llm_scenario_returns_none_on_skip(generator, endpoint):
    """Returns None when scenario is skipped (e.g. no injection points for security)."""
    with patch.object(generator, "_prepare_scenario_precomputation", return_value=None):
        result = await generator._generate_llm_scenario(
            ScenarioType.SECURITY,
            endpoint,
            "# base",
            "# test_data",
        )
        assert result is None


@pytest.mark.asyncio
async def test_generate_llm_scenario_raises_on_all_retries_fail(
    generator, mock_ai_client, endpoint
):
    """Raises CodeValidationError after all retries fail."""
    mock_ai_client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response("<code>def bad(:\n  pass</code>")
    )
    with pytest.raises(CodeValidationError):
        await generator._generate_llm_scenario(
            ScenarioType.POSITIVE,
            endpoint,
            "# base",
            "# test_data",
        )


# ===========================================================================
# 10. generate_tag_orchestrator tests
# ===========================================================================


@pytest.mark.asyncio
async def test_generate_tag_orchestrator_success(generator, mock_ai_client, endpoint):
    """Generates orchestrator code successfully."""
    mock_ai_client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(f"<code>{VALID_ORCHESTRATOR_CODE}</code>")
    )
    result = await generator.generate_tag_orchestrator(
        tag_name="users",
        tag_endpoints=[endpoint],
        base_workflow_content="# base",
        test_data_content="# test data",
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_tag_orchestrator_retries_on_syntax_error(
    generator, mock_ai_client, endpoint
):
    """Retries on syntax error then succeeds."""
    bad_resp = _make_llm_response("<code>def bad(:\n  pass</code>")
    good_resp = _make_llm_response(f"<code>{VALID_ORCHESTRATOR_CODE}</code>")
    mock_ai_client.chat.completions.create = AsyncMock(
        side_effect=[bad_resp, good_resp]
    )
    result = await generator.generate_tag_orchestrator(
        tag_name="users",
        tag_endpoints=[endpoint],
        base_workflow_content="# base",
        test_data_content="# test data",
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_generate_tag_orchestrator_raises_after_max_retries(
    generator, mock_ai_client, endpoint
):
    """Raises CodeValidationError after all retries fail."""
    bad_resp = _make_llm_response("<code>def bad(:\n  pass</code>")
    mock_ai_client.chat.completions.create = AsyncMock(return_value=bad_resp)
    with pytest.raises(CodeValidationError, match="orchestrator"):
        await generator.generate_tag_orchestrator(
            tag_name="users",
            tag_endpoints=[endpoint],
            base_workflow_content="# base",
            test_data_content="# test data",
        )


# ===========================================================================
# 11. _run_llm_retry_loop tests
# ===========================================================================


@pytest.mark.asyncio
async def test_run_llm_retry_loop_first_success(generator, mock_ai_client, endpoint):
    """Returns code on first success."""
    mock_ai_client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(f"<code>{VALID_POSITIVE_CODE}</code>")
    )
    with patch.object(generator, "_run_semantic_check", return_value=None):
        result = await generator._run_llm_retry_loop(
            ScenarioType.POSITIVE,
            endpoint,
            "GET /users",
            "positive",
            "GetUsers",
            "default",
            [endpoint],
            [200],
            "test prompt",
        )
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_llm_retry_loop_retries_on_syntax_error(
    generator, mock_ai_client, endpoint
):
    """Retries when first attempt has syntax error."""
    bad_resp = _make_llm_response("<code>def bad(:\n  pass</code>")
    good_resp = _make_llm_response(f"<code>{VALID_POSITIVE_CODE}</code>")
    mock_ai_client.chat.completions.create = AsyncMock(
        side_effect=[bad_resp, good_resp]
    )
    with patch.object(generator, "_run_semantic_check", return_value=None):
        result = await generator._run_llm_retry_loop(
            ScenarioType.POSITIVE,
            endpoint,
            "GET /users",
            "positive",
            "GetUsers",
            "default",
            [endpoint],
            [200],
            "test prompt",
        )
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_llm_retry_loop_raises_after_max_retries(
    generator, mock_ai_client, endpoint
):
    """Raises CodeValidationError after exhausting retries."""
    bad_resp = _make_llm_response("<code>def bad(:\n  pass</code>")
    mock_ai_client.chat.completions.create = AsyncMock(return_value=bad_resp)
    with pytest.raises(CodeValidationError):
        await generator._run_llm_retry_loop(
            ScenarioType.POSITIVE,
            endpoint,
            "GET /users",
            "positive",
            "GetUsers",
            "default",
            [endpoint],
            [200],
            "test prompt",
        )


@pytest.mark.asyncio
async def test_run_llm_retry_loop_semantic_failure_triggers_retry(
    generator, mock_ai_client, endpoint
):
    """Semantic failure triggers retry with semantic fix prompt."""
    mock_ai_client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(f"<code>{VALID_POSITIVE_CODE}</code>")
    )
    mock_sem_result = Mock()
    mock_sem_result.is_valid = False
    mock_sem_result.violations = [Mock(rule="F", message="hallucinated", line_number=5)]
    mock_sem_result.error_message = "semantic error"

    # First call: semantic fail, second call: success
    with patch.object(
        generator, "_run_semantic_check", side_effect=[mock_sem_result, None]
    ):
        result = await generator._run_llm_retry_loop(
            ScenarioType.POSITIVE,
            endpoint,
            "GET /users",
            "positive",
            "GetUsers",
            "default",
            [endpoint],
            [200],
            "test prompt",
        )
        assert isinstance(result, str)


# ===========================================================================
# 12. update_rate_limit tests
# ===========================================================================


def test_update_rate_limit_parses_headers(generator):
    """Parses rate limit headers correctly."""
    headers = {
        "x-ratelimit-limit": "10",
        "x-ratelimit-remaining": "8",
        "x-ratelimit-reset": "1.0",
    }
    info = generator.update_rate_limit(headers)
    assert isinstance(info, RateLimitInfo)
    assert info.requests_per_second == 10
    assert info.remaining == 8


def test_update_rate_limit_updates_concurrency(generator):
    """Updates concurrency based on rate limit."""
    headers = {
        "x-ratelimit-limit": "100",
        "x-ratelimit-remaining": "90",
        "x-ratelimit-reset": "1.0",
    }
    generator.update_rate_limit(headers)
    # Concurrency should be adjusted (may or may not change depending on values)
    assert generator.current_concurrency >= 2


def test_update_rate_limit_invalid_headers(generator):
    """Handles missing/invalid headers gracefully."""
    headers = {}
    info = generator.update_rate_limit(headers)
    assert isinstance(info, RateLimitInfo)


def test_update_rate_limit_concurrency_never_below_2(generator):
    """Concurrency never drops below 2."""
    headers = {
        "x-ratelimit-limit": "1",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "60.0",
    }
    generator.update_rate_limit(headers)
    assert generator.current_concurrency >= 2


# ===========================================================================
# Additional tests for edge cases
# ===========================================================================


@pytest.mark.asyncio
async def test_try_ai_call_updates_rate_limit_from_headers(generator, mock_ai_client):
    """Updates rate limit when response has headers."""
    resp = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = f"<code>{VALID_POSITIVE_CODE}</code>"
    mock_choice.message = mock_message
    resp.choices = [mock_choice]
    resp.headers = {
        "x-ratelimit-limit": "50",
        "x-ratelimit-remaining": "49",
        "x-ratelimit-reset": "1.0",
    }
    mock_ai_client.chat.completions.create = AsyncMock(return_value=resp)
    messages = [{"role": "user", "content": "test"}]
    result = await generator._try_ai_call(messages, "positive", 0)
    assert isinstance(result, str)
    assert generator._rate_limit_info is not None


def test_validate_llm_response_accepts_whitespace_content(generator, endpoint):
    """Does not raise on content that is only whitespace (truthy but empty-ish)."""
    # "   " is truthy so should not raise
    generator._validate_llm_response("   ", ScenarioType.POSITIVE, endpoint)


@pytest.mark.asyncio
async def test_call_ai_service_builds_correct_messages(generator, mock_ai_client):
    """Verifies that _call_ai_service passes system + user messages."""
    await generator._call_ai_service("my prompt", "test")
    call_args = mock_ai_client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "my prompt"


def test_get_rate_limit_info_default(generator):
    """Returns default rate limit when no headers have been seen."""
    info = generator.get_rate_limit_info()
    assert isinstance(info, RateLimitInfo)
    assert info.requests_per_minute == 60
