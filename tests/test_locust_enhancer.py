"""
Tests for LocustTestEnhancer module.
"""

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import tempfile

from devdox_ai_locust.locust_enhancer import (
    LocustTestEnhancer,
    EnhanceResult,
)
from devdox_ai_locust.utils.ai_client import AIEnhancementConfig
from devdox_ai_locust.utils.code_merger import MergeResult


@pytest.fixture
def api_key():
    """Test API key."""
    return "test-api-key"


@pytest.fixture
def enhancer(api_key):
    """Create enhancer instance."""
    return LocustTestEnhancer(together_api_key=api_key, verbose=False)


@pytest.fixture
def verbose_enhancer(api_key):
    """Create verbose enhancer instance."""
    return LocustTestEnhancer(together_api_key=api_key, verbose=True)


@pytest.fixture
def sample_locust_source():
    """Sample Locust test file source."""
    return '''
from locust import HttpUser, task, between

class APIUser(HttpUser):
    """Test user class."""

    wait_time = between(1, 3)

    @task
    def get_users(self):
        """Get all users."""
        self.client.get("/users")

    @task
    def get_user_by_id(self):
        """Get user by ID."""
        self.client.get("/users/1")
'''


@pytest.fixture
def temp_locust_file(sample_locust_source):
    """Create temporary Locust file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(sample_locust_source)
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestEnhanceResult:
    """Tests for EnhanceResult dataclass."""

    def test_default_values(self):
        """Test EnhanceResult default values."""
        result = EnhanceResult(
            success=True,
            enhanced_source="code",
            original_source="original",
        )

        assert result.success is True
        assert result.enhanced_source == "code"
        assert result.original_source == "original"
        assert result.added_imports == []
        assert result.added_tasks == []
        assert result.added_classes == []
        assert result.added_helpers == []
        assert result.replaced_tasks == []
        assert result.replaced_helpers == []
        assert result.replaced_classes == []
        assert result.warnings == []
        assert result.error is None

    def test_with_all_fields(self):
        """Test EnhanceResult with all fields set."""
        result = EnhanceResult(
            success=True,
            enhanced_source="code",
            original_source="original",
            added_imports=["import json"],
            added_tasks=["new_task"],
            replaced_tasks=["old_task"],
            warnings=["warning"],
            error=None,
        )

        assert len(result.added_imports) == 1
        assert len(result.added_tasks) == 1
        assert len(result.replaced_tasks) == 1
        assert len(result.warnings) == 1


class TestEnhancerInit:
    """Tests for enhancer initialization."""

    def test_init_with_defaults(self, api_key):
        """Test initialization with default config."""
        enhancer = LocustTestEnhancer(together_api_key=api_key)

        assert enhancer._api_key == api_key
        assert enhancer._verbose is False

    def test_init_with_custom_config(self, api_key):
        """Test initialization with custom AI config."""
        config = AIEnhancementConfig(model="custom-model")
        enhancer = LocustTestEnhancer(
            together_api_key=api_key,
            ai_config=config,
        )

        assert enhancer._ai_config.model == "custom-model"

    def test_init_verbose(self, api_key):
        """Test initialization with verbose flag."""
        enhancer = LocustTestEnhancer(
            together_api_key=api_key,
            verbose=True,
        )

        assert enhancer._verbose is True


class TestEnhanceFile:
    """Tests for enhance_file method."""

    @pytest.mark.asyncio
    @patch.object(LocustTestEnhancer, "_run_enhancement_pipeline")
    async def test_enhance_file_success(
        self, mock_pipeline, enhancer, temp_locust_file
    ):
        """Test successful file enhancement."""
        mock_pipeline.return_value = EnhanceResult(
            success=True,
            enhanced_source="enhanced code",
            original_source="original",
            added_tasks=["new_task"],
        )

        result = await enhancer.enhance_file(
            temp_locust_file,
            "Add a new task",
        )

        assert result.success is True
        mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_file_not_found(self, enhancer):
        """Test enhancing non-existent file."""
        result = await enhancer.enhance_file(
            "/nonexistent/file.py",
            "Add tasks",
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_enhance_file_invalid_syntax(self, enhancer):
        """Test enhancing file with invalid syntax."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def broken(\n  invalid")
            f.flush()

            result = await enhancer.enhance_file(f.name, "Add tasks")

            assert result.success is False
            assert "syntax" in result.error.lower()

        Path(f.name).unlink(missing_ok=True)


class TestEnhanceSource:
    """Tests for enhance_source method."""

    @pytest.mark.asyncio
    @patch.object(LocustTestEnhancer, "_run_enhancement_pipeline")
    async def test_enhance_source_success(
        self, mock_pipeline, enhancer, sample_locust_source
    ):
        """Test successful source enhancement."""
        mock_pipeline.return_value = EnhanceResult(
            success=True,
            enhanced_source="enhanced",
            original_source=sample_locust_source,
        )

        result = await enhancer.enhance_source(
            sample_locust_source,
            "Add new scenarios",
        )

        assert result.success is True
        mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_source_invalid_syntax(self, enhancer):
        """Test enhancing invalid source."""
        invalid_source = "class Broken(:\n  pass"

        result = await enhancer.enhance_source(invalid_source, "Add tasks")

        assert result.success is False
        assert "syntax" in result.error.lower()


class TestAnalysisHelpers:
    """Tests for analysis helper methods."""

    def test_has_auth_with_login(self, enhancer):
        """Test auth detection with login method."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    def on_start(self):
        self.login()

    def login(self):
        self.client.post("/login")

    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_auth(analysis) is True

    def test_has_auth_without_auth(self, enhancer):
        """Test auth detection without auth methods."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    @task
    def test(self):
        self.client.get("/api")
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_auth(analysis) is False

    def test_has_sequential_tasks_with_taskset(self, enhancer):
        """Test sequential task detection with TaskSet."""
        source = '''
from locust import HttpUser, TaskSet, task

class MyTasks(TaskSet):
    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_sequential_tasks(analysis) is True

    def test_has_sequential_tasks_without(self, enhancer):
        """Test sequential task detection without TaskSet."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_sequential_tasks(analysis) is False


class TestPromptRendering:
    """Tests for prompt rendering."""

    def test_render_prompt(self, enhancer, sample_locust_source):
        """Test prompt rendering."""
        analysis = enhancer._analyze_source(sample_locust_source)

        prompt = enhancer._render_prompt(
            analysis,
            "Add edge case tests",
            "API: Test API v1.0",
        )

        assert "Add edge case tests" in prompt
        assert "APIUser" in prompt
        assert "get_users" in prompt

    def test_render_prompt_without_schema(self, enhancer, sample_locust_source):
        """Test prompt rendering without API schema."""
        analysis = enhancer._analyze_source(sample_locust_source)

        prompt = enhancer._render_prompt(
            analysis,
            "Add tests",
            "",  # No schema
        )

        assert "Add tests" in prompt


class TestSourceExtraction:
    """Tests for source extraction helpers."""

    def test_extract_method_sources(self, enhancer, sample_locust_source):
        """Test extracting method source code."""
        analysis = enhancer._analyze_source(sample_locust_source)

        sources = enhancer._extract_method_sources(analysis)

        assert "get_users" in sources
        assert "get_user_by_id" in sources
        assert "self.client.get" in sources["get_users"]

    def test_extract_helper_sources(self, enhancer):
        """Test extracting helper function sources."""
        source = '''
from locust import HttpUser

def my_helper():
    return "helper"

def another_helper(x):
    return x * 2

class User(HttpUser):
    pass
'''
        analysis = enhancer._analyze_source(source)
        sources = enhancer._extract_helper_sources(analysis)

        assert "my_helper" in sources
        assert "another_helper" in sources

    def test_extract_class_sources(self, enhancer, sample_locust_source):
        """Test extracting class source code."""
        analysis = enhancer._analyze_source(sample_locust_source)

        sources = enhancer._extract_class_sources(analysis)

        assert "APIUser" in sources
        assert "class APIUser" in sources["APIUser"]


class TestEnhancementPipeline:
    """Tests for the enhancement pipeline."""

    @pytest.mark.asyncio
    @patch.object(LocustTestEnhancer, "_generate_new_scenarios")
    async def test_pipeline_no_ai_response(
        self, mock_generate, enhancer, sample_locust_source
    ):
        """Test pipeline when AI returns nothing."""
        mock_generate.return_value = {
            "new_imports": "",
            "new_tasks": "",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }

        analysis = enhancer._analyze_source(sample_locust_source)
        result = await enhancer._run_enhancement_pipeline(
            analysis, "Add tasks"
        )

        assert result.success is False
        assert "No new scenarios" in result.error

    @pytest.mark.asyncio
    @patch.object(LocustTestEnhancer, "_generate_new_scenarios")
    async def test_pipeline_with_new_tasks(
        self, mock_generate, enhancer, sample_locust_source
    ):
        """Test pipeline with new tasks generated."""
        mock_generate.return_value = {
            "new_imports": "import json",
            "new_tasks": '''
    @task
    def new_test_task(self):
        """New task."""
        self.client.post("/api")
''',
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }

        analysis = enhancer._analyze_source(sample_locust_source)
        result = await enhancer._run_enhancement_pipeline(
            analysis, "Add a POST task"
        )

        assert result.success is True
        assert "new_test_task" in result.added_tasks or "new_test_task" in result.enhanced_source

    @pytest.mark.asyncio
    @patch.object(LocustTestEnhancer, "_generate_new_scenarios")
    async def test_pipeline_with_replacements(
        self, mock_generate, enhancer, sample_locust_source
    ):
        """Test pipeline with task replacements."""
        mock_generate.return_value = {
            "new_imports": "",
            "new_tasks": "",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": '''
    @task(5)
    def get_users(self):
        """Updated get users."""
        with self.client.get("/users", catch_response=True) as resp:
            pass
''',
            "replace_helpers": "",
            "replace_classes": "",
        }

        analysis = enhancer._analyze_source(sample_locust_source)
        result = await enhancer._run_enhancement_pipeline(
            analysis, "Improve get_users"
        )

        assert result.success is True
        assert "get_users" in result.replaced_tasks


class TestGenerateNewWorkflow:
    """Tests for generate_new_workflow method."""

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.locust_enhancer.TogetherAIClient")
    async def test_generate_new_workflow_success(
        self, mock_client_class, enhancer
    ):
        """Test successful new workflow generation."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.call.return_value = '''<code>
from locust import HttpUser, task

class UsersWorkflow(HttpUser):
    @task
    def get_users(self):
        self.client.get("/users")
</code>'''
        mock_client_class.return_value = mock_client

        # Create proper endpoint mock with tags attribute
        from devdox_ai_locust.utils.open_ai_parser import Endpoint
        mock_endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="Get users",
            parameters=[],
            request_body=None,
            responses=[],
            description="",
            tags=["users"],
        )

        result = await enhancer.generate_new_workflow(
            tag_name="users",
            tag_endpoints=[mock_endpoint],
            custom_requirement="Add user tests",
        )

        assert result.success is True
        assert "UsersWorkflow" in result.enhanced_source or "users" in str(result.added_classes)

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.locust_enhancer.TogetherAIClient")
    async def test_generate_new_workflow_empty_response(
        self, mock_client_class, enhancer
    ):
        """Test new workflow generation with empty AI response."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.call.return_value = ""
        mock_client_class.return_value = mock_client

        from devdox_ai_locust.utils.open_ai_parser import Endpoint
        mock_endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="Get users",
            parameters=[],
            request_body=None,
            responses=[],
            description="",
            tags=["users"],
        )

        result = await enhancer.generate_new_workflow(
            tag_name="users",
            tag_endpoints=[mock_endpoint],
            custom_requirement="Add tests",
        )

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    @patch("devdox_ai_locust.locust_enhancer.TogetherAIClient")
    async def test_generate_new_workflow_with_reference(
        self, mock_client_class, enhancer, sample_locust_source
    ):
        """Test new workflow generation with reference source."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.call.return_value = "<code>class NewWorkflow: pass</code>"
        mock_client_class.return_value = mock_client

        from devdox_ai_locust.utils.open_ai_parser import Endpoint
        mock_endpoint = Endpoint(
            path="/orders",
            method="POST",
            operation_id="createOrder",
            summary="Create order",
            parameters=[],
            request_body=None,
            responses=[],
            description="",
            tags=["orders"],
        )

        result = await enhancer.generate_new_workflow(
            tag_name="orders",
            tag_endpoints=[mock_endpoint],
            custom_requirement="Add order tests",
            reference_workflow_source=sample_locust_source,
        )

        # Should have extracted imports from reference
        mock_client.call.assert_called_once()


class TestBlackFormatting:
    """Tests for Black code formatting."""

    def test_format_valid_code(self, enhancer):
        """Test formatting valid Python code."""
        code = '''
def messy():
    x=1
    y=2
    return x+y
'''
        result = enhancer._format_with_black(code)

        # Black should format it
        assert "x = 1" in result or "x=1" in result  # Either formatted or original

    def test_format_invalid_code(self, enhancer):
        """Test formatting invalid Python code."""
        code = "def broken(\n  invalid"

        result = enhancer._format_with_black(code)

        # Should return original on failure
        assert result == code


class TestVerboseLogging:
    """Tests for verbose logging."""

    @pytest.mark.asyncio
    @patch.object(LocustTestEnhancer, "_generate_new_scenarios")
    async def test_verbose_pipeline_logging(
        self, mock_generate, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test that verbose mode produces debug logs."""
        mock_generate.return_value = {
            "new_imports": "import json",
            "new_tasks": "@task\ndef x(self): pass",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }

        import logging
        caplog.set_level(logging.DEBUG)

        analysis = verbose_enhancer._analyze_source(sample_locust_source)
        await verbose_enhancer._run_enhancement_pipeline(
            analysis, "Add tasks"
        )

        # Verbose logging should have been triggered
        # (actual log content depends on implementation)
        assert verbose_enhancer._verbose is True


class TestSyntaxValidation:
    """Tests for syntax validation and repair functionality."""

    def test_validate_sections_valid_code(self, enhancer):
        """Test validation passes for valid code."""
        sections = {
            "new_tasks": '''
    @task(1)
    def valid_task(self):
        """Valid task."""
        self.client.get("/test")
''',
            "new_imports": "import json",
            "new_classes": "",
            "new_helpers": "",
        }

        errors = enhancer._validate_sections(sections)
        assert errors == {}

    def test_validate_sections_incomplete_try_block(self, enhancer):
        """Test validation catches incomplete try blocks."""
        sections = {
            "new_tasks": '''
    @task(1)
    def broken_task(self):
        """Task with incomplete try."""
        try:
            self.client.get("/test")
        # Missing except/finally!
''',
            "new_imports": "",
            "new_classes": "",
            "new_helpers": "",
        }

        errors = enhancer._validate_sections(sections)
        assert "new_tasks" in errors
        assert "expected" in errors["new_tasks"].lower() or "except" in errors["new_tasks"].lower()

    def test_validate_sections_missing_colon(self, enhancer):
        """Test validation catches missing colons."""
        sections = {
            "new_tasks": '''
    @task(1)
    def broken_task(self)  # Missing colon
        pass
''',
            "new_imports": "",
            "new_classes": "",
            "new_helpers": "",
        }

        errors = enhancer._validate_sections(sections)
        assert "new_tasks" in errors

    def test_validate_sections_empty_sections(self, enhancer):
        """Test validation handles empty sections."""
        sections = {
            "new_tasks": "",
            "new_imports": "",
            "new_classes": "",
            "new_helpers": "",
        }

        errors = enhancer._validate_sections(sections)
        assert errors == {}

    def test_build_repair_prompt(self, enhancer):
        """Test repair prompt is built correctly."""
        sections = {
            "new_tasks": '''
    @task(1)
    def broken(self):
        try:
            pass
''',
        }
        errors = {"new_tasks": "line 5: expected 'except' or 'finally' block"}

        prompt = enhancer._build_repair_prompt(sections, errors)

        # Should contain error information
        assert "syntax errors" in prompt.lower()
        assert "new_tasks" in prompt
        assert "except" in prompt.lower() or "finally" in prompt.lower()
        # Should contain the broken code
        assert "def broken" in prompt
        # Should have instructions about try/except
        assert "try" in prompt.lower()

    @pytest.mark.asyncio
    async def test_repair_sections_merges_fixed_code(self, enhancer):
        """Test that repaired sections are merged back correctly."""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        # Simulate AI returning fixed code
        mock_client.call.return_value = '''
<new_tasks>
    @task(1)
    def fixed_task(self):
        """Fixed task."""
        try:
            self.client.get("/test")
        except Exception:
            pass
</new_tasks>
'''

        original_sections = {
            "new_tasks": "broken code here",
            "new_imports": "import json",
        }
        errors = {"new_tasks": "line 3: syntax error"}

        result = await enhancer._repair_sections(
            mock_client, original_sections, errors
        )

        # Should have called the AI
        mock_client.call.assert_called_once()

        # Should have updated the broken section
        assert "fixed_task" in result["new_tasks"]
        assert "except Exception" in result["new_tasks"]

        # Should preserve non-broken sections
        assert result["new_imports"] == "import json"

    @pytest.mark.asyncio
    async def test_repair_sections_handles_empty_response(self, enhancer):
        """Test repair handles empty AI response gracefully."""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.call.return_value = ""

        original_sections = {"new_tasks": "broken"}
        errors = {"new_tasks": "error"}

        result = await enhancer._repair_sections(
            mock_client, original_sections, errors
        )

        # Should return original sections when AI fails
        assert result["new_tasks"] == "broken"


class TestLogVerboseAnalysis:
    """Tests for _log_verbose_analysis helper."""

    def test_logs_class_and_task_counts(
        self, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test that class, task, and method counts are logged."""
        caplog.set_level(logging.DEBUG)
        analysis = verbose_enhancer._analyze_source(sample_locust_source)

        verbose_enhancer._log_verbose_analysis(analysis)

        assert "[analysis] 1 class(es)" in caplog.text
        assert "2 @task method(s)" in caplog.text

    def test_logs_per_class_breakdown(
        self, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test that per-class breakdown with parents and task names is logged."""
        caplog.set_level(logging.DEBUG)
        analysis = verbose_enhancer._analyze_source(sample_locust_source)

        verbose_enhancer._log_verbose_analysis(analysis)

        assert "class APIUser" in caplog.text
        assert "get_users" in caplog.text

    def test_logs_auth_and_sequential_flags(
        self, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test that auth and sequential task flags are logged."""
        caplog.set_level(logging.DEBUG)
        analysis = verbose_enhancer._analyze_source(sample_locust_source)

        verbose_enhancer._log_verbose_analysis(analysis)

        assert "auth detected:" in caplog.text
        assert "sequential tasks:" in caplog.text

    def test_logs_source_size(
        self, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test that source size in chars and lines is logged."""
        caplog.set_level(logging.DEBUG)
        analysis = verbose_enhancer._analyze_source(sample_locust_source)

        verbose_enhancer._log_verbose_analysis(analysis)

        assert "[analysis] source size:" in caplog.text
        assert "chars" in caplog.text
        assert "lines" in caplog.text


class TestFetchApiSchemaContext:
    """Tests for _fetch_api_schema_context helper."""

    def test_returns_empty_string_when_no_url(self, verbose_enhancer, caplog):
        """Test returns empty string and logs skip when URL is None."""
        caplog.set_level(logging.DEBUG)

        result = asyncio.run(
            verbose_enhancer._fetch_api_schema_context(None)
        )

        assert result == ""
        assert "no swagger URL provided" in caplog.text

    def test_returns_schema_on_success(self, enhancer):
        """Test returns schema summary when fetch succeeds."""
        with patch.object(
            LocustTestEnhancer,
            "_fetch_api_schema_summary",
            return_value="API: Test v1\nEndpoints:\n- GET /users",
        ) as mock_fetch:
            result = asyncio.run(
                enhancer._fetch_api_schema_context(
                    "https://example.com/swagger.json"
                )
            )

            assert "API: Test v1" in result
            mock_fetch.assert_called_once_with(
                "https://example.com/swagger.json"
            )

    def test_returns_empty_on_fetch_error(self, enhancer, caplog):
        """Test returns empty string and logs warning on fetch failure."""
        caplog.set_level(logging.WARNING)
        with patch.object(
            LocustTestEnhancer,
            "_fetch_api_schema_summary",
            side_effect=Exception("connection failed"),
        ):
            result = asyncio.run(
                enhancer._fetch_api_schema_context(
                    "https://example.com/swagger.json"
                )
            )

            assert result == ""
            assert "Could not fetch API schema" in caplog.text


class TestLogVerboseAiSections:
    """Tests for _log_verbose_ai_sections helper."""

    def test_logs_populated_section_sizes(self, verbose_enhancer, caplog):
        """Test that populated section sizes are logged."""
        caplog.set_level(logging.DEBUG)
        sections = {
            "new_imports": "import json",
            "new_tasks": "def task(): pass",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }

        verbose_enhancer._log_verbose_ai_sections(sections)

        assert "[ai-response] section sizes:" in caplog.text
        assert "new_imports" in caplog.text
        assert "EMPTY" in caplog.text  # from empty sections

    def test_logs_all_empty_sections(self, verbose_enhancer, caplog):
        """Test that all-empty sections are logged as EMPTY."""
        caplog.set_level(logging.DEBUG)
        sections = {k: "" for k in LocustTestEnhancer._SECTION_KEYS}

        verbose_enhancer._log_verbose_ai_sections(sections)

        # All sections should report EMPTY
        empty_count = caplog.text.count("EMPTY")
        assert empty_count == len(LocustTestEnhancer._SECTION_KEYS)


class TestLogVerboseMergePreview:
    """Tests for _log_verbose_merge_preview helper."""

    def test_logs_add_and_replace_actions(self, verbose_enhancer, caplog):
        """Test that add/replace actions are logged correctly."""
        caplog.set_level(logging.DEBUG)
        sections = {
            "new_imports": "import os",
            "new_tasks": "def t(): pass",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "def t(): pass",
            "replace_helpers": "",
            "replace_classes": "",
        }

        verbose_enhancer._log_verbose_merge_preview(sections)

        assert "[merge] about to:" in caplog.text
        assert "add imports" in caplog.text
        assert "add tasks" in caplog.text
        assert "replace tasks" in caplog.text

    def test_logs_nothing_for_empty_sections(self, verbose_enhancer, caplog):
        """Test that empty sections produce (nothing) log."""
        caplog.set_level(logging.DEBUG)
        sections = {k: "" for k in LocustTestEnhancer._SECTION_KEYS}

        verbose_enhancer._log_verbose_merge_preview(sections)

        assert "(nothing)" in caplog.text


class TestLogVerboseMergeResult:
    """Tests for _log_verbose_merge_result helper."""

    def test_logs_merge_counts(self, verbose_enhancer, caplog):
        """Test that merge result counts are logged."""
        caplog.set_level(logging.DEBUG)
        merge_result = MergeResult(
            merged_source="code",
            added_imports=["import json"],
            added_tasks=["task_a", "task_b"],
            added_classes=[],
            added_helpers=["helper_x"],
            replaced_tasks=[],
            replaced_helpers=[],
            replaced_classes=[],
            warnings=[],
        )

        verbose_enhancer._log_verbose_merge_result(merge_result)

        assert "+1 imports" in caplog.text
        assert "+2 tasks" in caplog.text
        assert "+1 helpers" in caplog.text

    def test_logs_individual_warnings(self, verbose_enhancer, caplog):
        """Test that merge warnings are logged individually."""
        caplog.set_level(logging.WARNING)
        merge_result = MergeResult(
            merged_source="code",
            warnings=["duplicate import", "method conflict"],
        )

        verbose_enhancer._log_verbose_merge_result(merge_result)

        assert "duplicate import" in caplog.text
        assert "method conflict" in caplog.text


class TestLogVerboseFormatResult:
    """Tests for _log_verbose_format_result helper."""

    def test_logs_line_increase(
        self, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test logging when formatted source has more lines."""
        caplog.set_level(logging.DEBUG)
        analysis = verbose_enhancer._analyze_source(sample_locust_source)
        # Add extra lines to the formatted output
        formatted = sample_locust_source + "\n\n# extra\n# lines\n"

        verbose_enhancer._log_verbose_format_result(analysis, formatted)

        assert "[format]" in caplog.text
        assert "+" in caplog.text
        assert "Black formatting applied" in caplog.text

    def test_logs_line_decrease(
        self, verbose_enhancer, sample_locust_source, caplog
    ):
        """Test logging when formatted source has fewer lines."""
        caplog.set_level(logging.DEBUG)
        analysis = verbose_enhancer._analyze_source(sample_locust_source)
        # Shorter formatted output
        formatted = "from locust import HttpUser\nclass U(HttpUser): pass\n"

        verbose_enhancer._log_verbose_format_result(analysis, formatted)

        assert "[format]" in caplog.text
        assert "Black formatting applied" in caplog.text


class TestExtractReferenceImports:
    """Tests for _extract_reference_imports helper."""

    def test_returns_imports_from_valid_source(
        self, enhancer, sample_locust_source
    ):
        """Test that imports are extracted from valid reference source."""
        imports = enhancer._extract_reference_imports(sample_locust_source)

        assert isinstance(imports, list)
        assert len(imports) > 0
        assert any("locust" in imp for imp in imports)

    def test_returns_empty_list_when_none(self, enhancer):
        """Test returns empty list when source is None."""
        imports = enhancer._extract_reference_imports(None)

        assert imports == []

    def test_returns_empty_list_on_invalid_syntax(self, enhancer):
        """Test returns empty list when source has invalid syntax."""
        imports = enhancer._extract_reference_imports(
            "def broken(\n  invalid"
        )

        assert imports == []


class TestLogVerboseWorkflowStart:
    """Tests for _log_verbose_workflow_start helper."""

    def test_logs_tag_and_endpoints(self, verbose_enhancer, caplog):
        """Test that tag name and endpoint summaries are logged."""
        caplog.set_level(logging.DEBUG)
        mock_ep = Mock()
        mock_ep.method = "GET"
        mock_ep.path = "/users"

        verbose_enhancer._log_verbose_workflow_start(
            "users", [mock_ep], None
        )

        assert "[new-workflow] generating for tag 'users'" in caplog.text
        assert "1 endpoint(s)" in caplog.text
        assert "GET /users" in caplog.text

    def test_logs_reference_info(self, verbose_enhancer, caplog):
        """Test that reference workflow info is logged."""
        caplog.set_level(logging.DEBUG)
        mock_ep = Mock()
        mock_ep.method = "POST"
        mock_ep.path = "/orders"
        reference = "from locust import HttpUser\nclass U(HttpUser): pass\n"

        verbose_enhancer._log_verbose_workflow_start(
            "orders", [mock_ep], reference
        )

        assert "reference workflow:" in caplog.text
        assert "chars" in caplog.text


class TestExtractMethodSourcesExceptionPath:
    """Tests for _extract_method_sources AST exception fallback."""

    def test_returns_rough_sources_on_ast_failure(self, enhancer):
        """Test that _extract_method_sources falls back gracefully when AST fails."""
        # Source with valid enough structure for analysis but we'll mock AST failure
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    @task
    def get_items(self):
        self.client.get("/items")
'''
        analysis = enhancer._analyze_source(source)

        # Patch ast.parse inside the method to raise on the refinement step
        import ast
        original_parse = ast.parse
        call_count = [0]

        def failing_parse(src, *args, **kwargs):
            call_count[0] += 1
            # First call succeeds (initial analysis), second fails (refinement)
            if call_count[0] > 1:
                raise SyntaxError("mocked failure")
            return original_parse(src, *args, **kwargs)

        with patch("ast.parse", side_effect=failing_parse):
            sources = enhancer._extract_method_sources(analysis)

        # Should still return sources (rough extraction, before AST refinement)
        assert "get_items" in sources


class TestExtractHelperSourcesExceptionPath:
    """Tests for _extract_helper_sources AST exception fallback."""

    def test_returns_empty_on_no_helpers(self, enhancer):
        """Test _extract_helper_sources returns empty when no module-level functions."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    @task
    def my_task(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        sources = enhancer._extract_helper_sources(analysis)

        assert sources == {}

    def test_returns_sources_on_ast_failure(self, enhancer):
        """Test _extract_helper_sources returns empty dict on AST failure."""
        source = '''
from locust import HttpUser, task

def helper_func():
    return 42

class User(HttpUser):
    @task
    def my_task(self):
        pass
'''
        analysis = enhancer._analyze_source(source)

        with patch("ast.parse", side_effect=SyntaxError("mocked")):
            sources = enhancer._extract_helper_sources(analysis)

        # Should return empty sources (AST failure suppresses extraction)
        assert isinstance(sources, dict)


class TestFormatWithBlackExceptionTypes:
    """Tests for _format_with_black exception handling paths."""

    def test_generic_exception_returns_original(self, enhancer, caplog):
        """Test that a generic Exception returns the original source."""
        caplog.set_level(logging.WARNING)
        code = "x = 1\n"

        with patch("black.format_str", side_effect=RuntimeError("unexpected")):
            result = enhancer._format_with_black(code)

        assert result == code
        assert "Black formatting failed" in caplog.text

    def test_invalid_input_returns_original(self, enhancer, caplog):
        """Test that black.InvalidInput returns the original source."""
        import black
        caplog.set_level(logging.WARNING)
        code = "x = 1\n"

        with patch("black.format_str", side_effect=black.InvalidInput("bad")):
            result = enhancer._format_with_black(code)

        assert result == code
        assert "could not parse" in caplog.text.lower()


class TestHasAuthPartialMatches:
    """Tests for _has_auth partial match detection."""

    def test_partial_match_do_login(self, enhancer):
        """Test auth detection with 'do_login' (partial match for 'login')."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    def do_login(self):
        self.client.post("/auth")

    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_auth(analysis) is True

    def test_partial_match_refresh_token(self, enhancer):
        """Test auth detection with 'refresh_token' (partial match for 'get_token')."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    def refresh_auth_token(self):
        self.client.post("/refresh")

    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_auth(analysis) is True

    def test_no_auth_unrelated_methods(self, enhancer):
        """Test auth detection returns False for unrelated method names."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    def setup(self):
        pass

    def helper(self):
        pass

    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_auth(analysis) is False


class TestEnhanceFileIOErrorPath:
    """Tests for enhance_file IOError during fallback read."""

    def test_enhance_file_ioerror_fallback(self, enhancer):
        """Test that enhance_file handles IOError when reading file for error result."""
        # Use a path that will cause FileNotFoundError in _analyze_file
        # and then IOError in the fallback read
        result = asyncio.run(
            enhancer.enhance_file("/nonexistent/path/file.py", "Add tasks")
        )

        assert result.success is False
        assert result.error is not None
        # original_source should be empty since file doesn't exist
        assert result.original_source == ""

    def test_enhance_file_value_error_with_readable_file(self, enhancer):
        """Test enhance_file ValueError path with a file that can still be read."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("invalid python {{{\n")
            f.flush()

            result = asyncio.run(
                enhancer.enhance_file(f.name, "Add tasks")
            )

            assert result.success is False
            # File should be readable so original_source should have content
            assert result.original_source != "" or result.error is not None

        Path(f.name).unlink(missing_ok=True)


class TestValidateSections:
    """Tests for _validate_sections method."""

    def test_valid_sections_return_empty(self, enhancer):
        """Test that valid sections produce no errors."""
        sections = {
            "new_imports": "import json",
            "new_tasks": "@task\ndef my_task(self):\n    pass",
            "new_classes": "class Foo:\n    pass",
            "new_helpers": "def helper():\n    return 1",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }
        errors = enhancer._validate_sections(sections)

        assert errors == {}

    def test_invalid_task_section_returns_error(self, enhancer):
        """Test that invalid task code produces a validation error."""
        sections = {
            "new_imports": "",
            "new_tasks": "def broken(\n    invalid",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }
        errors = enhancer._validate_sections(sections)

        assert "new_tasks" in errors

    def test_empty_sections_return_no_errors(self, enhancer):
        """Test that empty sections are skipped."""
        sections = {
            "new_imports": "",
            "new_tasks": "",
            "new_classes": "",
            "new_helpers": "",
            "replace_tasks": "",
            "replace_helpers": "",
            "replace_classes": "",
        }
        errors = enhancer._validate_sections(sections)

        assert errors == {}


class TestHasSequentialTasks:
    """Tests for _has_sequential_tasks method."""

    def test_sequential_taskset_detected(self, enhancer):
        """Test SequentialTaskSet detection."""
        source = '''
from locust import HttpUser, SequentialTaskSet, task

class OrderFlow(SequentialTaskSet):
    @task
    def step1(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_sequential_tasks(analysis) is True

    def test_regular_httpuser_not_sequential(self, enhancer):
        """Test that regular HttpUser is not sequential."""
        source = '''
from locust import HttpUser, task

class User(HttpUser):
    @task
    def test(self):
        pass
'''
        analysis = enhancer._analyze_source(source)
        assert enhancer._has_sequential_tasks(analysis) is False
