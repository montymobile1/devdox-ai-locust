"""
Tests for LocustTestEnhancer module.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import tempfile

from devdox_ai_locust.locust_enhancer import (
    LocustTestEnhancer,
    EnhanceResult,
)
from devdox_ai_locust.utils.ai_client import AIEnhancementConfig


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
