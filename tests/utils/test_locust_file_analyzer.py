"""
Tests for LocustFileAnalyzer module.
"""

import pytest
import tempfile
from pathlib import Path

from devdox_ai_locust.utils.locust_file_analyzer import (
    LocustFileAnalyzer,
    LocustFileAnalysis,
    UserClassInfo,
    TaskMethodInfo,
    ImportInfo,
)


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return LocustFileAnalyzer()


@pytest.fixture
def sample_locust_source():
    """Sample Locust test file source code."""
    return '''
import logging
from locust import HttpUser, task, between, events

logger = logging.getLogger(__name__)


def helper_function():
    """A module-level helper function."""
    return "helper"


class APIUser(HttpUser):
    """Test user class for API."""

    wait_time = between(1, 3)

    def on_start(self):
        """Login before starting."""
        self.client.post("/login", json={"user": "test"})

    @task(3)
    def get_users(self):
        """Get all users."""
        self.client.get("/users")

    @task
    def get_user_by_id(self):
        """Get a specific user."""
        self.client.get("/users/1")

    @task(2)
    def create_user(self):
        """Create a new user."""
        self.client.post("/users", json={"name": "new"})


class AdminUser(HttpUser):
    """Admin user class."""

    wait_time = between(2, 5)

    @task
    def admin_action(self):
        """Perform admin action."""
        self.client.delete("/admin/cleanup")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Event handler for test start."""
    print("Test starting!")


if __name__ == "__main__":
    print("Running locust")
'''


@pytest.fixture
def temp_locust_file(sample_locust_source):
    """Create a temporary Locust test file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(sample_locust_source)
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestLocustFileAnalyzer:
    """Tests for LocustFileAnalyzer."""

    def test_analyze_file(self, analyzer, temp_locust_file):
        """Test analyzing a file from disk."""
        result = analyzer.analyze(temp_locust_file)

        assert isinstance(result, LocustFileAnalysis)
        assert len(result.user_classes) == 2
        assert result.has_main_block is True

    def test_analyze_file_not_found(self, analyzer):
        """Test analyzing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/path/file.py")

    def test_analyze_source(self, analyzer, sample_locust_source):
        """Test analyzing source code directly."""
        result = analyzer.analyze_source(sample_locust_source)

        assert isinstance(result, LocustFileAnalysis)
        assert result.raw_source == sample_locust_source
        assert result.total_lines > 0

    def test_analyze_source_invalid_syntax(self, analyzer):
        """Test analyzing invalid Python syntax."""
        invalid_source = "def broken(\n  invalid syntax here"

        with pytest.raises(ValueError, match="Invalid Python syntax"):
            analyzer.analyze_source(invalid_source)


class TestClassExtraction:
    """Tests for class extraction functionality."""

    def test_extract_user_classes(self, analyzer, sample_locust_source):
        """Test extracting user classes."""
        result = analyzer.analyze_source(sample_locust_source)

        assert len(result.user_classes) == 2

        api_user = result.user_classes[0]
        assert api_user.name == "APIUser"
        assert "HttpUser" in api_user.parent_classes

        admin_user = result.user_classes[1]
        assert admin_user.name == "AdminUser"

    def test_extract_parent_classes(self, analyzer):
        """Test extracting parent classes."""
        source = '''
from locust import HttpUser, TaskSet

class MyTaskSet(TaskSet):
    pass

class MultiInherit(HttpUser, SomeMixin):
    pass
'''
        result = analyzer.analyze_source(source)

        assert len(result.user_classes) == 2
        assert "TaskSet" in result.user_classes[0].parent_classes
        assert "HttpUser" in result.user_classes[1].parent_classes
        assert "SomeMixin" in result.user_classes[1].parent_classes

    def test_class_line_numbers(self, analyzer, sample_locust_source):
        """Test class line number extraction."""
        result = analyzer.analyze_source(sample_locust_source)

        for cls in result.user_classes:
            assert cls.line_number > 0
            assert cls.end_line_number >= cls.line_number


class TestTaskMethodExtraction:
    """Tests for task method extraction."""

    def test_extract_task_methods(self, analyzer, sample_locust_source):
        """Test extracting @task decorated methods."""
        result = analyzer.analyze_source(sample_locust_source)

        api_user = result.user_classes[0]
        assert len(api_user.task_methods) == 3

        task_names = [t.name for t in api_user.task_methods]
        assert "get_users" in task_names
        assert "get_user_by_id" in task_names
        assert "create_user" in task_names

    def test_task_weight_extraction(self, analyzer, sample_locust_source):
        """Test extracting task weights."""
        result = analyzer.analyze_source(sample_locust_source)

        api_user = result.user_classes[0]
        weights = {t.name: t.weight for t in api_user.task_methods}

        assert weights["get_users"] == 3
        assert weights["get_user_by_id"] == 1  # Default weight
        assert weights["create_user"] == 2

    def test_task_weight_no_args(self, analyzer):
        """Test @task with no parentheses."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        pass
'''
        result = analyzer.analyze_source(source)
        assert result.user_classes[0].task_methods[0].weight == 1

    def test_task_weight_empty_parens(self, analyzer):
        """Test @task() with empty parentheses."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task()
    def my_task(self):
        pass
'''
        result = analyzer.analyze_source(source)
        assert result.user_classes[0].task_methods[0].weight == 1

    def test_other_methods_extraction(self, analyzer, sample_locust_source):
        """Test extracting non-task methods."""
        result = analyzer.analyze_source(sample_locust_source)

        api_user = result.user_classes[0]
        assert "on_start" in api_user.other_methods


class TestHttpCallDetection:
    """Tests for HTTP call detection in task methods."""

    def test_detect_get_call(self, analyzer):
        """Test detecting self.client.get() calls."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        self.client.get("/api/users")
'''
        result = analyzer.analyze_source(source)
        task = result.user_classes[0].task_methods[0]

        assert task.http_method == "GET"
        assert task.http_path == "/api/users"

    def test_detect_post_call(self, analyzer):
        """Test detecting self.client.post() calls."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        self.client.post("/api/users", json={"name": "test"})
'''
        result = analyzer.analyze_source(source)
        task = result.user_classes[0].task_methods[0]

        assert task.http_method == "POST"
        assert task.http_path == "/api/users"

    def test_detect_all_http_methods(self, analyzer):
        """Test detecting all HTTP methods."""
        methods = ["get", "post", "put", "patch", "delete", "head", "options"]

        for method in methods:
            source = f'''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        self.client.{method}("/test")
'''
            result = analyzer.analyze_source(source)
            task = result.user_classes[0].task_methods[0]
            assert task.http_method == method.upper()

    def test_detect_fstring_path(self, analyzer):
        """Test detecting f-string paths."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        user_id = 1
        self.client.get(f"/users/{user_id}")
'''
        result = analyzer.analyze_source(source)
        task = result.user_classes[0].task_methods[0]

        assert task.http_method == "GET"
        assert task.http_path is not None
        assert "user_id" in task.http_path

    def test_no_http_call(self, analyzer):
        """Test method without HTTP call."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        print("No HTTP call here")
'''
        result = analyzer.analyze_source(source)
        task = result.user_classes[0].task_methods[0]

        assert task.http_method is None
        assert task.http_path is None


class TestImportExtraction:
    """Tests for import statement extraction."""

    def test_extract_imports(self, analyzer, sample_locust_source):
        """Test extracting import statements."""
        result = analyzer.analyze_source(sample_locust_source)

        assert len(result.imports) >= 2

        statements = [imp.statement for imp in result.imports]
        assert any("logging" in s for s in statements)
        assert any("locust" in s for s in statements)

    def test_import_line_numbers(self, analyzer, sample_locust_source):
        """Test import line number extraction."""
        result = analyzer.analyze_source(sample_locust_source)

        for imp in result.imports:
            assert imp.line_number > 0

    def test_import_from_statement(self, analyzer):
        """Test 'from ... import ...' statements."""
        source = '''
from locust import HttpUser, task
from datetime import datetime
import json
'''
        result = analyzer.analyze_source(source)

        assert len(result.imports) == 3
        statements = [imp.statement for imp in result.imports]
        assert any("from locust import" in s for s in statements)
        assert any("import json" in s for s in statements)


class TestModuleLevelFunctions:
    """Tests for module-level function extraction."""

    def test_extract_module_functions(self, analyzer, sample_locust_source):
        """Test extracting module-level functions."""
        result = analyzer.analyze_source(sample_locust_source)

        assert "helper_function" in result.module_level_functions

    def test_event_handler_not_in_module_functions(self, analyzer, sample_locust_source):
        """Test that event handlers are also in module functions."""
        result = analyzer.analyze_source(sample_locust_source)

        # Event handlers are module-level functions too
        assert "on_test_start" in result.module_level_functions


class TestEventHandlers:
    """Tests for event handler detection."""

    def test_detect_event_handlers(self, analyzer, sample_locust_source):
        """Test detecting @events decorated functions."""
        result = analyzer.analyze_source(sample_locust_source)

        assert "on_test_start" in result.event_handlers

    def test_multiple_event_handlers(self, analyzer):
        """Test detecting multiple event handlers."""
        source = '''
from locust import events

@events.test_start.add_listener
def on_start(environment, **kwargs):
    pass

@events.test_stop.add_listener
def on_stop(environment, **kwargs):
    pass

@events.request.add_listener
def on_request(request_type, name, response_time, **kwargs):
    pass
'''
        result = analyzer.analyze_source(source)

        assert len(result.event_handlers) == 3
        assert "on_start" in result.event_handlers
        assert "on_stop" in result.event_handlers
        assert "on_request" in result.event_handlers


class TestMainBlockDetection:
    """Tests for __main__ block detection."""

    def test_detect_main_block(self, analyzer, sample_locust_source):
        """Test detecting if __name__ == '__main__' block."""
        result = analyzer.analyze_source(sample_locust_source)

        assert result.has_main_block is True
        assert result.main_block_line is not None
        assert result.main_block_line > 0

    def test_no_main_block(self, analyzer):
        """Test when no main block exists."""
        source = '''
from locust import HttpUser, task

class TestUser(HttpUser):
    @task
    def my_task(self):
        pass
'''
        result = analyzer.analyze_source(source)

        assert result.has_main_block is False
        assert result.main_block_line is None

    def test_reverse_main_check(self, analyzer):
        """Test reversed main check: '__main__' == __name__."""
        source = '''
from locust import HttpUser

class TestUser(HttpUser):
    pass

if "__main__" == __name__:
    print("Running")
'''
        result = analyzer.analyze_source(source)

        assert result.has_main_block is True


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_source(self, analyzer):
        """Test analyzing empty source."""
        result = analyzer.analyze_source("")

        assert result.user_classes == []
        assert result.imports == []
        assert result.has_main_block is False

    def test_no_classes(self, analyzer):
        """Test source with no classes."""
        source = '''
import logging

def my_function():
    pass

x = 1 + 2
'''
        result = analyzer.analyze_source(source)

        assert result.user_classes == []
        assert len(result.module_level_functions) == 1

    def test_class_without_task_methods(self, analyzer):
        """Test class without @task methods."""
        source = '''
from locust import HttpUser

class TestUser(HttpUser):
    def on_start(self):
        pass

    def helper(self):
        pass
'''
        result = analyzer.analyze_source(source)

        assert len(result.user_classes) == 1
        assert result.user_classes[0].task_methods == []
        assert len(result.user_classes[0].other_methods) == 2

    def test_nested_class(self, analyzer):
        """Test that nested classes are not extracted."""
        source = '''
from locust import HttpUser, task

class OuterUser(HttpUser):
    class InnerClass:
        pass

    @task
    def my_task(self):
        pass
'''
        result = analyzer.analyze_source(source)

        # Only the outer class should be extracted
        assert len(result.user_classes) == 1
        assert result.user_classes[0].name == "OuterUser"

    def test_decorated_class(self, analyzer):
        """Test class with decorators."""
        source = '''
from locust import HttpUser, task

@some_decorator
class DecoratedUser(HttpUser):
    @task
    def my_task(self):
        pass
'''
        result = analyzer.analyze_source(source)

        assert len(result.user_classes) == 1
        assert result.user_classes[0].name == "DecoratedUser"
