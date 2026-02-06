"""
Tests for LocustCodeMerger module.
"""

import pytest

from devdox_ai_locust.utils.locust_file_analyzer import (
    LocustFileAnalyzer,
    LocustFileAnalysis,
)
from devdox_ai_locust.utils.code_merger import (
    LocustCodeMerger,
    MergeResult,
)


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return LocustFileAnalyzer()


@pytest.fixture
def sample_source():
    """Sample Locust file source for testing."""
    return '''import logging
from locust import HttpUser, task, between

logger = logging.getLogger(__name__)


def existing_helper():
    """An existing helper function."""
    return "helper"


class APIUser(HttpUser):
    """Test user class."""

    wait_time = between(1, 3)

    @task(3)
    def get_users(self):
        """Get all users."""
        self.client.get("/users")

    @task
    def get_user_by_id(self):
        """Get a specific user."""
        self.client.get("/users/1")


if __name__ == "__main__":
    print("Running")
'''


@pytest.fixture
def analysis(analyzer, sample_source):
    """Create analysis from sample source."""
    return analyzer.analyze_source(sample_source)


@pytest.fixture
def merger(analysis):
    """Create merger instance."""
    return LocustCodeMerger(analysis)


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_merge_result_defaults(self):
        """Test MergeResult default values."""
        result = MergeResult(merged_source="test")

        assert result.merged_source == "test"
        assert result.added_imports == []
        assert result.added_tasks == []
        assert result.added_classes == []
        assert result.added_helpers == []
        assert result.replaced_tasks == []
        assert result.replaced_helpers == []
        assert result.replaced_classes == []
        assert result.warnings == []


class TestImportMerging:
    """Tests for import statement merging."""

    def test_merge_new_imports(self, merger):
        """Test merging new import statements."""
        new_imports = '''
import json
from datetime import datetime
'''
        result = merger.merge(new_imports=new_imports)

        assert "import json" in result.merged_source
        assert "from datetime import datetime" in result.merged_source
        assert "json" in result.added_imports[0] or "datetime" in result.added_imports[0]
        assert len(result.added_imports) == 2

    def test_no_duplicate_imports(self, merger):
        """Test that duplicate imports are not added."""
        new_imports = '''
import logging
from locust import HttpUser
'''
        result = merger.merge(new_imports=new_imports)

        # Count occurrences of 'import logging'
        count = result.merged_source.count("import logging")
        assert count == 1

    def test_import_normalization(self, merger):
        """Test that imports are normalized for comparison."""
        # Exact same import should be detected as duplicate
        new_imports = "from locust import HttpUser, task, between"
        result = merger.merge(new_imports=new_imports)

        # Should not add duplicate (normalized comparison)
        assert len(result.added_imports) == 0

    def test_empty_imports(self, merger):
        """Test with empty imports string."""
        result = merger.merge(new_imports="")

        assert result.added_imports == []


class TestTaskMerging:
    """Tests for @task method merging."""

    def test_merge_new_tasks(self, merger):
        """Test merging new task methods."""
        new_tasks = '''
    @task(2)
    def create_user(self):
        """Create a new user."""
        self.client.post("/users", json={"name": "test"})
'''
        result = merger.merge(new_tasks=new_tasks)

        assert "create_user" in result.merged_source
        assert "create_user" in result.added_tasks

    def test_no_duplicate_tasks(self, merger):
        """Test that duplicate task names are not added."""
        new_tasks = '''
    @task
    def get_users(self):
        """Duplicate task."""
        pass
'''
        result = merger.merge(new_tasks=new_tasks)

        # Should not add duplicate
        assert "get_users" not in result.added_tasks

    def test_task_indentation(self, merger):
        """Test that tasks are properly indented."""
        new_tasks = '''
@task
def new_task(self):
    """New task."""
    self.client.get("/new")
'''
        result = merger.merge(new_tasks=new_tasks)

        # Find the new task in the merged source
        lines = result.merged_source.split("\n")
        for i, line in enumerate(lines):
            if "def new_task" in line:
                # Check it has proper indentation (4 spaces for class method)
                assert line.startswith("    ")
                break

    def test_multiple_new_tasks(self, merger):
        """Test merging multiple new tasks."""
        new_tasks = '''
    @task
    def task_one(self):
        pass

    @task(5)
    def task_two(self):
        pass
'''
        result = merger.merge(new_tasks=new_tasks)

        assert "task_one" in result.added_tasks
        assert "task_two" in result.added_tasks
        assert len(result.added_tasks) == 2

    def test_nested_task_indentation_normalization(self, merger):
        """Test that nested/inconsistent @task indentation is normalized.

        The AI sometimes outputs multiple @task methods where later methods
        are incorrectly indented inside the body of earlier methods (e.g.,
        inside a try/except block). This test verifies that the merger
        normalizes all @task decorators to the same indentation level.
        """
        # Simulates AI output where second task is nested inside first task's body
        new_tasks = '''
    @task(1)
    def first_task(self):
        """First task."""
        try:
            self.client.get("/first")
        except Exception:
            pass

        @task(2)
        def second_task(self):
            """Second task - incorrectly indented by AI."""
            self.client.get("/second")

            @task(3)
            def third_task(self):
                """Third task - even more nested."""
                self.client.get("/third")
'''
        result = merger.merge(new_tasks=new_tasks)

        # All three tasks should be added
        assert "first_task" in result.added_tasks
        assert "second_task" in result.added_tasks
        assert "third_task" in result.added_tasks

        # Verify the merged code has valid Python syntax
        import ast
        try:
            ast.parse(result.merged_source)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        assert syntax_valid, "Merged code should have valid Python syntax"

        # Verify all @task decorators have consistent indentation
        lines = result.merged_source.split("\n")
        task_indents = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("@task"):
                indent = len(line) - len(stripped)
                task_indents.append(indent)

        # All @task decorators should have the same indentation (4 spaces for class methods)
        assert len(set(task_indents)) == 1, (
            f"All @task decorators should have same indentation, got: {task_indents}"
        )
        assert task_indents[0] == 4, (
            f"@task decorators should have 4-space indent, got: {task_indents[0]}"
        )

    def test_misaligned_decorator_and_def(self, merger):
        """Test that misaligned decorator and def are fixed.

        The AI sometimes outputs code where the @task decorator is at one
        indentation level but the 'def' line has extra indentation.
        """
        new_tasks = '''
@task(1)
    def misaligned_task(self):
        """Task with misaligned def."""
        self.client.get("/test")
'''
        result = merger.merge(new_tasks=new_tasks)

        assert "misaligned_task" in result.added_tasks

        # Verify syntax is valid
        import ast
        try:
            ast.parse(result.merged_source)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        assert syntax_valid, "Merged code should have valid Python syntax"


class TestClassMerging:
    """Tests for class definition merging."""

    def test_merge_new_class(self, merger):
        """Test merging a new class definition."""
        new_classes = '''
class AdminUser(HttpUser):
    """Admin user class."""

    wait_time = between(2, 5)

    @task
    def admin_task(self):
        pass
'''
        result = merger.merge(new_classes=new_classes)

        assert "class AdminUser" in result.merged_source
        assert "AdminUser" in result.added_classes

    def test_class_before_main_block(self, merger, sample_source):
        """Test that new classes are added before __main__ block."""
        new_classes = '''
class NewUser(HttpUser):
    pass
'''
        result = merger.merge(new_classes=new_classes)

        # Find positions
        class_pos = result.merged_source.find("class NewUser")
        main_pos = result.merged_source.find('if __name__')

        assert class_pos < main_pos

    def test_multiple_new_classes(self, merger):
        """Test merging multiple new classes."""
        new_classes = '''
class UserOne(HttpUser):
    pass


class UserTwo(HttpUser):
    pass
'''
        result = merger.merge(new_classes=new_classes)

        assert "UserOne" in result.added_classes
        assert "UserTwo" in result.added_classes


class TestHelperMerging:
    """Tests for helper function merging."""

    def test_merge_new_helper(self, merger):
        """Test merging a new helper function."""
        new_helpers = '''
def new_helper():
    """A new helper function."""
    return "new"
'''
        result = merger.merge(new_helpers=new_helpers)

        assert "def new_helper" in result.merged_source
        assert "new_helper" in result.added_helpers

    def test_helper_position(self, merger):
        """Test that helpers are added before classes."""
        new_helpers = '''
def my_helper():
    pass
'''
        result = merger.merge(new_helpers=new_helpers)

        helper_pos = result.merged_source.find("def my_helper")
        class_pos = result.merged_source.find("class APIUser")

        assert helper_pos < class_pos


class TestTaskReplacement:
    """Tests for task method replacement."""

    def test_replace_task(self, merger):
        """Test replacing an existing task method."""
        replace_tasks = '''
    @task(10)
    def get_users(self):
        """Updated get users task."""
        with self.client.get("/users", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("Failed")
'''
        result = merger.merge(replace_tasks=replace_tasks)

        assert "get_users" in result.replaced_tasks
        assert "@task(10)" in result.merged_source
        assert "Updated get users task" in result.merged_source

    def test_replace_nonexistent_task(self, merger):
        """Test replacing a task that doesn't exist."""
        replace_tasks = '''
    @task
    def nonexistent_task(self):
        pass
'''
        result = merger.merge(replace_tasks=replace_tasks)

        assert "nonexistent_task" not in result.replaced_tasks
        assert len(result.warnings) > 0

    def test_replace_multiple_tasks(self, merger):
        """Test replacing multiple tasks."""
        replace_tasks = '''
    @task(5)
    def get_users(self):
        """Updated."""
        pass

    @task(7)
    def get_user_by_id(self):
        """Also updated."""
        pass
'''
        result = merger.merge(replace_tasks=replace_tasks)

        assert "get_users" in result.replaced_tasks
        assert "get_user_by_id" in result.replaced_tasks


class TestHelperReplacement:
    """Tests for helper function replacement."""

    def test_replace_helper(self, merger):
        """Test replacing an existing helper function."""
        replace_helpers = '''
def existing_helper():
    """Updated helper function."""
    return "updated helper"
'''
        result = merger.merge(replace_helpers=replace_helpers)

        assert "existing_helper" in result.replaced_helpers
        assert "Updated helper function" in result.merged_source
        assert "updated helper" in result.merged_source

    def test_replace_nonexistent_helper(self, merger):
        """Test replacing a helper that doesn't exist."""
        replace_helpers = '''
def nonexistent_helper():
    pass
'''
        result = merger.merge(replace_helpers=replace_helpers)

        assert "nonexistent_helper" not in result.replaced_helpers
        assert len(result.warnings) > 0


class TestClassReplacement:
    """Tests for class replacement."""

    def test_replace_class(self, merger):
        """Test replacing an existing class."""
        replace_classes = '''
class APIUser(HttpUser):
    """Completely replaced APIUser class."""

    wait_time = between(0.5, 1.5)

    @task(10)
    def new_implementation(self):
        """New implementation."""
        self.client.get("/v2/users")
'''
        result = merger.merge(replace_classes=replace_classes)

        assert "APIUser" in result.replaced_classes
        assert "Completely replaced APIUser class" in result.merged_source
        assert "new_implementation" in result.merged_source
        # Old methods should be gone
        assert result.merged_source.count("def get_users") == 0

    def test_replace_nonexistent_class(self, merger):
        """Test replacing a class that doesn't exist."""
        replace_classes = '''
class NonexistentClass(HttpUser):
    pass
'''
        result = merger.merge(replace_classes=replace_classes)

        assert "NonexistentClass" not in result.replaced_classes
        assert len(result.warnings) > 0


class TestCombinedOperations:
    """Tests for combining multiple operations."""

    def test_combined_add_and_replace(self, merger):
        """Test combining add and replace operations."""
        result = merger.merge(
            new_imports="import json",
            new_tasks='''
    @task
    def new_task(self):
        pass
''',
            replace_tasks='''
    @task(5)
    def get_users(self):
        """Replaced."""
        pass
''',
        )

        assert "json" in result.added_imports[0]
        assert "new_task" in result.added_tasks
        assert "get_users" in result.replaced_tasks

    def test_operation_order(self, merger):
        """Test that replacements happen before insertions."""
        # This tests that line numbers don't drift due to order issues
        result = merger.merge(
            new_tasks='''
    @task
    def brand_new_task(self):
        pass
''',
            replace_tasks='''
    @task(99)
    def get_users(self):
        """Weight 99."""
        pass
''',
        )

        # Both operations should succeed without corrupting the file
        assert "@task(99)" in result.merged_source
        assert "brand_new_task" in result.merged_source
        assert result.warnings == [] or all("syntax" not in w.lower() for w in result.warnings)


class TestCodeValidation:
    """Tests for merged code validation."""

    def test_valid_merged_code(self, merger):
        """Test that merged code is syntactically valid."""
        result = merger.merge(
            new_imports="import json",
            new_tasks='''
    @task
    def valid_task(self):
        self.client.get("/test")
''',
        )

        # Should have no syntax error warnings
        syntax_warnings = [w for w in result.warnings if "syntax" in w.lower()]
        assert len(syntax_warnings) == 0

    def test_syntax_validation_warning(self, analyzer):
        """Test that syntax errors produce warnings."""
        # Create a source that will produce invalid code after merge
        broken_source = '''
class Broken(HttpUser:  # Missing closing paren
    pass
'''
        try:
            analysis = analyzer.analyze_source(broken_source)
            merger = LocustCodeMerger(analysis)
            result = merger.merge()
            # If we get here, check for warnings
            assert len(result.warnings) > 0
        except ValueError:
            # Expected - source is invalid
            pass


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_merge(self, merger):
        """Test merge with no new content."""
        result = merger.merge()

        assert result.added_imports == []
        assert result.added_tasks == []
        assert result.added_classes == []
        assert result.added_helpers == []

    def test_whitespace_only_content(self, merger):
        """Test merge with whitespace-only content."""
        result = merger.merge(
            new_imports="   \n\t  ",
            new_tasks="  \n  ",
        )

        assert result.added_imports == []
        assert result.added_tasks == []

    def test_source_without_classes(self, analyzer):
        """Test merging into source without user classes."""
        source = '''
import logging

def helper():
    pass
'''
        analysis = analyzer.analyze_source(source)
        merger = LocustCodeMerger(analysis)

        new_tasks = '''
@task
def my_task(self):
    pass
'''
        result = merger.merge(new_tasks=new_tasks)

        # Should warn about no classes
        assert result.added_tasks == []

    def test_preserve_trailing_newline(self, merger):
        """Test that trailing newlines are preserved."""
        result = merger.merge(new_imports="import json")

        assert result.merged_source.endswith("\n")

    def test_indentation_detection(self, analyzer):
        """Test indentation detection with tabs."""
        source = '''
from locust import HttpUser, task

class TabUser(HttpUser):
\t@task
\tdef tabbed_task(self):
\t\tpass
'''
        analysis = analyzer.analyze_source(source)
        merger = LocustCodeMerger(analysis)

        new_tasks = '''
@task
def new_task(self):
    pass
'''
        result = merger.merge(new_tasks=new_tasks)

        # New task should be added (indentation should be handled)
        assert "new_task" in result.added_tasks


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_merge_without_replace_params(self, merger):
        """Test that merge works without replace parameters (old API)."""
        result = merger.merge(
            new_imports="import json",
            new_tasks='''
    @task
    def new_task(self):
        pass
''',
            new_classes="",
            new_helpers="",
        )

        assert len(result.added_imports) == 1
        assert "new_task" in result.added_tasks
        # Replace lists should be empty
        assert result.replaced_tasks == []
        assert result.replaced_helpers == []
        assert result.replaced_classes == []
