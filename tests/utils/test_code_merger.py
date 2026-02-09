"""
Tests for LocustCodeMerger module.
"""

import logging
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


class TestLogSectionSizes:
    """Tests for _log_section_sizes helper."""

    def test_logs_populated_section_sizes(self, merger, caplog):
        """Test that populated section sizes are logged correctly."""
        caplog.set_level(logging.DEBUG)

        merger._log_section_sizes(
            new_imports="import json",
            new_tasks="def task(): pass",
            new_classes="",
        )

        assert "merge() called" in caplog.text
        assert "new_imports=11" in caplog.text
        assert "new_tasks=16" in caplog.text
        assert "new_classes=0" in caplog.text

    def test_handles_empty_and_none_sections(self, merger, caplog):
        """Test that empty strings produce size 0."""
        caplog.set_level(logging.DEBUG)

        merger._log_section_sizes(
            new_imports="", new_tasks="", new_classes="",
        )

        assert "new_imports=0" in caplog.text
        assert "new_tasks=0" in caplog.text


class TestValidateAndLogResult:
    """Tests for _validate_and_log_result helper."""

    def test_appends_warning_on_invalid_code(self, merger, caplog):
        """Test that invalid code appends a warning."""
        caplog.set_level(logging.WARNING)
        warnings = []

        merger._validate_and_log_result("def broken(\n  invalid", warnings)

        assert len(warnings) == 1
        assert "syntax errors" in warnings[0].lower()
        assert "validation failed" in caplog.text

    def test_logs_success_on_valid_code(self, merger, caplog):
        """Test that valid code logs success without warnings."""
        caplog.set_level(logging.DEBUG)
        warnings = []

        merger._validate_and_log_result("x = 1\n", warnings)

        assert len(warnings) == 0
        assert "passed syntax validation" in caplog.text


class TestCollectDecorators:
    """Tests for _collect_decorators helper."""

    def test_collects_single_decorator(self):
        """Test collecting a single decorator line."""
        lines = ["    @task\n", "    def foo(self):\n", "        pass\n"]
        decorators, j = LocustCodeMerger._collect_decorators(lines, 0)

        assert decorators == ["    @task\n"]
        assert j == 1

    def test_collects_multiple_decorators_with_gaps(self):
        """Test collecting multiple decorators with empty lines."""
        lines = [
            "@task(1)\n",
            "\n",
            "@other\n",
            "def foo(self):\n",
            "    pass\n",
        ]
        decorators, j = LocustCodeMerger._collect_decorators(lines, 0)

        assert len(decorators) == 3
        assert decorators[0] == "@task(1)\n"
        assert decorators[2] == "@other\n"
        assert j == 3

    def test_stops_at_non_decorator(self):
        """Test that collection stops at non-decorator, non-empty line."""
        lines = ["@task\n", "def foo(self):\n"]
        decorators, j = LocustCodeMerger._collect_decorators(lines, 0)

        assert decorators == ["@task\n"]
        assert j == 1


class TestFixMisalignedDef:
    """Tests for _fix_misaligned_def helper."""

    def test_fixes_misaligned_def(self):
        """Test that a misaligned def is realigned."""
        lines = [
            "        def foo(self):\n",
            "            pass\n",
        ]
        result = LocustCodeMerger._fix_misaligned_def(lines, 0, deco_indent=4)

        assert result is not None
        fixed_lines, next_idx = result
        assert fixed_lines[0] == "    def foo(self):\n"
        assert next_idx == 2

    def test_returns_none_when_aligned(self):
        """Test returns None when def indent matches decorator indent."""
        lines = ["    def foo(self):\n", "        pass\n"]
        result = LocustCodeMerger._fix_misaligned_def(lines, 0, deco_indent=4)

        assert result is None

    def test_returns_none_for_non_def(self):
        """Test returns None when line is not a def."""
        lines = ["    x = 1\n"]
        result = LocustCodeMerger._fix_misaligned_def(lines, 0, deco_indent=0)

        assert result is None

    def test_fixes_async_def(self):
        """Test that a misaligned async def is also fixed."""
        lines = [
            "        async def bar(self):\n",
            "            await something()\n",
        ]
        result = LocustCodeMerger._fix_misaligned_def(lines, 0, deco_indent=4)

        assert result is not None
        fixed_lines, _ = result
        assert fixed_lines[0] == "    async def bar(self):\n"


class TestSyntaxErrorFallbacks:
    """Tests for AST SyntaxError fallback paths in code_merger."""

    def test_find_method_in_class_syntax_error(self, merger):
        """Test _find_method_in_class returns (None, None) on SyntaxError."""
        broken_lines = ["class Foo(\n", "    def bar(self):\n", "        pass\n"]
        result = merger._find_method_in_class(broken_lines, 0, 2, "bar")

        assert result == (None, None)

    def test_find_module_function_boundaries_syntax_error(self, merger):
        """Test _find_module_function_boundaries returns (None, None) on SyntaxError."""
        broken_lines = ["def broken(\n", "    pass\n"]
        result = merger._find_module_function_boundaries(broken_lines, "broken")

        assert result == (None, None)

    def test_find_class_boundaries_syntax_error(self, merger):
        """Test _find_class_boundaries returns (None, None) on SyntaxError."""
        broken_lines = ["class Broken(\n", "    pass\n"]
        result = merger._find_class_boundaries(broken_lines, "Broken")

        assert result == (None, None)

    def test_split_code_into_functions_regex_fallback(self, merger):
        """Test _split_code_into_functions falls back to regex on SyntaxError."""
        # Code that is invalid Python but has recognizable def statements
        broken_code = "def func_a():\n    pass\n\ndef func_b():\n    pass\n    unterminated = '\n"
        segments = merger._split_code_into_functions(broken_code)

        names = [name for name, _ in segments]
        assert "func_a" in names
        assert "func_b" in names

    def test_split_code_into_classes_regex_fallback(self, merger):
        """Test _split_code_into_classes falls back to regex on SyntaxError."""
        broken_code = "class Foo:\n    pass\n\nclass Bar:\n    pass\n    unterminated = '\n"
        segments = merger._split_code_into_classes(broken_code)

        names = [name for name, _ in segments]
        assert "Foo" in names
        assert "Bar" in names

    def test_filter_methods_syntax_error(self, merger):
        """Test _filter_methods returns full dedented code on SyntaxError."""
        broken_code = "    def foo():\n        pass\n    unterminated = '\n"
        result = merger._filter_methods(broken_code, ["foo"])

        # Should return the dedented full code block
        assert "def foo" in result

    def test_extract_names_from_code_regex_fallback_functions(self, merger):
        """Test _extract_names_from_code falls back to regex for functions."""
        broken_code = "def helper_a():\n    pass\n    unterminated = '\n"
        names = merger._extract_names_from_code(broken_code, "function")

        assert "helper_a" in names

    def test_extract_names_from_code_regex_fallback_classes(self, merger):
        """Test _extract_names_from_code falls back to regex for classes."""
        broken_code = "class MyClass:\n    pass\n    unterminated = '\n"
        names = merger._extract_names_from_code(broken_code, "class")

        assert "MyClass" in names


class TestEdgeCasePaths:
    """Tests for edge case paths in code_merger that improve coverage."""

    def test_detect_indentation_default(self, merger):
        """Test _detect_indentation returns default 4-space when no def/@ found."""
        lines = ["class Foo:\n", "    x = 1\n", "    y = 2\n"]
        # x = 1 doesn't start with 'def ' or '@', so default is returned
        # Actually 'x = 1' line doesn't match def or @, let's use lines with no methods
        lines_no_methods = ["class Foo:\n", "\n", "\n"]
        result = merger._detect_indentation(lines_no_methods, 0, 2)

        assert result == "    "

    def test_detect_indentation_finds_decorator(self, merger):
        """Test _detect_indentation detects indent from @ decorator."""
        lines = [
            "class Foo:\n",
            "        @task\n",
            "        def bar(self):\n",
            "            pass\n",
        ]
        result = merger._detect_indentation(lines, 0, 3)

        assert result == "        "

    def test_reindent_no_indent_unit(self, merger):
        """Test _reindent when source code has no indentation (empty indent unit)."""
        code = "x = 1\ny = 2\n"
        result = merger._reindent(code, "    ")

        # All lines should get target_indent prepended
        for line in result.splitlines():
            if line.strip():
                assert line.startswith("    ")

    def test_pick_target_class_fallback(self, analyzer):
        """Test _pick_target_class falls back to first class when none has tasks."""
        source = '''from locust import HttpUser, task, between

class EmptyUser(HttpUser):
    wait_time = between(1, 2)

class AnotherUser(HttpUser):
    wait_time = between(1, 2)
'''
        analysis = analyzer.analyze_source(source)
        merger = LocustCodeMerger(analysis)

        target = merger._pick_target_class()
        # Should fall back to first user class
        assert target.name == "EmptyUser"

    def test_find_class_node_at_no_match(self):
        """Test _find_class_node_at returns None when no class at that line."""
        import ast
        source = "class Foo:\n    pass\n"
        tree = ast.parse(source)

        result = LocustCodeMerger._find_class_node_at(tree, 999)
        assert result is None

    def test_node_start_line_without_decorators(self):
        """Test _node_start_line for a node without decorators."""
        import ast
        source = "def foo():\n    pass\n"
        tree = ast.parse(source)
        func_node = tree.body[0]

        result = LocustCodeMerger._node_start_line(func_node)
        assert result == 0

    def test_node_start_line_with_decorator(self):
        """Test _node_start_line accounts for decorators."""
        import ast
        source = "@my_decorator\ndef foo():\n    pass\n"
        tree = ast.parse(source)
        func_node = tree.body[0]

        result = LocustCodeMerger._node_start_line(func_node)
        assert result == 0  # Decorator is on line 1, so 0-based is 0

    def test_extract_names_from_code_unknown_kind(self, merger):
        """Test _extract_names_from_code returns [] for unknown kind."""
        result = merger._extract_names_from_code("x = 1\n", "unknown")
        assert result == []

    def test_extract_names_regex_unknown_kind(self, merger):
        """Test _extract_names_regex returns [] for unknown kind."""
        result = merger._extract_names_regex("x = 1\n", "unknown")
        assert result == []

    def test_merge_imports_at_line_zero(self, analyzer):
        """Test _merge_imports inserts at line 0 when no existing imports."""
        source = '''
class Foo:
    pass
'''
        analysis = analyzer.analyze_source(source)
        merger = LocustCodeMerger(analysis)

        lines = list(source.splitlines(keepends=True))
        updated_lines, added = merger._merge_imports(
            lines, "import json\n"
        )

        assert "import json" in added
        # The import should be near the top
        joined = "".join(updated_lines)
        assert "import json" in joined

    def test_filter_methods_no_match(self, merger):
        """Test _filter_methods returns full code when no names match."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        result = merger._filter_methods(code, ["nonexistent"])

        # When no kept_segments, returns full dedented code
        assert "def foo" in result
        assert "def bar" in result

    def test_find_main_block_line_not_found(self, merger):
        """Test _find_main_block_line returns None when no __main__ block."""
        lines = ["import os\n", "x = 1\n"]
        result = merger._find_main_block_line(lines)

        assert result is None

    def test_find_main_block_line_found(self, merger):
        """Test _find_main_block_line finds the __main__ guard."""
        lines = ["import os\n", "\n", 'if __name__ == "__main__":\n', "    pass\n"]
        result = merger._find_main_block_line(lines)

        assert result == 2

    def test_find_first_class_line_not_found(self, merger):
        """Test _find_first_class_line returns None with no classes."""
        lines = ["import os\n", "x = 1\n"]
        result = merger._find_first_class_line(lines)

        assert result is None

    def test_find_first_class_line_found(self, merger):
        """Test _find_first_class_line finds the first class statement."""
        lines = ["import os\n", "\n", "class Foo:\n", "    pass\n"]
        result = merger._find_first_class_line(lines)

        assert result == 2

    def test_validate_merged_code_valid(self, merger):
        """Test _validate_merged_code returns (True, None) for valid code."""
        is_valid, error = merger._validate_merged_code("x = 1\n")

        assert is_valid is True
        assert error is None

    def test_validate_merged_code_invalid(self, merger):
        """Test _validate_merged_code returns (False, error) for invalid code."""
        is_valid, error = merger._validate_merged_code("def broken(\n")

        assert is_valid is False
        assert error is not None
        assert "line" in error

    def test_normalize_import(self, merger):
        """Test _normalize_import collapses whitespace."""
        result = merger._normalize_import("  from   os   import   path  ")
        assert result == "from os import path"

    def test_detect_indent_unit_no_indented_lines(self, merger):
        """Test _detect_indent_unit returns '' when no indented lines."""
        result = merger._detect_indent_unit("x = 1\ny = 2\n")
        assert result == ""

    def test_detect_indent_unit_tabs(self, merger):
        """Test _detect_indent_unit detects tab indentation."""
        result = merger._detect_indent_unit("def foo():\n\tpass\n")
        assert result == "\t"

    def test_basic_dedent(self, merger):
        """Test _basic_dedent strips common leading whitespace."""
        code = "    x = 1\n    y = 2\n"
        result = merger._basic_dedent(code)
        assert result == "x = 1\ny = 2\n"

    def test_split_code_into_functions_regex_directly(self, merger):
        """Test _split_code_into_functions_regex with function kind."""
        code = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        segments = merger._split_code_into_functions_regex(code, "function")

        names = [name for name, _ in segments]
        assert "alpha" in names
        assert "beta" in names

    def test_split_code_into_functions_regex_class_kind(self, merger):
        """Test _split_code_into_functions_regex with class kind."""
        code = "class Alpha:\n    pass\n\nclass Beta:\n    pass\n"
        segments = merger._split_code_into_functions_regex(code, "class")

        names = [name for name, _ in segments]
        assert "Alpha" in names
        assert "Beta" in names
