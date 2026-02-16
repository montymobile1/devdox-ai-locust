"""
Locust Test File Analyzer

Analyzes existing Locust test files using Python's ast module to extract
their structure. The analysis is used by the AI enhancement system to
understand what already exists so it can add new test scenarios without
duplication.
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TaskMethodInfo:
    """Represents a @task decorated method within a Locust user class."""

    name: str
    weight: int
    http_method: Optional[str] = None
    http_path: Optional[str] = None
    line_number: int = 0


@dataclass
class UserClassInfo:
    """Represents a Locust user class or task set definition."""

    name: str
    parent_classes: List[str]
    task_methods: List[TaskMethodInfo]
    other_methods: List[str]
    line_number: int = 0
    end_line_number: int = 0


@dataclass
class ImportInfo:
    """Represents an import statement in the analyzed file."""

    statement: str
    line_number: int = 0


@dataclass
class LocustFileAnalysis:
    """Complete analysis result for a Locust test file."""

    user_classes: List[UserClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    module_level_functions: List[str] = field(default_factory=list)
    event_handlers: List[str] = field(default_factory=list)
    has_main_block: bool = False
    main_block_line: Optional[int] = None
    raw_source: str = ""
    total_lines: int = 0


class LocustFileAnalyzer:
    """Analyzes existing Locust test files to extract structure information.

    Uses Python's ast module to parse the source and walk the syntax tree,
    extracting classes, methods, decorators, imports, and other structural
    elements relevant to Locust test generation.
    """

    _HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}

    def analyze(self, file_path: str) -> LocustFileAnalysis:
        """
        Read and analyze a Locust test file.

        Args:
            file_path: Path to the Locust test file to analyze.

        Returns:
            LocustFileAnalysis with the extracted structure.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid Python syntax.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except FileNotFoundError:
            logger.error(f"Locust file not found: {file_path}")
            raise

        return self.analyze_source(source)

    def analyze_source(self, source: str) -> LocustFileAnalysis:
        """
        Analyze Locust test source code directly without reading from disk.

        Args:
            source: Python source code as a string.

        Returns:
            LocustFileAnalysis with the extracted structure.

        Raises:
            ValueError: If the source contains invalid Python syntax.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(
                f"Invalid Python syntax at line {e.lineno}: {e.msg}"
            )

        has_main, main_line = self._find_main_block(tree)

        return LocustFileAnalysis(
            user_classes=self._extract_classes(tree),
            imports=self._extract_imports(tree),
            module_level_functions=self._extract_module_functions(tree),
            event_handlers=self._extract_event_handlers(tree),
            has_main_block=has_main,
            main_block_line=main_line,
            raw_source=source,
            total_lines=len(source.splitlines()),
        )

    def _extract_classes(self, tree: ast.Module) -> List[UserClassInfo]:
        """
        Extract class definitions and their methods from the module.

        Args:
            tree: The parsed AST module.

        Returns:
            List of UserClassInfo for each class found at module level.
        """
        classes: List[UserClassInfo] = []

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            parent_classes = [
                ast.unparse(base) for base in node.bases
            ]

            task_methods = self._extract_task_methods(node)
            task_method_names = {m.name for m in task_methods}

            other_methods = [
                func.name
                for func in ast.iter_child_nodes(node)
                if isinstance(func, ast.FunctionDef)
                and func.name not in task_method_names
            ]

            end_line = node.end_lineno if node.end_lineno else node.lineno

            classes.append(
                UserClassInfo(
                    name=node.name,
                    parent_classes=parent_classes,
                    task_methods=task_methods,
                    other_methods=other_methods,
                    line_number=node.lineno,
                    end_line_number=end_line,
                )
            )

        return classes

    def _extract_task_methods(
        self, class_node: ast.ClassDef
    ) -> List[TaskMethodInfo]:
        """
        Extract @task decorated methods from a class definition.

        Args:
            class_node: The AST class definition node.

        Returns:
            List of TaskMethodInfo for each @task method found.
        """
        task_methods: List[TaskMethodInfo] = []

        for node in ast.iter_child_nodes(class_node):
            if not isinstance(node, ast.FunctionDef):
                continue

            task_weight = self._get_task_weight_from_decorators(node.decorator_list)
            if task_weight is None:
                continue

            http_method, http_path = self._detect_http_call(node)

            task_methods.append(
                TaskMethodInfo(
                    name=node.name,
                    weight=task_weight,
                    http_method=http_method,
                    http_path=http_path,
                    line_number=node.lineno,
                )
            )

        return task_methods

    def _get_task_weight_from_decorators(
        self, decorators: List[ast.expr]
    ) -> Optional[int]:
        """
        Check if a list of decorators contains @task and return its weight.

        Args:
            decorators: List of decorator AST expressions.

        Returns:
            The task weight if @task is found, otherwise None.
        """
        for decorator in decorators:
            weight = self._get_task_weight(decorator)
            if weight is not None:
                return weight
        return None

    def _get_task_weight(self, decorator: ast.expr) -> Optional[int]:
        """
        Extract weight from a @task decorator.

        Handles both ``@task`` (weight defaults to 1) and ``@task(N)``
        (weight is N).

        Args:
            decorator: A single decorator AST expression.

        Returns:
            The task weight as an integer, or None if the decorator
            is not a @task decorator.
        """
        # @task — no arguments
        if isinstance(decorator, ast.Name) and decorator.id == "task":
            return 1

        # @task(N) — called with arguments
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "task":
                if decorator.args:
                    first_arg = decorator.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(
                        first_arg.value, int
                    ):
                        return first_arg.value
                # @task() with no arguments defaults to weight 1
                return 1

        return None

    def _detect_http_call(
        self, func_node: ast.FunctionDef
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to detect self.client.<method>(...) calls and extract the path.

        Walks the function body looking for attribute calls on
        ``self.client`` that match known HTTP methods (get, post, put,
        patch, delete, head, options). Returns the first match found.

        Args:
            func_node: The AST function definition node.

        Returns:
            A tuple of (http_method, http_path). Both are None if no
            HTTP call is detected.
        """
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            method_name = func.attr
            if method_name not in self._HTTP_METHODS:
                continue

            # Check that the call target is self.client
            if not self._is_self_client(func.value):
                continue

            http_path = self._extract_first_string_arg(node)

            return method_name.upper(), http_path

        return None, None

    def _is_self_client(self, node: ast.expr) -> bool:
        """
        Check whether an AST node represents ``self.client``.

        Args:
            node: The AST expression to check.

        Returns:
            True if the node is ``self.client``, False otherwise.
        """
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "client"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def _extract_first_string_arg(self, call_node: ast.Call) -> Optional[str]:
        """
        Extract the first string argument from a function call.

        Handles plain string literals and f-strings. For f-strings the
        result is the unparsed representation, which may contain
        placeholder expressions.

        Args:
            call_node: The AST Call node.

        Returns:
            The string value of the first argument, or None if it cannot
            be extracted.
        """
        if not call_node.args:
            return None

        first_arg = call_node.args[0]

        # Plain string literal
        if isinstance(first_arg, ast.Constant) and isinstance(
            first_arg.value, str
        ):
            return first_arg.value

        # f-string — unparse to get a readable representation
        if isinstance(first_arg, ast.JoinedStr):
            try:
                return ast.unparse(first_arg)
            except Exception:
                return None

        return None

    def _extract_imports(self, tree: ast.Module) -> List[ImportInfo]:
        """
        Extract all import statements from the module.

        Args:
            tree: The parsed AST module.

        Returns:
            List of ImportInfo for each import statement.
        """
        imports: List[ImportInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                statement = ast.unparse(node)
                imports.append(
                    ImportInfo(
                        statement=statement,
                        line_number=node.lineno,
                    )
                )

        return imports

    def _find_main_block(self, tree: ast.Module) -> Tuple[bool, Optional[int]]:
        """
        Find whether an ``if __name__ == '__main__':`` block exists.

        Args:
            tree: The parsed AST module.

        Returns:
            A tuple of (has_main_block, line_number). line_number is None
            if no main block is found.
        """
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.If):
                continue

            if self._is_name_main_check(node.test):
                return True, node.lineno

        return False, None

    def _is_name_main_check(self, test_node: ast.expr) -> bool:
        """
        Check if an AST expression is ``__name__ == '__main__'``.

        Handles both orderings: ``__name__ == '__main__'`` and
        ``'__main__' == __name__``.

        Args:
            test_node: The AST expression from an if-statement test.

        Returns:
            True if the expression matches the __name__ == '__main__' pattern.
        """
        if not isinstance(test_node, ast.Compare):
            return False

        if len(test_node.ops) != 1 or not isinstance(test_node.ops[0], ast.Eq):
            return False

        if len(test_node.comparators) != 1:
            return False

        left = test_node.left
        right = test_node.comparators[0]

        # __name__ == "__main__"
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ):
            return True

        # "__main__" == __name__
        if (
            isinstance(right, ast.Name)
            and right.id == "__name__"
            and isinstance(left, ast.Constant)
            and left.value == "__main__"
        ):
            return True

        return False

    def _extract_event_handlers(self, tree: ast.Module) -> List[str]:
        """
        Find functions decorated with event-related decorators.

        Looks for decorators that reference ``events`` (e.g.,
        ``@events.test_start.add_listener``).

        Args:
            tree: The parsed AST module.

        Returns:
            List of function names that are event handlers.
        """
        handlers: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            for decorator in node.decorator_list:
                if self._is_events_decorator(decorator):
                    handlers.append(node.name)
                    break

        return handlers

    def _is_events_decorator(self, decorator: ast.expr) -> bool:
        """
        Check if a decorator references the ``events`` object.

        Traverses the attribute chain to find ``events`` at the root.

        Args:
            decorator: A single decorator AST expression.

        Returns:
            True if the decorator chain starts with ``events``.
        """
        node = decorator

        # Handle Call wrappers: @events.test_start.add_listener
        # may appear as a Call node wrapping the attribute chain
        if isinstance(node, ast.Call):
            node = node.func

        # Walk down the attribute chain to find the root name
        while isinstance(node, ast.Attribute):
            node = node.value

        return isinstance(node, ast.Name) and node.id == "events"

    def _extract_module_functions(self, tree: ast.Module) -> List[str]:
        """
        Extract top-level function definitions (not inside classes).

        Args:
            tree: The parsed AST module.

        Returns:
            List of function names defined at module level.
        """
        return [
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef)
        ]
