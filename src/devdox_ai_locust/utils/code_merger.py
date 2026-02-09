"""
Locust Code Merger

Intelligently merges AI-generated new test scenarios into an existing Locust
test file. Receives structured sections of new code (imports, tasks, classes,
helpers) and inserts them into the correct locations in the existing file while
preserving all existing functionality.
"""

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from devdox_ai_locust.utils.locust_file_analyzer import (
    LocustFileAnalysis,
    UserClassInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of merging AI-generated code sections into an existing file."""

    merged_source: str
    added_imports: List[str] = field(default_factory=list)
    added_tasks: List[str] = field(default_factory=list)
    added_classes: List[str] = field(default_factory=list)
    added_helpers: List[str] = field(default_factory=list)
    replaced_tasks: List[str] = field(default_factory=list)
    replaced_helpers: List[str] = field(default_factory=list)
    replaced_classes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class LocustCodeMerger:
    """Merges AI-generated code sections into an existing Locust test file.

    Works with a ``LocustFileAnalysis`` produced by
    ``LocustFileAnalyzer`` to understand the structure of the existing
    file and insert new imports, task methods, helper functions, and
    class definitions at the correct locations without breaking
    existing functionality.
    """

    def __init__(self, analysis: LocustFileAnalysis) -> None:
        """
        Args:
            analysis: The LocustFileAnalysis from the analyzer for the
                existing file.
        """
        self._analysis = analysis
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        new_imports: str = "",
        new_tasks: str = "",
        new_classes: str = "",
        new_helpers: str = "",
        replace_tasks: str = "",
        replace_helpers: str = "",
        replace_classes: str = "",
    ) -> MergeResult:
        """
        Merge all new code sections into the existing source.

        The merge is performed in two phases:

        **Phase 1 — Replacements** (applied first to minimise line-number
        drift):

        1. ``replace_classes`` — replace entire class definitions.
        2. ``replace_helpers`` — replace module-level helper functions.
        3. ``replace_tasks`` — replace ``@task`` methods inside a user
           class.

        **Phase 2 — Insertions** (existing behaviour):

        4. **Imports** — inserted near the top of the file.
        5. **Helpers** — inserted after imports, before class definitions.
        6. **Tasks** — injected inside an existing user class.
        7. **Classes** — appended at the end (before ``__main__`` block
           if present).

        Args:
            new_imports: New import statements (may be empty string).
            new_tasks: New ``@task`` methods to add to an existing user
                class (may be empty).
            new_classes: New class definitions (may be empty).
            new_helpers: New helper functions (may be empty).
            replace_tasks: Replacement code for existing ``@task`` methods
                (may be empty).
            replace_helpers: Replacement code for existing module-level
                helper functions (may be empty).
            replace_classes: Replacement code for existing class
                definitions (may be empty).

        Returns:
            MergeResult with the merged source and a summary of changes.
        """
        lines = self._analysis.raw_source.splitlines(keepends=True)

        # Ensure every line ends with a newline so joins are consistent.
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        self._log_section_sizes(
            new_imports=new_imports, new_tasks=new_tasks,
            new_classes=new_classes, new_helpers=new_helpers,
            replace_tasks=replace_tasks, replace_helpers=replace_helpers,
            replace_classes=replace_classes,
        )

        warnings: List[str] = []

        # ---- Phase 1: Replace operations (largest scope first) ----
        replaced_names: List[List[str]] = []
        replace_ops = [
            (replace_classes, self._replace_classes),
            (replace_helpers, self._replace_helpers),
            (replace_tasks, self._replace_tasks),
        ]
        for code, handler in replace_ops:
            if code and code.strip():
                lines, names, rw = handler(lines, code)
                replaced_names.append(names)
                warnings.extend(rw)
            else:
                replaced_names.append([])
        replaced_classes_list, replaced_helpers_list, replaced_tasks_list = replaced_names

        # ---- Phase 2: Insert operations (existing behaviour) ----
        inserted_names: List[List[str]] = []
        insert_ops = [
            (new_imports, self._merge_imports),
            (new_helpers, self._merge_helpers),
            (new_tasks, self._merge_tasks),
            (new_classes, self._merge_classes),
        ]
        for code, handler in insert_ops:
            if code and code.strip():
                lines, names = handler(lines, code)
                inserted_names.append(names)
            else:
                inserted_names.append([])
        added_imports, added_helpers_list, added_tasks, added_classes = inserted_names

        merged_source = "".join(lines)
        self._validate_and_log_result(merged_source, warnings)

        self._logger.info(
            "merge() complete — +%d imports, +%d tasks, +%d classes, "
            "+%d helpers, ~%d tasks replaced, ~%d helpers replaced, "
            "~%d classes replaced, %d warning(s)",
            len(added_imports),
            len(added_tasks),
            len(added_classes),
            len(added_helpers_list),
            len(replaced_tasks_list),
            len(replaced_helpers_list),
            len(replaced_classes_list),
            len(warnings),
        )

        return MergeResult(
            merged_source=merged_source,
            added_imports=added_imports,
            added_tasks=added_tasks,
            added_classes=added_classes,
            added_helpers=added_helpers_list,
            replaced_tasks=replaced_tasks_list,
            replaced_helpers=replaced_helpers_list,
            replaced_classes=replaced_classes_list,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _log_section_sizes(self, **sections: str) -> None:
        """Log incoming section sizes for traceability."""
        parts = ", ".join(
            f"{name}={len(code.strip()) if code else 0}"
            for name, code in sections.items()
        )
        self._logger.debug("merge() called — section sizes: %s", parts)

    def _validate_and_log_result(
        self, merged_source: str, warnings: List[str]
    ) -> None:
        """Validate the merged source and append a warning if invalid."""
        is_valid, error = self._validate_merged_code(merged_source)
        if not is_valid:
            warnings.append(f"Merged code has syntax errors: {error}")
            self._logger.warning("Merged code validation failed: %s", error)
        else:
            self._logger.debug("Merged code passed syntax validation")

    # ------------------------------------------------------------------
    # Replacement operations
    # ------------------------------------------------------------------

    def _replace_tasks(
        self, lines: List[str], replace_tasks_code: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Find existing ``@task`` methods by name inside the target user
        class and replace their entire body (including decorators) with
        the upgraded version.

        The replacement code is parsed to discover function names.  For
        each name the *current* lines are re-parsed (since earlier
        replacements may have shifted line numbers) to locate the method
        boundaries.

        Args:
            lines: Current file lines.
            replace_tasks_code: Replacement ``@task`` method code block.

        Returns:
            Tuple of (updated lines, list of replaced method names,
            list of warning strings).
        """
        warnings: List[str] = []
        replaced_names: List[str] = []

        # Parse replacement code to get individual function segments.
        func_segments = self._split_code_into_functions(replace_tasks_code)
        if not func_segments:
            return lines, replaced_names, warnings

        if not self._analysis.user_classes:
            warnings.append(
                "No user classes found; cannot replace task methods."
            )
            return lines, replaced_names, warnings

        target_class = self._pick_target_class()

        for func_name, func_code in func_segments:
            # Re-parse lines on every iteration because earlier
            # replacements shift line numbers.
            class_start, class_end = self._find_class_boundaries(
                lines, target_class.name
            )
            if class_start is None or class_end is None:
                warnings.append(
                    f"Could not locate class '{target_class.name}' "
                    f"when trying to replace task '{func_name}'."
                )
                continue

            method_start, method_end = self._find_method_in_class(
                lines, class_start, class_end, func_name
            )
            if method_start is None or method_end is None:
                warnings.append(
                    f"Task method '{func_name}' not found in class "
                    f"'{target_class.name}'; skipping replacement."
                )
                continue

            # Detect the class body indentation to re-indent the
            # replacement code consistently.
            indent = self._detect_indentation(lines, class_start, class_end)
            reindented = self._reindent(func_code, indent)

            # Build replacement lines (ensure trailing newline).
            if not reindented.endswith("\n"):
                reindented += "\n"
            replacement_lines = reindented.splitlines(keepends=True)

            # Splice: remove old method lines, insert new ones.
            lines[method_start : method_end + 1] = replacement_lines

            replaced_names.append(func_name)
            self._logger.info(
                "Replaced task method '%s' in class '%s' "
                "(lines %d–%d -> %d lines)",
                func_name,
                target_class.name,
                method_start + 1,
                method_end + 1,
                len(replacement_lines),
            )

        return lines, replaced_names, warnings

    def _replace_helpers(
        self, lines: List[str], replace_helpers_code: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Find existing module-level functions by name and replace them
        with the upgraded version.

        Args:
            lines: Current file lines.
            replace_helpers_code: Replacement helper function code block.

        Returns:
            Tuple of (updated lines, list of replaced function names,
            list of warning strings).
        """
        warnings: List[str] = []
        replaced_names: List[str] = []

        func_segments = self._split_code_into_functions(replace_helpers_code)
        if not func_segments:
            return lines, replaced_names, warnings

        for func_name, func_code in func_segments:
            # Re-parse every iteration because earlier replacements
            # shift line numbers.
            func_start, func_end = self._find_module_function_boundaries(
                lines, func_name
            )
            if func_start is None or func_end is None:
                warnings.append(
                    f"Module-level function '{func_name}' not found; "
                    f"skipping replacement."
                )
                continue

            # Module-level functions are at zero indent.
            dedented = textwrap.dedent(func_code)
            if not dedented.endswith("\n"):
                dedented += "\n"
            replacement_lines = dedented.splitlines(keepends=True)

            lines[func_start : func_end + 1] = replacement_lines

            replaced_names.append(func_name)
            self._logger.info(
                "Replaced module-level function '%s' "
                "(lines %d–%d -> %d lines)",
                func_name,
                func_start + 1,
                func_end + 1,
                len(replacement_lines),
            )

        return lines, replaced_names, warnings

    def _replace_classes(
        self, lines: List[str], replace_classes_code: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Find existing class definitions by name and replace them with
        the upgraded version.

        Args:
            lines: Current file lines.
            replace_classes_code: Replacement class definition code block.

        Returns:
            Tuple of (updated lines, list of replaced class names,
            list of warning strings).
        """
        warnings: List[str] = []
        replaced_names: List[str] = []

        class_segments = self._split_code_into_classes(replace_classes_code)
        if not class_segments:
            return lines, replaced_names, warnings

        for class_name, class_code in class_segments:
            # Re-parse every iteration.
            class_start, class_end = self._find_class_boundaries(
                lines, class_name
            )
            if class_start is None or class_end is None:
                warnings.append(
                    f"Class '{class_name}' not found; "
                    f"skipping replacement."
                )
                continue

            # Classes live at module level (zero indent).
            dedented = textwrap.dedent(class_code)
            if not dedented.endswith("\n"):
                dedented += "\n"
            replacement_lines = dedented.splitlines(keepends=True)

            lines[class_start : class_end + 1] = replacement_lines

            replaced_names.append(class_name)
            self._logger.info(
                "Replaced class '%s' (lines %d–%d -> %d lines)",
                class_name,
                class_start + 1,
                class_end + 1,
                len(replacement_lines),
            )

        return lines, replaced_names, warnings

    # ------------------------------------------------------------------
    # Replacement helpers (locating boundaries)
    # ------------------------------------------------------------------

    @staticmethod
    def _node_start_line(node: ast.AST) -> int:
        """Return the 0-based start line of an AST node, including decorators."""
        start = node.lineno - 1
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list) - 1
        return start

    @staticmethod
    def _find_class_node_at(
        tree: ast.Module, class_start: int
    ) -> Optional[ast.ClassDef]:
        """Find the ClassDef node whose lineno matches *class_start*."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and (node.lineno - 1) == class_start:
                return node
        return None

    def _find_method_in_class(
        self,
        lines: List[str],
        class_start: int,
        class_end: int, #NOSONAR
        method_name: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Locate a method (including its decorators) inside a class in the
        current source lines.

        Re-parses the full source via AST so that line numbers are
        accurate even after earlier replacements have shifted lines.

        Args:
            lines: Current file lines.
            class_start: 0-based start index of the enclosing class.
            class_end: 0-based end index of the enclosing class.
            method_name: Name of the method to find.

        Returns:
            Tuple of (start_idx, end_idx) as 0-based inclusive indices,
            or (None, None) if the method is not found.
        """
        source = "".join(lines)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None, None

        class_node = self._find_class_node_at(tree, class_start)
        if class_node is None:
            return None, None

        for child in ast.iter_child_nodes(class_node):
            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                start = self._node_start_line(child)
                end = child.end_lineno - 1 if child.end_lineno else start
                return start, end

        return None, None

    def _find_module_function_boundaries(
        self, lines: List[str], func_name: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the start and end line indices of a module-level function
        in the current lines by re-parsing the source.

        Includes decorator lines that precede the function definition.

        Args:
            lines: Current file lines.
            func_name: Name of the function to locate.

        Returns:
            Tuple of (start_idx, end_idx) as 0-based inclusive indices,
            or (None, None) if the function is not found.
        """
        source = "".join(lines)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None, None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start = self._node_start_line(node)
                end = node.end_lineno - 1 if node.end_lineno else start
                return start, end

        return None, None

    def _split_code_into_functions(
        self, code: str
    ) -> List[Tuple[str, str]]:
        """
        Split a code block containing one or more function definitions
        into a list of ``(name, source)`` tuples.

        Uses ``ast.parse`` to identify function boundaries (including
        decorators).  Falls back to a regex-based splitter when parsing
        fails.

        Args:
            code: Source code containing one or more function definitions.

        Returns:
            List of (function_name, function_source) tuples.
        """
        try:
            dedented = textwrap.dedent(code)
            tree = ast.parse(dedented)
        except SyntaxError:
            return self._split_code_into_functions_regex(code, "function")

        source_lines = dedented.splitlines(keepends=True)
        segments: List[Tuple[str, str]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                start = self._node_start_line(node)
                end = node.end_lineno if node.end_lineno else start + 1
                segment = "".join(source_lines[start:end])
                segments.append((node.name, segment))

        return segments

    def _split_code_into_classes(
        self, code: str
    ) -> List[Tuple[str, str]]:
        """
        Split a code block containing one or more class definitions
        into a list of ``(name, source)`` tuples.

        Uses ``ast.parse`` to identify class boundaries (including
        decorators).  Falls back to a regex-based splitter when parsing
        fails.

        Args:
            code: Source code containing one or more class definitions.

        Returns:
            List of (class_name, class_source) tuples.
        """
        try:
            dedented = textwrap.dedent(code)
            tree = ast.parse(dedented)
        except SyntaxError:
            return self._split_code_into_functions_regex(code, "class")

        source_lines = dedented.splitlines(keepends=True)
        segments: List[Tuple[str, str]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                start = self._node_start_line(node)
                end = node.end_lineno if node.end_lineno else start + 1
                segment = "".join(source_lines[start:end])
                segments.append((node.name, segment))

        return segments

    def _split_code_into_functions_regex(
        self, code: str, kind: str
    ) -> List[Tuple[str, str]]:
        """
        Regex fallback for splitting code into function or class
        segments when AST parsing fails.

        Splits on unindented ``def`` or ``class`` keywords and captures
        all lines until the next unindented definition or end of string.

        Args:
            code: Source code string.
            kind: ``"function"`` or ``"class"``.

        Returns:
            List of (name, source) tuples.
        """
        keyword = "def" if kind == "function" else "class"
        # Match decorator lines followed by a def/class at zero indent.
        pattern = re.compile(
            rf"^((?:@\w[^\n]*\n)*)({keyword}\s+(\w+)[^\n]*\n"
            r"(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*)",
            re.MULTILINE,
        )
        segments: List[Tuple[str, str]] = []
        dedented = textwrap.dedent(code)
        for m in pattern.finditer(dedented):
            decorators = m.group(1)
            body = m.group(2)
            name = m.group(3)
            segments.append((name, decorators + body))
        return segments

    # ------------------------------------------------------------------
    # Import merging
    # ------------------------------------------------------------------

    def _merge_imports(
        self, lines: List[str], new_imports: str
    ) -> Tuple[List[str], List[str]]:
        """
        Insert non-duplicate import statements after the last existing
        import line.

        Args:
            lines: Current file lines.
            new_imports: New import statements block.

        Returns:
            Tuple of (updated lines, list of added import strings).
        """
        existing_normalized = {
            self._normalize_import(imp.statement)
            for imp in self._analysis.imports
        }

        import_lines = [
            line
            for line in new_imports.splitlines()
            if line.strip() and (
                line.strip().startswith("import ")
                or line.strip().startswith("from ")
            )
        ]

        added: List[str] = []
        to_insert: List[str] = []
        for imp_line in import_lines:
            normalized = self._normalize_import(imp_line.strip())
            if normalized not in existing_normalized:
                to_insert.append(imp_line.strip() + "\n")
                added.append(imp_line.strip())
                # Prevent inserting the same import twice within one batch.
                existing_normalized.add(normalized)

        if not to_insert:
            return lines, added

        # Find the last import line number (1-based in the analysis).
        last_import_lineno = 0
        for imp in self._analysis.imports:
            if imp.line_number > last_import_lineno:
                last_import_lineno = imp.line_number

        # Convert to 0-based index and insert after that line.
        insert_idx = last_import_lineno if last_import_lineno > 0 else 0
        for offset, new_line in enumerate(to_insert):
            lines.insert(insert_idx + offset, new_line)

        self._logger.info(
            "Inserted %d new import(s) at line %d", len(to_insert), insert_idx + 1
        )
        return lines, added

    # ------------------------------------------------------------------
    # Task merging
    # ------------------------------------------------------------------

    def _merge_tasks(
        self, lines: List[str], new_tasks: str
    ) -> Tuple[List[str], List[str]]:
        """
        Insert new ``@task`` methods into an existing user class.

        Because earlier merge steps (imports, helpers) may have shifted
        line numbers, this method re-scans the current lines to locate
        the target class rather than relying on the original analysis
        line numbers.

        Args:
            lines: Current file lines.
            new_tasks: New task method code block.

        Returns:
            Tuple of (updated lines, list of added task method names).
        """
        if not self._analysis.user_classes:
            self._logger.warning(
                "No user classes found in existing file; "
                "skipping task insertion."
            )
            return lines, []

        target_class = self._pick_target_class()

        # Names of tasks already present in the target class.
        existing_task_names = {
            m.name for m in target_class.task_methods
        }
        # Also include non-task methods to avoid name collisions.
        existing_method_names = existing_task_names | set(target_class.other_methods)

        # Normalize indentation early so nested @task methods become
        # top-level functions visible to AST-based name extraction.
        new_tasks = self._normalize_decorator_indentation(new_tasks)

        # Parse new tasks to discover their names and filter duplicates.
        new_task_names = self._extract_names_from_code(new_tasks, "function")
        names_to_add = [
            name for name in new_task_names
            if name not in existing_method_names
        ]

        if not names_to_add:
            self._logger.info("All new tasks already exist; nothing to add.")
            return lines, []

        # Filter to keep only the methods we need.
        tasks_code = self._filter_methods(new_tasks, names_to_add)

        if not tasks_code.strip():
            return lines, []

        # Find the target class boundaries in the *current* lines.
        class_start, class_end = self._find_class_boundaries(
            lines, target_class.name
        )
        if class_start is None or class_end is None:
            self._logger.warning(
                "Could not locate class '%s' in current source; "
                "skipping task insertion.",
                target_class.name,
            )
            return lines, []

        # Detect indentation from the class body in the current lines.
        indent = self._detect_indentation(lines, class_start, class_end)

        # Re-indent the new tasks to match the target class.
        tasks_code = self._reindent(tasks_code, indent)

        # Build the block to insert: blank line separator + tasks.
        block = "\n" + tasks_code
        if not block.endswith("\n"):
            block += "\n"

        insert_lines = block.splitlines(keepends=True)

        # Insert just before the class_end line (so the code stays
        # inside the class body).  class_end is 0-based inclusive
        # index of the last line of the class.
        insert_idx = class_end + 1
        for offset, new_line in enumerate(insert_lines):
            lines.insert(insert_idx + offset, new_line)

        self._logger.info(
            "Inserted %d task(s) into class '%s'",
            len(names_to_add),
            target_class.name,
        )
        return lines, names_to_add

    def _pick_target_class(self) -> UserClassInfo:
        """
        Return the user class that should receive new task methods.

        Prefers the first class that already contains task methods; falls
        back to the first user class.

        Returns:
            The selected UserClassInfo.
        """
        for cls in self._analysis.user_classes:
            if cls.task_methods:
                return cls
        return self._analysis.user_classes[0]

    def _find_class_boundaries(
        self, lines: List[str], class_name: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the start and end line indices of a class in the current
        lines by re-parsing the source.

        This is necessary because earlier merge operations may have
        shifted line numbers relative to the original analysis.

        Args:
            lines: Current file lines.
            class_name: Name of the class to locate.

        Returns:
            Tuple of (start_idx, end_idx) as 0-based indices, or
            (None, None) if the class cannot be found.
        """
        source = "".join(lines)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None, None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                start = node.lineno - 1  # 0-based
                end = (node.end_lineno - 1) if node.end_lineno else start
                return start, end

        return None, None

    def _filter_methods(
        self, code: str, keep_names: List[str]
    ) -> str:
        """
        Keep only the function definitions whose names are in
        *keep_names*.

        Uses ``ast.parse`` to identify boundaries and reassembles the
        kept functions. Falls back to returning the full code block when
        parsing fails.

        Args:
            code: Source code containing one or more function definitions.
            keep_names: Function names to keep.

        Returns:
            Filtered source code string (dedented to zero indent).
        """
        keep_set = set(keep_names)

        try:
            dedented = textwrap.dedent(code)
            tree = ast.parse(dedented)
        except SyntaxError:
            self._logger.debug(
                "Could not parse task code for filtering; "
                "inserting full block."
            )
            return textwrap.dedent(code)

        source_lines = dedented.splitlines(keepends=True)
        kept_segments: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name in keep_set:
                start = self._node_start_line(node)
                end = node.end_lineno if node.end_lineno else start + 1
                segment = "".join(source_lines[start:end])
                kept_segments.append(segment)

        if not kept_segments:
            return textwrap.dedent(code)

        # Join with a single blank line between methods.
        return "\n".join(seg.rstrip("\n") for seg in kept_segments) + "\n"

    # ------------------------------------------------------------------
    # Class merging
    # ------------------------------------------------------------------

    def _merge_classes(
        self, lines: List[str], new_classes: str
    ) -> Tuple[List[str], List[str]]:
        """
        Append new class definitions at the end of the file.

        If an ``if __name__ == "__main__":`` block exists the new
        classes are inserted immediately before it. Two blank lines are
        added before each new class to comply with PEP 8.

        Args:
            lines: Current file lines.
            new_classes: New class definitions code block.

        Returns:
            Tuple of (updated lines, list of new class names).
        """
        class_names = self._extract_names_from_code(new_classes, "class")

        block = "\n\n" + new_classes.rstrip("\n") + "\n"
        insert_lines = block.splitlines(keepends=True)

        # Determine insertion point.
        insert_idx = self._find_main_block_line(lines)
        if insert_idx is None:
            insert_idx = len(lines)

        for offset, new_line in enumerate(insert_lines):
            lines.insert(insert_idx + offset, new_line)

        self._logger.info("Inserted %d new class(es)", len(class_names))
        return lines, class_names

    def _find_main_block_line(self, lines: List[str]) -> Optional[int]:
        """
        Re-scan lines for ``if __name__`` since earlier insertions may
        have shifted line numbers compared to the original analysis.

        Args:
            lines: Current file lines.

        Returns:
            0-based line index of the ``if __name__`` block, or None.
        """
        pattern = re.compile(
            r"""^\s*if\s+__name__\s*==\s*['"]__main__['"]\s*:"""
        )
        for idx, line in enumerate(lines):
            if pattern.match(line):
                return idx
        return None

    # ------------------------------------------------------------------
    # Helper function merging
    # ------------------------------------------------------------------

    def _merge_helpers(
        self, lines: List[str], new_helpers: str
    ) -> Tuple[List[str], List[str]]:
        """
        Insert helper functions after imports but before class
        definitions.

        Args:
            lines: Current file lines.
            new_helpers: New helper function code block.

        Returns:
            Tuple of (updated lines, list of helper function names).
        """
        helper_names = self._extract_names_from_code(new_helpers, "function")

        block = "\n\n" + new_helpers.rstrip("\n") + "\n"
        insert_lines = block.splitlines(keepends=True)

        # Find the line number of the first class definition (0-based).
        first_class_idx = self._find_first_class_line(lines)
        if first_class_idx is not None:
            insert_idx = first_class_idx
        else:
            # No classes — insert at the end of file.
            insert_idx = len(lines)

        for offset, new_line in enumerate(insert_lines):
            lines.insert(insert_idx + offset, new_line)

        self._logger.info("Inserted %d helper function(s)", len(helper_names))
        return lines, helper_names

    def _find_first_class_line(self, lines: List[str]) -> Optional[int]:
        """
        Scan for the first ``class`` statement at module level.

        Args:
            lines: Current file lines.

        Returns:
            0-based line index, or None if no class found.
        """
        pattern = re.compile(r"^class\s+\w+")
        for idx, line in enumerate(lines):
            if pattern.match(line):
                return idx
        return None

    # ------------------------------------------------------------------
    # Indentation helpers
    # ------------------------------------------------------------------

    def _detect_indentation(
        self, lines: List[str], class_start: int, class_end: int
    ) -> str:
        """
        Detect the indentation string used for methods inside the given
        class by scanning the class body in the current lines.

        Args:
            lines: Current file lines.
            class_start: 0-based start index of the class.
            class_end: 0-based end index of the class.

        Returns:
            Indentation string (e.g. ``"    "`` for four spaces).
        """
        # Scan lines within the class body for 'def ' or '@' decorators.
        for idx in range(class_start + 1, min(class_end + 1, len(lines))):
            stripped = lines[idx].lstrip()
            if stripped.startswith("def ") or stripped.startswith("@"):
                leading = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
                if leading:
                    return leading
        return "    "

    def _reindent(self, code: str, target_indent: str) -> str:
        """
        Adjust the indentation of a code block to match *target_indent*.

        Strips the common leading whitespace from the block, then
        re-indents every line so that the top-level statements sit at
        *target_indent* and deeper levels are indented using the same
        indent unit (e.g. tabs when the target is a tab, four spaces
        when the target is four spaces).

        Args:
            code: The code block to re-indent.
            target_indent: The desired indentation string for the first
                level.

        Returns:
            Re-indented code string.
        """
        # First normalize decorator/def indentation issues from AI output
        normalized = self._normalize_decorator_indentation(code)
        dedented = textwrap.dedent(normalized)

        # Detect the indent unit used in the dedented code by looking at
        # the first indented non-empty line.
        source_indent_unit = self._detect_indent_unit(dedented)

        result_lines: List[str] = []
        for line in dedented.splitlines(keepends=True):
            if not line.strip():
                result_lines.append(line)
                continue

            # Count how many levels of indentation this line has.
            leading = line[: len(line) - len(line.lstrip())]
            if source_indent_unit and leading:
                depth = len(leading) // len(source_indent_unit)
                remainder = len(leading) % len(source_indent_unit)
                new_leading = (
                    target_indent
                    + target_indent * depth
                    + (" " * remainder if remainder else "")
                )
            else:
                new_leading = target_indent

            result_lines.append(new_leading + line.lstrip())
        return "".join(result_lines)

    def _normalize_decorator_indentation(self, code: str) -> str:
        """
        Normalize indentation issues in AI-generated code.

        The AI sometimes outputs multiple @task methods where some are
        incorrectly indented (nested inside other methods). This function
        extracts all @task decorated methods and normalizes them to the
        same base indentation level.

        Args:
            code: Source code that may have inconsistent indentation.

        Returns:
            Code with all @task methods at the same indentation level.
        """
        lines = code.splitlines(keepends=True)
        if not lines:
            return code

        # Find all @task decorators (regardless of nesting level)
        task_starts: List[int] = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("@task"):
                task_starts.append(i)

        if not task_starts:
            # No @task decorators - just do basic normalization
            return self._basic_dedent(code)

        # Find the minimum indentation among all @task decorators
        min_indent = min(
            len(lines[i]) - len(lines[i].lstrip())
            for i in task_starts
        )

        # Extract each method (from @task to next @task or end)
        methods: List[str] = []
        for idx, start in enumerate(task_starts):
            # End is either the next @task or end of code
            end = task_starts[idx + 1] if idx + 1 < len(task_starts) else len(lines)

            # Extract lines for this method
            method_lines = lines[start:end]

            # Calculate the indent of this method's @task decorator
            first_line = method_lines[0]
            current_indent = len(first_line) - len(first_line.lstrip())
            indent_diff = current_indent - min_indent

            # Dedent all lines of this method
            dedented_method: List[str] = []
            for line in method_lines:
                if not line.strip():
                    dedented_method.append(line)
                else:
                    line_indent = len(line) - len(line.lstrip())
                    new_indent = max(0, line_indent - indent_diff)
                    dedented_method.append(" " * new_indent + line.lstrip())

            methods.append("".join(dedented_method))

        # Get any content before the first @task
        prefix = "".join(lines[:task_starts[0]]) if task_starts[0] > 0 else ""

        # Combine all methods with proper separation
        result = prefix + "".join(methods)

        # Final pass: ensure decorator and def are aligned
        return self._fix_decorator_def_alignment(result)

    def _basic_dedent(self, code: str) -> str:
        """Apply basic dedent for code without @task decorators."""
        return textwrap.dedent(code)

    def _fix_decorator_def_alignment(self, code: str) -> str:
        """
        Fix cases where a decorator and its def have different indentation.

        For example:
            @task(1)
                def foo(self):  # 4 extra spaces

        Becomes:
            @task(1)
            def foo(self):
        """
        lines = code.splitlines(keepends=True)
        result: List[str] = []
        i = 0

        while i < len(lines):
            stripped = lines[i].lstrip()

            if not stripped.startswith("@"):
                result.append(lines[i])
                i += 1
                continue

            deco_indent = len(lines[i]) - len(stripped)
            decorators, j = self._collect_decorators(lines, i)

            fix = self._fix_misaligned_def(lines, j, deco_indent) if j < len(lines) else None
            if fix is not None:
                result.extend(decorators)
                fixed_lines, j = fix
                result.extend(fixed_lines)
            else:
                result.extend(decorators)
            i = j

        return "".join(result)

    @staticmethod
    def _collect_decorators(
        lines: List[str], start: int
    ) -> Tuple[List[str], int]:
        """Collect consecutive decorator lines (and empty lines between them).

        Returns the decorator lines and the index of the next non-decorator line.
        """
        decorators = [lines[start]]
        j = start + 1
        while j < len(lines):
            next_stripped = lines[j].lstrip()
            if next_stripped.startswith("@") or not next_stripped:
                decorators.append(lines[j])
                j += 1
            else:
                break
        return decorators, j

    @staticmethod
    def _fix_misaligned_def(
        lines: List[str], def_idx: int, deco_indent: int
    ) -> Optional[Tuple[List[str], int]]:
        """Fix a misaligned def/async def line and its body.

        Returns ``(fixed_lines, next_idx)`` when a misalignment is found
        and corrected, or ``None`` when no fix is needed.
        """
        def_line = lines[def_idx]
        def_stripped = def_line.lstrip()

        if not (def_stripped.startswith("def ") or def_stripped.startswith("async def ")):
            return None

        def_indent = len(def_line) - len(def_stripped)
        if def_indent == deco_indent:
            return None

        # Realign the def line to match the decorator indent.
        fixed: List[str] = []
        indent_str = " " * deco_indent
        fixed.append(indent_str + def_stripped)

        # Adjust body indentation.
        diff = def_indent - deco_indent
        j = def_idx + 1
        while j < len(lines):
            body_stripped = lines[j].lstrip()

            if not body_stripped:
                fixed.append(lines[j])
                j += 1
                continue

            if body_stripped.startswith("@"):
                break

            body_indent = len(lines[j]) - len(body_stripped)
            new_indent = max(0, body_indent - diff)
            fixed.append(" " * new_indent + body_stripped)
            j += 1

        return fixed, j

    def _detect_indent_unit(self, code: str) -> str:
        """
        Detect the base indentation unit in a dedented code block.

        Looks at the first indented non-empty line and returns its
        leading whitespace. Returns an empty string if no indented
        lines are found.

        Args:
            code: Dedented source code.

        Returns:
            The indentation unit string (e.g. ``"    "`` or ``"\\t"``).
        """
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped and line != stripped:
                return line[: len(line) - len(stripped)]
        return ""

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _normalize_import(self, import_str: str) -> str:
        """
        Normalize an import statement for duplicate comparison.

        Strips surrounding whitespace and collapses internal runs of
        whitespace to single spaces so that cosmetic differences do not
        cause false negatives.

        Args:
            import_str: The raw import statement string.

        Returns:
            Normalized import string.
        """
        return " ".join(import_str.strip().split())

    def _extract_names_from_code(self, code: str, kind: str) -> List[str]:
        """
        Extract function or class names from a code snippet.

        Uses ``ast.parse`` for reliable extraction. Falls back to a
        simple regex scan when the snippet is not valid standalone
        Python.

        Args:
            code: Source code snippet.
            kind: ``"function"`` or ``"class"``.

        Returns:
            List of extracted names.
        """
        try:
            dedented = textwrap.dedent(code)
            tree = ast.parse(dedented)
        except SyntaxError:
            return self._extract_names_regex(code, kind)

        if kind == "function":
            return [
                node.name
                for node in ast.iter_child_nodes(tree)
                if isinstance(node, ast.FunctionDef)
            ]
        if kind == "class":
            return [
                node.name
                for node in ast.iter_child_nodes(tree)
                if isinstance(node, ast.ClassDef)
            ]
        return []

    def _extract_names_regex(self, code: str, kind: str) -> List[str]:
        """
        Fallback name extraction using regular expressions.

        Args:
            code: Source code snippet.
            kind: ``"function"`` or ``"class"``.

        Returns:
            List of extracted names.
        """
        if kind == "function":
            pattern = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
        elif kind == "class":
            pattern = re.compile(r"^\s*class\s+(\w+)\s*[:(]", re.MULTILINE)
        else:
            return []
        return pattern.findall(code)

    def _validate_merged_code(
        self, source: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that the merged source is syntactically valid Python.

        Args:
            source: The complete merged source code.

        Returns:
            Tuple of (is_valid, error_message). ``error_message`` is
            None when the source is valid.
        """
        try:
            ast.parse(source)
            return True, None
        except SyntaxError as exc:
            return False, f"line {exc.lineno}: {exc.msg}"
