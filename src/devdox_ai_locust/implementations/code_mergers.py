"""
Code Merger Implementations

Safe code merging that only adds new code without removing existing functionality.
"""

import re
import logging
from typing import Set, Optional

from devdox_ai_locust.abstractions.code_merger import (
    CodeMerger,
    MergeResult,
    MergeContext,
    ProtectedSymbolRemovedError,
    InvalidMergeError,
)
from devdox_ai_locust.abstractions.code_parser import CodeParser, CodeExports
from devdox_ai_locust.implementations.code_parsers import CompositeCodeParser

logger = logging.getLogger(__name__)


class SafeCodeMergerImpl:
    """
    Safe code merger that only adds new code.

    Never removes or modifies existing classes, functions, or methods.
    Uses AST parsing when possible, with regex fallback.
    """

    def __init__(self, code_parser: Optional[CodeParser] = None):
        """
        Initialize merger.

        Args:
            code_parser: Parser for code analysis (default: CompositeCodeParser)
        """
        self.parser = code_parser or CompositeCodeParser()

    def merge(
        self,
        original: str,
        additions: str,
        context: Optional[MergeContext] = None,
    ) -> MergeResult:
        """
        Merge additions into original code safely.

        Only adds new methods/functions, never removes existing ones.
        """
        context = context or MergeContext()

        try:
            # Parse both files
            orig_exports = self.parser.extract_exports(original)
            add_exports = self.parser.extract_exports(additions)

            # Extract only new methods
            new_methods = self.extract_new_methods(
                original,
                additions,
                context.target_class,
            )

            if not new_methods.strip():
                # Nothing new to add
                return MergeResult(
                    success=True,
                    merged_code=original,
                    original_preserved=True,
                    additions=set(),
                )

            # Merge the new methods
            merged = self._do_merge(original, new_methods, context.target_class)

            # Validate merge
            if context.protected_symbols:
                if not self.validate_merge(original, merged, context.protected_symbols):
                    return MergeResult(
                        success=False,
                        merged_code=original,
                        error="Protected symbols were removed",
                    )

            # Validate syntax
            if not self.parser.is_valid_python(merged):
                return MergeResult(
                    success=False,
                    merged_code=original,
                    error="Merged code has syntax errors",
                )

            # Calculate what was added
            merged_exports = self.parser.extract_exports(merged)
            new_funcs = merged_exports.functions - orig_exports.functions
            new_classes = merged_exports.classes - orig_exports.classes

            added = new_funcs | new_classes
            for class_name, methods in merged_exports.methods.items():
                orig_methods = orig_exports.methods.get(class_name, set())
                new_meths = methods - orig_methods
                added |= {f"{class_name}.{m}" for m in new_meths}

            return MergeResult(
                success=True,
                merged_code=merged,
                original_preserved=True,
                additions=added,
            )

        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return MergeResult(
                success=False,
                merged_code=original,
                error=str(e),
            )

    def validate_merge(
        self,
        original: str,
        merged: str,
        protected_symbols: Set[str],
    ) -> bool:
        """Validate that protected symbols are preserved."""
        try:
            merged_exports = self.parser.extract_exports(merged)

            # Check all symbols present
            all_merged = (
                merged_exports.classes |
                merged_exports.functions |
                merged_exports.constants
            )

            # Also include methods
            for class_name, methods in merged_exports.methods.items():
                all_merged.add(class_name)
                for method in methods:
                    all_merged.add(f"{class_name}.{method}")

            for symbol in protected_symbols:
                if symbol not in all_merged:
                    # Check if it's a class.method reference
                    if "." in symbol:
                        class_name, method = symbol.split(".", 1)
                        class_methods = merged_exports.methods.get(class_name, set())
                        if method not in class_methods:
                            logger.warning(f"Protected method missing: {symbol}")
                            return False
                    else:
                        logger.warning(f"Protected symbol missing: {symbol}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def extract_new_methods(
        self,
        original: str,
        additions: str,
        target_class: Optional[str] = None,
    ) -> str:
        """Extract only new methods from additions."""
        try:
            orig_exports = self.parser.extract_exports(original)
            add_exports = self.parser.extract_exports(additions)

            if target_class:
                # Extract only new methods for target class
                orig_methods = orig_exports.methods.get(target_class, set())
                add_methods = add_exports.methods.get(target_class, set())
                new_methods = add_methods - orig_methods

                if not new_methods:
                    return ""

                return self._extract_methods_by_name(additions, new_methods)
            else:
                # Extract new top-level functions
                new_functions = add_exports.functions - orig_exports.functions

                if not new_functions:
                    return ""

                return self._extract_functions_by_name(additions, new_functions)

        except Exception as e:
            logger.warning(f"Failed to extract new methods: {e}")
            # Fallback to regex extraction
            return self._extract_methods_regex(additions, original)

    def _do_merge(
        self,
        original: str,
        new_methods: str,
        target_class: Optional[str] = None,
    ) -> str:
        """Perform the actual merge."""
        if not new_methods.strip():
            return original

        if target_class:
            # Find the class and append methods
            return self._append_to_class(original, new_methods, target_class)
        else:
            # Append at end of file
            return original.rstrip() + "\n\n# AI-enhanced additions\n" + new_methods

    def _append_to_class(self, code: str, methods: str, class_name: str) -> str:
        """Append methods to a class definition."""
        # Find the class definition
        pattern = rf"(class\s+{class_name}\s*(?:\([^)]*\))?\s*:)"
        match = re.search(pattern, code)

        if not match:
            # Class not found, append at end
            return code.rstrip() + "\n\n" + methods

        # Find the end of the class
        class_start = match.end()
        indent = self._detect_class_body_indent(code[class_start:])

        # Find where to insert (before next top-level definition or end)
        remaining = code[class_start:]
        next_class = re.search(r"\nclass\s+\w+", remaining)
        next_func = re.search(r"\ndef\s+\w+", remaining)

        insert_pos = len(code)
        if next_class:
            insert_pos = min(insert_pos, class_start + next_class.start())
        if next_func:
            insert_pos = min(insert_pos, class_start + next_func.start())

        # Indent methods properly
        indented_methods = self._indent_code(methods, len(indent))

        # Insert methods
        return (
            code[:insert_pos].rstrip() +
            "\n\n" +
            indented_methods +
            "\n" +
            code[insert_pos:]
        )

    def _detect_class_body_indent(self, code: str) -> str:
        """Detect the indentation used in class body."""
        # Look for the first method or statement
        match = re.search(r"\n(\s+)(?:def |pass|\"\"\")", code)
        if match:
            return match.group(1)
        return "    "  # Default to 4 spaces

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified spaces."""
        indent = " " * spaces
        lines = code.split("\n")
        return "\n".join(indent + line if line.strip() else line for line in lines)

    def _extract_methods_by_name(self, code: str, method_names: Set[str]) -> str:
        """Extract method definitions by name using regex."""
        methods = []

        for name in method_names:
            # Match method definition
            pattern = rf"(\s*def\s+{name}\s*\([^)]*\)[^:]*:.*?)(?=\n\s*def\s|\n\s*class\s|\Z)"
            match = re.search(pattern, code, re.DOTALL)
            if match:
                methods.append(match.group(1).strip())

        return "\n\n".join(methods)

    def _extract_functions_by_name(self, code: str, function_names: Set[str]) -> str:
        """Extract function definitions by name using regex."""
        functions = []

        for name in function_names:
            # Match function definition (no leading whitespace)
            pattern = rf"(^def\s+{name}\s*\([^)]*\)[^:]*:.*?)(?=\ndef\s|\nclass\s|\Z)"
            match = re.search(pattern, code, re.DOTALL | re.MULTILINE)
            if match:
                functions.append(match.group(1).strip())

        return "\n\n".join(functions)

    def _extract_methods_regex(self, additions: str, original: str) -> str:
        """Fallback regex-based extraction of new methods."""
        # Extract all method names from original
        orig_methods = set(re.findall(r"def\s+(\w+)\s*\(", original))

        # Extract methods from additions that aren't in original
        new_methods = []
        current_method = []
        in_method = False
        base_indent = None

        for line in additions.split("\n"):
            method_match = re.match(r"(\s*)def\s+(\w+)\s*\(", line)

            if method_match:
                # Save previous method if exists
                if current_method and in_method:
                    new_methods.append("\n".join(current_method))

                method_name = method_match.group(2)
                base_indent = len(method_match.group(1))

                if method_name not in orig_methods:
                    current_method = [line]
                    in_method = True
                else:
                    in_method = False
                    current_method = []

            elif in_method:
                # Check if still in method (same or greater indentation)
                stripped = line.lstrip()
                if stripped:
                    current_indent = len(line) - len(stripped)
                    if current_indent > base_indent or not stripped:
                        current_method.append(line)
                    else:
                        # Method ended
                        new_methods.append("\n".join(current_method))
                        in_method = False
                        current_method = []
                else:
                    current_method.append(line)

        # Don't forget last method
        if current_method and in_method:
            new_methods.append("\n".join(current_method))

        return "\n\n".join(new_methods)
