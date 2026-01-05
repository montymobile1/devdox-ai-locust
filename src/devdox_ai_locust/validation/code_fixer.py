"""
Code fixer for AI-generated Python code.

Attempts to automatically fix common issues in AI-generated code.
"""
import ast
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .code_validator import CodeValidator, ValidationResult
from .unreachable import UnreachableCodeDetector

logger = logging.getLogger(__name__)


@dataclass
class FixResult:
    """Result of code fix attempt."""
    success: bool
    fixed_code: str
    fixes_applied: List[str]
    remaining_issues: List[str]


class CodeFixer:
    """
    Attempts to fix common issues in AI-generated code.

    This is an injectable service that can be used to repair
    AI output before writing files.
    """

    def __init__(self):
        self.validator = CodeValidator()
        self.unreachable_detector = UnreachableCodeDetector()

    def fix(self, code: str, max_attempts: int = 3) -> FixResult:
        """
        Attempt to fix code issues.

        Args:
            code: Python code to fix
            max_attempts: Maximum fix iterations

        Returns:
            FixResult with fixed code and applied fixes
        """
        fixes_applied = []
        current_code = code

        for attempt in range(max_attempts):
            # Validate current state
            result = self.validator.validate(current_code)

            if result.is_valid:
                # Check for unreachable code (warning level)
                has_unreachable, findings = self.unreachable_detector.has_unreachable(current_code)
                if has_unreachable:
                    fixed, fix_desc = self._fix_unreachable_code(current_code, findings)
                    if fixed != current_code:
                        current_code = fixed
                        fixes_applied.append(fix_desc)
                        continue

                # All good
                return FixResult(
                    success=True,
                    fixed_code=current_code,
                    fixes_applied=fixes_applied,
                    remaining_issues=[]
                )

            # Try to fix syntax errors
            if not result.syntax_valid:
                fixed, fix_desc = self._fix_syntax_error(current_code, result.syntax_error)
                if fixed != current_code:
                    current_code = fixed
                    fixes_applied.append(fix_desc)
                    continue

            # Try to fix other issues
            for issue in result.issues:
                fixed, fix_desc = self._fix_issue(current_code, issue)
                if fixed != current_code:
                    current_code = fixed
                    fixes_applied.append(fix_desc)
                    break  # Re-validate after each fix

        # Final validation
        final_result = self.validator.validate(current_code)
        return FixResult(
            success=final_result.is_valid,
            fixed_code=current_code,
            fixes_applied=fixes_applied,
            remaining_issues=final_result.issues
        )

    def _fix_syntax_error(self, code: str, error: Optional[str]) -> Tuple[str, str]:
        """Attempt to fix syntax errors."""
        if not error:
            return code, ""

        # Try common fixes
        fixed = code

        # Fix: Missing colon after def/if/for/while/with/class/try/except
        fixed = re.sub(
            r'^(\s*)(def\s+\w+\s*\([^)]*\))\s*$',
            r'\1\2:',
            fixed,
            flags=re.MULTILINE
        )
        fixed = re.sub(
            r'^(\s*)(if\s+.+?)\s*$(?!\s*:)',
            r'\1\2:',
            fixed,
            flags=re.MULTILINE
        )
        fixed = re.sub(
            r'^(\s*)(else)\s*$',
            r'\1\2:',
            fixed,
            flags=re.MULTILINE
        )
        fixed = re.sub(
            r'^(\s*)(elif\s+.+?)\s*$(?!\s*:)',
            r'\1\2:',
            fixed,
            flags=re.MULTILINE
        )

        if fixed != code:
            return fixed, "Added missing colons"

        # Fix: Unclosed string
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            # Count quotes
            single_quotes = line.count("'") - line.count("\\'")
            double_quotes = line.count('"') - line.count('\\"')
            if single_quotes % 2 == 1:
                line = line + "'"
            if double_quotes % 2 == 1:
                line = line + '"'
            fixed_lines.append(line)
        fixed = '\n'.join(fixed_lines)

        if fixed != code:
            return fixed, "Closed unclosed strings"

        return code, ""

    def _fix_unreachable_code(
        self,
        code: str,
        findings: List
    ) -> Tuple[str, str]:
        """
        Attempt to fix unreachable code by adjusting indentation.

        The most common case is code after return that should be
        outside the if block.
        """
        if not findings:
            return code, ""

        lines = code.split('\n')
        fixed_lines = lines.copy()
        fixes_made = 0

        for finding in findings:
            line_idx = finding.line_number - 1
            if line_idx >= len(fixed_lines):
                continue

            # Find the return/raise line before this
            return_line_idx = None
            for i in range(line_idx - 1, -1, -1):
                stripped = fixed_lines[i].strip()
                if stripped.startswith('return') or stripped.startswith('raise'):
                    return_line_idx = i
                    break

            if return_line_idx is None:
                continue

            # Get indent of the return line
            return_indent = len(fixed_lines[return_line_idx]) - len(fixed_lines[return_line_idx].lstrip())

            # Find the if/with block that contains the return
            block_indent = None
            for i in range(return_line_idx - 1, -1, -1):
                stripped = fixed_lines[i].strip()
                line_indent = len(fixed_lines[i]) - len(fixed_lines[i].lstrip())
                if stripped.endswith(':') and line_indent < return_indent:
                    block_indent = line_indent
                    break

            if block_indent is None:
                continue

            # Dedent the unreachable line to be outside the if block
            current_indent = len(fixed_lines[line_idx]) - len(fixed_lines[line_idx].lstrip())
            new_indent = block_indent
            content = fixed_lines[line_idx].lstrip()
            fixed_lines[line_idx] = ' ' * new_indent + content
            fixes_made += 1

        if fixes_made > 0:
            return '\n'.join(fixed_lines), f"Fixed {fixes_made} unreachable code blocks"

        return code, ""

    def _fix_issue(self, code: str, issue: str) -> Tuple[str, str]:
        """Attempt to fix a specific issue."""
        # Duplicate class definitions
        if "Duplicate class" in issue:
            return self._remove_duplicate_classes(code)

        # Import and definition conflict
        if "both imported and defined" in issue:
            return self._remove_conflicting_imports(code, issue)

        return code, ""

    def _remove_duplicate_classes(self, code: str) -> Tuple[str, str]:
        """Remove duplicate class definitions, keeping the first."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, ""

        seen_classes = set()
        lines_to_remove = set()
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in seen_classes:
                    # Mark all lines of this class for removal
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                    for i in range(start, end):
                        lines_to_remove.add(i)
                else:
                    seen_classes.add(node.name)

        if lines_to_remove:
            fixed_lines = [
                line for i, line in enumerate(lines)
                if i not in lines_to_remove
            ]
            return '\n'.join(fixed_lines), "Removed duplicate class definitions"

        return code, ""

    def _remove_conflicting_imports(self, code: str, issue: str) -> Tuple[str, str]:
        """Remove imports that conflict with local definitions."""
        # Extract conflicting names from issue
        match = re.search(r'\[(.*?)\]', issue)
        if not match:
            return code, ""

        conflicts = [s.strip().strip("'\"") for s in match.group(1).split(',')]

        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('from ') or stripped.startswith('import '):
                # Check if this import contains any conflicting name
                should_remove = any(
                    re.search(rf'\b{re.escape(name)}\b', line)
                    for name in conflicts
                )
                if should_remove:
                    continue
            fixed_lines.append(line)

        if len(fixed_lines) != len(lines):
            return '\n'.join(fixed_lines), f"Removed conflicting imports: {conflicts}"

        return code, ""

    def fix_method_code(self, code: str) -> FixResult:
        """
        Fix code that represents method bodies (not full classes).

        Wraps code in a dummy class for fixing, then unwraps.
        """
        # Wrap in dummy class
        wrapped = "class _DummyClass:\n"
        for line in code.split('\n'):
            wrapped += f"    {line}\n"

        # Fix the wrapped code
        result = self.fix(wrapped)

        if result.success:
            # Unwrap - remove the class wrapper
            lines = result.fixed_code.split('\n')
            unwrapped_lines = []
            for line in lines[1:]:  # Skip class line
                if line.startswith('    '):
                    unwrapped_lines.append(line[4:])
                elif line.strip() == '':
                    unwrapped_lines.append('')

            return FixResult(
                success=True,
                fixed_code='\n'.join(unwrapped_lines),
                fixes_applied=result.fixes_applied,
                remaining_issues=result.remaining_issues
            )

        return result
