"""
Unreachable code detector for AI-generated Python code.

Detects dead code patterns like code after unconditional return/raise statements.
"""
import ast
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class UnreachableCode:
    """Represents detected unreachable code."""
    line_number: int
    description: str
    code_snippet: str


class UnreachableCodeDetector:
    """
    Detects unreachable code in Python AST.

    Common patterns detected:
    - Code after unconditional return
    - Code after unconditional raise
    - Code after break/continue in loops
    """

    def detect(self, code: str) -> List[UnreachableCode]:
        """
        Detect unreachable code in the given Python code.

        Args:
            code: Python source code

        Returns:
            List of UnreachableCode findings
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.warning("Cannot detect unreachable code - syntax error in source")
            return []

        findings = []
        self._walk_body(tree.body, findings, code.split('\n'))
        return findings

    def _walk_body(
        self,
        body: List[ast.stmt],
        findings: List[UnreachableCode],
        lines: List[str]
    ) -> None:
        """Walk a body of statements looking for unreachable code."""
        for i, stmt in enumerate(body):
            # Check if this is a terminating statement
            if self._is_terminating(stmt):
                # Any statements after this in the same block are unreachable
                remaining = body[i + 1:]
                for unreachable_stmt in remaining:
                    line_no = unreachable_stmt.lineno
                    snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                    findings.append(UnreachableCode(
                        line_number=line_no,
                        description=f"Code after {self._get_terminator_name(stmt)} is unreachable",
                        code_snippet=snippet
                    ))
                # Don't continue checking this block
                break

            # Recurse into nested structures
            self._check_nested(stmt, findings, lines)

    def _check_nested(
        self,
        stmt: ast.stmt,
        findings: List[UnreachableCode],
        lines: List[str]
    ) -> None:
        """Check nested statements for unreachable code."""
        if isinstance(stmt, ast.FunctionDef):
            self._walk_body(stmt.body, findings, lines)
        elif isinstance(stmt, ast.ClassDef):
            for item in stmt.body:
                if isinstance(item, ast.FunctionDef):
                    self._walk_body(item.body, findings, lines)
        elif isinstance(stmt, ast.If):
            self._walk_body(stmt.body, findings, lines)
            self._walk_body(stmt.orelse, findings, lines)
        elif isinstance(stmt, (ast.For, ast.While)):
            self._walk_body(stmt.body, findings, lines)
            self._walk_body(stmt.orelse, findings, lines)
        elif isinstance(stmt, ast.With):
            self._walk_body(stmt.body, findings, lines)
        elif isinstance(stmt, ast.Try):
            self._walk_body(stmt.body, findings, lines)
            for handler in stmt.handlers:
                self._walk_body(handler.body, findings, lines)
            self._walk_body(stmt.orelse, findings, lines)
            self._walk_body(stmt.finalbody, findings, lines)

    def _is_terminating(self, stmt: ast.stmt) -> bool:
        """Check if statement unconditionally terminates flow."""
        return isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue))

    def _get_terminator_name(self, stmt: ast.stmt) -> str:
        """Get human-readable name of terminating statement."""
        if isinstance(stmt, ast.Return):
            return "return"
        elif isinstance(stmt, ast.Raise):
            return "raise"
        elif isinstance(stmt, ast.Break):
            return "break"
        elif isinstance(stmt, ast.Continue):
            return "continue"
        return "terminator"

    def has_unreachable(self, code: str) -> Tuple[bool, List[UnreachableCode]]:
        """
        Check if code has unreachable code.

        Returns:
            Tuple of (has_unreachable, list of findings)
        """
        findings = self.detect(code)
        return bool(findings), findings
