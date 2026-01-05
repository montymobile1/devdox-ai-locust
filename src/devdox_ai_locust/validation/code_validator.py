"""
Code validator for AI-generated Python code.

Provides AST-based validation for syntax, structure, and common issues.
Promoted and enhanced from tests/live/generate/test_validation.py.
"""
import ast
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    syntax_valid: bool = True
    syntax_error: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_issue(self, issue: str) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(warning)


class CodeValidator:
    """
    Validates generated Python code for correctness and quality.

    This is an injectable service that can be used by generators
    to validate AI output before writing files.
    """

    # Classes that should only appear once across all files
    SINGLETON_CLASSES = {
        "TestDataGenerator",
        "ResponseValidator",
        "RequestLogger",
        "PerformanceMonitor",
        "DataManager",
        "LoadTestConfig",
    }

    # Keywords indicating auth-related functionality
    AUTH_KEYWORDS = {"auth", "login", "logout", "token", "session", "credential"}

    def validate(self, code: str, context: Optional[str] = None) -> ValidationResult:
        """
        Perform full validation on code.

        Args:
            code: Python code to validate
            context: Optional context for error messages (e.g., file name)

        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(is_valid=True)

        # Check syntax first
        syntax_valid, syntax_error = self.is_valid_python(code)
        result.syntax_valid = syntax_valid
        result.syntax_error = syntax_error

        if not syntax_valid:
            result.add_issue(f"Syntax error: {syntax_error}")
            return result  # Can't do further validation with syntax errors

        # Check for duplicates
        has_dups, duplicates = self.has_duplicate_class(code)
        if has_dups:
            result.add_issue(f"Duplicate class definitions: {duplicates}")

        # Check for import/definition conflicts
        has_conflicts, conflicts = self.has_import_and_definition(code)
        if has_conflicts:
            result.add_issue(f"Classes both imported and defined: {conflicts}")

        # Check Locust structure
        proper_structure, structure_issues = self.has_proper_locust_structure(code)
        if not proper_structure:
            for issue in structure_issues:
                result.add_warning(issue)

        # Check for hardcoded secrets
        has_secrets, secret_issues = self.has_hardcoded_secrets(code)
        if has_secrets:
            for issue in secret_issues:
                result.add_warning(issue)

        return result

    @staticmethod
    def is_valid_python(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    @staticmethod
    def compile_check(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code compiles (catches more issues than ast.parse)."""
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    @staticmethod
    def get_class_definitions(code: str) -> List[str]:
        """Extract all class names defined in code."""
        try:
            tree = ast.parse(code)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except SyntaxError:
            return re.findall(r"^class\s+(\w+)", code, re.MULTILINE)

    @staticmethod
    def get_function_definitions(code: str) -> List[str]:
        """Extract all function names defined in code."""
        try:
            tree = ast.parse(code)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            return re.findall(r"^def\s+(\w+)", code, re.MULTILINE)

    @staticmethod
    def get_imports(code: str) -> Set[str]:
        """Extract all imported names."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.add(alias.asname or alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.asname or alias.name)
        except SyntaxError:
            imports.update(re.findall(r"from\s+\S+\s+import\s+(\w+)", code))
            imports.update(re.findall(r"^import\s+(\w+)", code, re.MULTILINE))
        return imports

    @staticmethod
    def has_duplicate_class(code: str) -> Tuple[bool, List[str]]:
        """Check if any class is defined more than once."""
        classes = CodeValidator.get_class_definitions(code)
        duplicates = [c for c in set(classes) if classes.count(c) > 1]
        return bool(duplicates), duplicates

    @staticmethod
    def has_import_and_definition(code: str) -> Tuple[bool, List[str]]:
        """Check if a class is both imported and defined."""
        imports = CodeValidator.get_imports(code)
        definitions = set(CodeValidator.get_class_definitions(code))
        conflicts = imports & definitions
        return bool(conflicts), list(conflicts)

    @staticmethod
    def has_proper_locust_structure(code: str) -> Tuple[bool, List[str]]:
        """Validate Locust-specific patterns."""
        issues = []

        if "HttpUser" in code and "class" in code:
            if not re.search(r"class\s+\w+.*HttpUser", code):
                issues.append("HttpUser should be used as base class")

        if "catch_response=True" in code:
            if not re.search(r"with\s+self\.client\.[a-z]+\(.*catch_response\s*=\s*True", code, re.DOTALL):
                issues.append("catch_response=True should be used in 'with' block")

        return len(issues) == 0, issues

    @staticmethod
    def has_hardcoded_secrets(code: str) -> Tuple[bool, List[str]]:
        """Check for hardcoded secrets, API keys, or passwords."""
        issues = []
        secret_patterns = [
            (r'api_key\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']', "Possible hardcoded API key"),
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "Possible hardcoded password"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
            (r'token\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']', "Possible hardcoded token"),
        ]

        for pattern, description in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                # Skip if it looks like a placeholder
                if not re.search(r'(test|example|placeholder|your_|changeme)', code, re.IGNORECASE):
                    issues.append(description)

        return bool(issues), issues

    @staticmethod
    def count_auth_classes(code: str) -> int:
        """Count auth-related classes in code."""
        classes = CodeValidator.get_class_definitions(code)
        return sum(
            1 for c in classes
            if any(kw in c.lower() for kw in CodeValidator.AUTH_KEYWORDS)
        )

    def validate_method_code(self, code: str) -> ValidationResult:
        """
        Validate code that represents method bodies (not full classes).

        Wraps code in a dummy class for AST parsing.
        """
        # Wrap in dummy class to make it parseable
        wrapped = f"class _DummyClass:\n"
        for line in code.split('\n'):
            wrapped += f"    {line}\n"

        return self.validate(wrapped)
