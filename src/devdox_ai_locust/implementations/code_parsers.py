"""
Code Parser Implementations

AST-based and regex-based implementations of the CodeParser protocol.
"""

import ast
import re
import logging
from typing import Set, Optional

from devdox_ai_locust.abstractions.code_parser import (
    CodeParser,
    CodeExports,
    CodeImports,
    SyntaxParseError,
)

logger = logging.getLogger(__name__)


class ASTCodeParser:
    """
    Code parser using Python AST module.

    Most accurate but fails on syntax errors.
    """

    def extract_exports(self, code: str) -> CodeExports:
        """Extract exports using AST."""
        exports = CodeExports()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxParseError(str(e), e.lineno)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                exports.classes.add(node.name)
                # Extract methods
                methods = set()
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.add(item.name)
                exports.methods[node.name] = methods

            elif isinstance(node, ast.FunctionDef):
                # Top-level functions only
                if isinstance(node, ast.FunctionDef):
                    parent = getattr(node, 'parent', None)
                    if parent is None or isinstance(parent, ast.Module):
                        exports.functions.add(node.name)

            elif isinstance(node, ast.Assign):
                # Top-level constants (uppercase names)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            exports.constants.add(target.id)

        # Fix: Walk tree manually to get proper parent relationships
        exports = self._extract_with_parents(tree)
        return exports

    def _extract_with_parents(self, tree: ast.AST) -> CodeExports:
        """Extract exports with proper parent tracking."""
        exports = CodeExports()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                exports.classes.add(node.name)
                methods = set()
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.add(item.name)
                exports.methods[node.name] = methods

            elif isinstance(node, ast.FunctionDef):
                exports.functions.add(node.name)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        exports.constants.add(target.id)

        return exports

    def extract_imports(self, code: str) -> CodeImports:
        """Extract imports using AST."""
        imports = CodeImports()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxParseError(str(e), e.lineno)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module not in imports.imports:
                        imports.imports[module] = set()
                    imports.imports[module].add(alias.asname or alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                # Check if local import (relative or from .)
                if node.level > 0 or module.startswith("."):
                    target = imports.local_imports
                else:
                    target = imports.imports

                if module not in target:
                    target[module] = set()

                for alias in node.names:
                    if alias.name == "*":
                        target[module].add("*")
                    else:
                        target[module].add(alias.name)

        return imports

    def is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def get_class_methods(self, code: str, class_name: str) -> Set[str]:
        """Get methods of a specific class."""
        exports = self.extract_exports(code)
        return exports.methods.get(class_name, set())


class RegexCodeParser:
    """
    Code parser using regex patterns.

    Less accurate but works on invalid Python syntax.
    Used as fallback when AST parsing fails.
    """

    # Patterns for extraction
    CLASS_PATTERN = re.compile(r"^class\s+(\w+)", re.MULTILINE)
    FUNCTION_PATTERN = re.compile(r"^def\s+(\w+)", re.MULTILINE)
    METHOD_PATTERN = re.compile(r"^\s{4}def\s+(\w+)", re.MULTILINE)
    IMPORT_PATTERN = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)$", re.MULTILINE)
    CONSTANT_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)

    def extract_exports(self, code: str) -> CodeExports:
        """Extract exports using regex."""
        exports = CodeExports()

        # Extract classes
        exports.classes = set(self.CLASS_PATTERN.findall(code))

        # Extract top-level functions (not indented)
        for match in re.finditer(r"^def\s+(\w+)", code, re.MULTILINE):
            # Check if at start of line (not a method)
            line_start = code.rfind("\n", 0, match.start()) + 1
            if match.start() == line_start:
                exports.functions.add(match.group(1))

        # Extract methods per class
        class_positions = [
            (m.start(), m.group(1))
            for m in self.CLASS_PATTERN.finditer(code)
        ]

        for i, (pos, class_name) in enumerate(class_positions):
            # Find end of class (next class or end of file)
            end = class_positions[i + 1][0] if i + 1 < len(class_positions) else len(code)
            class_code = code[pos:end]

            methods = set()
            for match in self.METHOD_PATTERN.finditer(class_code):
                methods.add(match.group(1))
            exports.methods[class_name] = methods

        # Extract constants
        exports.constants = set(self.CONSTANT_PATTERN.findall(code))

        return exports

    def extract_imports(self, code: str) -> CodeImports:
        """Extract imports using regex."""
        imports = CodeImports()

        for match in self.IMPORT_PATTERN.finditer(code):
            from_module = match.group(1)
            import_part = match.group(2)

            if from_module:
                # from X import Y
                module = from_module
                names = [n.strip().split(" as ")[0] for n in import_part.split(",")]

                is_local = from_module.startswith(".")
                target = imports.local_imports if is_local else imports.imports

                if module not in target:
                    target[module] = set()
                target[module].update(names)
            else:
                # import X
                modules = [m.strip().split(" as ")[0] for m in import_part.split(",")]
                for module in modules:
                    if module not in imports.imports:
                        imports.imports[module] = set()
                    imports.imports[module].add(module)

        return imports

    def is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python (using AST)."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def get_class_methods(self, code: str, class_name: str) -> Set[str]:
        """Get methods of a specific class."""
        exports = self.extract_exports(code)
        return exports.methods.get(class_name, set())


class CompositeCodeParser:
    """
    Composite parser that tries AST first, falls back to regex.

    Best of both worlds: accuracy when possible, fallback when needed.
    """

    def __init__(
        self,
        primary: Optional[CodeParser] = None,
        fallback: Optional[CodeParser] = None,
    ):
        """
        Initialize composite parser.

        Args:
            primary: Primary parser (default: ASTCodeParser)
            fallback: Fallback parser (default: RegexCodeParser)
        """
        self.primary = primary or ASTCodeParser()
        self.fallback = fallback or RegexCodeParser()

    def extract_exports(self, code: str) -> CodeExports:
        """Extract exports, falling back on error."""
        try:
            return self.primary.extract_exports(code)
        except (SyntaxParseError, SyntaxError) as e:
            logger.debug(f"AST parsing failed, using regex fallback: {e}")
            return self.fallback.extract_exports(code)

    def extract_imports(self, code: str) -> CodeImports:
        """Extract imports, falling back on error."""
        try:
            return self.primary.extract_imports(code)
        except (SyntaxParseError, SyntaxError) as e:
            logger.debug(f"AST parsing failed, using regex fallback: {e}")
            return self.fallback.extract_imports(code)

    def is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python."""
        return self.primary.is_valid_python(code)

    def get_class_methods(self, code: str, class_name: str) -> Set[str]:
        """Get methods of a specific class."""
        try:
            return self.primary.get_class_methods(code, class_name)
        except (SyntaxParseError, SyntaxError):
            return self.fallback.get_class_methods(code, class_name)
