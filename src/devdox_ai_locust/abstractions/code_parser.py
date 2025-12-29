"""
Code Parser Protocol

Defines the contract for parsing Python code to extract
classes, functions, imports, and other symbols.
"""

from typing import Protocol, Set, Dict, List, Optional, Tuple, runtime_checkable
from pydantic import BaseModel, Field


class CodeExports(BaseModel):
    """Symbols exported/defined by a code file"""
    classes: Set[str] = Field(default_factory=set)
    functions: Set[str] = Field(default_factory=set)
    methods: Dict[str, Set[str]] = Field(default_factory=dict)  # class -> methods
    constants: Set[str] = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True


class CodeImports(BaseModel):
    """Symbols imported by a code file"""
    # module -> set of imported names (or '*' for star import)
    imports: Dict[str, Set[str]] = Field(default_factory=dict)
    # Relative imports from local files
    local_imports: Dict[str, Set[str]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


@runtime_checkable
class CodeParser(Protocol):
    """
    Protocol for Python code parsing implementations.

    Implementations:
        - ASTCodeParser: Uses Python AST module
        - RegexCodeParser: Uses regex patterns (fallback)
        - CompositeCodeParser: Tries AST first, falls back to regex

    Example:
        class ASTCodeParser:
            def extract_exports(self, code: str) -> CodeExports:
                tree = ast.parse(code)
                exports = CodeExports()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        exports.classes.add(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        exports.functions.add(node.name)
                return exports
    """

    def extract_exports(self, code: str) -> CodeExports:
        """
        Extract defined symbols from code.

        Args:
            code: Python source code

        Returns:
            CodeExports with classes, functions, methods
        """
        ...

    def extract_imports(self, code: str) -> CodeImports:
        """
        Extract imported symbols from code.

        Args:
            code: Python source code

        Returns:
            CodeImports with module imports and local imports
        """
        ...

    def is_valid_python(self, code: str) -> bool:
        """
        Check if code is valid Python syntax.

        Args:
            code: Python source code

        Returns:
            True if syntax is valid
        """
        ...

    def get_class_methods(self, code: str, class_name: str) -> Set[str]:
        """
        Get method names for a specific class.

        Args:
            code: Python source code
            class_name: Name of the class

        Returns:
            Set of method names in the class
        """
        ...


class CodeParserError(Exception):
    """Base exception for code parser errors"""
    pass


class SyntaxParseError(CodeParserError):
    """Code has syntax errors"""

    def __init__(self, message: str, line: Optional[int] = None):
        super().__init__(message)
        self.line = line
