"""
Code Merger Protocol

Defines the contract for merging AI-generated code with existing code.
Ensures additive-only changes that preserve existing functionality.
"""

from typing import Protocol, Optional, Set, runtime_checkable
from pydantic import BaseModel, Field


class MergeResult(BaseModel):
    """Result of a code merge operation"""
    success: bool
    merged_code: str
    original_preserved: bool = True
    additions: Set[str] = Field(default_factory=set)  # New methods/classes added
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class MergeContext(BaseModel):
    """Context for merge operation"""
    target_class: Optional[str] = None  # Class to merge into
    protected_symbols: Set[str] = Field(default_factory=set)  # Symbols that must not be removed
    allow_override: bool = False  # Allow overriding existing methods

    class Config:
        arbitrary_types_allowed = True


@runtime_checkable
class CodeMerger(Protocol):
    """
    Protocol for code merging implementations.

    Implementations:
        - SafeCodeMerger: Only adds new code, never removes
        - ASTCodeMerger: Uses AST for precise merging
        - SimpleCodeMerger: Appends code at end

    Example:
        class SafeCodeMerger:
            def __init__(self, code_parser: CodeParser):
                self.parser = code_parser

            def merge(
                self,
                original: str,
                additions: str,
                context: MergeContext
            ) -> MergeResult:
                # Parse both files
                orig_exports = self.parser.extract_exports(original)
                add_exports = self.parser.extract_exports(additions)

                # Only add new symbols
                new_symbols = add_exports.functions - orig_exports.functions
                # ...
    """

    def merge(
        self,
        original: str,
        additions: str,
        context: Optional[MergeContext] = None,
    ) -> MergeResult:
        """
        Merge additions into original code.

        Args:
            original: Original source code
            additions: Code to add/merge
            context: Merge context with options

        Returns:
            MergeResult with merged code
        """
        ...

    def validate_merge(
        self,
        original: str,
        merged: str,
        protected_symbols: Set[str],
    ) -> bool:
        """
        Validate that a merge preserved protected symbols.

        Args:
            original: Original source code
            merged: Merged source code
            protected_symbols: Symbols that must exist in merged code

        Returns:
            True if all protected symbols are preserved
        """
        ...

    def extract_new_methods(
        self,
        original: str,
        additions: str,
        target_class: Optional[str] = None,
    ) -> str:
        """
        Extract only new methods from additions.

        Args:
            original: Original source code
            additions: Code with potential additions
            target_class: If set, only extract methods for this class

        Returns:
            String containing only new methods
        """
        ...


class MergeError(Exception):
    """Base exception for merge errors"""
    pass


class ProtectedSymbolRemovedError(MergeError):
    """A protected symbol was removed during merge"""

    def __init__(self, symbol: str):
        super().__init__(f"Protected symbol removed: {symbol}")
        self.symbol = symbol


class InvalidMergeError(MergeError):
    """Merge would produce invalid code"""

    def __init__(self, message: str, syntax_error: Optional[str] = None):
        super().__init__(message)
        self.syntax_error = syntax_error
