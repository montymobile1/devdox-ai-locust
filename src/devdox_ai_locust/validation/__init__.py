"""
Code validation and fixing module for AI-generated code.

This module provides:
- CodeValidator: AST-based syntax and structure validation
- UnreachableCodeDetector: Dead code detection after return/raise
- CodeFixer: Auto-fix common AI generation issues
"""
from .code_validator import CodeValidator, ValidationResult
from .unreachable import UnreachableCodeDetector
from .code_fixer import CodeFixer

__all__ = [
    "CodeValidator",
    "ValidationResult",
    "UnreachableCodeDetector",
    "CodeFixer",
]
