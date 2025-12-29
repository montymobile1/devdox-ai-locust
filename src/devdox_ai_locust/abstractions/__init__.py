"""
Abstractions Package - Protocol Definitions for DevDox AI Locust

This package contains Protocol definitions (interfaces) that define contracts
between components. By depending on abstractions rather than concrete
implementations, we achieve:

1. **Dependency Inversion** - High-level modules don't depend on low-level modules
2. **Testability** - Components can be tested in isolation using mock implementations
3. **Extensibility** - New implementations can be added without modifying existing code
4. **Loose Coupling** - Components interact through well-defined interfaces

Usage:
    from devdox_ai_locust.abstractions import AIClient, TemplateEngine, FileSystem

    class MyService:
        def __init__(
            self,
            ai_client: AIClient,
            template_engine: TemplateEngine,
            file_system: FileSystem,
        ):
            self.ai_client = ai_client
            self.template_engine = template_engine
            self.file_system = file_system
"""

from .ai_client import AIClient, AICompletionRequest, AICompletionResponse
from .template_engine import TemplateEngine
from .file_system import FileSystem, WriteResult, FileInfo
from .code_parser import CodeParser, CodeExports, CodeImports
from .retry_policy import RetryPolicy, RetryDecision
from .code_merger import CodeMerger, MergeResult, MergeContext
from .test_generator import TestGenerator, GenerationResult, GenerationContext

__all__ = [
    # AI Client
    "AIClient",
    "AICompletionRequest",
    "AICompletionResponse",
    # Template Engine
    "TemplateEngine",
    # File System
    "FileSystem",
    "WriteResult",
    "FileInfo",
    # Code Parser
    "CodeParser",
    "CodeExports",
    "CodeImports",
    # Retry Policy
    "RetryPolicy",
    "RetryDecision",
    # Code Merger
    "CodeMerger",
    "MergeResult",
    "MergeContext",
    # Test Generator
    "TestGenerator",
    "GenerationResult",
    "GenerationContext",
]
