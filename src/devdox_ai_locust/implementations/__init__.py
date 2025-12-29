"""
Implementations Package - Concrete Implementations of Protocols

This package contains concrete implementations of the protocols defined
in the abstractions package. Each implementation can be swapped at runtime
through dependency injection.

Production Implementations:
    - TogetherAIClient: Together AI API client
    - JinjaTemplateEngine: Jinja2 template engine
    - LocalFileSystem: Real file system operations
    - ASTCodeParser: Python AST-based code parser

Testing Implementations:
    - MockAIClient: Returns predefined responses
    - InMemoryTemplateEngine: Templates stored in dict
    - InMemoryFileSystem: All operations in memory
    - RecordingFileSystem: Records operations for verification
"""

from .ai_clients import TogetherAIClient, MockAIClient
from .template_engines import JinjaTemplateEngine, InMemoryTemplateEngine
from .file_systems import LocalFileSystem, InMemoryFileSystem
from .code_parsers import ASTCodeParser, RegexCodeParser, CompositeCodeParser
from .retry_policies import (
    ExponentialBackoffPolicy,
    RateLimitAwarePolicy,
    NoRetryPolicy,
)
from .code_mergers import SafeCodeMergerImpl

__all__ = [
    # AI Clients
    "TogetherAIClient",
    "MockAIClient",
    # Template Engines
    "JinjaTemplateEngine",
    "InMemoryTemplateEngine",
    # File Systems
    "LocalFileSystem",
    "InMemoryFileSystem",
    # Code Parsers
    "ASTCodeParser",
    "RegexCodeParser",
    "CompositeCodeParser",
    # Retry Policies
    "ExponentialBackoffPolicy",
    "RateLimitAwarePolicy",
    "NoRetryPolicy",
    # Code Mergers
    "SafeCodeMergerImpl",
]
