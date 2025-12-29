"""
Dependency Injection Container

Provides a central place to configure and wire all dependencies.
Supports different configurations for production, testing, and development.

Usage:
    # Production
    container = Container.create_production(api_key="...", output_dir=Path(...))
    generator = container.get_generator()

    # Testing
    container = Container.create_testing()
    generator = container.get_generator()

    # Custom
    container = Container()
    container.register_ai_client(MockAIClient())
    container.register_file_system(InMemoryFileSystem())
"""

from pathlib import Path
from typing import Optional, Dict, Any, Type, TypeVar
from dataclasses import dataclass, field
import logging

from devdox_ai_locust.abstractions import (
    AIClient,
    TemplateEngine,
    FileSystem,
    CodeParser,
    RetryPolicy,
    CodeMerger,
)
from devdox_ai_locust.implementations import (
    TogetherAIClient,
    MockAIClient,
    JinjaTemplateEngine,
    InMemoryTemplateEngine,
    LocalFileSystem,
    InMemoryFileSystem,
    CompositeCodeParser,
    ExponentialBackoffPolicy,
    RateLimitAwarePolicy,
    NoRetryPolicy,
    SafeCodeMergerImpl,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ContainerConfig:
    """Configuration for the dependency container"""
    # Paths
    template_dir: Optional[Path] = None
    prompt_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # AI Configuration
    ai_api_key: Optional[str] = None
    ai_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    ai_max_tokens: int = 8000
    ai_temperature: float = 0.3
    ai_timeout: int = 60

    # Retry Configuration
    max_retries: int = 3
    base_backoff: float = 1.0
    rate_limit_backoff: float = 10.0

    # Feature Flags
    enable_ai_enhancement: bool = True
    enable_patch_tracking: bool = True
    dry_run: bool = False

    # Additional settings
    extra: Dict[str, Any] = field(default_factory=dict)


class Container:
    """
    Dependency Injection Container

    Manages creation and wiring of all system dependencies.
    Supports lazy initialization and singleton patterns.
    """

    def __init__(self, config: Optional[ContainerConfig] = None):
        """
        Initialize container with configuration.

        Args:
            config: Container configuration
        """
        self.config = config or ContainerConfig()
        self._instances: Dict[Type, Any] = {}

        # Pre-registered implementations
        self._ai_client: Optional[AIClient] = None
        self._template_engine: Optional[TemplateEngine] = None
        self._prompt_engine: Optional[TemplateEngine] = None
        self._file_system: Optional[FileSystem] = None
        self._code_parser: Optional[CodeParser] = None
        self._retry_policy: Optional[RetryPolicy] = None
        self._code_merger: Optional[CodeMerger] = None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create_production(
        cls,
        api_key: str,
        output_dir: Path,
        template_dir: Optional[Path] = None,
        prompt_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> "Container":
        """
        Create a production container with real implementations.

        Args:
            api_key: Together AI API key
            output_dir: Directory for output files
            template_dir: Directory for Jinja2 templates
            prompt_dir: Directory for AI prompt templates
            **kwargs: Additional configuration options

        Returns:
            Configured Container instance
        """
        # Find default directories if not specified
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        if prompt_dir is None:
            prompt_dir = Path(__file__).parent / "prompt"

        config = ContainerConfig(
            ai_api_key=api_key,
            output_dir=output_dir,
            template_dir=template_dir,
            prompt_dir=prompt_dir,
            **kwargs,
        )

        container = cls(config)

        # Pre-configure with production implementations
        try:
            from together import AsyncTogether
            together_client = AsyncTogether(api_key=api_key)
            container._ai_client = TogetherAIClient(together_client)
        except ImportError:
            logger.warning("together package not installed, AI features disabled")
            container._ai_client = None

        container._file_system = LocalFileSystem(output_dir)
        container._template_engine = JinjaTemplateEngine(template_dir)
        container._prompt_engine = JinjaTemplateEngine(prompt_dir)
        container._code_parser = CompositeCodeParser()
        container._retry_policy = RateLimitAwarePolicy(
            ExponentialBackoffPolicy(
                max_attempts=config.max_retries,
                base_backoff=config.base_backoff,
            ),
            rate_limit_backoff=config.rate_limit_backoff,
        )
        container._code_merger = SafeCodeMergerImpl(container._code_parser)

        return container

    @classmethod
    def create_testing(
        cls,
        templates: Optional[Dict[str, str]] = None,
        ai_responses: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> "Container":
        """
        Create a testing container with mock implementations.

        All operations are in-memory, no disk or network access.

        Args:
            templates: Optional dict of template name -> content
            ai_responses: Optional dict of prompt patterns -> responses
            **kwargs: Additional configuration options

        Returns:
            Configured Container instance
        """
        config = ContainerConfig(
            enable_ai_enhancement=True,
            dry_run=True,
            **kwargs,
        )

        container = cls(config)

        # Configure with mock implementations
        container._ai_client = MockAIClient(
            responses=ai_responses,
            default_response="# AI generated\npass",
        )
        container._file_system = InMemoryFileSystem()
        container._template_engine = InMemoryTemplateEngine(templates or {})
        container._prompt_engine = InMemoryTemplateEngine({})
        container._code_parser = CompositeCodeParser()
        container._retry_policy = NoRetryPolicy()
        container._code_merger = SafeCodeMergerImpl(container._code_parser)

        return container

    @classmethod
    def create_dry_run(
        cls,
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> "Container":
        """
        Create a dry-run container that logs but doesn't execute.

        Useful for testing configuration without side effects.

        Args:
            api_key: Optional API key (won't be used)
            output_dir: Optional output directory
            **kwargs: Additional configuration options

        Returns:
            Configured Container instance
        """
        config = ContainerConfig(
            dry_run=True,
            enable_ai_enhancement=False,
            **kwargs,
        )

        container = cls(config)

        # Use mock AI and in-memory file system
        container._ai_client = MockAIClient(should_fail=True)
        container._file_system = InMemoryFileSystem()

        return container

    # =========================================================================
    # Registration Methods
    # =========================================================================

    def register_ai_client(self, client: AIClient) -> "Container":
        """Register a custom AI client."""
        self._ai_client = client
        return self

    def register_template_engine(self, engine: TemplateEngine) -> "Container":
        """Register a custom template engine for code templates."""
        self._template_engine = engine
        return self

    def register_prompt_engine(self, engine: TemplateEngine) -> "Container":
        """Register a custom template engine for AI prompts."""
        self._prompt_engine = engine
        return self

    def register_file_system(self, fs: FileSystem) -> "Container":
        """Register a custom file system."""
        self._file_system = fs
        return self

    def register_code_parser(self, parser: CodeParser) -> "Container":
        """Register a custom code parser."""
        self._code_parser = parser
        return self

    def register_retry_policy(self, policy: RetryPolicy) -> "Container":
        """Register a custom retry policy."""
        self._retry_policy = policy
        return self

    def register_code_merger(self, merger: CodeMerger) -> "Container":
        """Register a custom code merger."""
        self._code_merger = merger
        return self

    # =========================================================================
    # Getter Methods
    # =========================================================================

    def get_ai_client(self) -> Optional[AIClient]:
        """Get the AI client."""
        return self._ai_client

    def get_template_engine(self) -> TemplateEngine:
        """Get the template engine for code templates."""
        if self._template_engine is None:
            if self.config.template_dir:
                self._template_engine = JinjaTemplateEngine(self.config.template_dir)
            else:
                self._template_engine = InMemoryTemplateEngine()
        return self._template_engine

    def get_prompt_engine(self) -> TemplateEngine:
        """Get the template engine for AI prompts."""
        if self._prompt_engine is None:
            if self.config.prompt_dir:
                self._prompt_engine = JinjaTemplateEngine(self.config.prompt_dir)
            else:
                self._prompt_engine = InMemoryTemplateEngine()
        return self._prompt_engine

    def get_file_system(self) -> FileSystem:
        """Get the file system."""
        if self._file_system is None:
            if self.config.output_dir:
                self._file_system = LocalFileSystem(self.config.output_dir)
            else:
                self._file_system = InMemoryFileSystem()
        return self._file_system

    def get_code_parser(self) -> CodeParser:
        """Get the code parser."""
        if self._code_parser is None:
            self._code_parser = CompositeCodeParser()
        return self._code_parser

    def get_retry_policy(self) -> RetryPolicy:
        """Get the retry policy."""
        if self._retry_policy is None:
            self._retry_policy = RateLimitAwarePolicy(
                ExponentialBackoffPolicy(
                    max_attempts=self.config.max_retries,
                    base_backoff=self.config.base_backoff,
                ),
                rate_limit_backoff=self.config.rate_limit_backoff,
            )
        return self._retry_policy

    def get_code_merger(self) -> CodeMerger:
        """Get the code merger."""
        if self._code_merger is None:
            self._code_merger = SafeCodeMergerImpl(self.get_code_parser())
        return self._code_merger

    # =========================================================================
    # Component Factory Methods
    # =========================================================================

    def get_metadata_manager(self, output_dir: Optional[Path] = None):
        """Get a metadata manager instance."""
        from devdox_ai_locust.utils.metadata_manager import MetadataManager
        return MetadataManager(output_dir or self.config.output_dir or Path.cwd())

    def get_patch_tracker(self, metadata_manager=None):
        """Get a patch tracker instance."""
        from devdox_ai_locust.utils.patch_tracker import PatchTracker
        if metadata_manager is None:
            metadata_manager = self.get_metadata_manager()
        return PatchTracker.from_metadata_manager(metadata_manager)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_ai_available(self) -> bool:
        """Check if AI client is available and configured."""
        return (
            self._ai_client is not None and
            self._ai_client.is_available() and
            self.config.enable_ai_enhancement
        )

    def get_config(self) -> ContainerConfig:
        """Get the container configuration."""
        return self.config

    def __repr__(self) -> str:
        return (
            f"Container("
            f"ai={'available' if self.is_ai_available() else 'unavailable'}, "
            f"dry_run={self.config.dry_run})"
        )
