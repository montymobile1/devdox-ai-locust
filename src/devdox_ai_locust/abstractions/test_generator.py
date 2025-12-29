"""
Test Generator Protocol

Defines the contract for Locust test generation implementations.
Separates template-based and AI-enhanced generation strategies.
"""

from typing import Protocol, Optional, List, Dict, Any, runtime_checkable
from pathlib import Path
from pydantic import BaseModel, Field

from devdox_ai_locust.utils.open_ai_parser import Endpoint


class GenerationContext(BaseModel):
    """Context for test generation"""
    endpoints: List[Any] = Field(default_factory=list)  # List[Endpoint]
    api_info: Dict[str, Any] = Field(default_factory=dict)
    target_host: Optional[str] = None
    include_auth: bool = True
    db_type: str = ""
    custom_requirement: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class GeneratedFile(BaseModel):
    """A single generated file"""
    filename: str
    content: str
    is_workflow: bool = False
    group: Optional[str] = None  # Workflow group if applicable


class GenerationResult(BaseModel):
    """Result of test generation"""
    success: bool
    main_files: Dict[str, str] = Field(default_factory=dict)  # filename -> content
    workflow_files: List[Dict[str, str]] = Field(default_factory=list)  # [{filename: content}]
    grouped_endpoints: Dict[str, List[Any]] = Field(default_factory=dict)
    enhancements_applied: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0

    class Config:
        arbitrary_types_allowed = True


@runtime_checkable
class TestGenerator(Protocol):
    """
    Protocol for test generation implementations.

    Implementations:
        - TemplateTestGenerator: Uses Jinja2 templates only
        - AIEnhancedTestGenerator: Combines templates + AI
        - MockTestGenerator: Returns predefined output (for testing)

    Example:
        class TemplateTestGenerator:
            def __init__(
                self,
                template_engine: TemplateEngine,
                config: TestDataConfig,
            ):
                self.template_engine = template_engine
                self.config = config

            async def generate(self, context: GenerationContext) -> GenerationResult:
                # Generate using templates
                locustfile = self.template_engine.render(
                    "locust.py.j2",
                    endpoints=context.endpoints,
                    api_info=context.api_info,
                )
                return GenerationResult(
                    success=True,
                    main_files={"locustfile.py": locustfile},
                )
    """

    async def generate(self, context: GenerationContext) -> GenerationResult:
        """
        Generate Locust test files.

        Args:
            context: Generation context with endpoints and options

        Returns:
            GenerationResult with generated files
        """
        ...

    def validate_output(self, result: GenerationResult) -> bool:
        """
        Validate generated output.

        Args:
            result: Generation result to validate

        Returns:
            True if all generated files are valid Python
        """
        ...


class EnhancementStrategy(Protocol):
    """
    Protocol for individual enhancement strategies.

    Each strategy handles one type of enhancement:
    - Locustfile enhancement
    - Test data enhancement
    - Validation utilities enhancement
    - Domain flows generation
    - Workflow enhancement
    """

    async def enhance(
        self,
        base_content: str,
        context: GenerationContext,
    ) -> tuple[str, List[str]]:
        """
        Apply enhancement to base content.

        Args:
            base_content: Original generated content
            context: Generation context

        Returns:
            Tuple of (enhanced_content, list of enhancements applied)
        """
        ...

    def is_applicable(self, context: GenerationContext) -> bool:
        """
        Check if this enhancement is applicable.

        Args:
            context: Generation context

        Returns:
            True if enhancement should be applied
        """
        ...


class GenerationError(Exception):
    """Base exception for generation errors"""
    pass


class TemplateGenerationError(GenerationError):
    """Template generation failed"""

    def __init__(self, template: str, message: str):
        super().__init__(f"Template {template} failed: {message}")
        self.template = template


class EnhancementError(GenerationError):
    """Enhancement failed"""

    def __init__(self, enhancement: str, message: str):
        super().__init__(f"Enhancement {enhancement} failed: {message}")
        self.enhancement = enhancement
