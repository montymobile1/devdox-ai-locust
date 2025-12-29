"""
Template Engine Protocol

Defines the contract for template rendering implementations.
Allows swapping between Jinja2, in-memory templates for testing,
or other template engines.
"""

from typing import Protocol, Any, Dict, List, Optional, runtime_checkable


@runtime_checkable
class TemplateEngine(Protocol):
    """
    Protocol for template engine implementations.

    Implementations:
        - JinjaTemplateEngine: Uses Jinja2 with FileSystemLoader
        - InMemoryTemplateEngine: Uses dict of templates (for testing)
        - CachingTemplateEngine: Wraps another engine with caching

    Example:
        class JinjaTemplateEngine:
            def __init__(self, template_dir: Path):
                self.env = Environment(loader=FileSystemLoader(str(template_dir)))

            def render(self, template_name: str, **context: Any) -> str:
                template = self.env.get_template(template_name)
                return template.render(**context)
    """

    def render(self, template_name: str, **context: Any) -> str:
        """
        Render a template with the given context.

        Args:
            template_name: Name of the template file (e.g., "locust.j2")
            **context: Variables to pass to the template

        Returns:
            Rendered template as string

        Raises:
            TemplateNotFoundError: If template doesn't exist
            TemplateRenderError: If rendering fails
        """
        ...

    def has_template(self, template_name: str) -> bool:
        """
        Check if a template exists.

        Args:
            template_name: Name of the template file

        Returns:
            True if template exists
        """
        ...

    def list_templates(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List of template names
        """
        ...


class TemplateError(Exception):
    """Base exception for template errors"""
    pass


class TemplateNotFoundError(TemplateError):
    """Template file not found"""

    def __init__(self, template_name: str):
        super().__init__(f"Template not found: {template_name}")
        self.template_name = template_name


class TemplateRenderError(TemplateError):
    """Template rendering failed"""

    def __init__(self, template_name: str, message: str):
        super().__init__(f"Failed to render {template_name}: {message}")
        self.template_name = template_name
