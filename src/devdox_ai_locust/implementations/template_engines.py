"""
Template Engine Implementations

Production and testing implementations of the TemplateEngine protocol.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound

from devdox_ai_locust.abstractions.template_engine import (
    TemplateEngine,
    TemplateNotFoundError,
    TemplateRenderError,
)

logger = logging.getLogger(__name__)


class JinjaTemplateEngine:
    """
    Production template engine using Jinja2 with FileSystemLoader.
    """

    def __init__(self, template_dir: Path, **jinja_options: Any):
        """
        Initialize Jinja2 environment.

        Args:
            template_dir: Directory containing template files
            **jinja_options: Additional options for Jinja2 Environment
        """
        self.template_dir = Path(template_dir)

        default_options = {
            "trim_blocks": True,
            "lstrip_blocks": True,
            "keep_trailing_newline": True,
            "autoescape": False,
        }
        default_options.update(jinja_options)

        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            **default_options,
        )

    def render(self, template_name: str, **context: Any) -> str:
        """Render a template with context."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound:
            raise TemplateNotFoundError(template_name)
        except Exception as e:
            raise TemplateRenderError(template_name, str(e))

    def has_template(self, template_name: str) -> bool:
        """Check if template exists."""
        try:
            self.env.get_template(template_name)
            return True
        except TemplateNotFound:
            return False

    def list_templates(self) -> List[str]:
        """List all available templates."""
        return self.env.list_templates()


class DictLoader(BaseLoader):
    """Jinja2 loader that loads templates from a dict."""

    def __init__(self, templates: Dict[str, str]):
        self.templates = templates

    def get_source(self, environment: Environment, template: str):
        if template not in self.templates:
            raise TemplateNotFound(template)
        source = self.templates[template]
        return source, template, lambda: True

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())


class InMemoryTemplateEngine:
    """
    In-memory template engine for testing.

    Templates are stored in a dictionary instead of on the file system.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, str]] = None,
        **jinja_options: Any,
    ):
        """
        Initialize with in-memory templates.

        Args:
            templates: Dict mapping template names to template content
            **jinja_options: Additional options for Jinja2 Environment
        """
        self.templates = templates or {}

        default_options = {
            "trim_blocks": True,
            "lstrip_blocks": True,
            "keep_trailing_newline": True,
            "autoescape": False,
        }
        default_options.update(jinja_options)

        self.env = Environment(
            loader=DictLoader(self.templates),
            **default_options,
        )

    def add_template(self, name: str, content: str) -> None:
        """Add or update a template."""
        self.templates[name] = content
        # Recreate environment with updated templates
        self.env = Environment(
            loader=DictLoader(self.templates),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: Any) -> str:
        """Render a template with context."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound:
            raise TemplateNotFoundError(template_name)
        except Exception as e:
            raise TemplateRenderError(template_name, str(e))

    def has_template(self, template_name: str) -> bool:
        """Check if template exists."""
        return template_name in self.templates

    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())


class PassthroughTemplateEngine:
    """
    Simple template engine that returns templates without rendering.

    Useful for testing when you just need the raw template content.
    """

    def __init__(self, templates: Optional[Dict[str, str]] = None):
        """
        Initialize with templates.

        Args:
            templates: Dict mapping template names to content
        """
        self.templates = templates or {}

    def render(self, template_name: str, **context: Any) -> str:
        """Return template content directly (no rendering)."""
        if template_name not in self.templates:
            raise TemplateNotFoundError(template_name)
        return self.templates[template_name]

    def has_template(self, template_name: str) -> bool:
        """Check if template exists."""
        return template_name in self.templates

    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())
