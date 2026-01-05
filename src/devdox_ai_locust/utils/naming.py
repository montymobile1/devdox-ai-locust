"""
Centralized Naming Strategy Module

Provides consistent naming transformations for:
- Workflow module names (python-safe snake_case)
- Workflow filenames
- TaskMethods class names (PascalCase)
- Method names from endpoints
- Path parameter variable names

All naming decisions should flow through this module to ensure consistency
between generated imports, class names, and file names.
"""

import re
import keyword
from typing import Optional, Set, Protocol
from dataclasses import dataclass, field


class NamingStrategy(Protocol):
    """Protocol for naming strategy implementations"""

    def to_workflow_module(self, group_label: str) -> str:
        """Convert group label to workflow module name (e.g., 'authentication')"""
        ...

    def to_workflow_filename(self, group_label: str) -> str:
        """Convert group label to workflow filename (e.g., 'authentication_workflow.py')"""
        ...

    def to_task_methods_class(self, group_label: str) -> str:
        """Convert group label to TaskMethods class name (e.g., 'AuthenticationTaskMethods')"""
        ...

    def to_method_name(
        self, operation_id: Optional[str], method: str, path: str
    ) -> str:
        """Generate valid Python method name from endpoint info"""
        ...

    def to_param_var(self, raw_param_name: str) -> str:
        """Convert raw parameter name to valid Python variable name"""
        ...


@dataclass
class DefaultNamingStrategy:
    """
    Default implementation of naming strategy.

    Ensures all generated names are valid Python identifiers and
    consistent across the codebase.
    """

    _used_method_names: Set[str] = field(default_factory=set)

    def _sanitize_identifier(self, name: str) -> str:
        """
        Convert any string to a valid Python identifier.

        - Replaces non-alphanumeric characters with underscores
        - Ensures starts with letter or underscore
        - Removes consecutive underscores
        - Converts to lowercase
        """
        if not name:
            return "unnamed"

        # Replace all non-word characters with underscores
        sanitized = re.sub(r"[^\w]", "_", name)

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Strip leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Convert to lowercase
        sanitized = sanitized.lower()

        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"n_{sanitized}"

        # Handle empty result
        if not sanitized:
            sanitized = "unnamed"

        # Handle Python keywords
        if keyword.iskeyword(sanitized):
            sanitized = f"{sanitized}_"

        return sanitized

    def _to_pascal_case(self, name: str) -> str:
        """
        Convert a string to PascalCase.

        Examples:
            'user_management' -> 'UserManagement'
            'api-key' -> 'ApiKey'
            'git_tokens' -> 'GitTokens'
        """
        if not name:
            return "Unnamed"

        # First sanitize to get consistent word boundaries
        sanitized = re.sub(r"[^\w]", "_", name)

        # Split on underscores and capitalize each word
        words = [w for w in sanitized.split("_") if w]

        if not words:
            return "Unnamed"

        # Capitalize each word
        pascal = "".join(word.capitalize() for word in words)

        # Ensure starts with letter
        if pascal and pascal[0].isdigit():
            pascal = f"N{pascal}"

        return pascal

    def to_workflow_module(self, group_label: str) -> str:
        """
        Convert group label to workflow module name.

        Examples:
            'User Management' -> 'user_management'
            'api-key' -> 'api_key'
            'GitTokens' -> 'gittokens'
        """
        return self._sanitize_identifier(group_label)

    def to_workflow_filename(self, group_label: str) -> str:
        """
        Convert group label to workflow filename.

        Examples:
            'User Management' -> 'user_management_workflow.py'
            'api-key' -> 'api_key_workflow.py'
        """
        module_name = self.to_workflow_module(group_label)
        return f"{module_name}_workflow.py"

    def to_task_methods_class(self, group_label: str) -> str:
        """
        Convert group label to TaskMethods class name.

        Examples:
            'User Management' -> 'UserManagementTaskMethods'
            'api-key' -> 'ApiKeyTaskMethods'
            'authentication' -> 'AuthenticationTaskMethods'
        """
        pascal_name = self._to_pascal_case(group_label)
        return f"{pascal_name}TaskMethods"

    def to_api_user_class(self, group_label: str) -> str:
        """
        Convert group label to APIUser class name.

        Examples:
            'User Management' -> 'UserManagementAPIUser'
            'api-key' -> 'ApiKeyAPIUser'
        """
        pascal_name = self._to_pascal_case(group_label)
        return f"{pascal_name}APIUser"

    def to_method_name(
        self,
        operation_id: Optional[str],
        method: str,
        path: str,
        ensure_unique: bool = True,
    ) -> str:
        """
        Generate valid Python method name from endpoint info.

        Args:
            operation_id: OpenAPI operationId if available
            method: HTTP method (GET, POST, etc.)
            path: API path (/api/v1/users/{id})
            ensure_unique: If True, appends suffix if name was already used

        Returns:
            Valid Python method name
        """
        if operation_id:
            base_name = self._sanitize_identifier(operation_id)
        else:
            # Generate from method and path
            path_parts = [
                part for part in path.split("/") if part and not part.startswith("{")
            ]
            base_name = f"{method.lower()}_{'_'.join(path_parts)}"
            base_name = self._sanitize_identifier(base_name)

        if not base_name:
            base_name = f"{method.lower()}_endpoint"

        # Ensure uniqueness if requested
        if ensure_unique:
            final_name = base_name
            counter = 2
            while final_name in self._used_method_names:
                final_name = f"{base_name}_{counter}"
                counter += 1
            self._used_method_names.add(final_name)
            return final_name

        return base_name

    def to_param_var(self, raw_param_name: str) -> str:
        """
        Convert raw parameter name to valid Python variable name.

        Examples:
            'user-id' -> 'user_id'
            'api_key' -> 'api_key'
            'Content-Type' -> 'content_type'
            '123abc' -> 'n_123abc'
        """
        return self._sanitize_identifier(raw_param_name)

    def to_path_with_safe_params(self, path: str, params: list) -> str:
        """
        Convert path with parameters to use safe Python variable names.

        Args:
            path: Original path like '/users/{user-id}/posts/{post-id}'
            params: List of parameter objects with 'name' attribute

        Returns:
            Path with sanitized parameter names like '/users/{user_id}/posts/{post_id}'
        """
        result = path
        for param in params:
            if hasattr(param, "name") and hasattr(param, "location"):
                if param.location.value == "path":
                    safe_name = self.to_param_var(param.name)
                    result = result.replace(f"{{{param.name}}}", f"{{{safe_name}}}")
        return result

    def reset_used_names(self) -> None:
        """Reset the set of used method names (call between workflows)"""
        self._used_method_names.clear()


# Global default instance for convenience
default_naming = DefaultNamingStrategy()


def to_workflow_module(group_label: str) -> str:
    """Convenience function using default naming strategy"""
    return default_naming.to_workflow_module(group_label)


def to_workflow_filename(group_label: str) -> str:
    """Convenience function using default naming strategy"""
    return default_naming.to_workflow_filename(group_label)


def to_task_methods_class(group_label: str) -> str:
    """Convenience function using default naming strategy"""
    return default_naming.to_task_methods_class(group_label)


def to_api_user_class(group_label: str) -> str:
    """Convenience function using default naming strategy"""
    return default_naming.to_api_user_class(group_label)


def to_method_name(
    operation_id: Optional[str],
    method: str,
    path: str,
    ensure_unique: bool = False,
) -> str:
    """Convenience function using default naming strategy"""
    return default_naming.to_method_name(operation_id, method, path, ensure_unique)


def to_param_var(raw_param_name: str) -> str:
    """Convenience function using default naming strategy"""
    return default_naming.to_param_var(raw_param_name)
