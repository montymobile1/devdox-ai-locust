"""
OpenAPI 3.x Specification Parser

Parses OpenAPI/Swagger specifications and extracts endpoint information
for automated test generation.
"""

import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

application_json_type = "application/json"
localhost_url = "http://localhost"


class ParameterType(Enum):
    QUERY = "query"
    PATH = "path"
    HEADER = "header"
    COOKIE = "cookie"


@dataclass
class Parameter:
    """Represents an OpenAPI parameter"""

    name: str
    location: ParameterType
    required: bool
    type: str
    description: Optional[str] = None
    example: Optional[Any] = None
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    format: Optional[str] = None


@dataclass
class RequestBody:
    """Represents an OpenAPI request body"""

    content_type: str
    schema: Dict[str, Any]
    required: bool = True
    description: Optional[str] = None
    examples: Optional[Dict[str, Any]] = None


@dataclass
class Response:
    """Represents an OpenAPI response"""

    status_code: str
    description: str
    content_type: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, Any]] = None


@dataclass
class Endpoint:
    """Represents a parsed API endpoint"""

    path: str
    method: str
    operation_id: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    parameters: List[Parameter]
    request_body: Optional[RequestBody]
    responses: List[Response]
    tags: List[str]
    security: Optional[List[Dict[str, Any]]] = None

    def requires_auth(self, global_security: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Determine if this endpoint requires authentication based on OpenAPI security spec.

        OpenAPI security rules:
        - If endpoint has `security: []` (empty array), it's explicitly public
        - If endpoint has `security: null` or missing, it inherits global security
        - If endpoint has non-empty security array, it requires that auth

        Args:
            global_security: The global security requirements from the API spec root

        Returns:
            True if the endpoint requires authentication, False if it's public
        """
        # If endpoint has explicit security defined
        if self.security is not None:
            # Empty array means explicitly public (no auth required)
            if len(self.security) == 0:
                return False
            # Non-empty array means auth is required
            return True

        # Endpoint doesn't have security defined, inherit from global
        if global_security:
            # Global security exists and is non-empty
            return len(global_security) > 0

        # No security defined anywhere - endpoint is public
        return False

    def get_security_schemes(
        self, global_security: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Get the security scheme names required by this endpoint.

        Args:
            global_security: The global security requirements from the API spec root

        Returns:
            List of security scheme names (e.g., ["bearerAuth", "apiKey"])
        """
        security_requirements = self.security
        if security_requirements is None:
            security_requirements = global_security or []

        scheme_names = []
        for req in security_requirements:
            if isinstance(req, dict):
                scheme_names.extend(req.keys())
        return scheme_names


class OpenAPIParser:
    """Parser for OpenAPI 3.x specifications"""

    def __init__(self) -> None:
        self.spec_data: Optional[Dict[str, Any]] = None
        self.components: Optional[Dict[str, Any]] = None

    def parse_schema(self, schema_content: str) -> Dict[str, Any]:
        """
        Parse OpenAPI schema from string content

        Args:
            schema_content: Raw OpenAPI schema as string (JSON or YAML)

        Returns:
            Parsed schema dictionary

        Raises:
            ValueError: If schema is invalid or cannot be parsed
        """
        try:
            # Try parsing as JSON first
            try:
                self.spec_data = json.loads(schema_content)
            except json.JSONDecodeError:
                # If JSON fails, try YAML
                self.spec_data = yaml.safe_load(schema_content)

            # Validate OpenAPI structure
            self._validate_openapi_schema()

            # Store components for reference resolution
            self.components = self.spec_data.get("components", {})

            return self.spec_data

        except Exception as e:
            logger.error(f"Failed to parse OpenAPI schema: {e}")
            raise ValueError(f"Invalid OpenAPI schema: {e}")

    def _validate_openapi_schema(self) -> None:
        """Validate that the schema has required OpenAPI structure"""
        if not self.spec_data:
            raise ValueError("No schema data to validate")

        required_fields = ["openapi", "info", "paths"]
        missing_fields = [
            field for field in required_fields if field not in self.spec_data
        ]

        if missing_fields:
            raise ValueError(f"Missing required OpenAPI fields: {missing_fields}")

        # Check OpenAPI version
        version = self.spec_data.get("openapi", "")
        if not version.startswith("3."):
            raise ValueError(
                f"Unsupported OpenAPI version: {version}. Only 3.x is supported."
            )

    def parse_endpoints(self) -> List[Endpoint]:
        """
        Extract all endpoints from the OpenAPI specification

        Returns:
            List of parsed Endpoint objects
        """
        if not self.spec_data:
            raise ValueError("Schema must be parsed first. Call parse_schema().")

        endpoints = []
        paths = self.spec_data.get("paths", {})

        for path, path_item in paths.items():
            # Skip parameters defined at path level for now
            # (they apply to all operations in the path)
            path_parameters = path_item.get("parameters", [])

            # Process each HTTP method
            http_methods = [
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
                "trace",
            ]

            for method in http_methods:
                if method in path_item:
                    operation = path_item[method]

                    endpoint = Endpoint(
                        path=path,
                        method=method.upper(),
                        operation_id=operation.get("operationId"),
                        summary=operation.get("summary"),
                        description=operation.get("description"),
                        parameters=self._extract_parameters(operation, path_parameters),
                        request_body=self._extract_request_body(operation),
                        responses=self._extract_responses(operation),
                        tags=operation.get("tags", []),
                        security=operation.get("security"),
                    )

                    endpoints.append(endpoint)
        return endpoints

    def _extract_parameters(
        self,
        operation: Dict[str, Any],
        path_parameters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Parameter]:
        """
        Extract parameters from an OpenAPI operation

        Args:
            operation: OpenAPI operation object
            path_parameters: Parameters defined at path level

        Returns:
            List of Parameter objects
        """
        parameters = []

        # Combine path-level and operation-level parameters
        all_params = (path_parameters or []) + operation.get("parameters", [])

        for param in all_params:
            # Resolve reference if needed
            param = self._resolve_reference(param)

            if not param:
                continue

            # Extract parameter type from schema
            param_schema = param.get("schema", {})
            param_type = param_schema.get("type", "string")
            param_format = param_schema.get("format")

            # Handle array types
            if param_type == "array":
                items = param_schema.get("items", {})
                param_type = f"array[{items.get('type', 'string')}]"

            parameter = Parameter(
                name=param.get("name", ""),
                location=ParameterType(param.get("in", "query")),
                required=param.get("required", False),
                type=param_type,
                description=param.get("description"),
                example=param.get("example") or param_schema.get("example"),
                enum=param_schema.get("enum"),
                default=param_schema.get("default"),
                format=param_format,
            )

            parameters.append(parameter)

        return parameters

    def _extract_request_body(self, operation: Dict[str, Any]) -> Optional[RequestBody]:
        """
        Extract request body from an OpenAPI operation

        Args:
            operation: OpenAPI operation object

        Returns:
            RequestBody object or None if no request body
        """
        request_body_def = operation.get("requestBody")
        if not request_body_def:
            return None

        # Resolve reference if needed
        request_body_def = self._resolve_reference(request_body_def)
        if not request_body_def:
            return None

        content = request_body_def.get("content", {})
        if not content:
            return None

        # Get the first content type (prioritize JSON)
        content_types = list(content.keys())
        preferred_types = [
            application_json_type,
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ]

        content_type = None
        for preferred in preferred_types:
            if preferred in content_types:
                content_type = preferred
                break

        if not content_type:
            content_type = content_types[0]

        media_type = content[content_type]
        schema = media_type.get("schema", {})

        # Resolve schema reference if needed
        schema = self._resolve_reference(schema)

        return RequestBody(
            content_type=content_type,
            schema=schema or {},
            required=request_body_def.get("required", True),
            description=request_body_def.get("description"),
            examples=media_type.get("examples"),
        )

    def _extract_responses(self, operation: Dict[str, Any]) -> List[Response]:
        """
        Extract responses from an OpenAPI operation

        Args:
            operation: OpenAPI operation object

        Returns:
            List of Response objects
        """
        responses = []
        responses_def = operation.get("responses", {})

        for status_code, response_def in responses_def.items():
            # Resolve reference if needed
            response_def = self._resolve_reference(response_def)
            if not response_def:
                continue

            # Extract content information
            content = response_def.get("content", {})
            content_type = None
            schema = None

            if content:
                # Prioritize JSON content type
                if application_json_type in content:
                    content_type = application_json_type
                    media_type = content[application_json_type]
                else:
                    content_type = list(content.keys())[0]
                    media_type = content[content_type]

                schema = media_type.get("schema")
                if schema:
                    schema = self._resolve_reference(schema)

            response = Response(
                status_code=status_code,
                description=response_def.get("description", ""),
                content_type=content_type,
                schema=schema,
                headers=response_def.get("headers"),
            )

            responses.append(response)

        return responses

    def _resolve_reference(
        self, obj: Union[Dict[str, Any], str]
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve OpenAPI reference ($ref) to actual object

        Args:
            obj: Object that might contain a $ref

        Returns:
            Resolved object or None if resolution fails
        """
        if not isinstance(obj, dict):
            return None

        ref = obj.get("$ref")
        if not ref:
            return obj

        # Parse reference path (e.g., "#/components/schemas/User")
        if not ref.startswith("#/"):
            logger.warning(f"External references not supported: {ref}")
            return None

        try:
            # Split path and navigate through the spec
            path_parts = ref[2:].split("/")  # Remove "#/" prefix
            current = self.spec_data
            if current is None:
                return None

            for part in path_parts:
                if not isinstance(current, dict):
                    return None
                current = current.get(part)
                if current is None:
                    return None

            return current

        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to resolve reference {ref}: {e}")
            return None

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get basic information about the API

        Returns:
            Dictionary with API information
        """
        if not self.spec_data:
            return {}

        info = self.spec_data.get("info", {})

        return {
            "title": info.get("title", "Unknown API"),
            "version": info.get("version", "Unknown"),
            "description": info.get("description", ""),
            "base_url": self._extract_base_url(),
            "security_schemes": self._extract_security_schemes(),
            "global_security": self._extract_global_security(),
        }

    def _extract_global_security(self) -> List[Dict[str, Any]]:
        """
        Extract global security requirements from the root of the OpenAPI spec.

        Global security applies to all operations unless overridden at the operation level.
        An empty array [] at the operation level means the endpoint is public.

        Returns:
            List of security requirement objects
        """
        if not isinstance(self.spec_data, dict):
            return []

        security = self.spec_data.get("security", [])
        if isinstance(security, list):
            return security
        return []

    def _extract_base_url(self) -> str:
        """Extract base URL from servers section"""
        if not isinstance(self.spec_data, dict):
            return localhost_url

        servers = self.spec_data.get("servers", [])
        if servers and isinstance(servers[0], dict):
            url = servers[0].get("url", localhost_url)
            if isinstance(url, str):
                return url
        return localhost_url

    def _extract_security_schemes(self) -> Dict[str, Any]:
        """Extract security schemes from components"""
        if not isinstance(self.components, dict):
            return {}

        schemes = self.components.get("securitySchemes", {})
        if isinstance(schemes, dict):
            return schemes
        return {}
