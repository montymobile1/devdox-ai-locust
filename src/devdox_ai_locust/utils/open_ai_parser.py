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
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


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
    operation_id: str  # Required - generated from method+path if not in spec
    summary: Optional[str]
    description: Optional[str]
    parameters: List[Parameter]
    request_body: Optional[RequestBody]
    responses: List[Response]
    tags: List[str]
    security: Optional[List[Dict[str, Any]]] = None


class OpenAPIParser:
    """Parser for OpenAPI 3.x specifications"""

    def __init__(self) -> None:
        self.spec_data: Optional[Dict[str, Any]] = None
        self.components: Optional[Dict[str, Any]] = None

    def _generate_operation_id(self, method: str, path: str) -> str:
        """
        Generate operation_id from HTTP method and path.

        This is called when operationId is not present in the OpenAPI spec.
        Produces a valid Python identifier like 'get_users_userId'.

        Args:
            method: HTTP method (e.g., 'GET', 'POST')
            path: API path (e.g., '/users/{userId}')

        Returns:
            A valid Python identifier for the operation
        """
        import re
        # Remove path parameter braces and leading slash
        path_parts = path.strip("/").replace("{", "").replace("}", "")
        # Combine method and path
        raw_id = f"{method.lower()}_{path_parts.replace('/', '_')}"
        # Sanitize: replace common separators with underscores
        raw_id = raw_id.replace("-", "_").replace(".", "_")
        # Remove non-alphanumeric chars except underscore
        raw_id = re.sub(r'[^a-zA-Z0-9_]', '', raw_id)
        # Remove consecutive underscores
        raw_id = re.sub(r'_+', '_', raw_id)
        # Remove leading/trailing underscores
        raw_id = raw_id.strip('_')
        # Ensure doesn't start with a number (shouldn't happen with method prefix)
        if raw_id and raw_id[0].isdigit():
            raw_id = f"n{raw_id}"
        return raw_id or f"{method.lower()}_endpoint"

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

                    # Get operation_id from spec, or generate from method+path
                    operation_id = operation.get("operationId")
                    if not operation_id:
                        operation_id = self._generate_operation_id(method.upper(), path)
                        logger.debug(
                            f"Generated operation_id '{operation_id}' for {method.upper()} {path}"
                        )

                    endpoint = Endpoint(
                        path=path,
                        method=method.upper(),
                        operation_id=operation_id,
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

            # Unwrap OpenAPI 3.1 anyOf nullable pattern
            effective_schema = param_schema
            any_of = param_schema.get("anyOf") or param_schema.get("oneOf")
            if any_of and isinstance(any_of, list):
                # Resolve $refs within anyOf variants before filtering
                resolved_variants = []
                for v in any_of:
                    if isinstance(v, dict) and "$ref" in v:
                        resolved_variants.append(self._resolve_reference(v) or v)
                    elif isinstance(v, dict):
                        resolved_variants.append(v)
                real_variants = [v for v in resolved_variants if v.get("type") != "null"]
                if len(real_variants) == 1:
                    effective_schema = real_variants[0]

            param_type = effective_schema.get("type", "string")
            param_format = effective_schema.get("format")

            # Handle array types
            if param_type == "array":
                items = effective_schema.get("items", {})
                param_type = f"array[{items.get('type', 'string')}]"

            parameter = Parameter(
                name=param.get("name", ""),
                location=ParameterType(param.get("in", "query")),
                required=param.get("required", False),
                type=param_type,
                description=param.get("description"),
                example=param.get("example") or effective_schema.get("example"),
                enum=effective_schema.get("enum"),
                default=effective_schema.get("default"),
                format=param_format,
                pattern=effective_schema.get("pattern"),
                min_length=effective_schema.get("minLength"),
                max_length=effective_schema.get("maxLength"),
                minimum=effective_schema.get("minimum"),
                maximum=effective_schema.get("maximum"),
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

        # Recursively resolve all $refs in the schema
        schema = self._resolve_schema_deep(schema)

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
                    schema = self._resolve_schema_deep(schema)

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

    def _resolve_schema_deep(
        self, schema: Optional[Dict[str, Any]], _ancestors: Optional[set] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Recursively resolve all $ref references in a schema tree.

        Uses ancestry tracking to detect circular references. When a circular
        reference is detected, the schema at that point is returned with only
        its direct properties resolved (no further recursion into the cycle).

        Args:
            schema: Schema dict that may contain $ref at any level
            _ancestors: Set of $ref paths seen in the current resolution chain

        Returns:
            Fully resolved schema dict, or None if resolution fails
        """
        if not isinstance(schema, dict):
            return schema

        if _ancestors is None:
            _ancestors = set()

        # If this schema is a $ref, resolve it
        ref = schema.get("$ref")
        if ref:
            if ref in _ancestors:
                # Circular reference detected - return the resolved target
                # without further recursion (one level of fields only)
                resolved = self._resolve_reference(schema)
                if not resolved:
                    return None
                # Return a shallow copy without recursing into its nested refs
                result = dict(resolved)
                result.pop("$ref", None)
                return result

            # Track this ref in ancestry
            new_ancestors = _ancestors | {ref}
            resolved = self._resolve_reference(schema)
            if not resolved:
                return None
            # Continue resolving the resolved schema
            return self._resolve_schema_deep(resolved, new_ancestors)

        # Not a $ref - recursively resolve nested structures
        result = dict(schema)

        # Resolve properties
        if "properties" in result and isinstance(result["properties"], dict):
            resolved_props = {}
            for prop_name, prop_schema in result["properties"].items():
                resolved_props[prop_name] = self._resolve_schema_deep(
                    prop_schema, _ancestors
                ) or prop_schema
            result["properties"] = resolved_props

        # Resolve additionalProperties
        if "additionalProperties" in result and isinstance(
            result["additionalProperties"], dict
        ):
            result["additionalProperties"] = self._resolve_schema_deep(
                result["additionalProperties"], _ancestors
            ) or result["additionalProperties"]

        # Resolve items (for arrays)
        if "items" in result and isinstance(result["items"], dict):
            result["items"] = self._resolve_schema_deep(
                result["items"], _ancestors
            ) or result["items"]

        # Resolve allOf, oneOf, anyOf
        for keyword in ("allOf", "oneOf", "anyOf"):
            if keyword in result and isinstance(result[keyword], list):
                resolved_list = []
                for item in result[keyword]:
                    resolved_item = self._resolve_schema_deep(item, _ancestors)
                    resolved_list.append(resolved_item or item)
                result[keyword] = resolved_list

        return result

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
        }

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
