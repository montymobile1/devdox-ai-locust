"""
Comprehensive tests for open_ai_parser.py module.

Tests cover:
- ParameterType enum
- Parameter, RequestBody, Response, Endpoint dataclasses
- OpenAPIParser class initialization
- Schema parsing (JSON and YAML)
- Schema validation
- Endpoint extraction
- Parameter extraction
- Request body extraction
- Response extraction
- Reference resolution
- Schema info extraction
"""

import pytest
import json
from unittest.mock import patch

from devdox_ai_locust.utils.open_ai_parser import (
    ParameterType,
    Parameter,
    RequestBody,
    Response,
    Endpoint,
    OpenAPIParser,
    application_json_type,
    localhost_url,
)


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_application_json_type(self):
        """application_json_type should be 'application/json'."""
        assert application_json_type == "application/json"

    def test_localhost_url(self):
        """localhost_url should be 'http://localhost'."""
        assert localhost_url == "http://localhost"


# =============================================================================
# ParameterType Enum Tests
# =============================================================================


class TestParameterType:
    """Tests for ParameterType enum."""

    def test_query_value(self):
        """QUERY should have value 'query'."""
        assert ParameterType.QUERY.value == "query"

    def test_path_value(self):
        """PATH should have value 'path'."""
        assert ParameterType.PATH.value == "path"

    def test_header_value(self):
        """HEADER should have value 'header'."""
        assert ParameterType.HEADER.value == "header"

    def test_cookie_value(self):
        """COOKIE should have value 'cookie'."""
        assert ParameterType.COOKIE.value == "cookie"

    def test_enum_count(self):
        """Should have exactly 4 parameter types."""
        assert len(ParameterType) == 4

    def test_create_from_value(self):
        """Should be able to create from string value."""
        assert ParameterType("query") == ParameterType.QUERY
        assert ParameterType("path") == ParameterType.PATH


# =============================================================================
# Parameter Dataclass Tests
# =============================================================================


class TestParameterDataclass:
    """Tests for Parameter dataclass."""

    def test_required_fields(self):
        """Should require name, location, required, and type fields."""
        param = Parameter(
            name="user_id",
            location=ParameterType.PATH,
            required=True,
            type="integer",
        )
        assert param.name == "user_id"
        assert param.location == ParameterType.PATH
        assert param.required is True
        assert param.type == "integer"

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        param = Parameter(
            name="test",
            location=ParameterType.QUERY,
            required=False,
            type="string",
        )
        assert param.description is None
        assert param.example is None
        assert param.enum is None
        assert param.default is None
        assert param.format is None

    def test_all_fields(self):
        """Should accept all fields."""
        param = Parameter(
            name="status",
            location=ParameterType.QUERY,
            required=False,
            type="string",
            description="Filter by status",
            example="active",
            enum=["active", "inactive"],
            default="active",
            format="enum",
        )
        assert param.description == "Filter by status"
        assert param.example == "active"
        assert param.enum == ["active", "inactive"]
        assert param.default == "active"
        assert param.format == "enum"


# =============================================================================
# RequestBody Dataclass Tests
# =============================================================================


class TestRequestBodyDataclass:
    """Tests for RequestBody dataclass."""

    def test_required_fields(self):
        """Should require content_type and schema fields."""
        body = RequestBody(
            content_type="application/json",
            schema={"type": "object"},
        )
        assert body.content_type == "application/json"
        assert body.schema == {"type": "object"}

    def test_required_defaults_true(self):
        """required field should default to True."""
        body = RequestBody(
            content_type="application/json",
            schema={},
        )
        assert body.required is True

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        body = RequestBody(
            content_type="application/json",
            schema={},
        )
        assert body.description is None
        assert body.examples is None

    def test_all_fields(self):
        """Should accept all fields."""
        body = RequestBody(
            content_type="application/json",
            schema={"type": "object"},
            required=False,
            description="User data",
            examples={"example1": {"value": {}}},
        )
        assert body.required is False
        assert body.description == "User data"
        assert body.examples == {"example1": {"value": {}}}


# =============================================================================
# Response Dataclass Tests
# =============================================================================


class TestResponseDataclass:
    """Tests for Response dataclass."""

    def test_required_fields(self):
        """Should require status_code and description fields."""
        response = Response(
            status_code="200",
            description="Successful response",
        )
        assert response.status_code == "200"
        assert response.description == "Successful response"

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        response = Response(
            status_code="200",
            description="OK",
        )
        assert response.content_type is None
        assert response.schema is None
        assert response.headers is None

    def test_all_fields(self):
        """Should accept all fields."""
        response = Response(
            status_code="200",
            description="OK",
            content_type="application/json",
            schema={"type": "object"},
            headers={"X-Rate-Limit": {"schema": {"type": "integer"}}},
        )
        assert response.content_type == "application/json"
        assert response.schema == {"type": "object"}
        assert response.headers is not None


# =============================================================================
# Endpoint Dataclass Tests
# =============================================================================


class TestEndpointDataclass:
    """Tests for Endpoint dataclass."""

    def test_required_fields(self):
        """Should require all main fields."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="Get users",
            description="Returns all users",
            parameters=[],
            request_body=None,
            responses=[],
            tags=["users"],
        )
        assert endpoint.path == "/users"
        assert endpoint.method == "GET"
        assert endpoint.operation_id == "getUsers"

    def test_security_defaults_none(self):
        """security field should default to None."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        assert endpoint.security is None

    def test_with_security(self):
        """Should accept security field."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[{"bearerAuth": []}],
        )
        assert endpoint.security == [{"bearerAuth": []}]


# =============================================================================
# Endpoint.requires_auth Tests
# =============================================================================


class TestEndpointRequiresAuth:
    """Tests for Endpoint.requires_auth method."""

    def test_returns_true_with_security_defined(self):
        """Should return True when endpoint has security defined."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[{"bearerAuth": []}],
        )
        assert endpoint.requires_auth() is True

    def test_returns_false_with_empty_security(self):
        """Should return False when security is empty array (explicitly public)."""
        endpoint = Endpoint(
            path="/health",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[],  # Explicitly public
        )
        assert endpoint.requires_auth() is False

    def test_returns_false_when_no_security(self):
        """Should return False when no security defined and no global security."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=None,
        )
        assert endpoint.requires_auth() is False

    def test_inherits_global_security(self):
        """Should inherit global security when endpoint has no security."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=None,  # Will inherit global
        )
        global_security = [{"bearerAuth": []}]
        assert endpoint.requires_auth(global_security) is True

    def test_empty_security_overrides_global(self):
        """Should override global security when endpoint has empty security."""
        endpoint = Endpoint(
            path="/health",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[],  # Explicitly public, overrides global
        )
        global_security = [{"bearerAuth": []}]
        assert endpoint.requires_auth(global_security) is False

    def test_endpoint_security_takes_precedence(self):
        """Endpoint-level security should take precedence over global."""
        endpoint = Endpoint(
            path="/admin",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[{"adminAuth": []}],  # Different from global
        )
        global_security = [{"bearerAuth": []}]
        # Should require auth (from endpoint, not global)
        assert endpoint.requires_auth(global_security) is True


# =============================================================================
# Endpoint.get_security_schemes Tests
# =============================================================================


class TestEndpointGetSecuritySchemes:
    """Tests for Endpoint.get_security_schemes method."""

    def test_returns_scheme_names(self):
        """Should return list of security scheme names."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[{"bearerAuth": []}, {"apiKey": []}],
        )
        schemes = endpoint.get_security_schemes()
        assert "bearerAuth" in schemes
        assert "apiKey" in schemes

    def test_returns_empty_for_public(self):
        """Should return empty list for public endpoints."""
        endpoint = Endpoint(
            path="/health",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=[],
        )
        assert endpoint.get_security_schemes() == []

    def test_inherits_global_schemes(self):
        """Should return global schemes when endpoint has no security."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
            security=None,
        )
        global_security = [{"bearerAuth": []}]
        schemes = endpoint.get_security_schemes(global_security)
        assert schemes == ["bearerAuth"]


# =============================================================================
# OpenAPIParser Initialization Tests
# =============================================================================


class TestOpenAPIParserInit:
    """Tests for OpenAPIParser initialization."""

    def test_init_spec_data_none(self):
        """spec_data should be None initially."""
        parser = OpenAPIParser()
        assert parser.spec_data is None

    def test_init_components_none(self):
        """components should be None initially."""
        parser = OpenAPIParser()
        assert parser.components is None


# =============================================================================
# OpenAPIParser.parse_schema Tests
# =============================================================================


class TestParseSchemaJSON:
    """Tests for parse_schema with JSON input."""

    def test_parses_valid_json(self):
        """Should parse valid JSON schema."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0.0"},
                "paths": {},
            }
        )
        result = parser.parse_schema(schema)
        assert result["openapi"] == "3.0.0"

    def test_stores_spec_data(self):
        """Should store parsed data in spec_data."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0.0"},
                "paths": {},
            }
        )
        parser.parse_schema(schema)
        assert parser.spec_data is not None
        assert parser.spec_data["openapi"] == "3.0.0"

    def test_stores_components(self):
        """Should store components from schema."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0.0"},
                "paths": {},
                "components": {
                    "schemas": {"User": {"type": "object"}},
                },
            }
        )
        parser.parse_schema(schema)
        assert parser.components is not None
        assert "schemas" in parser.components


class TestParseSchemaYAML:
    """Tests for parse_schema with YAML input."""

    def test_parses_valid_yaml(self):
        """Should parse valid YAML schema."""
        parser = OpenAPIParser()
        schema = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
paths: {}
"""
        result = parser.parse_schema(schema)
        assert result["openapi"] == "3.0.0"

    def test_parses_yaml_when_json_fails(self):
        """Should fall back to YAML when JSON parsing fails."""
        parser = OpenAPIParser()
        schema = """
openapi: '3.0.0'
info:
  title: Test API
  version: '1.0.0'
paths:
  /users:
    get:
      summary: Get users
"""
        result = parser.parse_schema(schema)
        assert "/users" in result["paths"]


class TestParseSchemaValidation:
    """Tests for schema validation in parse_schema."""

    def test_raises_for_missing_openapi(self):
        """Should raise ValueError when openapi field is missing."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "info": {"title": "Test", "version": "1.0"},
                "paths": {},
            }
        )
        with pytest.raises(ValueError) as exc_info:
            parser.parse_schema(schema)
        assert "Missing required OpenAPI fields" in str(exc_info.value)

    def test_raises_for_missing_info(self):
        """Should raise ValueError when info field is missing."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "3.0.0",
                "paths": {},
            }
        )
        with pytest.raises(ValueError) as exc_info:
            parser.parse_schema(schema)
        assert "Missing required OpenAPI fields" in str(exc_info.value)

    def test_raises_for_missing_paths(self):
        """Should raise ValueError when paths field is missing."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "3.0.0",
                "info": {"title": "Test", "version": "1.0"},
            }
        )
        with pytest.raises(ValueError) as exc_info:
            parser.parse_schema(schema)
        assert "Missing required OpenAPI fields" in str(exc_info.value)

    def test_raises_for_unsupported_version(self):
        """Should raise ValueError for OpenAPI 2.x (Swagger)."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "2.0",
                "info": {"title": "Test", "version": "1.0"},
                "paths": {},
            }
        )
        with pytest.raises(ValueError) as exc_info:
            parser.parse_schema(schema)
        assert "Unsupported OpenAPI version" in str(exc_info.value)

    def test_accepts_openapi_31(self):
        """Should accept OpenAPI 3.1.x."""
        parser = OpenAPIParser()
        schema = json.dumps(
            {
                "openapi": "3.1.0",
                "info": {"title": "Test", "version": "1.0"},
                "paths": {},
            }
        )
        result = parser.parse_schema(schema)
        assert result["openapi"] == "3.1.0"


class TestParseSchemaErrors:
    """Tests for error handling in parse_schema."""

    def test_raises_for_invalid_json_and_yaml(self):
        """Should raise ValueError for invalid content."""
        parser = OpenAPIParser()
        with pytest.raises(ValueError):
            parser.parse_schema("not valid { json or yaml")

    def test_logs_error_on_failure(self):
        """Should log error when parsing fails."""
        parser = OpenAPIParser()
        with patch("devdox_ai_locust.utils.open_ai_parser.logger") as mock_logger:
            with pytest.raises(ValueError):
                parser.parse_schema("invalid content")
            mock_logger.error.assert_called_once()


# =============================================================================
# OpenAPIParser.parse_endpoints Tests
# =============================================================================


class TestParseEndpointsBasic:
    """Basic tests for parse_endpoints method."""

    def test_raises_without_schema(self):
        """Should raise ValueError if schema not parsed first."""
        parser = OpenAPIParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse_endpoints()
        assert "Schema must be parsed first" in str(exc_info.value)

    def test_returns_empty_list_for_empty_paths(self):
        """Should return empty list when no paths defined."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints == []

    def test_extracts_single_endpoint(self):
        """Should extract a single endpoint."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "summary": "Get users",
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/users"
        assert endpoints[0].method == "GET"


class TestParseEndpointsHTTPMethods:
    """Tests for HTTP method extraction in parse_endpoints."""

    def test_extracts_all_http_methods(self):
        """Should extract all standard HTTP methods."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/resource": {
                            "get": {"responses": {"200": {"description": "OK"}}},
                            "post": {"responses": {"201": {"description": "Created"}}},
                            "put": {"responses": {"200": {"description": "OK"}}},
                            "patch": {"responses": {"200": {"description": "OK"}}},
                            "delete": {
                                "responses": {"204": {"description": "Deleted"}}
                            },
                            "head": {"responses": {"200": {"description": "OK"}}},
                            "options": {"responses": {"200": {"description": "OK"}}},
                            "trace": {"responses": {"200": {"description": "OK"}}},
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        methods = {e.method for e in endpoints}
        assert methods == {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "HEAD",
            "OPTIONS",
            "TRACE",
        }

    def test_method_is_uppercase(self):
        """Method should be converted to uppercase."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {"responses": {"200": {"description": "OK"}}},
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].method == "GET"


class TestParseEndpointsFields:
    """Tests for endpoint field extraction."""

    def test_extracts_operation_id(self):
        """Should extract operationId."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "operationId": "getUsers",
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].operation_id == "getUsers"

    def test_extracts_summary(self):
        """Should extract summary."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "summary": "Get all users",
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].summary == "Get all users"

    def test_extracts_description(self):
        """Should extract description."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "description": "Returns a list of users",
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].description == "Returns a list of users"

    def test_extracts_tags(self):
        """Should extract tags."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "tags": ["users", "admin"],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].tags == ["users", "admin"]

    def test_extracts_security(self):
        """Should extract security requirements."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "security": [{"bearerAuth": []}],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].security == [{"bearerAuth": []}]


# =============================================================================
# OpenAPIParser._extract_parameters Tests
# =============================================================================


class TestExtractParameters:
    """Tests for _extract_parameters method."""

    def test_extracts_query_parameter(self):
        """Should extract query parameters."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "parameters": [
                                    {
                                        "name": "limit",
                                        "in": "query",
                                        "required": False,
                                        "schema": {"type": "integer"},
                                    }
                                ],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        params = endpoints[0].parameters
        assert len(params) == 1
        assert params[0].name == "limit"
        assert params[0].location == ParameterType.QUERY
        assert params[0].required is False
        assert params[0].type == "integer"

    def test_extracts_path_parameter(self):
        """Should extract path parameters."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users/{id}": {
                            "get": {
                                "parameters": [
                                    {
                                        "name": "id",
                                        "in": "path",
                                        "required": True,
                                        "schema": {"type": "string"},
                                    }
                                ],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        params = endpoints[0].parameters
        assert params[0].location == ParameterType.PATH
        assert params[0].required is True

    def test_extracts_header_parameter(self):
        """Should extract header parameters."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "parameters": [
                                    {
                                        "name": "X-Request-ID",
                                        "in": "header",
                                        "schema": {"type": "string"},
                                    }
                                ],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        params = endpoints[0].parameters
        assert params[0].location == ParameterType.HEADER

    def test_combines_path_and_operation_parameters(self):
        """Should combine path-level and operation-level parameters."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users/{id}": {
                            "parameters": [
                                {
                                    "name": "id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                },
                            ],
                            "get": {
                                "parameters": [
                                    {
                                        "name": "include",
                                        "in": "query",
                                        "schema": {"type": "string"},
                                    },
                                ],
                                "responses": {"200": {"description": "OK"}},
                            },
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        params = endpoints[0].parameters
        assert len(params) == 2
        param_names = {p.name for p in params}
        assert param_names == {"id", "include"}

    def test_extracts_array_type(self):
        """Should handle array type parameters."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "parameters": [
                                    {
                                        "name": "ids",
                                        "in": "query",
                                        "schema": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                        },
                                    }
                                ],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        params = endpoints[0].parameters
        assert params[0].type == "array[integer]"

    def test_extracts_enum(self):
        """Should extract enum values."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "parameters": [
                                    {
                                        "name": "status",
                                        "in": "query",
                                        "schema": {
                                            "type": "string",
                                            "enum": ["active", "inactive"],
                                        },
                                    }
                                ],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        params = endpoints[0].parameters
        assert params[0].enum == ["active", "inactive"]


# =============================================================================
# OpenAPIParser._extract_request_body Tests
# =============================================================================


class TestExtractRequestBody:
    """Tests for _extract_request_body method."""

    def test_returns_none_without_request_body(self):
        """Should return None when no request body defined."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {"responses": {"200": {"description": "OK"}}},
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        assert endpoints[0].request_body is None

    def test_extracts_json_request_body(self):
        """Should extract JSON request body."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "post": {
                                "requestBody": {
                                    "required": True,
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"}
                                                },
                                            }
                                        }
                                    },
                                },
                                "responses": {"201": {"description": "Created"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        body = endpoints[0].request_body
        assert body is not None
        assert body.content_type == "application/json"
        assert body.required is True
        assert body.schema["type"] == "object"

    def test_prioritizes_json_content_type(self):
        """Should prioritize JSON over other content types."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "post": {
                                "requestBody": {
                                    "content": {
                                        "multipart/form-data": {"schema": {}},
                                        "application/json": {
                                            "schema": {"type": "object"}
                                        },
                                        "application/xml": {"schema": {}},
                                    },
                                },
                                "responses": {"201": {"description": "Created"}},
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        body = endpoints[0].request_body
        assert body.content_type == "application/json"


# =============================================================================
# OpenAPIParser._extract_responses Tests
# =============================================================================


class TestExtractResponses:
    """Tests for _extract_responses method."""

    def test_extracts_single_response(self):
        """Should extract a single response."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "responses": {
                                    "200": {"description": "Successful response"},
                                }
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        responses = endpoints[0].responses
        assert len(responses) == 1
        assert responses[0].status_code == "200"
        assert responses[0].description == "Successful response"

    def test_extracts_multiple_responses(self):
        """Should extract multiple responses."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "responses": {
                                    "200": {"description": "OK"},
                                    "400": {"description": "Bad Request"},
                                    "500": {"description": "Server Error"},
                                }
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        responses = endpoints[0].responses
        assert len(responses) == 3
        status_codes = {r.status_code for r in responses}
        assert status_codes == {"200", "400", "500"}

    def test_extracts_response_content_type(self):
        """Should extract response content type."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {
                        "/users": {
                            "get": {
                                "responses": {
                                    "200": {
                                        "description": "OK",
                                        "content": {
                                            "application/json": {
                                                "schema": {"type": "array"},
                                            }
                                        },
                                    }
                                }
                            }
                        }
                    },
                }
            )
        )
        endpoints = parser.parse_endpoints()
        response = endpoints[0].responses[0]
        assert response.content_type == "application/json"
        assert response.schema == {"type": "array"}


# =============================================================================
# OpenAPIParser._resolve_reference Tests
# =============================================================================


class TestResolveReference:
    """Tests for _resolve_reference method."""

    def test_returns_object_without_ref(self):
        """Should return object as-is when no $ref present."""
        parser = OpenAPIParser()
        parser.spec_data = {"test": "data"}
        obj = {"type": "string"}
        result = parser._resolve_reference(obj)
        assert result == {"type": "string"}

    def test_returns_none_for_non_dict(self):
        """Should return None for non-dict input."""
        parser = OpenAPIParser()
        assert parser._resolve_reference("string") is None
        assert parser._resolve_reference(123) is None
        assert parser._resolve_reference(None) is None

    def test_resolves_schema_reference(self):
        """Should resolve schema reference."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                    "components": {
                        "schemas": {
                            "User": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            }
                        }
                    },
                }
            )
        )
        result = parser._resolve_reference({"$ref": "#/components/schemas/User"})
        assert result is not None
        assert result["type"] == "object"
        assert "properties" in result

    def test_returns_none_for_external_reference(self):
        """Should return None for external references."""
        parser = OpenAPIParser()
        parser.spec_data = {}
        with patch("devdox_ai_locust.utils.open_ai_parser.logger") as mock_logger:
            result = parser._resolve_reference({"$ref": "external.yaml#/User"})
            assert result is None
            mock_logger.warning.assert_called_once()

    def test_returns_none_for_missing_reference(self):
        """Should return None when reference path doesn't exist."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                    "components": {},
                }
            )
        )
        result = parser._resolve_reference({"$ref": "#/components/schemas/NonExistent"})
        assert result is None


# =============================================================================
# OpenAPIParser.get_schema_info Tests
# =============================================================================


class TestGetSchemaInfo:
    """Tests for get_schema_info method."""

    def test_returns_empty_dict_without_schema(self):
        """Should return empty dict when no schema parsed."""
        parser = OpenAPIParser()
        assert parser.get_schema_info() == {}

    def test_extracts_title(self):
        """Should extract API title."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "My API", "version": "1.0"},
                    "paths": {},
                }
            )
        )
        info = parser.get_schema_info()
        assert info["title"] == "My API"

    def test_extracts_version(self):
        """Should extract API version."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "2.1.0"},
                    "paths": {},
                }
            )
        )
        info = parser.get_schema_info()
        assert info["version"] == "2.1.0"

    def test_extracts_description(self):
        """Should extract API description."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {
                        "title": "Test",
                        "version": "1.0",
                        "description": "A test API",
                    },
                    "paths": {},
                }
            )
        )
        info = parser.get_schema_info()
        assert info["description"] == "A test API"

    def test_extracts_base_url(self):
        """Should extract base URL from servers."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                    "servers": [{"url": "https://api.example.com"}],
                }
            )
        )
        info = parser.get_schema_info()
        assert info["base_url"] == "https://api.example.com"

    def test_defaults_base_url_to_localhost(self):
        """Should default base URL to localhost."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                }
            )
        )
        info = parser.get_schema_info()
        assert info["base_url"] == "http://localhost"

    def test_extracts_security_schemes(self):
        """Should extract security schemes."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                    "components": {
                        "securitySchemes": {
                            "bearerAuth": {
                                "type": "http",
                                "scheme": "bearer",
                            }
                        }
                    },
                }
            )
        )
        info = parser.get_schema_info()
        assert "bearerAuth" in info["security_schemes"]

    def test_extracts_global_security(self):
        """Should extract global security requirements."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                    "security": [{"bearerAuth": []}],
                    "components": {
                        "securitySchemes": {
                            "bearerAuth": {
                                "type": "http",
                                "scheme": "bearer",
                            }
                        }
                    },
                }
            )
        )
        info = parser.get_schema_info()
        assert "global_security" in info
        assert info["global_security"] == [{"bearerAuth": []}]

    def test_global_security_defaults_empty(self):
        """Should default global_security to empty list when not defined."""
        parser = OpenAPIParser()
        parser.parse_schema(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Test", "version": "1.0"},
                    "paths": {},
                }
            )
        )
        info = parser.get_schema_info()
        assert info["global_security"] == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestOpenAPIParserIntegration:
    """Integration tests for OpenAPIParser."""

    def test_full_parsing_workflow(self):
        """Test complete parsing workflow with realistic schema."""
        schema = {
            "openapi": "3.0.0",
            "info": {
                "title": "Pet Store API",
                "version": "1.0.0",
                "description": "A sample pet store API",
            },
            "servers": [{"url": "https://petstore.example.com/v1"}],
            "paths": {
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "summary": "List all pets",
                        "tags": ["pets"],
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {"type": "integer", "default": 10},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "A list of pets",
                                "content": {
                                    "application/json": {"schema": {"type": "array"}}
                                },
                            }
                        },
                    },
                    "post": {
                        "operationId": "createPet",
                        "summary": "Create a pet",
                        "tags": ["pets"],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Pet"}
                                }
                            },
                        },
                        "responses": {
                            "201": {"description": "Pet created"},
                        },
                    },
                },
                "/pets/{petId}": {
                    "get": {
                        "operationId": "getPet",
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {"description": "Pet found"},
                            "404": {"description": "Pet not found"},
                        },
                    },
                },
            },
            "components": {
                "schemas": {
                    "Pet": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                    }
                },
                "securitySchemes": {
                    "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
                },
            },
        }

        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(schema))

        # Check schema info
        info = parser.get_schema_info()
        assert info["title"] == "Pet Store API"
        assert info["base_url"] == "https://petstore.example.com/v1"
        assert "apiKey" in info["security_schemes"]

        # Check endpoints
        endpoints = parser.parse_endpoints()
        assert len(endpoints) == 3

        # Check GET /pets
        get_pets = next(e for e in endpoints if e.path == "/pets" and e.method == "GET")
        assert get_pets.operation_id == "listPets"
        assert len(get_pets.parameters) == 1
        assert get_pets.parameters[0].name == "limit"

        # Check POST /pets
        post_pets = next(
            e for e in endpoints if e.path == "/pets" and e.method == "POST"
        )
        assert post_pets.request_body is not None
        assert post_pets.request_body.content_type == "application/json"

        # Check GET /pets/{petId}
        get_pet = next(e for e in endpoints if e.path == "/pets/{petId}")
        assert len(get_pet.parameters) == 1
        assert get_pet.parameters[0].location == ParameterType.PATH
        assert len(get_pet.responses) == 2

    def test_yaml_schema_parsing(self):
        """Test parsing YAML schema."""
        yaml_schema = """
openapi: "3.0.0"
info:
  title: YAML API
  version: "1.0.0"
paths:
  /items:
    get:
      summary: Get items
      responses:
        "200":
          description: Success
"""
        parser = OpenAPIParser()
        parser.parse_schema(yaml_schema)
        endpoints = parser.parse_endpoints()

        assert len(endpoints) == 1
        assert endpoints[0].path == "/items"
        assert endpoints[0].method == "GET"
