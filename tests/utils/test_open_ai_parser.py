"""
Tests for OpenAPI Parser module
"""

import pytest
import json
import yaml
from unittest.mock import Mock, patch

from devdox_ai_locust.utils.open_ai_parser import (
    OpenAPIParser, 
    Endpoint, 
    Parameter, 
    RequestBody, 
    Response, 
    ParameterType
)


class TestOpenAPIParser:
    """Test OpenAPIParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = OpenAPIParser()
        
        assert parser.spec_data is None
        assert parser.components is None

    def test_parse_schema_json(self, sample_openapi_schema):
        """Test parsing JSON schema."""
        parser = OpenAPIParser()
        schema_json = json.dumps(sample_openapi_schema)
        
        result = parser.parse_schema(schema_json)
        
        assert result == sample_openapi_schema
        assert parser.spec_data == sample_openapi_schema
        assert parser.components == sample_openapi_schema.get('components', {})

    def test_parse_schema_yaml(self, sample_yaml_schema):
        """Test parsing YAML schema."""
        parser = OpenAPIParser()
        
        result = parser.parse_schema(sample_yaml_schema)
        
        assert result['openapi'] == '3.0.0'
        assert result['info']['title'] == 'Test API'
        assert parser.spec_data is not None

    def test_parse_schema_invalid_json(self):
        """Test parsing invalid JSON."""
        parser = OpenAPIParser()
        
        with pytest.raises(ValueError, match="Invalid OpenAPI schema"):
            parser.parse_schema("invalid json {")

    def test_parse_schema_invalid_yaml(self):
        """Test parsing invalid YAML."""
        parser = OpenAPIParser()
        
        with pytest.raises(ValueError, match="Invalid OpenAPI schema"):
            parser.parse_schema("invalid: yaml: [unclosed")

    def test_parse_schema_missing_required_fields(self):
        """Test parsing schema with missing required fields."""
        parser = OpenAPIParser()
        invalid_schema = {"info": {"title": "Test"}}  # Missing openapi and paths
        
        with pytest.raises(ValueError, match="Missing required OpenAPI fields"):
            parser.parse_schema(json.dumps(invalid_schema))

    def test_parse_schema_unsupported_version(self):
        """Test parsing schema with unsupported OpenAPI version."""
        parser = OpenAPIParser()
        invalid_schema = {
            "openapi": "2.0",  # Unsupported version
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {}
        }
        
        with pytest.raises(ValueError, match="Unsupported OpenAPI version"):
            parser.parse_schema(json.dumps(invalid_schema))

    def test_parse_endpoints(self, sample_openapi_schema):
        """Test parsing endpoints from schema."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        endpoints = parser.parse_endpoints()
        
        assert len(endpoints) == 4  # GET /users, POST /users, GET /users/{id}, POST /auth/login
        
        # Test first endpoint (GET /users)
        get_users = next(ep for ep in endpoints if ep.method == "GET" and ep.path == "/users")
        assert get_users.operation_id == "getUsers"
        assert get_users.summary == "Get all users"
        assert len(get_users.parameters) == 1
        assert get_users.parameters[0].name == "limit"
        assert get_users.request_body is None
        assert len(get_users.responses) == 1

    def test_parse_endpoints_without_schema(self):
        """Test parsing endpoints without parsed schema."""
        parser = OpenAPIParser()
        
        with pytest.raises(ValueError, match="Schema must be parsed first"):
            parser.parse_endpoints()

    def test_extract_parameters(self, sample_openapi_schema):
        """Test parameter extraction."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        # Test query parameter
        operation = sample_openapi_schema["paths"]["/users"]["get"]
        parameters = parser._extract_parameters(operation)
        
        assert len(parameters) == 1
        param = parameters[0]
        assert param.name == "limit"
        assert param.location == ParameterType.QUERY
        assert param.required is False
        assert param.type == "integer"

    def test_extract_path_parameters(self, sample_openapi_schema):
        """Test path parameter extraction."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        operation = sample_openapi_schema["paths"]["/users/{id}"]["get"]
        parameters = parser._extract_parameters(operation)
        
        assert len(parameters) == 1
        param = parameters[0]
        assert param.name == "id"
        assert param.location == ParameterType.PATH
        assert param.required is True
        assert param.type == "integer"

    def test_extract_request_body(self, sample_openapi_schema):
        """Test request body extraction."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        operation = sample_openapi_schema["paths"]["/users"]["post"]
        request_body = parser._extract_request_body(operation)
        
        assert request_body is not None
        assert request_body.content_type == "application/json"
        assert request_body.required is True
        assert request_body.schema is not None

    def test_extract_request_body_none(self, sample_openapi_schema):
        """Test request body extraction when none exists."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        operation = sample_openapi_schema["paths"]["/users"]["get"]
        request_body = parser._extract_request_body(operation)
        
        assert request_body is None

    def test_extract_responses(self, sample_openapi_schema):
        """Test response extraction."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        operation = sample_openapi_schema["paths"]["/users/{id}"]["get"]
        responses = parser._extract_responses(operation)
        
        assert len(responses) == 2  # 200 and 404
        
        success_response = next(r for r in responses if r.status_code == "200")
        assert success_response.description == "User found"
        assert success_response.content_type == "application/json"
        
        error_response = next(r for r in responses if r.status_code == "404")
        assert error_response.description == "User not found"

    def test_resolve_reference(self, sample_openapi_schema):
        """Test reference resolution."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        # Test resolving a component reference
        ref_obj = {"$ref": "#/components/schemas/User"}
        resolved = parser._resolve_reference(ref_obj)
        
        assert resolved is not None
        assert resolved == sample_openapi_schema["components"]["schemas"]["User"]

    def test_resolve_reference_invalid(self, sample_openapi_schema):
        """Test invalid reference resolution."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        # Test non-existent reference
        ref_obj = {"$ref": "#/components/schemas/NonExistent"}
        resolved = parser._resolve_reference(ref_obj)
        
        assert resolved is None

    def test_resolve_reference_external(self, sample_openapi_schema):
        """Test external reference resolution."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        # Test external reference (should not be supported)
        ref_obj = {"$ref": "external.json#/schemas/User"}
        resolved = parser._resolve_reference(ref_obj)
        
        assert resolved is None

    def test_resolve_reference_no_ref(self):
        """Test resolving object without $ref."""
        parser = OpenAPIParser()
        
        obj = {"type": "string"}
        resolved = parser._resolve_reference(obj)
        
        assert resolved == obj

    def test_get_schema_info(self, sample_openapi_schema):
        """Test schema info extraction."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        info = parser.get_schema_info()
        
        assert info['title'] == 'Test API'
        assert info['version'] == '1.0.0'
        assert info['description'] == 'A test API for load testing'
        assert info['base_url'] == 'https://api.example.com/v1'
        assert 'security_schemes' in info

    def test_get_schema_info_no_schema(self):
        """Test schema info extraction without parsed schema."""
        parser = OpenAPIParser()
        
        info = parser.get_schema_info()
        
        assert info == {}

    def test_extract_base_url_with_servers(self, sample_openapi_schema):
        """Test base URL extraction with servers."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        base_url = parser._extract_base_url()
        
        assert base_url == "https://api.example.com/v1"

    def test_extract_base_url_no_servers(self):
        """Test base URL extraction without servers."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {}
        }
        parser.parse_schema(json.dumps(schema))
        
        base_url = parser._extract_base_url()
        
        assert base_url == "http://localhost"

    def test_extract_security_schemes(self, sample_openapi_schema):
        """Test security schemes extraction."""
        parser = OpenAPIParser()
        parser.parse_schema(json.dumps(sample_openapi_schema))
        
        security_schemes = parser._extract_security_schemes()
        
        assert isinstance(security_schemes, dict)

    def test_extract_security_schemes_no_components(self):
        """Test security schemes extraction without components."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {}
        }
        parser.parse_schema(json.dumps(schema))
        
        security_schemes = parser._extract_security_schemes()
        
        assert security_schemes == {}


class TestParameterType:
    """Test ParameterType enum."""

    def test_parameter_type_values(self):
        """Test ParameterType enum values."""
        assert ParameterType.QUERY.value == "query"
        assert ParameterType.PATH.value == "path"
        assert ParameterType.HEADER.value == "header"
        assert ParameterType.COOKIE.value == "cookie"


class TestDataClasses:
    """Test data classes."""

    def test_parameter_creation(self):
        """Test Parameter dataclass creation."""
        param = Parameter(
            name="test_param",
            location=ParameterType.QUERY,
            required=True,
            type="string",
            description="Test parameter"
        )
        
        assert param.name == "test_param"
        assert param.location == ParameterType.QUERY
        assert param.required is True
        assert param.type == "string"
        assert param.description == "Test parameter"

    def test_request_body_creation(self):
        """Test RequestBody dataclass creation."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        request_body = RequestBody(
            content_type="application/json",
            schema=schema,
            required=True,
            description="Test request body"
        )
        
        assert request_body.content_type == "application/json"
        assert request_body.schema == schema
        assert request_body.required is True
        assert request_body.description == "Test request body"

    def test_response_creation(self):
        """Test Response dataclass creation."""
        response = Response(
            status_code="200",
            description="Success",
            content_type="application/json",
            schema={"type": "object"}
        )
        
        assert response.status_code == "200"
        assert response.description == "Success"
        assert response.content_type == "application/json"
        assert response.schema == {"type": "object"}

    def test_endpoint_creation(self):
        """Test Endpoint dataclass creation."""
        endpoint = Endpoint(
            path="/test",
            method="GET",
            operation_id="testEndpoint",
            summary="Test endpoint",
            description="A test endpoint",
            parameters=[],
            request_body=None,
            responses=[],
            tags=["test"]
        )
        
        assert endpoint.path == "/test"
        assert endpoint.method == "GET"
        assert endpoint.operation_id == "testEndpoint"
        assert endpoint.summary == "Test endpoint"
        assert endpoint.description == "A test endpoint"
        assert endpoint.parameters == []
        assert endpoint.request_body is None
        assert endpoint.responses == []
        assert endpoint.tags == ["test"]


class TestOpenAPIParserEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_paths(self):
        """Test handling empty paths section."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {}
        }
        parser.parse_schema(json.dumps(schema))
        
        endpoints = parser.parse_endpoints()
        
        assert endpoints == []

    def test_array_parameter_type(self):
        """Test handling array parameter types."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "parameters": [{
                            "name": "tags",
                            "in": "query",
                            "schema": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }]
                    }
                }
            }
        }
        parser.parse_schema(json.dumps(schema))
        
        endpoints = parser.parse_endpoints()
        param = endpoints[0].parameters[0]
        
        assert param.type == "array[string]"

    def test_parameter_with_enum(self):
        """Test parameter with enum values."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "parameters": [{
                            "name": "status",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["active", "inactive"]
                            }
                        }]
                    }
                }
            }
        }
        parser.parse_schema(json.dumps(schema))
        
        endpoints = parser.parse_endpoints()
        param = endpoints[0].parameters[0]
        
        assert param.enum == ["active", "inactive"]

    def test_multiple_content_types(self):
        """Test handling multiple content types in request body."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/xml": {"schema": {"type": "string"}},
                                "application/json": {"schema": {"type": "object"}},
                                "text/plain": {"schema": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        }
        parser.parse_schema(json.dumps(schema))
        
        endpoints = parser.parse_endpoints()
        request_body = endpoints[0].request_body
        
        # Should prioritize application/json
        assert request_body.content_type == "application/json"

    def test_form_data_content_type(self):
        """Test handling form data content type."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            }
        }
        parser.parse_schema(json.dumps(schema))
        
        endpoints = parser.parse_endpoints()
        request_body = endpoints[0].request_body
        
        assert request_body.content_type == "application/x-www-form-urlencoded"

    def test_missing_operation_id(self):
        """Test handling missing operation ID."""
        parser = OpenAPIParser()
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "summary": "Test endpoint"
                        # No operationId
                    }
                }
            }
        }
        parser.parse_schema(json.dumps(schema))
        
        endpoints = parser.parse_endpoints()
        
        assert endpoints[0].operation_id is None
        assert endpoints[0].summary == "Test endpoint"
