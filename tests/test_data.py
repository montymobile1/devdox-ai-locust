"""
Test data fixtures and utilities

This module contains common test data, mock factories, and utility functions
that can be shared across multiple test files.
"""

from typing import Dict, List, Any
from unittest.mock import Mock

from devdox_ai_locust.utils.open_ai_parser import (
    Endpoint,
    Parameter,
    RequestBody,
    Response,
    ParameterType,
)


class MockDataFactory:
    """Factory for creating mock test data."""

    @staticmethod
    def create_mock_endpoint(
        path: str = "/test",
        method: str = "GET",
        operation_id: str = None,
        summary: str = None,
        parameters: List[Parameter] = None,
        request_body: RequestBody = None,
        responses: List[Response] = None,
        tags: List[str] = None,
    ) -> Endpoint:
        """Create a mock endpoint with default values."""
        return Endpoint(
            path=path,
            method=method,
            operation_id=operation_id or f"{method.lower()}{path.replace('/', '_')}",
            summary=summary or f"{method} {path}",
            description=f"Test endpoint for {method} {path}",
            parameters=parameters or [],
            request_body=request_body,
            responses=responses or [MockDataFactory.create_mock_response()],
            tags=tags or ["test"],
        )

    @staticmethod
    def create_mock_parameter(
        name: str = "test_param",
        location: ParameterType = ParameterType.QUERY,
        required: bool = False,
        param_type: str = "string",
        description: str = None,
    ) -> Parameter:
        """Create a mock parameter."""
        return Parameter(
            name=name,
            location=location,
            required=required,
            type=param_type,
            description=description or f"Test {name} parameter",
        )

    @staticmethod
    def create_mock_request_body(
        content_type: str = "application/json",
        schema: Dict[str, Any] = None,
        required: bool = True,
    ) -> RequestBody:
        """Create a mock request body."""
        return RequestBody(
            content_type=content_type,
            schema=schema
            or {"type": "object", "properties": {"test": {"type": "string"}}},
            required=required,
            description="Test request body",
        )

    @staticmethod
    def create_mock_response(
        status_code: str = "200",
        description: str = "Success",
        content_type: str = "application/json",
        schema: Dict[str, Any] = None,
    ) -> Response:
        """Create a mock response."""
        return Response(
            status_code=status_code,
            description=description,
            content_type=content_type,
            schema=schema or {"type": "object"},
        )

    @staticmethod
    def create_crud_endpoints(resource: str = "users") -> List[Endpoint]:
        """Create a set of CRUD endpoints for a resource."""
        return [
            MockDataFactory.create_mock_endpoint(
                path=f"/{resource}",
                method="GET",
                operation_id=f"get{resource.capitalize()}",
                summary=f"Get all {resource}",
                parameters=[
                    MockDataFactory.create_mock_parameter(
                        "limit", ParameterType.QUERY, False, "integer"
                    ),
                    MockDataFactory.create_mock_parameter(
                        "offset", ParameterType.QUERY, False, "integer"
                    ),
                ],
                tags=[resource],
            ),
            MockDataFactory.create_mock_endpoint(
                path=f"/{resource}",
                method="POST",
                operation_id=f"create{resource.capitalize()[:-1]}",
                summary=f"Create a new {resource[:-1]}",
                request_body=MockDataFactory.create_mock_request_body(),
                responses=[
                    MockDataFactory.create_mock_response("201", "Created"),
                    MockDataFactory.create_mock_response("400", "Bad Request"),
                ],
                tags=[resource],
            ),
            MockDataFactory.create_mock_endpoint(
                path=f"/{resource}/{{id}}",
                method="GET",
                operation_id=f"get{resource.capitalize()[:-1]}ById",
                summary=f"Get {resource[:-1]} by ID",
                parameters=[
                    MockDataFactory.create_mock_parameter(
                        "id", ParameterType.PATH, True, "integer"
                    )
                ],
                responses=[
                    MockDataFactory.create_mock_response("200", "Success"),
                    MockDataFactory.create_mock_response("404", "Not Found"),
                ],
                tags=[resource],
            ),
            MockDataFactory.create_mock_endpoint(
                path=f"/{resource}/{{id}}",
                method="PUT",
                operation_id=f"update{resource.capitalize()[:-1]}",
                summary=f"Update {resource[:-1]}",
                parameters=[
                    MockDataFactory.create_mock_parameter(
                        "id", ParameterType.PATH, True, "integer"
                    )
                ],
                request_body=MockDataFactory.create_mock_request_body(),
                tags=[resource],
            ),
            MockDataFactory.create_mock_endpoint(
                path=f"/{resource}/{{id}}",
                method="DELETE",
                operation_id=f"delete{resource.capitalize()[:-1]}",
                summary=f"Delete {resource[:-1]}",
                parameters=[
                    MockDataFactory.create_mock_parameter(
                        "id", ParameterType.PATH, True, "integer"
                    )
                ],
                responses=[
                    MockDataFactory.create_mock_response("204", "No Content"),
                    MockDataFactory.create_mock_response("404", "Not Found"),
                ],
                tags=[resource],
            ),
        ]

    @staticmethod
    def create_auth_endpoints() -> List[Endpoint]:
        """Create authentication-related endpoints."""
        return [
            MockDataFactory.create_mock_endpoint(
                path="/auth/login",
                method="POST",
                operation_id="login",
                summary="User login",
                request_body=RequestBody(
                    content_type="application/json",
                    schema={
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "format": "email"},
                            "password": {"type": "string"},
                        },
                        "required": ["email", "password"],
                    },
                    required=True,
                ),
                responses=[
                    Response(
                        status_code="200",
                        description="Login successful",
                        content_type="application/json",
                        schema={
                            "type": "object",
                            "properties": {
                                "token": {"type": "string"},
                                "user": {"type": "object"},
                            },
                        },
                    ),
                    MockDataFactory.create_mock_response("401", "Unauthorized"),
                ],
                tags=["auth"],
            ),
            MockDataFactory.create_mock_endpoint(
                path="/auth/logout",
                method="POST",
                operation_id="logout",
                summary="User logout",
                responses=[
                    MockDataFactory.create_mock_response("200", "Logout successful")
                ],
                tags=["auth"],
            ),
            MockDataFactory.create_mock_endpoint(
                path="/auth/refresh",
                method="POST",
                operation_id="refreshToken",
                summary="Refresh access token",
                request_body=RequestBody(
                    content_type="application/json",
                    schema={
                        "type": "object",
                        "properties": {"refresh_token": {"type": "string"}},
                        "required": ["refresh_token"],
                    },
                ),
                tags=["auth"],
            ),
        ]


class TestAPISchemas:
    """Common API schemas for testing."""

    @staticmethod
    def get_petstore_schema() -> Dict[str, Any]:
        """Get a simplified Petstore OpenAPI schema."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Swagger Petstore",
                "description": "This is a sample server Petstore server.",
                "version": "1.0.0",
            },
            "servers": [{"url": "https://petstore.swagger.io/v2"}],
            "paths": {
                "/pet": {
                    "post": {
                        "tags": ["pet"],
                        "summary": "Add a new pet to the store",
                        "operationId": "addPet",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Pet"}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "successful operation",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Pet"}
                                    }
                                },
                            }
                        },
                    }
                },
                "/pet/{petId}": {
                    "get": {
                        "tags": ["pet"],
                        "summary": "Find pet by ID",
                        "operationId": "getPetById",
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "integer", "format": "int64"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "successful operation",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Pet"}
                                    }
                                },
                            },
                            "404": {"description": "Pet not found"},
                        },
                    }
                },
            },
            "components": {
                "schemas": {
                    "Pet": {
                        "type": "object",
                        "required": ["name", "photoUrls"],
                        "properties": {
                            "id": {"type": "integer", "format": "int64"},
                            "name": {"type": "string", "example": "doggie"},
                            "photoUrls": {"type": "array", "items": {"type": "string"}},
                            "status": {
                                "type": "string",
                                "enum": ["available", "pending", "sold"],
                            },
                        },
                    }
                }
            },
        }

    @staticmethod
    def get_ecommerce_schema() -> Dict[str, Any]:
        """Get an e-commerce API schema."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "E-commerce API",
                "description": "API for managing products and orders",
                "version": "1.0.0",
            },
            "servers": [{"url": "https://api.ecommerce.com/v1"}],
            "paths": {
                "/products": {
                    "get": {
                        "tags": ["products"],
                        "summary": "Get all products",
                        "parameters": [
                            {
                                "name": "category",
                                "in": "query",
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                            },
                        ],
                        "responses": {
                            "200": {
                                "description": "successful operation",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/Product"
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
                "/cart": {
                    "post": {
                        "tags": ["cart"],
                        "summary": "Add item to cart",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CartItem"}
                                }
                            },
                        },
                        "responses": {"200": {"description": "Item added to cart"}},
                    }
                },
                "/orders": {
                    "post": {
                        "tags": ["orders"],
                        "summary": "Create order",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Order"}
                                }
                            },
                        },
                        "responses": {"201": {"description": "Order created"}},
                    }
                },
            },
            "components": {
                "schemas": {
                    "Product": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "price": {"type": "number", "format": "float"},
                            "category": {"type": "string"},
                        },
                    },
                    "CartItem": {
                        "type": "object",
                        "properties": {
                            "productId": {"type": "integer"},
                            "quantity": {"type": "integer", "minimum": 1},
                        },
                    },
                    "Order": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/CartItem"},
                            },
                            "totalAmount": {"type": "number", "format": "float"},
                        },
                    },
                }
            },
        }


class MockTogetherClient:
    """Mock Together AI client for testing."""

    def __init__(self, response_content: str = None):
        self.response_content = response_content or self._default_response_content()
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock()

        # Configure mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = self.response_content
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        self.chat.completions.create.return_value = mock_response

    def _default_response_content(self) -> str:
        """Default response content for mock AI calls."""
        return """<code>
import locust
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def test_endpoint(self):
        response = self.client.get("/test")
        assert response.status_code == 200
</code>"""


# Common test constants
TEST_API_INFO = {
    "title": "Test API",
    "version": "1.0.0",
    "description": "A test API for load testing",
    "base_url": "https://api.example.com/v1",
    "security_schemes": {},
}

TEST_GENERATED_FILES = {
    "locustfile.py": """
import locust
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def test_users(self):
        self.client.get("/users")
""",
    "test_data.py": """
class TestDataGenerator:
    def generate_user_data(self):
        return {"username": "testuser", "email": "test@example.com"}
""",
    "config.py": """
API_BASE_URL = "https://api.example.com"
""",
    "requirements.txt": """
locust>=2.0.0
requests>=2.28.0
""",
}
