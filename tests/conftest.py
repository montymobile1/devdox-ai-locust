"""
Pytest configuration and shared fixtures for devdox-ai-locust tests
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil

from devdox_ai_locust.utils.open_ai_parser import Endpoint, Parameter, RequestBody, Response, ParameterType
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_openapi_schema():
    """Sample OpenAPI 3.0 schema for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "description": "A test API for load testing"
        },
        "servers": [
            {"url": "https://api.example.com/v1"}
        ],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "getUsers",
                    "summary": "Get all users",
                    "description": "Retrieve a list of all users",
                    "tags": ["users"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 1, "maximum": 100}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/User"}
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "operationId": "createUser",
                    "summary": "Create a new user",
                    "tags": ["users"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CreateUser"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "User created",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        }
                    }
                }
            },
            "/users/{id}": {
                "get": {
                    "operationId": "getUserById",
                    "summary": "Get user by ID",
                    "tags": ["users"],
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "User found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        },
                        "404": {
                            "description": "User not found"
                        }
                    }
                }
            },
            "/auth/login": {
                "post": {
                    "operationId": "login",
                    "summary": "User login",
                    "tags": ["auth"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "username": {"type": "string"},
                                        "password": {"type": "string"}
                                    },
                                    "required": ["username", "password"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "token": {"type": "string"},
                                            "user": {"$ref": "#/components/schemas/User"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "username": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                        "createdAt": {"type": "string", "format": "date-time"}
                    },
                    "required": ["id", "username", "email"]
                },
                "CreateUser": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                        "password": {"type": "string"}
                    },
                    "required": ["username", "email", "password"]
                }
            }
        }
    }


@pytest.fixture
def sample_endpoints():
    """Sample parsed endpoints for testing."""
    return [
        Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary="Get all users",
            description="Retrieve a list of all users",
            parameters=[
                Parameter(
                    name="limit",
                    location=ParameterType.QUERY,
                    required=False,
                    type="integer",
                    description="Maximum number of users to return"
                )
            ],
            request_body=None,
            responses=[
                Response(
                    status_code="200",
                    description="Successful response",
                    content_type="application/json",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            tags=["users"]
        ),
        Endpoint(
            path="/users",
            method="POST",
            operation_id="createUser",
            summary="Create a new user",
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/json",
                schema={
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                        "email": {"type": "string"},
                        "password": {"type": "string"}
                    },
                    "required": ["username", "email", "password"]
                },
                required=True
            ),
            responses=[
                Response(
                    status_code="201",
                    description="User created",
                    content_type="application/json",
                    schema={"type": "object"}
                )
            ],
            tags=["users"]
        ),
        Endpoint(
            path="/users/{id}",
            method="GET",
            operation_id="getUserById",
            summary="Get user by ID",
            description=None,
            parameters=[
                Parameter(
                    name="id",
                    location=ParameterType.PATH,
                    required=True,
                    type="integer",
                    description="User ID"
                )
            ],
            request_body=None,
            responses=[
                Response(
                    status_code="200",
                    description="User found",
                    content_type="application/json",
                    schema={"type": "object"}
                ),
                Response(
                    status_code="404",
                    description="User not found"
                )
            ],
            tags=["users"]
        ),
        Endpoint(
            path="/auth/login",
            method="POST",
            operation_id="login",
            summary="User login",
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/json",
                schema={
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                        "password": {"type": "string"}
                    },
                    "required": ["username", "password"]
                },
                required=True
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
                            "user": {"type": "object"}
                        }
                    }
                )
            ],
            tags=["auth"]
        )
    ]


@pytest.fixture
def sample_api_info():
    """Sample API info for testing."""
    return {
        "title": "Test API",
        "version": "1.0.0",
        "description": "A test API for load testing",
        "base_url": "https://api.example.com/v1",
        "security_schemes": {}
    }


@pytest.fixture
def mock_together_client():
    """Mock Together AI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    
    # Configure the mock response
    mock_message.content = """<code>
import locust
from locust import HttpUser, task, between

class TestUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def test_endpoint(self):
        response = self.client.get("/test")
        assert response.status_code == 200
</code>"""
    
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    # Mock the chat completions create method
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_httpx_client():
    """Mock httpx async client for HTTP requests."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"openapi": "3.0.0", "info": {"title": "Test API"}}'
    mock_response.headers = {"content-type": "application/json"}
    mock_response.raise_for_status.return_value = None
    
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    
    return mock_client


@pytest.fixture
def swagger_processing_request():
    """Sample SwaggerProcessingRequest for testing."""
    return SwaggerProcessingRequest(
        swagger_url="https://api.example.com/swagger.json"
    )


@pytest.fixture
def sample_generated_files():
    """Sample generated files for testing."""
    return {
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
        """
    }


@pytest.fixture
def mock_file_system(temp_dir):
    """Mock file system operations for testing."""
    def mock_write_file(path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    
    def mock_read_file(path: Path) -> str:
        return path.read_text()
    
    return {
        "write_file": mock_write_file,
        "read_file": mock_read_file,
        "temp_dir": temp_dir
    }


@pytest.fixture
def mock_jinja_template():
    """Mock Jinja2 template for testing."""
    mock_template = Mock()
    mock_template.render.return_value = "Generated content from template"
    return mock_template


@pytest.fixture
def mock_jinja_env(mock_jinja_template):
    """Mock Jinja2 environment for testing."""
    mock_env = Mock()
    mock_env.get_template.return_value = mock_jinja_template
    return mock_env


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sample_yaml_schema():
    """Sample OpenAPI schema in YAML format."""
    return """
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
  description: A test API for load testing
servers:
  - url: https://api.example.com/v1
paths:
  /health:
    get:
      operationId: healthCheck
      summary: Health check
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: ok
"""


@pytest.fixture
def ai_enhancement_config():
    """Sample AI enhancement configuration."""
    from devdox_ai_locust.hybrid_loctus_generator import AIEnhancementConfig
    return AIEnhancementConfig(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        max_tokens=4000,
        temperature=0.3,
        timeout=30,
        enhance_workflows=True,
        enhance_test_data=True,
        enhance_validation=True,
        create_domain_flows=True,
        update_main_locust=True
    )
