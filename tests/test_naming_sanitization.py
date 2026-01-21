"""
Tests for naming sanitization functions across the codebase.

These tests ensure that all user-provided names (tags, operation IDs, paths)
are properly sanitized to create valid:
- Python identifiers (class names, method names)
- File system paths (directory names, file names)
- Import statements
"""

import pytest
from unittest.mock import Mock
from pathlib import Path


class TestScenarioGeneratorSanitization:
    """Tests for scenario_generator.py sanitization functions"""

    @pytest.fixture
    def generator(self):
        """Create a ScenarioWorkflowGenerator instance for testing"""
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator

        # Mock the dependencies
        mock_client = Mock()
        mock_config = Mock()
        mock_config.model = "test-model"
        mock_config.max_tokens = 1000
        mock_config.temperature = 0.3
        mock_config.timeout = 30

        # Use a temp directory for prompts
        prompt_dir = Path(__file__).parent.parent / "src" / "devdox_ai_locust" / "prompt"

        return ScenarioWorkflowGenerator(
            prompt_dir=prompt_dir,
            ai_client=mock_client,
            ai_config=mock_config,
        )

    @pytest.mark.parametrize("input_name,expected", [
        # Basic cases
        ("simple", "simple"),
        ("Simple", "Simple"),
        ("UPPERCASE", "UPPERCASE"),

        # Spaces
        ("nested resources", "nested_resources"),
        ("  leading spaces", "leading_spaces"),
        ("trailing spaces  ", "trailing_spaces"),
        ("multiple   spaces", "multiple_spaces"),

        # Dashes
        ("kebab-case", "kebab_case"),
        ("multi-dash-name", "multi_dash_name"),
        ("--leading-dashes", "leading_dashes"),
        ("trailing-dashes--", "trailing_dashes"),

        # Dots
        ("api.v1.users", "api_v1_users"),
        ("file.json", "file_json"),
        ("...dots...", "dots"),

        # Mixed separators
        ("mixed-case name.v1", "mixed_case_name_v1"),
        ("weird--name__here", "weird_name_here"),

        # Special characters
        ("user@domain", "userdomain"),
        ("path/to/resource", "path_to_resource"),
        ("query?param=value", "queryparamvalue"),
        ("hash#tag", "hashtag"),
        ("percent%encoded", "percentencoded"),
        ("ampersand&test", "ampersandtest"),
        ("colon:separated", "colonseparated"),

        # Numbers
        ("123numeric", "n123numeric"),
        ("0starting", "n0starting"),
        ("name123", "name123"),
        ("v1api", "v1api"),

        # Path parameters
        ("{userId}", "userId"),
        ("{id}", "id"),
        ("users/{userId}/posts", "users_userId_posts"),

        # Edge cases
        ("", "unnamed"),
        ("   ", "unnamed"),
        ("---", "unnamed"),
        ("...", "unnamed"),
        ("@#$%", "unnamed"),
        ("_", "unnamed"),
        ("__", "unnamed"),
    ])
    def test_sanitize_identifier(self, generator, input_name, expected):
        """Test _sanitize_identifier handles various inputs correctly"""
        result = generator._sanitize_identifier(input_name)
        assert result == expected, f"Input '{input_name}' should become '{expected}', got '{result}'"

    @pytest.mark.parametrize("input_name,expected", [
        # Basic cases
        ("simple", "Simple"),
        ("get_users", "GetUsers"),
        ("create_user_post", "CreateUserPost"),

        # Spaces
        ("nested resources", "NestedResources"),
        ("advanced features", "AdvancedFeatures"),

        # Dashes
        ("kebab-case", "KebabCase"),
        ("get-user-by-id", "GetUserById"),

        # Dots
        ("api.v1.users", "ApiV1Users"),

        # Numbers
        ("123numeric", "N123numeric"),
        ("get2users", "Get2users"),

        # Mixed
        ("POST_api_v1_users", "PostApiV1Users"),
        ("GET-users-{id}", "GetUsersId"),

        # Edge cases - Note: empty/invalid operation_id falls back to method+path
        # So we don't test empty strings here, they're covered in test_get_endpoint_dir_name
    ])
    def test_operation_to_class_name(self, generator, input_name, expected):
        """Test _operation_to_class_name creates valid PascalCase class names"""
        # Create a mock endpoint with operation_id
        endpoint = Mock()
        endpoint.operation_id = input_name
        endpoint.method = "GET"
        endpoint.path = "/test"

        result = generator._operation_to_class_name(endpoint)
        assert result == expected, f"Input '{input_name}' should become '{expected}', got '{result}'"

    def test_operation_to_class_name_fallback(self, generator):
        """Test that empty operation_id falls back to method+path"""
        endpoint = Mock()
        endpoint.operation_id = ""
        endpoint.method = "GET"
        endpoint.path = "/users"

        result = generator._operation_to_class_name(endpoint)
        assert result == "GetUsers", f"Empty operation_id should fallback to method+path, got '{result}'"

    @pytest.mark.parametrize("method,path,expected", [
        ("GET", "/users", "get_users"),
        ("POST", "/api/v1/users", "post_api_v1_users"),
        ("DELETE", "/users/{id}", "delete_users_id"),
        ("PUT", "/users/{userId}/posts/{postId}", "put_users_userId_posts_postId"),
        ("GET", "/api.v1/resource", "get_api_v1_resource"),
        ("POST", "/path-with-dashes", "post_path_with_dashes"),
    ])
    def test_generate_operation_id(self, generator, method, path, expected):
        """Test _generate_operation_id creates valid IDs from method and path"""
        endpoint = Mock()
        endpoint.operation_id = None  # Force generation
        endpoint.method = method
        endpoint.path = path

        result = generator._generate_operation_id(endpoint)
        assert result == expected, f"{method} {path} should become '{expected}', got '{result}'"

    @pytest.mark.parametrize("operation_id,method,path,expected", [
        # With operation_id
        ("getUsers", "GET", "/users", "getusers"),
        ("CreateUserPost", "POST", "/users", "createuserpost"),
        ("get-user-by-id", "GET", "/users/{id}", "get_user_by_id"),

        # Without operation_id (uses method+path)
        (None, "GET", "/users", "get_users"),
        (None, "POST", "/api/v1/users", "post_api_v1_users"),
        ("", "DELETE", "/users/{id}", "delete_users_id"),

        # Special characters
        ("user@test", "GET", "/test", "usertest"),
        ("api.v1.get", "GET", "/test", "api_v1_get"),
    ])
    def test_get_endpoint_dir_name(self, generator, operation_id, method, path, expected):
        """Test get_endpoint_dir_name creates valid directory names"""
        endpoint = Mock()
        endpoint.operation_id = operation_id
        endpoint.method = method
        endpoint.path = path

        result = generator.get_endpoint_dir_name(endpoint)
        assert result == expected, f"Operation '{operation_id}' or {method} {path} should become '{expected}', got '{result}'"


class TestLocustGeneratorSanitization:
    """Tests for locust_generator.py sanitization functions"""

    @pytest.fixture
    def generator(self):
        """Create a LocustTestGenerator instance for testing"""
        from devdox_ai_locust.locust_generator import LocustTestGenerator
        return LocustTestGenerator()

    @pytest.mark.parametrize("input_name,expected", [
        # Same test cases as ScenarioGenerator
        ("simple", "simple"),
        ("nested resources", "nested_resources"),
        ("kebab-case", "kebab_case"),
        ("api.v1.users", "api_v1_users"),
        ("user@domain", "userdomain"),
        ("123numeric", "n123numeric"),
        ("", "unnamed"),
        ("---", "unnamed"),
    ])
    def test_sanitize_identifier(self, generator, input_name, expected):
        """Test _sanitize_identifier handles various inputs correctly"""
        result = generator._sanitize_identifier(input_name)
        assert result == expected, f"Input '{input_name}' should become '{expected}', got '{result}'"

    @pytest.mark.parametrize("input_name,expected", [
        ("simple", "Simple"),
        ("nested resources", "NestedResources"),
        ("kebab-case", "KebabCase"),
        ("api.v1.users", "ApiV1Users"),
        ("123numeric", "N123numeric"),
        ("", "Unnamed"),
    ])
    def test_to_class_name(self, generator, input_name, expected):
        """Test _to_class_name creates valid PascalCase class names"""
        result = generator._to_class_name(input_name)
        assert result == expected, f"Input '{input_name}' should become '{expected}', got '{result}'"


class TestCliSanitization:
    """Tests for cli.py sanitize_dir_name function"""

    @pytest.fixture
    def sanitize_dir_name(self):
        """Return the sanitize_dir_name function (mirrors cli.py implementation)"""
        import re

        def sanitize_dir_name(name: str) -> str:
            name = name.lower().replace("-", "_").replace(" ", "_").replace(".", "_").replace("/", "_")
            name = re.sub(r'[^a-z0-9_]', '', name)
            name = re.sub(r'_+', '_', name).strip('_')
            return name or "unnamed"

        return sanitize_dir_name

    @pytest.mark.parametrize("input_name,expected", [
        # Basic cases
        ("Simple", "simple"),
        ("UPPERCASE", "uppercase"),

        # Spaces
        ("Nested Resources", "nested_resources"),
        ("Advanced Features", "advanced_features"),

        # Dashes
        ("kebab-case", "kebab_case"),
        ("API-Gateway", "api_gateway"),

        # Dots
        ("api.v1.users", "api_v1_users"),

        # Special characters
        ("User@Domain", "userdomain"),
        ("Path/Resource", "path_resource"),

        # Numbers
        ("123Tag", "123tag"),  # CLI version doesn't prefix numbers

        # Edge cases
        ("", "unnamed"),
        ("   ", "unnamed"),
        ("---", "unnamed"),
    ])
    def test_sanitize_dir_name(self, sanitize_dir_name, input_name, expected):
        """Test sanitize_dir_name creates valid directory names"""
        result = sanitize_dir_name(input_name)
        assert result == expected, f"Input '{input_name}' should become '{expected}', got '{result}'"


class TestIntegration:
    """Integration tests for naming consistency across modules"""

    def test_same_input_produces_consistent_output(self):
        """Test that the same input produces consistent file/class names across modules"""
        from devdox_ai_locust.locust_generator import LocustTestGenerator
        from devdox_ai_locust.utils.scenario_generator import ScenarioWorkflowGenerator
        from unittest.mock import Mock

        locust_gen = LocustTestGenerator()

        mock_client = Mock()
        mock_config = Mock()
        mock_config.model = "test"
        mock_config.max_tokens = 1000
        mock_config.temperature = 0.3
        mock_config.timeout = 30
        prompt_dir = Path(__file__).parent.parent / "src" / "devdox_ai_locust" / "prompt"

        scenario_gen = ScenarioWorkflowGenerator(
            prompt_dir=prompt_dir,
            ai_client=mock_client,
            ai_config=mock_config,
        )

        test_names = [
            "Nested Resources",
            "Advanced-Features",
            "api.v1.users",
            "user@test",
        ]

        for name in test_names:
            # File names should match
            locust_file = locust_gen._sanitize_identifier(name).lower()
            scenario_file = scenario_gen._sanitize_identifier(name).lower()
            assert locust_file == scenario_file, f"File names don't match for '{name}': {locust_file} vs {scenario_file}"

            # Class names should match
            locust_class = locust_gen._to_class_name(name)
            # For scenario_gen, we need to mock an endpoint
            endpoint = Mock()
            endpoint.operation_id = name
            endpoint.method = "GET"
            endpoint.path = "/test"
            scenario_class = scenario_gen._operation_to_class_name(endpoint)
            assert locust_class == scenario_class, f"Class names don't match for '{name}': {locust_class} vs {scenario_class}"

    def test_generated_names_are_valid_python_identifiers(self):
        """Test that all generated names are valid Python identifiers"""
        from devdox_ai_locust.locust_generator import LocustTestGenerator
        import keyword

        generator = LocustTestGenerator()

        # Various edge case inputs
        test_inputs = [
            "Nested Resources",
            "123-numeric-start",
            "special@chars#here",
            "path/to/resource",
            "dots.in.name",
            "   spaces   ",
            "",
            "---",
            "___",
        ]

        for input_name in test_inputs:
            # Test identifier
            identifier = generator._sanitize_identifier(input_name)
            assert identifier.isidentifier() or identifier == "unnamed", \
                f"'{identifier}' from '{input_name}' is not a valid Python identifier"
            assert not keyword.iskeyword(identifier), \
                f"'{identifier}' from '{input_name}' is a Python keyword"

            # Test class name
            class_name = generator._to_class_name(input_name)
            assert class_name.isidentifier() or class_name == "Unnamed", \
                f"'{class_name}' from '{input_name}' is not a valid Python identifier"
            assert not keyword.iskeyword(class_name), \
                f"'{class_name}' from '{input_name}' is a Python keyword"


class TestOpenAPIParserOperationId:
    """Tests for OpenAPI parser operation_id generation"""

    @pytest.fixture
    def parser(self):
        """Create an OpenAPIParser instance for testing"""
        from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser
        return OpenAPIParser()

    @pytest.mark.parametrize("method,path,expected", [
        # Basic cases
        ("GET", "/users", "get_users"),
        ("POST", "/users", "post_users"),
        ("PUT", "/users/{id}", "put_users_id"),
        ("DELETE", "/users/{userId}", "delete_users_userId"),
        ("PATCH", "/api/v1/users", "patch_api_v1_users"),

        # Multiple path parameters
        ("GET", "/users/{userId}/posts/{postId}", "get_users_userId_posts_postId"),

        # Dashes and dots in path
        ("GET", "/api-gateway/v1.0/users", "get_api_gateway_v1_0_users"),
        ("POST", "/path-with-dashes", "post_path_with_dashes"),

        # Special characters
        ("GET", "/users@domain", "get_usersdomain"),
        ("GET", "/path/to/resource", "get_path_to_resource"),

        # Empty or minimal paths - just method becomes the operation_id
        ("GET", "/", "get"),
        ("POST", "", "post"),
    ])
    def test_generate_operation_id(self, parser, method, path, expected):
        """Test _generate_operation_id creates valid operation IDs"""
        result = parser._generate_operation_id(method, path)
        assert result == expected, f"{method} {path} should become '{expected}', got '{result}'"

    def test_parse_endpoints_generates_operation_id(self, parser):
        """Test that parse_endpoints generates operation_id when missing from spec"""
        spec = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0"
paths:
  /users:
    get:
      summary: Get all users
      responses:
        '200':
          description: Success
    post:
      operationId: createUser
      summary: Create a user
      responses:
        '201':
          description: Created
  /users/{id}:
    get:
      summary: Get user by ID
      responses:
        '200':
          description: Success
"""
        parser.parse_schema(spec)
        endpoints = parser.parse_endpoints()

        # Find endpoints by path and method
        get_users = next(e for e in endpoints if e.path == "/users" and e.method == "GET")
        post_users = next(e for e in endpoints if e.path == "/users" and e.method == "POST")
        get_user_id = next(e for e in endpoints if e.path == "/users/{id}" and e.method == "GET")

        # Verify operation_id is always present
        assert get_users.operation_id == "get_users", "Missing operationId should be generated"
        assert post_users.operation_id == "createUser", "Existing operationId should be preserved"
        assert get_user_id.operation_id == "get_users_id", "Path params should be included"

    def test_operation_id_is_always_string(self, parser):
        """Test that operation_id is never None or empty after parsing"""
        spec = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0"
paths:
  /test:
    get:
      responses:
        '200':
          description: Success
    post:
      responses:
        '200':
          description: Success
    put:
      responses:
        '200':
          description: Success
    delete:
      responses:
        '200':
          description: Success
"""
        parser.parse_schema(spec)
        endpoints = parser.parse_endpoints()

        for endpoint in endpoints:
            assert endpoint.operation_id is not None, f"{endpoint.method} {endpoint.path} has None operation_id"
            assert endpoint.operation_id != "", f"{endpoint.method} {endpoint.path} has empty operation_id"
            assert isinstance(endpoint.operation_id, str), f"{endpoint.method} {endpoint.path} operation_id is not string"


class TestRealWorldExamples:
    """Tests with real-world OpenAPI tag and operation names"""

    @pytest.fixture
    def generator(self):
        from devdox_ai_locust.locust_generator import LocustTestGenerator
        return LocustTestGenerator()

    @pytest.mark.parametrize("tag_name,expected_file,expected_class", [
        # Common OpenAPI tags
        ("Users", "users", "Users"),
        ("User Management", "user_management", "UserManagement"),
        ("API Gateway", "api_gateway", "ApiGateway"),
        ("v1-endpoints", "v1_endpoints", "V1Endpoints"),
        ("Auth/OAuth2", "auth_oauth2", "AuthOauth2"),
        ("Nested Resources", "nested_resources", "NestedResources"),
        ("Advanced Features", "advanced_features", "AdvancedFeatures"),

        # Edge cases from real APIs
        ("pet-store.v1", "pet_store_v1", "PetStoreV1"),
        ("AWS S3 Operations", "aws_s3_operations", "AwsS3Operations"),
        ("GraphQL Queries", "graphql_queries", "GraphqlQueries"),
    ])
    def test_real_world_tags(self, generator, tag_name, expected_file, expected_class):
        """Test with real-world OpenAPI tag names"""
        file_name = generator._sanitize_identifier(tag_name).lower()
        class_name = generator._to_class_name(tag_name)

        assert file_name == expected_file, f"File name for '{tag_name}': expected '{expected_file}', got '{file_name}'"
        assert class_name == expected_class, f"Class name for '{tag_name}': expected '{expected_class}', got '{class_name}'"

    @pytest.mark.parametrize("operation_id,expected_file,expected_class", [
        # Common operation IDs
        ("getUsers", "getusers", "Getusers"),
        ("createUser", "createuser", "Createuser"),
        ("get-user-by-id", "get_user_by_id", "GetUserById"),
        ("POST_api_v1_users", "post_api_v1_users", "PostApiV1Users"),
        ("listAllItems", "listallitems", "Listallitems"),

        # Swagger/OpenAPI generated IDs
        ("UsersController_getUsers", "userscontroller_getusers", "UserscontrollerGetusers"),
        ("api.v1.users.list", "api_v1_users_list", "ApiV1UsersList"),
    ])
    def test_real_world_operations(self, generator, operation_id, expected_file, expected_class):
        """Test with real-world operation IDs"""
        file_name = generator._sanitize_identifier(operation_id).lower()
        class_name = generator._to_class_name(operation_id)

        assert file_name == expected_file, f"File name for '{operation_id}': expected '{expected_file}', got '{file_name}'"
        assert class_name == expected_class, f"Class name for '{operation_id}': expected '{expected_class}', got '{class_name}'"
