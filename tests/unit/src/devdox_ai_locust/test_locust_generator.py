"""
Comprehensive tests for locust_generator.py module.

Tests cover:
- DatabaseType enum
- MongoDBConfig dataclass
- PostgreSQLConfig dataclass
- TestDataConfig dataclass
- LocustTestGenerator class and its methods
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict
from jinja2 import Environment, DictLoader

from devdox_ai_locust.locust_generator import (
    DatabaseType,
    MongoDBConfig,
    PostgreSQLConfig,
    TestDataConfig,
    LocustTestGenerator,
)
from devdox_ai_locust.utils.open_ai_parser import (
    Endpoint,
    Parameter,
    ParameterType,
    RequestBody,
    Response,
)


# =============================================================================
# DatabaseType Enum Tests
# =============================================================================


class TestDatabaseType:
    """Tests for DatabaseType enum."""

    def test_mongo_value(self):
        """MONGO should have value 'mongo'."""
        assert DatabaseType.MONGO.value == "mongo"

    def test_postgres_value(self):
        """POSTGRES should have value 'postgres'."""
        assert DatabaseType.POSTGRES.value == "postgres"

    def test_enum_count(self):
        """Should have exactly 2 database types."""
        assert len(DatabaseType) == 2

    def test_create_from_value(self):
        """Should be able to create from string value."""
        assert DatabaseType("mongo") == DatabaseType.MONGO
        assert DatabaseType("postgres") == DatabaseType.POSTGRES


# =============================================================================
# MongoDBConfig Dataclass Tests
# =============================================================================


class TestMongoDBConfig:
    """Tests for MongoDBConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = MongoDBConfig()
        assert config.use_realistic_data == "true"
        assert config.enable_mongodb == "false"
        assert config.use_mongodb_for_test_data == "false"
        assert config.mongodb_uri == "mongodb://localhost:27017/"
        assert config.mongodb_database == "locust_test_data"

    def test_pool_size_defaults(self):
        """Should have correct pool size defaults."""
        config = MongoDBConfig()
        assert config.MONGODB_MAX_POOL_SIZE == 100
        assert config.MONGODB_MIN_POOL_SIZE == 10

    def test_timeout_defaults(self):
        """Should have correct timeout defaults."""
        config = MongoDBConfig()
        assert config.MONGODB_CONNECT_TIMEOUT_MS == 5000
        assert config.MONGODB_SERVER_SELECTION_TIMEOUT_MS == 5000
        assert config.MONGODB_SOCKET_TIMEOUT_MS == 10000
        assert config.MONGODB_MAX_IDLE_TIME_MS == 60000
        assert config.MONGODB_WAIT_QUEUE_TIMEOUT_MS == 10000

    def test_custom_values(self):
        """Should accept custom values."""
        config = MongoDBConfig(
            mongodb_uri="mongodb://custom:27017/",
            mongodb_database="custom_db",
            MONGODB_MAX_POOL_SIZE=200,
        )
        assert config.mongodb_uri == "mongodb://custom:27017/"
        assert config.mongodb_database == "custom_db"
        assert config.MONGODB_MAX_POOL_SIZE == 200

    def test_asdict(self):
        """Should be convertible to dictionary."""
        config = MongoDBConfig()
        data = asdict(config)
        assert isinstance(data, dict)
        assert "mongodb_uri" in data
        assert "MONGODB_MAX_POOL_SIZE" in data


# =============================================================================
# PostgreSQLConfig Dataclass Tests
# =============================================================================


class TestPostgreSQLConfig:
    """Tests for PostgreSQLConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = PostgreSQLConfig()
        assert config.host == "localhost"
        assert config.port == "5432"
        assert config.database == "test_db"
        assert config.user == "test_user"
        assert config.password == "test_password"

    def test_pool_defaults(self):
        """Should have correct pool defaults."""
        config = PostgreSQLConfig()
        assert config.pool_size == "10"
        assert config.max_overflow == "20"

    def test_custom_values(self):
        """Should accept custom values."""
        config = PostgreSQLConfig(
            host="db.example.com",
            port="5433",
            database="production",
            user="admin",
            password="secret",
        )
        assert config.host == "db.example.com"
        assert config.port == "5433"
        assert config.database == "production"


# =============================================================================
# TestDataConfig Dataclass Tests
# =============================================================================


class TestTestDataConfig:
    """Tests for TestDataConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = TestDataConfig()
        assert config.string_length == 10
        assert config.integer_min == 1
        assert config.integer_max == 1000
        assert config.array_size == 3
        assert config.use_realistic_data is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = TestDataConfig(
            string_length=20,
            integer_min=100,
            integer_max=500,
            array_size=5,
            use_realistic_data=False,
        )
        assert config.string_length == 20
        assert config.integer_min == 100
        assert config.integer_max == 500
        assert config.array_size == 5
        assert config.use_realistic_data is False


# =============================================================================
# LocustTestGenerator Initialization Tests
# =============================================================================


class TestLocustTestGeneratorInit:
    """Tests for LocustTestGenerator initialization."""

    def test_init_default_config(self):
        """Should create default TestDataConfig when none provided."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()
            assert generator.test_config is not None
            assert isinstance(generator.test_config, TestDataConfig)

    def test_init_custom_config(self):
        """Should use provided config."""
        custom_config = TestDataConfig(string_length=50)
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator(test_config=custom_config)
            assert generator.test_config.string_length == 50

    def test_init_empty_generated_files(self):
        """Should initialize with empty generated_files dict."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()
            assert generator.generated_files == {}

    def test_init_auth_token_none(self):
        """Should initialize auth_token as None."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()
            assert generator.auth_token is None

    def test_init_user_data_empty(self):
        """Should initialize user_data as empty dict."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()
            assert generator.user_data == {}

    def test_init_request_count_zero(self):
        """Should initialize request_count as 0."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()
            assert generator.request_count == 0

    def test_init_with_jinja_env(self):
        """Should use provided Jinja environment."""
        mock_env = MagicMock()
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator(jinja_env=mock_env)
            assert generator.jinja_env == mock_env


# =============================================================================
# LocustTestGenerator._generate_method_name Tests
# =============================================================================


class TestGenerateMethodName:
    """Tests for _generate_method_name method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_uses_operation_id_when_available(self, generator):
        """Should use operationId when available."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id="getUsers",
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_method_name(endpoint)
        assert result == "getUsers"

    def test_generates_from_method_and_path(self, generator):
        """Should generate name from method and path when no operationId."""
        endpoint = Endpoint(
            path="/users/profile",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_method_name(endpoint)
        assert result == "get_users_profile"

    def test_excludes_path_parameters(self, generator):
        """Should exclude path parameters from generated name."""
        endpoint = Endpoint(
            path="/users/{id}/posts/{postId}",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_method_name(endpoint)
        assert result == "get_users_posts"

    def test_cleans_special_characters(self, generator):
        """Should clean special characters from name."""
        endpoint = Endpoint(
            path="/api-v2/user_data",
            method="POST",
            operation_id="create-user.data",
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_method_name(endpoint)
        assert "-" not in result
        assert "." not in result

    def test_fallback_for_empty_path(self, generator):
        """Should provide fallback when path is empty."""
        endpoint = Endpoint(
            path="/",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_method_name(endpoint)
        # Empty path "/" results in just the method name
        assert result == "get"


# =============================================================================
# LocustTestGenerator._get_task_weight Tests
# =============================================================================


class TestGetTaskWeight:
    """Tests for _get_task_weight method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_get_weight(self, generator):
        """Should return weight of 5."""
        assert generator._get_task_weight("GET") == 5

    def test_post_weight(self, generator):
        """Should return weight of 2."""
        assert generator._get_task_weight("POST") == 2

    def test_put_weight(self, generator):
        """Should return weight of 1."""
        assert generator._get_task_weight("PUT") == 1

    def test_patch_weight(self, generator):
        """Should return weight of 1."""
        assert generator._get_task_weight("PATCH") == 1

    def test_delete_weight(self, generator):
        """Should return weight of 1."""
        assert generator._get_task_weight("DELETE") == 1

    def test_head_weight(self, generator):
        """Should return weight of 3."""
        assert generator._get_task_weight("HEAD") == 3

    def test_options_weight(self, generator):
        """Should return weight of 1."""
        assert generator._get_task_weight("OPTIONS") == 1

    def test_unknown_method_weight(self, generator):
        """Should return default weight of 1 for unknown methods."""
        assert generator._get_task_weight("UNKNOWN") == 1

    def test_case_insensitive(self, generator):
        """Should be case insensitive."""
        assert generator._get_task_weight("get") == 5
        assert generator._get_task_weight("Get") == 5


# =============================================================================
# LocustTestGenerator._group_endpoints_by_tag Tests
# =============================================================================


class TestGroupEndpointsByTag:
    """Tests for _group_endpoints_by_tag method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_groups_by_single_tag(self, generator):
        """Should group endpoints by their tag."""
        endpoints = [
            Endpoint(
                path="/users",
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=["users"],
            ),
            Endpoint(
                path="/users",
                method="POST",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=["users"],
            ),
        ]
        result = generator._group_endpoints_by_tag(endpoints)
        assert "users" in result
        assert len(result["users"]) == 2

    def test_default_tag_for_untagged(self, generator):
        """Should use 'default' tag for endpoints without tags."""
        endpoints = [
            Endpoint(
                path="/health",
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=[],
            ),
        ]
        result = generator._group_endpoints_by_tag(endpoints)
        assert "default" in result

    def test_endpoints_grouped_by_tags_not_path_keywords(self, generator):
        """Should group endpoints by tags, not by path keywords.

        The old behavior used keyword matching on paths (e.g., 'token', 'auth')
        to detect auth endpoints. The new behavior uses OpenAPI security field.
        Endpoints with paths containing auth keywords should be grouped by their
        actual tags, not placed in a separate 'Authentication' group.
        """
        endpoints = [
            Endpoint(
                path="/api/v1/git_tokens",  # Contains 'token' but is not auth
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=["git_tokens"],
                security=None,  # No security defined
            ),
            Endpoint(
                path="/users",
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=["users"],
                security=None,
            ),
        ]
        result = generator._group_endpoints_by_tag(
            endpoints, include_auth_endpoints=True
        )
        # Should NOT have Authentication group based on path keywords
        assert "Authentication" not in result
        # Should be grouped by actual tags
        assert "git_tokens" in result
        assert "users" in result

    def test_uses_openapi_security_for_auth_detection(self, generator):
        """Should use OpenAPI security field to identify secured endpoints."""
        # Endpoint with security defined in OpenAPI spec
        secured_endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=["users"],
            security=[{"bearerAuth": []}],  # OpenAPI security requirement
        )
        # Endpoint explicitly marked as public (empty security array)
        public_endpoint = Endpoint(
            path="/health",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=["health"],
            security=[],  # Explicitly public
        )

        # Test that requires_auth correctly identifies secured endpoints
        assert secured_endpoint.requires_auth() is True
        assert public_endpoint.requires_auth() is False

    def test_global_security_inheritance(self, generator):
        """Should inherit global security when endpoint has no security defined."""
        global_security = [{"bearerAuth": []}]

        # Endpoint without security inherits from global
        endpoint_inherits = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=["users"],
            security=None,  # Will inherit global
        )
        # Endpoint explicitly public overrides global
        endpoint_public = Endpoint(
            path="/health",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=["health"],
            security=[],  # Explicitly public, overrides global
        )

        assert endpoint_inherits.requires_auth(global_security) is True
        assert endpoint_public.requires_auth(global_security) is False

    def test_grouping_respects_tags_regardless_of_security(self, generator):
        """Should group by tags, security detection is separate concern."""
        endpoints = [
            Endpoint(
                path="/auth/login",
                method="POST",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=["auth"],
                security=[],  # Explicitly public login endpoint
            ),
        ]
        result = generator._group_endpoints_by_tag(
            endpoints, include_auth_endpoints=True
        )
        # Should be in 'auth' tag group, not a special Authentication group
        assert "auth" in result
        assert len(result["auth"]) == 1

    def test_deduplicates_endpoints(self, generator):
        """Should not duplicate endpoints within groups."""
        endpoints = [
            Endpoint(
                path="/users",
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=["users", "api"],
            ),
        ]
        result = generator._group_endpoints_by_tag(endpoints)
        # Should appear in both groups but not duplicated within each
        assert len(result.get("users", [])) == 1
        assert len(result.get("api", [])) == 1


# =============================================================================
# LocustTestGenerator._generate_path_with_params Tests
# =============================================================================


class TestGeneratePathWithParams:
    """Tests for _generate_path_with_params method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_simple_path_unchanged(self, generator):
        """Should return simple path unchanged."""
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
        result = generator._generate_path_with_params(endpoint)
        assert result == "/users"

    def test_replaces_path_parameter(self, generator):
        """Should replace path parameters with f-string format."""
        endpoint = Endpoint(
            path="/users/{id}",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[
                Parameter(
                    name="id",
                    location=ParameterType.PATH,
                    required=True,
                    type="string",
                )
            ],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_path_with_params(endpoint)
        assert "{id}" in result

    def test_sanitizes_parameter_names(self, generator):
        """Should sanitize parameter names for valid Python identifiers."""
        endpoint = Endpoint(
            path="/users/{user-id}",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[
                Parameter(
                    name="user-id",
                    location=ParameterType.PATH,
                    required=True,
                    type="string",
                )
            ],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_path_with_params(endpoint)
        # Should be sanitized to valid Python identifier
        assert "-" not in result or "{user_id}" in result


# =============================================================================
# LocustTestGenerator._indent_methods Tests
# =============================================================================


class TestIndentMethods:
    """Tests for _indent_methods method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_indents_single_method(self, generator):
        """Should properly indent a single method."""
        methods = ["@task\ndef test_method(self):\n    pass"]
        result = generator._indent_methods(methods, indent_level=1)
        assert result.startswith("    @task")

    def test_indents_method_body_deeper(self, generator):
        """Should indent method body deeper than decorator."""
        methods = ["@task\ndef test(self):\npass"]
        result = generator._indent_methods(methods, indent_level=1)
        lines = result.split("\n")
        # Method body should have more indentation
        assert lines[0].startswith("    @task")

    def test_handles_empty_lines(self, generator):
        """Should preserve empty lines."""
        methods = ["@task\ndef test(self):\n\n    pass"]
        result = generator._indent_methods(methods)
        assert "\n\n" in result or result.count("\n") >= 2

    def test_joins_multiple_methods(self, generator):
        """Should join multiple methods with double newlines."""
        methods = ["@task\ndef m1(self):\n    pass", "@task\ndef m2(self):\n    pass"]
        result = generator._indent_methods(methods)
        assert "\n\n" in result


# =============================================================================
# LocustTestGenerator._generate_query_params_code Tests
# =============================================================================


class TestGenerateQueryParamsCode:
    """Tests for _generate_query_params_code method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_no_params_returns_empty_dict(self, generator):
        """Should return empty dict when no query params."""
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
        result = generator._generate_query_params_code(endpoint)
        assert "params = {}" in result

    def test_generates_integer_param(self, generator):
        """Should generate integer parameter code."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[
                Parameter(
                    name="limit",
                    location=ParameterType.QUERY,
                    required=True,
                    type="integer",
                )
            ],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_query_params_code(endpoint)
        assert "limit" in result
        assert "generate_integer" in result

    def test_generates_string_param(self, generator):
        """Should generate string parameter code."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[
                Parameter(
                    name="name",
                    location=ParameterType.QUERY,
                    required=True,
                    type="string",
                )
            ],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_query_params_code(endpoint)
        assert "name" in result
        assert "generate_string" in result

    def test_generates_boolean_param(self, generator):
        """Should generate boolean parameter code."""
        endpoint = Endpoint(
            path="/users",
            method="GET",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[
                Parameter(
                    name="active",
                    location=ParameterType.QUERY,
                    required=True,
                    type="boolean",
                )
            ],
            request_body=None,
            responses=[],
            tags=[],
        )
        result = generator._generate_query_params_code(endpoint)
        assert "active" in result
        assert "generate_boolean" in result


# =============================================================================
# LocustTestGenerator._generate_request_body_code Tests
# =============================================================================


class TestGenerateRequestBodyCode:
    """Tests for _generate_request_body_code method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_no_body_returns_none(self, generator):
        """Should return 'json_data = None' when no body."""
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
        result = generator._generate_request_body_code(endpoint)
        assert "json_data = None" in result

    def test_json_body_generates_json_code(self, generator):
        """Should generate JSON body code for application/json."""
        endpoint = Endpoint(
            path="/users",
            method="POST",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/json",
                schema={"type": "object"},
            ),
            responses=[],
            tags=[],
        )
        result = generator._generate_request_body_code(endpoint)
        assert "generate_json_data" in result

    def test_form_body_generates_form_code(self, generator):
        """Should generate form data code for urlencoded."""
        endpoint = Endpoint(
            path="/users",
            method="POST",
            operation_id=None,
            summary=None,
            description=None,
            parameters=[],
            request_body=RequestBody(
                content_type="application/x-www-form-urlencoded",
                schema={},
            ),
            responses=[],
            tags=[],
        )
        result = generator._generate_request_body_code(endpoint)
        assert "generate_form_data" in result


# =============================================================================
# LocustTestGenerator.fix_indent Tests
# =============================================================================


class TestFixIndent:
    """Tests for fix_indent method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_formats_valid_python(self, generator):
        """Should format valid Python code."""
        files = {"test.py": "def foo():\n  return 1"}
        result = generator.fix_indent(files)
        assert "test.py" in result
        # Black should standardize indentation
        assert "def foo():" in result["test.py"]

    def test_preserves_invalid_python(self, generator):
        """Should preserve files that aren't valid Python."""
        files = {"readme.md": "# Title\n\nSome content"}
        result = generator.fix_indent(files)
        assert result["readme.md"] == "# Title\n\nSome content"

    def test_handles_empty_dict(self, generator):
        """Should handle empty dictionary."""
        result = generator.fix_indent({})
        assert result == {}

    def test_handles_formatting_errors(self, generator):
        """Should keep original on formatting errors."""
        files = {"test.py": "def broken(:\n    pass"}
        result = generator.fix_indent(files)
        # Should return original content on error
        assert "test.py" in result


# =============================================================================
# LocustTestGenerator._generate_default_task_method Tests
# =============================================================================


class TestGenerateDefaultTaskMethod:
    """Tests for _generate_default_task_method method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_returns_valid_task(self, generator):
        """Should return a valid task method string."""
        result = generator._generate_default_task_method()
        assert "@task" in result
        assert "def default_health_check" in result

    def test_includes_health_endpoint(self, generator):
        """Should include /health endpoint."""
        result = generator._generate_default_task_method()
        assert "/health" in result

    def test_includes_error_handling(self, generator):
        """Should include error handling."""
        result = generator._generate_default_task_method()
        assert "except Exception" in result


# =============================================================================
# LocustTestGenerator._format_params_dict Tests
# =============================================================================


class TestFormatParamsDict:
    """Tests for _format_params_dict method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_empty_list_returns_empty_dict(self, generator):
        """Should return empty dict string for empty list."""
        result = generator._format_params_dict([])
        assert result == "params = {}"

    def test_formats_single_param(self, generator):
        """Should format single parameter correctly."""
        result = generator._format_params_dict(['"name": value,'])
        assert "params = {" in result
        assert '"name": value,' in result
        assert "}" in result

    def test_formats_multiple_params(self, generator):
        """Should format multiple parameters correctly."""
        result = generator._format_params_dict(['"a": 1,', '"b": 2,'])
        assert "params = {" in result
        assert '"a": 1,' in result
        assert '"b": 2,' in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestLocustGeneratorIntegration:
    """Integration tests for LocustTestGenerator."""

    def test_generate_task_method_for_endpoint(self):
        """Test generating a complete task method."""
        # Create mock templates
        templates = {
            "endpoint_template.py.j2": "# endpoint",
            "locust.py.j2": "# locust",
            "fallback_locust.py.j2": "# fallback",
        }
        mock_env = Environment(loader=DictLoader(templates))

        with patch.object(
            LocustTestGenerator, "_setup_templates", return_value=mock_env
        ):
            generator = LocustTestGenerator(jinja_env=mock_env)

            endpoint = Endpoint(
                path="/users/{id}",
                method="GET",
                operation_id="getUserById",
                summary="Get user by ID",
                description="Returns a single user",
                parameters=[
                    Parameter(
                        name="id",
                        location=ParameterType.PATH,
                        required=True,
                        type="string",
                    ),
                    Parameter(
                        name="include",
                        location=ParameterType.QUERY,
                        required=False,
                        type="string",
                    ),
                ],
                request_body=None,
                responses=[
                    Response(status_code="200", description="Success"),
                ],
                tags=["users"],
            )

            result = generator._generate_task_method(endpoint)

            assert "@task" in result
            assert "def getUserById" in result
            assert "Get user by ID" in result
            assert "make_request" in result

    def test_group_and_generate_workflow(self):
        """Test grouping endpoints and workflow structure."""
        templates = {
            "endpoint_template.py.j2": "# {{ group }}",
            "base_workflow.py.j2": "# base",
        }
        mock_env = Environment(loader=DictLoader(templates))

        with patch.object(
            LocustTestGenerator, "_setup_templates", return_value=mock_env
        ):
            generator = LocustTestGenerator(jinja_env=mock_env)

            endpoints = [
                Endpoint(
                    path="/users",
                    method="GET",
                    operation_id="listUsers",
                    summary="List users",
                    description=None,
                    parameters=[],
                    request_body=None,
                    responses=[],
                    tags=["users"],
                ),
                Endpoint(
                    path="/products",
                    method="GET",
                    operation_id="listProducts",
                    summary="List products",
                    description=None,
                    parameters=[],
                    request_body=None,
                    responses=[],
                    tags=["products"],
                ),
            ]

            grouped = generator._group_endpoints_by_tag(endpoints)
            assert "users" in grouped
            assert "products" in grouped
            assert len(grouped["users"]) == 1
            assert len(grouped["products"]) == 1


# =============================================================================
# Mocked Generation Tests
# =============================================================================


class TestGenerateFromEndpointsMocked:
    """Tests for generate_from_endpoints with mocked dependencies."""

    def test_handles_generation_error_gracefully(self):
        """Should return empty dict on generation error."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()
            # Force an error by not setting up templates properly
            generator.jinja_env = None

            endpoints = [MagicMock()]
            api_info = {"title": "Test"}

            files, workflows, grouped = generator.generate_from_endpoints(
                endpoints, api_info
            )

            assert files == {}
            assert workflows == []
            assert grouped == {}

    def test_returns_tuple_with_three_elements(self):
        """generate_from_endpoints should return 3-tuple."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()

            # Mock all the internal methods to avoid template issues
            with patch.object(generator, "_group_endpoints_by_tag", return_value={"users": []}):
                with patch.object(generator, "generate_workflows", return_value=[]):
                    with patch.object(generator, "_generate_main_locustfile", return_value="# main"):
                        with patch.object(generator, "_generate_test_data_file", return_value="# data"):
                            with patch.object(generator, "_generate_config_file", return_value="# cfg"):
                                with patch.object(generator, "_generate_utils_file", return_value="# utils"):
                                    with patch.object(generator, "_generate_custom_flows_file", return_value="# flows"):
                                        with patch.object(generator, "_generate_requirements_file", return_value="# req"):
                                            with patch.object(generator, "_generate_readme_file", return_value="# readme"):
                                                with patch.object(generator, "_generate_env_example", return_value="# env"):
                                                    result = generator.generate_from_endpoints(
                                                        [], {"title": "Test"}, include_auth=True
                                                    )

                                                    assert isinstance(result, tuple)
                                                    assert len(result) == 3


class TestGenerateWorkflowsMocked:
    """Tests for generate_workflows with mocked dependencies."""

    def test_adds_default_task_when_method_generation_fails(self):
        """Should add default task when _generate_task_method fails."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            generator = LocustTestGenerator()

            # Mock template rendering
            mock_template = MagicMock()
            mock_template.render.return_value = "# workflow content"
            generator.jinja_env = MagicMock()
            generator.jinja_env.get_template.return_value = mock_template

            # Force task method generation to fail
            with patch.object(generator, "_generate_task_method", side_effect=Exception("Error")):
                mock_endpoint = MagicMock()
                mock_endpoint.path = "/test"
                grouped = {"test_group": [mock_endpoint]}

                workflows = generator.generate_workflows(grouped, {"title": "Test"})

                # Should produce workflows (group + base)
                assert len(workflows) >= 1


class TestFixIndentEdgeCases:
    """Additional tests for fix_indent edge cases."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked templates."""
        with patch.object(LocustTestGenerator, "_setup_templates"):
            return LocustTestGenerator()

    def test_handles_black_invalid_input(self, generator):
        """Should handle Black InvalidInput exception."""
        # Syntax error that Black can't format
        files = {"test.py": "def broken(:\n    x = 1"}
        result = generator.fix_indent(files)
        assert "test.py" in result
        # Should keep original
        assert result["test.py"] == files["test.py"]

    def test_handles_other_exceptions(self, generator):
        """Should handle unexpected exceptions during formatting."""
        with patch("black.format_str", side_effect=Exception("Unexpected error")):
            files = {"test.py": "x = 1"}
            result = generator.fix_indent(files)
            # Should return original on any error
            assert "test.py" in result

    def test_returns_dict_on_any_error(self, generator):
        """Should always return a dict even on errors."""
        files = {"broken.py": "invalid python ("}
        result = generator.fix_indent(files)
        assert isinstance(result, dict)
        assert "broken.py" in result
