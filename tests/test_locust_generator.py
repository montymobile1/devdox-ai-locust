"""
Tests for locust_generator module
"""

from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

from devdox_ai_locust.locust_generator import LocustTestGenerator, TestDataConfig
from devdox_ai_locust.utils.open_ai_parser import (
    Endpoint,
    ParameterType,
)

mock_template = Mock()
mock_template.render.return_value = "# Test"

mock_env = Mock()
mock_env.get_template.return_value = mock_template


class TestTestDataConfig:
    """Test TestDataConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TestDataConfig()

        assert config.string_length == 10
        assert config.integer_min == 1
        assert config.integer_max == 1000
        assert config.array_size == 3
        assert config.use_realistic_data is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TestDataConfig(
            string_length=20,
            integer_min=10,
            integer_max=5000,
            array_size=5,
            use_realistic_data=False,
        )

        assert config.string_length == 20
        assert config.integer_min == 10
        assert config.integer_max == 5000
        assert config.array_size == 5
        assert config.use_realistic_data is False


class TestLocustTestGenerator:
    """Test LocustTestGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = LocustTestGenerator()

        assert isinstance(generator.test_config, TestDataConfig)
        assert generator.test_config.string_length == 10
        assert generator.generated_files == {}
        assert generator.auth_token is None
        assert generator.user_data == {}
        assert generator.request_count == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = TestDataConfig(string_length=25)
        generator = LocustTestGenerator(custom_config)

        assert generator.test_config.string_length == 25

    @patch("devdox_ai_locust.locust_generator.Path")
    @patch("pathlib.Path.mkdir")
    def test_find_project_root(self, mock_mkdir, mock_path):
        """Test finding project root."""
        mock_path.return_value = Path("/project/src/devdox_ai_locust/file.py")

        generator = LocustTestGenerator()
        root = generator._find_project_root()

        assert root == Path("/project/src/devdox_ai_locust")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("devdox_ai_locust.locust_generator.Environment")
    def test_setup_templates(self, mock_env):
        """Test template setup."""
        generator = LocustTestGenerator()

        # Should have created jinja environment
        assert hasattr(generator, "jinja_env")

    def test_fix_indent_valid_python(self):
        """Test fixing indentation for valid Python code."""
        generator = LocustTestGenerator()

        code = """
def test_function():
    print("hello")
    return True
"""
        files = {"test.py": code}

        result = generator.fix_indent(files)

        assert "test.py" in result
        assert "def test_function():" in result["test.py"]

    def test_fix_indent_invalid_python(self):
        """Test fixing indentation for invalid Python code."""
        generator = LocustTestGenerator()

        invalid_code = "def invalid_syntax(:"
        files = {"test.py": invalid_code}

        result = generator.fix_indent(files)

        # Should return original code if Black fails
        assert result["test.py"] == invalid_code

    def test_fix_indent_non_python_files(self):
        """Test fixing indentation for non-Python files."""
        generator = LocustTestGenerator()

        files = {"config.json": '{"key": "value"}', "readme.md": "# Title\n\nContent"}

        result = generator.fix_indent(files)

        def normalize(content_dict):
            return {k: "".join(v.split()) for k, v in content_dict.items()}

        normalized_files = normalize(files)
        normalized_result = normalize(result)

        # Compare
        assert normalized_files == normalized_result

    def test_generate_from_endpoints(self, sample_endpoints, sample_api_info):
        """Test generating files from endpoints."""
        generator = LocustTestGenerator()

        with (
            patch.object(generator, "_group_endpoints_by_tag") as mock_group,
            patch.object(generator, "generate_workflows") as mock_workflows,
            patch.object(generator, "_generate_main_locustfile") as mock_main,
            patch.object(generator, "_generate_test_data_file") as mock_test_data,
            patch.object(generator, "_generate_config_file") as mock_config,
            patch.object(generator, "_generate_utils_file") as mock_utils,
            patch.object(generator, "_generate_custom_flows_file") as mock_flows,
            patch.object(generator, "_generate_requirements_file") as mock_req,
            patch.object(generator, "_generate_readme_file") as mock_readme,
            patch.object(generator, "_generate_env_example") as mock_env,
        ):
            # Setup mocks
            mock_group.return_value = {"users": sample_endpoints}
            mock_workflows.return_value = []
            mock_main.return_value = "# Main locust file"
            mock_test_data.return_value = "# Test data"
            mock_config.return_value = "# Config"
            mock_utils.return_value = "# Utils"
            mock_flows.return_value = "# Custom flows"
            mock_req.return_value = "locust>=2.0.0"
            mock_readme.return_value = "# README"
            mock_env.return_value = "API_KEY=test"

            files, workflows, grouped = generator.generate_from_endpoints(
                sample_endpoints, sample_api_info
            )

            # Verify all expected files are generated
            expected_files = [
                "locustfile.py",
                "test_data.py",
                "config.py",
                "utils.py",
                "custom_flows.py",
                "requirements.txt",
                "README.md",
                ".env.example",
            ]

            for file_name in expected_files:
                assert file_name in files

    def test_group_endpoints_by_tag(self, sample_endpoints):
        """Test grouping endpoints by tags."""
        generator = LocustTestGenerator()

        grouped = generator._group_endpoints_by_tag(sample_endpoints)

        assert "users" in grouped
        assert "auth" in grouped

        # Check users group
        users_endpoints = grouped["users"]
        assert len(users_endpoints) == 3  # GET /users, POST /users, GET /users/{id}

        # Check auth group
        auth_endpoints = grouped["auth"]
        assert len(auth_endpoints) == 1  # POST /auth/login

    def test_group_endpoints_by_tag_with_auth_detection(self):
        """Test endpoint grouping with authentication detection."""
        generator = LocustTestGenerator()

        # Create endpoint that should be detected as auth
        auth_endpoint = Endpoint(
            path="/user/login",
            method="POST",
            operation_id="userLogin",
            summary="User login",
            description=None,
            parameters=[],
            request_body=None,
            responses=[],
            tags=["user"],  # Not auth tag, but path contains login
        )

        grouped = generator._group_endpoints_by_tag([auth_endpoint])

        # Should be in both Authentication and user groups
        assert "Authentication" in grouped
        assert "user" in grouped

    def test_generate_method_name(self):
        """Test generating method names from endpoints."""
        generator = LocustTestGenerator()

        # Test with operation ID
        endpoint_with_id = Mock()
        endpoint_with_id.operation_id = "getUserById"
        endpoint_with_id.path = "/users/{id}"
        endpoint_with_id.method = "GET"

        name = generator._generate_method_name(endpoint_with_id)
        assert name == "getUserById"

        # Test without operation ID
        endpoint_without_id = Mock()
        endpoint_without_id.operation_id = None
        endpoint_without_id.path = "/users/profile"
        endpoint_without_id.method = "GET"

        name = generator._generate_method_name(endpoint_without_id)
        assert name == "get_users_profile"

    def test_generate_method_name_cleanup(self):
        """Test method name cleanup for invalid characters."""
        generator = LocustTestGenerator()

        endpoint = Mock()
        endpoint.operation_id = "get-user-by-id!"
        endpoint.path = "/users/{id}"
        endpoint.method = "GET"

        name = generator._generate_method_name(endpoint)
        # Should clean up invalid characters
        assert name == "get_user_by_id"

    def test_generate_path_with_params(self):
        """Test generating path with parameters."""
        generator = LocustTestGenerator()

        endpoint = Mock()
        endpoint.path = "/users/{id}/posts/{postId}"
        endpoint.parameters = [
            Mock(name="id", location=ParameterType.PATH),
            Mock(name="postId", location=ParameterType.PATH),
        ]

        path = generator._generate_path_with_params(endpoint)
        assert path == "/users/{id}/posts/{postId}"

    def test_generate_path_params_code(self):
        """Test generating path parameters code."""
        generator = LocustTestGenerator()

        endpoint = SimpleNamespace()
        endpoint.parameters = [
            SimpleNamespace(name="id", location=ParameterType.PATH, type="integer"),
            SimpleNamespace(
                name="username", location=ParameterType.PATH, type="string"
            ),
        ]

        code = generator._generate_path_params_code(endpoint)

        assert "id = data_generator.generate_integer()" in code
        assert "username = data_generator.generate_string()" in code

    def test_generate_query_params_code(self):
        """Test generating query parameters code."""
        generator = LocustTestGenerator()

        endpoint = Mock()
        endpoint.parameters = [
            Mock(
                name="limit",
                location=ParameterType.QUERY,
                type="integer",
                required=True,
                default=None,
            ),
            Mock(
                name="filter",
                location=ParameterType.QUERY,
                type="string",
                required=False,
                default="active",
            ),
        ]

        code = generator._generate_query_params_code(endpoint)

        assert "params = {" in code
        assert "limit" in code

    def test_generate_request_body_code_json(self):
        """Test generating request body code for JSON."""
        generator = LocustTestGenerator()

        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.content_type = "application/json"
        endpoint.request_body.schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        code = generator._generate_request_body_code(endpoint)

        assert "json_data = data_generator.generate_json_data(" in code

    def test_generate_request_body_code_form(self):
        """Test generating request body code for form data."""
        generator = LocustTestGenerator()

        endpoint = Mock()
        endpoint.request_body = Mock()
        endpoint.request_body.content_type = "application/x-www-form-urlencoded"

        code = generator._generate_request_body_code(endpoint)

        assert "data = data_generator.generate_form_data()" in code

    def test_generate_request_body_code_none(self):
        """Test generating request body code when no body."""
        generator = LocustTestGenerator()

        endpoint = Mock()
        endpoint.request_body = None

        code = generator._generate_request_body_code(endpoint)

        assert code == "json_data = None"

    def test_get_task_weight(self):
        """Test getting task weights for different HTTP methods."""
        generator = LocustTestGenerator()

        assert generator._get_task_weight("GET") == 5
        assert generator._get_task_weight("POST") == 2
        assert generator._get_task_weight("PUT") == 1
        assert generator._get_task_weight("DELETE") == 1
        assert generator._get_task_weight("UNKNOWN") == 1

    def test_generate_task_method(self, sample_endpoints):
        """Test generating a complete task method."""
        generator = LocustTestGenerator()

        endpoint = sample_endpoints[0]  # GET /users

        task_method = generator._generate_task_method(endpoint)

        assert "@task(" in task_method
        assert "def " in task_method
        assert "make_request" in task_method
        assert endpoint.summary in task_method or "GET /users" in task_method

    def test_generate_task_method_error_handling(self):
        """Test task method generation with error handling."""
        generator = LocustTestGenerator()

        # Create a problematic endpoint
        endpoint = Mock()
        endpoint.operation_id = None
        endpoint.path = None
        endpoint.method = "GET"
        endpoint.summary = None
        endpoint.description = None
        endpoint.parameters = []
        endpoint.request_body = None

        # Should not crash, might return empty string
        result = generator._generate_task_method(endpoint)

        # Should either return valid code or empty string
        assert isinstance(result, str)

    def test_generate_default_task_method(self):
        """Test generating default task method."""
        generator = LocustTestGenerator()

        task_method = generator._generate_default_task_method()

        assert "@task(1)" in task_method
        assert "default_health_check" in task_method
        assert "/health" in task_method

    @patch.object(LocustTestGenerator, "_generate_task_method")
    def test_generate_main_locustfile(
        self, mock_task_method, sample_endpoints, sample_api_info
    ):
        """Test generating main locustfile."""
        generator = LocustTestGenerator()

        mock_task_method.return_value = "@task(1)\ndef test_task(self):\n    pass"

        with patch.object(generator, "_build_locustfile_template") as mock_build:
            mock_build.return_value = "# Generated locustfile"

            result = generator._generate_main_locustfile(
                sample_endpoints, sample_api_info, ["users", "auth"]
            )

            assert result == "# Generated locustfile"

    def test_indent_methods(self):
        """Test method indentation."""
        generator = LocustTestGenerator()

        methods = [
            "@task(1)\ndef test_method(self):\n    print('test')",
            "@task(2)\ndef another_method(self):\n    return True",
        ]

        result = generator._indent_methods(methods, indent_level=1)

        # Should be properly indented for class inclusion
        assert "    @task(1)" in result
        assert "    def test_method(self):" in result

    def test_generate_user_classes(self):
        """Test generating user classes."""
        generator = LocustTestGenerator()

        classes = generator._generate_user_classes()

        assert "LightUser" in classes
        assert "RegularUser" in classes
        assert "PowerUser" in classes
        assert "BaseAPIUser" in classes
        assert "BaseTaskMethods" in classes

    def test_generate_test_data_file(self):
        """Test generating test data file."""

        mock_template = Mock()
        mock_template.render.return_value = "# Test data file content"

        # Create a mock Jinja environment
        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        # Inject the mock environment
        generator = LocustTestGenerator(jinja_env=mock_env)
        result = generator._generate_test_data_file()

        assert result == "# Test data file content"
        mock_env.get_template.assert_called_once_with("test_data.py.j2")

    def test_generate_config_file(self, mock_jinja_env, sample_api_info):
        """Test generating config file."""

        mock_template = Mock()
        mock_template.render.return_value = "# Config file content"

        # Create a mock Jinja environment
        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        # Inject the mock environment
        generator = LocustTestGenerator(jinja_env=mock_env)
        result = generator._generate_config_file(sample_api_info)

        assert result == "# Config file content"
        mock_template.render.assert_called_once_with(api_info=sample_api_info)

    def test_generate_workflows(self, sample_endpoints, sample_api_info):
        """Test generating workflow files."""
        generator = LocustTestGenerator()

        grouped_endpoints = {
            "users": sample_endpoints[:2],
            "auth": sample_endpoints[2:],
        }

        with (
            patch.object(generator, "_generate_task_method") as mock_task,
            patch.object(generator, "_build_endpoint_template") as mock_build,
            patch.object(generator, "generate_base_common_file") as mock_base,
        ):
            mock_task.return_value = "@task(1)\ndef test(self): pass"
            mock_build.return_value = "# Endpoint template content"
            mock_base.return_value = "# Base workflow content"

            workflows = generator.generate_workflows(grouped_endpoints, sample_api_info)

            assert len(workflows) >= 2  # At least users and auth workflows
            assert any("users_workflow.py" in str(w) for w in workflows)

    def test_generate_workflows_error_handling(self, sample_api_info):
        """Test workflow generation with error handling."""
        mock_template = Mock()
        mock_template.render.return_value = "# BaseWorkflow"

        # Create a mock Jinja environment
        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        generator = LocustTestGenerator(jinja_env=mock_env)

        # Empty endpoints should still generate default workflows
        workflows = generator.generate_workflows({}, sample_api_info)

        # Should still generate base workflow
        assert len(workflows) >= 1


class TestLocustTestGeneratorTemplates:
    """Test template-related functionality."""

    def test_build_locustfile_template(self, sample_api_info):
        """Test building locustfile template."""
        generator = LocustTestGenerator()

        with patch.object(generator, "jinja_env") as mock_jinja_env:
            mock_template = Mock()
            mock_template.render.return_value = "# Generated locustfile"
            mock_jinja_env.get_template.return_value = mock_template

            result = generator._build_locustfile_template(
                sample_api_info, "# Task methods", ["users"]
            )

            assert result == "# Generated locustfile"

    def test_build_endpoint_template(self, sample_api_info):
        """Test building endpoint template."""
        generator = LocustTestGenerator()

        with patch.object(generator, "jinja_env") as mock_jinja_env:
            mock_template = Mock()
            mock_template.render.return_value = "# Endpoint template"
            mock_jinja_env.get_template.return_value = mock_template

            result = generator._build_endpoint_template(
                sample_api_info, "# Task methods", "users"
            )

            assert result == "# Endpoint template"

    def test_generate_env_example(self, sample_api_info):
        """Test generating .env.example file."""
        mock_template = Mock()
        mock_template.render.return_value = "API_BASE_URL=http://localhost:8000"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template
        generator = LocustTestGenerator(jinja_env=mock_env)
        result = generator._generate_env_example(sample_api_info)

        assert "API_BASE_URL" in result
        mock_template.render.assert_called_once()

    def test_generate_readme_file(self, sample_api_info):
        """Test generating README file."""
        mock_template = Mock()
        mock_template.render.return_value = "# Test API\n\nGenerated README"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template
        generator = LocustTestGenerator(jinja_env=mock_env)

        result = generator._generate_readme_file(sample_api_info)

        assert "# Test API" in result
        mock_template.render.assert_called_once()


class TestLocustTestGeneratorEdgeCases:
    """Test edge cases and error conditions."""

    def test_generate_from_endpoints_empty_list(self, sample_api_info):
        """Test generating from empty endpoints list."""
        generator = LocustTestGenerator(jinja_env=mock_env)

        files, workflows, grouped = generator.generate_from_endpoints(
            [], sample_api_info
        )

        # Should still generate basic files
        assert "locustfile.py" in files
        assert "requirements.txt" in files

    # def test_generate_task_method_with_complex_endpoint(self):
    #     """Test generating task method with complex endpoint."""
    #     generator = LocustTestGenerator()
    #
    #     # Create complex endpoint with many parameters
    #     endpoint = Mock()
    #     endpoint.operation_id = "complexOperation"
    #     endpoint.path = "/api/{version}/users/{id}/posts"
    #     endpoint.method = "POST"
    #     endpoint.summary = "Complex operation"
    #     endpoint.description = "A complex API operation"
    #     endpoint.parameters = [
    #         Mock(name="version", location=ParameterType.PATH, type="string", required=True),
    #         Mock(name="id", location=ParameterType.PATH, type="integer", required=True),
    #         Mock(name="limit", location=ParameterType.QUERY, type="integer", required=False),
    #         Mock(name="sort", location=ParameterType.QUERY, type="string", required=False)
    #     ]
    #     endpoint.request_body = Mock()
    #     endpoint.request_body.content_type = "application/json"
    #     endpoint.request_body.schema = {"type": "object"}
    #
    #     result = generator._generate_task_method(endpoint)
    #
    #     assert "complexOperation" in result or "complex_operation" in result
    #     assert "@task(" in result

    def test_parameter_generation_edge_cases(self):
        """Test parameter generation with edge cases."""
        generator = LocustTestGenerator()

        # Test parameter with enum
        param_with_enum = Mock()
        param_with_enum.name = "status"
        param_with_enum.type = "string"
        param_with_enum.enum = ["active", "inactive"]
        param_with_enum.required = True
        param_with_enum.default = None

        result = generator._generate_string_param(param_with_enum)
        assert "status" in result

    def test_fallback_locustfile_generation(self, sample_api_info):
        """Test fallback locustfile generation."""
        generator = LocustTestGenerator()

        with patch.object(generator, "jinja_env") as mock_jinja_env:
            mock_template = Mock()
            mock_template.render.return_value = "# Fallback locustfile"
            mock_jinja_env.get_template.return_value = mock_template

            result = generator._generate_fallback_locustfile(sample_api_info)

            assert result == "# Fallback locustfile"

    def test_template_error_handling(self, sample_api_info):
        """Test handling of template errors."""
        generator = LocustTestGenerator()

        with patch.object(generator, "jinja_env") as mock_jinja_env:
            mock_jinja_env.get_template.side_effect = Exception("Template error")

            # Should handle template errors gracefully
            result = generator._generate_env_example(sample_api_info)

            # Should return empty string or handle error
            assert isinstance(result, str)
