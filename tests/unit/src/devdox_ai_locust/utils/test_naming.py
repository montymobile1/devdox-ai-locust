"""
Comprehensive tests for devdox_ai_locust.utils.naming module.

Tests cover:
- DefaultNamingStrategy class and all its methods
- Sanitization of identifiers
- PascalCase conversion
- Workflow module/filename/class naming
- Method name generation with uniqueness
- Parameter variable naming
- Path parameter handling
- Edge cases and error handling
"""

from devdox_ai_locust.utils.naming import (
    DefaultNamingStrategy,
    default_naming,
    to_workflow_module,
    to_workflow_filename,
    to_task_methods_class,
    to_api_user_class,
    to_method_name,
    to_param_var,
)


class TestDefaultNamingStrategyInit:
    """Tests for DefaultNamingStrategy initialization."""

    def test_init_creates_empty_used_names(self):
        """Test that initialization creates an empty set of used names."""
        strategy = DefaultNamingStrategy()
        assert strategy._used_method_names == set()

    def test_multiple_instances_are_independent(self):
        """Test that multiple instances don't share state."""
        strategy1 = DefaultNamingStrategy()
        strategy2 = DefaultNamingStrategy()

        strategy1.to_method_name("op1", "GET", "/test", ensure_unique=True)

        assert "op1" in strategy1._used_method_names
        assert "op1" not in strategy2._used_method_names


class TestSanitizeIdentifier:
    """Tests for _sanitize_identifier method."""

    def test_empty_string_returns_unnamed(self):
        """Test that empty string returns 'unnamed'."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("") == "unnamed"

    def test_none_like_empty_returns_unnamed(self):
        """Test that None-like inputs are handled."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("") == "unnamed"

    def test_replaces_non_word_characters(self):
        """Test that non-word characters are replaced with underscores."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("user-name") == "user_name"
        assert strategy._sanitize_identifier("user.name") == "user_name"
        assert strategy._sanitize_identifier("user@name") == "user_name"
        assert strategy._sanitize_identifier("user/name") == "user_name"

    def test_removes_consecutive_underscores(self):
        """Test that consecutive underscores are reduced to one."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("user__name") == "user_name"
        assert strategy._sanitize_identifier("user---name") == "user_name"
        assert strategy._sanitize_identifier("a___b___c") == "a_b_c"

    def test_strips_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are removed."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("_user_") == "user"
        assert strategy._sanitize_identifier("___user___") == "user"

    def test_converts_to_lowercase(self):
        """Test that result is lowercase."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("UserName") == "username"
        assert strategy._sanitize_identifier("USER_NAME") == "user_name"
        assert strategy._sanitize_identifier("CamelCase") == "camelcase"

    def test_handles_digit_prefix(self):
        """Test that identifiers starting with digits get 'n_' prefix."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("123abc") == "n_123abc"
        assert strategy._sanitize_identifier("1user") == "n_1user"
        assert strategy._sanitize_identifier("2") == "n_2"

    def test_handles_python_keywords(self):
        """Test that Python keywords get underscore suffix."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("class") == "class_"
        assert strategy._sanitize_identifier("import") == "import_"
        assert strategy._sanitize_identifier("def") == "def_"
        assert strategy._sanitize_identifier("return") == "return_"
        assert strategy._sanitize_identifier("if") == "if_"

    def test_handles_special_characters_only(self):
        """Test string with only special characters returns unnamed."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("@#$%") == "unnamed"
        assert strategy._sanitize_identifier("---") == "unnamed"

    def test_handles_unicode_characters(self):
        """Test that unicode characters are handled."""
        strategy = DefaultNamingStrategy()
        # Unicode word characters should be kept
        result = strategy._sanitize_identifier("café")
        assert result == "café"

    def test_handles_mixed_separators(self):
        """Test handling of mixed separator types."""
        strategy = DefaultNamingStrategy()
        assert (
            strategy._sanitize_identifier("user-name_test.value")
            == "user_name_test_value"
        )


class TestToPascalCase:
    """Tests for _to_pascal_case method."""

    def test_empty_string_returns_unnamed(self):
        """Test that empty string returns 'Unnamed'."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("") == "Unnamed"

    def test_snake_case_conversion(self):
        """Test conversion of snake_case to PascalCase."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("user_management") == "UserManagement"
        assert strategy._to_pascal_case("api_key") == "ApiKey"
        assert strategy._to_pascal_case("git_tokens") == "GitTokens"

    def test_kebab_case_conversion(self):
        """Test conversion of kebab-case to PascalCase."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("user-management") == "UserManagement"
        assert strategy._to_pascal_case("api-key") == "ApiKey"

    def test_single_word(self):
        """Test single word conversion."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("user") == "User"
        assert strategy._to_pascal_case("authentication") == "Authentication"

    def test_handles_digit_prefix(self):
        """Test handling of digit prefix in PascalCase."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("123abc") == "N123abc"
        assert strategy._to_pascal_case("1_user") == "N1User"

    def test_handles_special_characters(self):
        """Test handling of special characters."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("user@name") == "UserName"
        assert strategy._to_pascal_case("user.name") == "UserName"

    def test_handles_only_special_characters(self):
        """Test string with only special characters returns Unnamed."""
        strategy = DefaultNamingStrategy()
        assert strategy._to_pascal_case("@#$") == "Unnamed"


class TestToWorkflowModule:
    """Tests for to_workflow_module method."""

    def test_basic_conversion(self):
        """Test basic group label conversion."""
        strategy = DefaultNamingStrategy()
        assert strategy.to_workflow_module("User Management") == "user_management"
        assert strategy.to_workflow_module("api-key") == "api_key"
        assert strategy.to_workflow_module("GitTokens") == "gittokens"

    def test_handles_special_characters(self):
        """Test handling of special characters in group labels."""
        strategy = DefaultNamingStrategy()
        assert strategy.to_workflow_module("user/management") == "user_management"
        assert strategy.to_workflow_module("api@key") == "api_key"


class TestToWorkflowFilename:
    """Tests for to_workflow_filename method."""

    def test_basic_conversion(self):
        """Test basic filename generation."""
        strategy = DefaultNamingStrategy()
        assert (
            strategy.to_workflow_filename("User Management")
            == "user_management_workflow.py"
        )
        assert strategy.to_workflow_filename("api-key") == "api_key_workflow.py"

    def test_appends_workflow_suffix(self):
        """Test that _workflow.py suffix is appended."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_workflow_filename("authentication")
        assert result.endswith("_workflow.py")
        assert result == "authentication_workflow.py"


class TestToTaskMethodsClass:
    """Tests for to_task_methods_class method."""

    def test_basic_conversion(self):
        """Test basic class name generation."""
        strategy = DefaultNamingStrategy()
        assert (
            strategy.to_task_methods_class("User Management")
            == "UserManagementTaskMethods"
        )
        assert strategy.to_task_methods_class("api-key") == "ApiKeyTaskMethods"
        assert (
            strategy.to_task_methods_class("authentication")
            == "AuthenticationTaskMethods"
        )

    def test_appends_task_methods_suffix(self):
        """Test that TaskMethods suffix is appended."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_task_methods_class("users")
        assert result.endswith("TaskMethods")


class TestToApiUserClass:
    """Tests for to_api_user_class method."""

    def test_basic_conversion(self):
        """Test basic API user class name generation."""
        strategy = DefaultNamingStrategy()
        assert strategy.to_api_user_class("User Management") == "UserManagementAPIUser"
        assert strategy.to_api_user_class("api-key") == "ApiKeyAPIUser"

    def test_appends_api_user_suffix(self):
        """Test that APIUser suffix is appended."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_api_user_class("users")
        assert result.endswith("APIUser")


class TestToMethodName:
    """Tests for to_method_name method."""

    def test_uses_operation_id_when_provided(self):
        """Test that operation_id is used when provided."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=False
        )
        assert result == "getusers"

    def test_generates_from_method_and_path(self):
        """Test generation from HTTP method and path."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_method_name(None, "GET", "/users", ensure_unique=False)
        assert result == "get_users"

        result = strategy.to_method_name(
            None, "POST", "/api/v1/users", ensure_unique=False
        )
        assert result == "post_api_v1_users"

    def test_excludes_path_parameters(self):
        """Test that path parameters are excluded from name."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_method_name(
            None, "GET", "/users/{id}", ensure_unique=False
        )
        assert result == "get_users"
        assert "{id}" not in result

    def test_ensure_unique_adds_suffix(self):
        """Test that ensure_unique adds suffix for duplicates."""
        strategy = DefaultNamingStrategy()

        result1 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=True
        )
        result2 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=True
        )
        result3 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=True
        )

        assert result1 == "getusers"
        assert result2 == "getusers_2"
        assert result3 == "getusers_3"

    def test_ensure_unique_false_allows_duplicates(self):
        """Test that ensure_unique=False allows duplicate names."""
        strategy = DefaultNamingStrategy()

        result1 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=False
        )
        result2 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=False
        )

        assert result1 == result2 == "getusers"

    def test_handles_empty_path(self):
        """Test handling of empty or root path."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_method_name(None, "GET", "/", ensure_unique=False)
        # Root path "/" results in just the method name
        assert result == "get"

    def test_handles_complex_paths(self):
        """Test handling of complex paths."""
        strategy = DefaultNamingStrategy()
        result = strategy.to_method_name(
            None, "DELETE", "/api/v2/users/{userId}/posts/{postId}", ensure_unique=False
        )
        assert result == "delete_api_v2_users_posts"


class TestToParamVar:
    """Tests for to_param_var method."""

    def test_basic_conversion(self):
        """Test basic parameter variable conversion."""
        strategy = DefaultNamingStrategy()
        assert strategy.to_param_var("user-id") == "user_id"
        assert strategy.to_param_var("api_key") == "api_key"
        assert strategy.to_param_var("Content-Type") == "content_type"

    def test_handles_digit_prefix(self):
        """Test handling of digit prefix."""
        strategy = DefaultNamingStrategy()
        assert strategy.to_param_var("123abc") == "n_123abc"


class TestToPathWithSafeParams:
    """Tests for to_path_with_safe_params method."""

    def test_converts_parameter_names(self):
        """Test conversion of parameter names in path."""
        strategy = DefaultNamingStrategy()

        # Create mock parameters
        class MockParam:
            def __init__(self, name, location_value):
                self.name = name
                self.location = type("obj", (object,), {"value": location_value})()

        params = [
            MockParam("user-id", "path"),
            MockParam("post-id", "path"),
        ]

        result = strategy.to_path_with_safe_params(
            "/users/{user-id}/posts/{post-id}", params
        )
        assert result == "/users/{user_id}/posts/{post_id}"

    def test_ignores_non_path_params(self):
        """Test that non-path parameters are ignored."""
        strategy = DefaultNamingStrategy()

        class MockParam:
            def __init__(self, name, location_value):
                self.name = name
                self.location = type("obj", (object,), {"value": location_value})()

        params = [
            MockParam("user-id", "path"),
            MockParam("limit", "query"),  # Query param should be ignored
        ]

        result = strategy.to_path_with_safe_params("/users/{user-id}", params)
        assert result == "/users/{user_id}"
        assert "{limit}" not in result


class TestResetUsedNames:
    """Tests for reset_used_names method."""

    def test_clears_used_names(self):
        """Test that reset clears all used names."""
        strategy = DefaultNamingStrategy()

        strategy.to_method_name("op1", "GET", "/test", ensure_unique=True)
        strategy.to_method_name("op2", "POST", "/test", ensure_unique=True)

        assert len(strategy._used_method_names) == 2

        strategy.reset_used_names()

        assert len(strategy._used_method_names) == 0

    def test_allows_reuse_after_reset(self):
        """Test that names can be reused after reset."""
        strategy = DefaultNamingStrategy()

        result1 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=True
        )
        strategy.reset_used_names()
        result2 = strategy.to_method_name(
            "getUsers", "GET", "/users", ensure_unique=True
        )

        assert result1 == result2 == "getusers"


class TestGlobalDefaultNaming:
    """Tests for global default_naming instance."""

    def test_default_naming_exists(self):
        """Test that default_naming instance exists."""
        assert default_naming is not None
        assert isinstance(default_naming, DefaultNamingStrategy)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_to_workflow_module_function(self):
        """Test to_workflow_module convenience function."""
        result = to_workflow_module("User Management")
        assert result == "user_management"

    def test_to_workflow_filename_function(self):
        """Test to_workflow_filename convenience function."""
        result = to_workflow_filename("users")
        assert result == "users_workflow.py"

    def test_to_task_methods_class_function(self):
        """Test to_task_methods_class convenience function."""
        result = to_task_methods_class("authentication")
        assert result == "AuthenticationTaskMethods"

    def test_to_api_user_class_function(self):
        """Test to_api_user_class convenience function."""
        result = to_api_user_class("users")
        assert result == "UsersAPIUser"

    def test_to_method_name_function(self):
        """Test to_method_name convenience function."""
        result = to_method_name("getUser", "GET", "/users/{id}", ensure_unique=False)
        assert result == "getuser"

    def test_to_param_var_function(self):
        """Test to_param_var convenience function."""
        result = to_param_var("user-id")
        assert result == "user_id"


class TestNamingStrategyProtocol:
    """Tests for NamingStrategy protocol compliance."""

    def test_default_naming_strategy_implements_protocol(self):
        """Test that DefaultNamingStrategy implements NamingStrategy protocol."""
        strategy = DefaultNamingStrategy()

        # Check that all required methods exist
        assert hasattr(strategy, "to_workflow_module")
        assert hasattr(strategy, "to_workflow_filename")
        assert hasattr(strategy, "to_task_methods_class")
        assert hasattr(strategy, "to_method_name")
        assert hasattr(strategy, "to_param_var")

        # Check that methods are callable
        assert callable(strategy.to_workflow_module)
        assert callable(strategy.to_workflow_filename)
        assert callable(strategy.to_task_methods_class)
        assert callable(strategy.to_method_name)
        assert callable(strategy.to_param_var)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_name(self):
        """Test handling of very long names."""
        strategy = DefaultNamingStrategy()
        long_name = "a" * 1000
        result = strategy._sanitize_identifier(long_name)
        assert len(result) <= 1000
        assert result.isidentifier() or result.endswith("_")

    def test_whitespace_only(self):
        """Test handling of whitespace-only input."""
        strategy = DefaultNamingStrategy()
        assert strategy._sanitize_identifier("   ") == "unnamed"
        assert strategy._sanitize_identifier("\t\n") == "unnamed"

    def test_numeric_only(self):
        """Test handling of numeric-only input."""
        strategy = DefaultNamingStrategy()
        result = strategy._sanitize_identifier("12345")
        assert result == "n_12345"
        assert result.isidentifier()

    def test_mixed_case_preserved_structure(self):
        """Test that structure is preserved while converting to lowercase."""
        strategy = DefaultNamingStrategy()
        result = strategy._sanitize_identifier("getUserByID")
        assert result == "getuserbyid"

    def test_lowercase_python_keywords(self):
        """Test handling of lowercase Python keywords.

        Note: The sanitizer lowercases input first, so 'False' becomes 'false'
        which is no longer a keyword. Only lowercase keywords get suffixed.
        """
        import keyword

        strategy = DefaultNamingStrategy()

        # Test keywords that remain keywords after lowercasing
        lowercase_keywords = [kw for kw in keyword.kwlist if kw.lower() == kw]
        for kw in lowercase_keywords:
            result = strategy._sanitize_identifier(kw)
            assert result.endswith("_"), f"Keyword {kw} not properly suffixed"
            assert result == f"{kw}_"

    def test_capitalized_keywords_become_lowercase(self):
        """Test that capitalized keywords like False/True/None become lowercase."""
        strategy = DefaultNamingStrategy()
        # These are lowercased first, so they're no longer keywords
        assert strategy._sanitize_identifier("False") == "false"
        assert strategy._sanitize_identifier("True") == "true"
        assert strategy._sanitize_identifier("None") == "none"
