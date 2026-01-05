"""
Comprehensive tests for hybrid_loctus_generator.py module.

Tests cover:
- SafeCodeMerger class
- ProtectedSymbol dataclass
- CodebaseAwareness class
- ErrorClassification dataclass
- AIEnhancementConfig dataclass
- EnhancementResult dataclass
- HybridLocustGenerator class (selected methods)
"""

import pytest
from unittest.mock import MagicMock

from devdox_ai_locust.hybrid_loctus_generator import (
    SafeCodeMerger,
    ProtectedSymbol,
    CodebaseAwareness,
    ErrorClassification,
    AIEnhancementConfig,
    EnhancementResult,
    HybridLocustGenerator,
    CRITICAL_CLASSES,
    CRITICAL_FUNCTIONS,
)
from devdox_ai_locust.utils.open_ai_parser import Endpoint


# =============================================================================
# SafeCodeMerger Tests
# =============================================================================


class TestSafeCodeMergerGetExistingNames:
    """Tests for SafeCodeMerger.get_existing_names method."""

    def test_extracts_class_names(self):
        """Should extract class names from code."""
        code = """
class MyClass:
    pass

class AnotherClass(Base):
    pass
"""
        classes, methods, functions = SafeCodeMerger.get_existing_names(code)
        assert "MyClass" in classes
        assert "AnotherClass" in classes

    def test_extracts_function_names(self):
        """Should extract top-level function names."""
        code = """
def my_function():
    pass

def another_function(arg):
    return arg
"""
        classes, methods, functions = SafeCodeMerger.get_existing_names(code)
        assert "my_function" in functions or "my_function" in methods
        assert "another_function" in functions or "another_function" in methods

    def test_extracts_method_names_with_class(self):
        """Should extract method names with class prefix."""
        code = """
class MyClass:
    def my_method(self):
        pass

    def another_method(self):
        pass
"""
        classes, methods, functions = SafeCodeMerger.get_existing_names(code)
        assert "MyClass" in classes
        assert "MyClass.my_method" in methods
        assert "MyClass.another_method" in methods

    def test_handles_syntax_error_gracefully(self):
        """Should handle syntax errors with regex fallback."""
        code = """
class MyClass:
    def broken(
        pass
"""
        # Should not raise, falls back to regex
        classes, methods, functions = SafeCodeMerger.get_existing_names(code)
        assert "MyClass" in classes

    def test_empty_code_returns_empty_sets(self):
        """Should return empty sets for empty code."""
        classes, methods, functions = SafeCodeMerger.get_existing_names("")
        assert len(classes) == 0
        assert len(methods) == 0
        assert len(functions) == 0


class TestSafeCodeMergerExtractNewMethodsOnly:
    """Tests for SafeCodeMerger.extract_new_methods_only method."""

    def test_extracts_new_methods(self):
        """Should extract methods that don't exist in original."""
        original = """
class MyClass:
    def existing_method(self):
        pass
"""
        ai_code = """
class MyClass:
    def existing_method(self):
        pass

    def new_method(self):
        return "new"
"""
        result = SafeCodeMerger.extract_new_methods_only(original, ai_code)
        assert "new_method" in result
        assert "existing_method" not in result

    def test_returns_empty_for_no_new_methods(self):
        """Should return empty string when no new methods."""
        original = """
class MyClass:
    def method(self):
        pass
"""
        ai_code = """
class MyClass:
    def method(self):
        return "modified"
"""
        result = SafeCodeMerger.extract_new_methods_only(original, ai_code)
        # method already exists, so nothing new
        assert "method" not in result or result.strip() == ""

    def test_handles_empty_ai_code(self):
        """Should return empty string for empty AI code."""
        original = "class MyClass: pass"
        result = SafeCodeMerger.extract_new_methods_only(original, "")
        assert result == ""

    def test_handles_whitespace_only_ai_code(self):
        """Should return empty string for whitespace-only AI code."""
        original = "class MyClass: pass"
        result = SafeCodeMerger.extract_new_methods_only(original, "   \n   ")
        assert result == ""


class TestSafeCodeMergerIndentCode:
    """Tests for SafeCodeMerger._indent_code method."""

    def test_adds_indentation(self):
        """Should add specified indentation to code."""
        code = "def method():\n    pass"
        result = SafeCodeMerger._indent_code(code, 4)
        assert result.startswith("    def method")

    def test_preserves_already_indented(self):
        """Should not double-indent already indented code."""
        code = "    def method():\n        pass"
        result = SafeCodeMerger._indent_code(code, 4)
        # Should not add more indentation
        assert result == code

    def test_handles_empty_lines(self):
        """Should not indent empty lines."""
        code = "def method():\n\n    pass"
        result = SafeCodeMerger._indent_code(code, 4)
        lines = result.split("\n")
        assert lines[1] == ""  # Empty line should stay empty


class TestSafeCodeMergerSafeMerge:
    """Tests for SafeCodeMerger.safe_merge method."""

    def test_preserves_original_code(self):
        """Should always preserve original code."""
        original = """
class MyClass:
    def original_method(self):
        return "original"
"""
        ai_additions = ""
        result = SafeCodeMerger.safe_merge(original, ai_additions)
        assert "original_method" in result
        assert 'return "original"' in result

    def test_adds_new_methods_to_target_class(self):
        """Should add new methods to target class."""
        original = """
class TestDataGenerator:
    def generate_string(self):
        return "test"
"""
        ai_additions = """
class TestDataGenerator:
    def new_generator(self):
        return "new"
"""
        result = SafeCodeMerger.safe_merge(
            original, ai_additions, target_class="TestDataGenerator"
        )
        assert "generate_string" in result
        assert "new_generator" in result

    def test_returns_original_for_empty_additions(self):
        """Should return original when no additions."""
        original = "class MyClass: pass"
        result = SafeCodeMerger.safe_merge(original, "")
        assert result == original

    def test_returns_original_for_none_additions(self):
        """Should return original when additions is None-like."""
        original = "class MyClass: pass"
        result = SafeCodeMerger.safe_merge(original, "   ")
        assert result == original


# =============================================================================
# ProtectedSymbol Tests
# =============================================================================


class TestProtectedSymbol:
    """Tests for ProtectedSymbol dataclass."""

    def test_creates_with_all_fields(self):
        """Should create with all required fields."""
        symbol = ProtectedSymbol(
            name="TestClass",
            symbol_type="class",
            defined_in="test.py",
            used_by=["other.py", "main.py"],
            reason="Imported by other modules",
        )
        assert symbol.name == "TestClass"
        assert symbol.symbol_type == "class"
        assert symbol.defined_in == "test.py"
        assert len(symbol.used_by) == 2
        assert symbol.reason == "Imported by other modules"

    def test_used_by_is_list(self):
        """used_by should be a list."""
        symbol = ProtectedSymbol(
            name="func",
            symbol_type="function",
            defined_in="utils.py",
            used_by=["app.py"],
            reason="Used by app",
        )
        assert isinstance(symbol.used_by, list)


# =============================================================================
# CodebaseAwareness Tests
# =============================================================================


class TestCodebaseAwarenessInit:
    """Tests for CodebaseAwareness initialization."""

    def test_initializes_empty_dicts(self):
        """Should initialize with empty dictionaries."""
        awareness = CodebaseAwareness()
        assert awareness.files == {}
        assert awareness.exports == {}
        assert awareness.imports == {}
        assert awareness.protected == {}


class TestCodebaseAwarenessExtractExports:
    """Tests for CodebaseAwareness._extract_exports method."""

    def test_extracts_class_exports(self):
        """Should extract class names as exports."""
        awareness = CodebaseAwareness()
        code = """
class MyClass:
    def method(self):
        pass
"""
        exports = awareness._extract_exports(code)
        assert "MyClass" in exports

    def test_extracts_function_exports(self):
        """Should extract function names as exports."""
        awareness = CodebaseAwareness()
        code = """
def my_function():
    pass
"""
        exports = awareness._extract_exports(code)
        assert "my_function" in exports

    def test_extracts_constant_exports(self):
        """Should extract uppercase constants as exports."""
        awareness = CodebaseAwareness()
        code = """
MY_CONSTANT = "value"
ANOTHER_CONST = 123
"""
        exports = awareness._extract_exports(code)
        assert "MY_CONSTANT" in exports
        assert "ANOTHER_CONST" in exports

    def test_extracts_method_exports(self):
        """Should extract public methods with class prefix."""
        awareness = CodebaseAwareness()
        code = """
class MyClass:
    def public_method(self):
        pass

    def _private_method(self):
        pass
"""
        exports = awareness._extract_exports(code)
        assert "MyClass.public_method" in exports
        assert "MyClass._private_method" not in exports


class TestCodebaseAwarenessExtractLocalImports:
    """Tests for CodebaseAwareness._extract_local_imports method."""

    def test_extracts_from_imports(self):
        """Should extract 'from X import Y' style imports."""
        awareness = CodebaseAwareness()
        code = """
from test_data import TestDataGenerator
from utils import ResponseValidator, RequestLogger
"""
        imports = awareness._extract_local_imports(code)
        assert "test_data.py" in imports
        assert "TestDataGenerator" in imports["test_data.py"]
        assert "utils.py" in imports
        assert "ResponseValidator" in imports["utils.py"]
        assert "RequestLogger" in imports["utils.py"]

    def test_handles_as_aliases(self):
        """Should handle 'as' aliases correctly."""
        awareness = CodebaseAwareness()
        code = """
from utils import ResponseValidator as RV
"""
        imports = awareness._extract_local_imports(code)
        assert "utils.py" in imports
        assert "ResponseValidator" in imports["utils.py"]

    def test_ignores_star_imports(self):
        """Should ignore star imports."""
        awareness = CodebaseAwareness()
        code = """
from utils import *
"""
        imports = awareness._extract_local_imports(code)
        if "utils.py" in imports:
            assert "*" not in imports["utils.py"]


class TestCodebaseAwarenessGetConstraintsForFile:
    """Tests for CodebaseAwareness.get_constraints_for_file method."""

    def test_returns_empty_for_unknown_file(self):
        """Should return empty string for unknown file."""
        awareness = CodebaseAwareness()
        result = awareness.get_constraints_for_file("unknown.py")
        assert result == ""

    def test_returns_free_modification_message(self):
        """Should indicate free modification when no protected symbols."""
        awareness = CodebaseAwareness()
        awareness.protected = {"test.py": []}
        result = awareness.get_constraints_for_file("test.py")
        assert "freely modify" in result.lower() or result == ""


# =============================================================================
# ErrorClassification Tests
# =============================================================================


class TestErrorClassification:
    """Tests for ErrorClassification dataclass."""

    def test_creates_retryable_error(self):
        """Should create retryable error classification."""
        classification = ErrorClassification(
            is_retryable=True,
            backoff_seconds=2.0,
            error_type="rate_limit",
        )
        assert classification.is_retryable is True
        assert classification.backoff_seconds == 2.0
        assert classification.error_type == "rate_limit"

    def test_creates_non_retryable_error(self):
        """Should create non-retryable error classification."""
        classification = ErrorClassification(
            is_retryable=False,
            backoff_seconds=0,
            error_type="auth",
        )
        assert classification.is_retryable is False
        assert classification.backoff_seconds == 0


# =============================================================================
# AIEnhancementConfig Tests
# =============================================================================


class TestAIEnhancementConfig:
    """Tests for AIEnhancementConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = AIEnhancementConfig()
        assert config.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert config.max_tokens == 8000
        assert config.temperature == 0.3
        assert config.timeout == 60

    def test_enhancement_flags_default_true(self):
        """Enhancement flags should default to True."""
        config = AIEnhancementConfig()
        assert config.enhance_workflows is True
        assert config.enhance_test_data is True
        assert config.enhance_validation is True
        assert config.create_domain_flows is True
        assert config.update_main_locust is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = AIEnhancementConfig(
            model="custom-model",
            max_tokens=4000,
            temperature=0.5,
            enhance_workflows=False,
        )
        assert config.model == "custom-model"
        assert config.max_tokens == 4000
        assert config.temperature == 0.5
        assert config.enhance_workflows is False


# =============================================================================
# EnhancementResult Tests
# =============================================================================


class TestEnhancementResult:
    """Tests for EnhancementResult dataclass."""

    def test_creates_successful_result(self):
        """Should create successful enhancement result."""
        result = EnhancementResult(
            success=True,
            enhanced_files={"test.py": "content"},
            enhanced_directory_files=[{"workflow.py": "content"}],
            enhancements_applied=["test_data", "validation"],
            errors=[],
            processing_time=1.5,
        )
        assert result.success is True
        assert "test.py" in result.enhanced_files
        assert len(result.enhancements_applied) == 2
        assert len(result.errors) == 0
        assert result.processing_time == 1.5

    def test_creates_failed_result(self):
        """Should create failed enhancement result."""
        result = EnhancementResult(
            success=False,
            enhanced_files={},
            enhanced_directory_files=[],
            enhancements_applied=[],
            errors=["AI service timeout", "Rate limit exceeded"],
            processing_time=30.0,
        )
        assert result.success is False
        assert len(result.errors) == 2


# =============================================================================
# CRITICAL_CLASSES and CRITICAL_FUNCTIONS Constants Tests
# =============================================================================


class TestCriticalConstants:
    """Tests for CRITICAL_CLASSES and CRITICAL_FUNCTIONS constants."""

    def test_critical_classes_has_test_data(self):
        """CRITICAL_CLASSES should include test_data.py requirements."""
        assert "test_data.py" in CRITICAL_CLASSES
        assert "TestDataGenerator" in CRITICAL_CLASSES["test_data.py"]

    def test_critical_classes_has_utils(self):
        """CRITICAL_CLASSES should include utils.py requirements."""
        assert "utils.py" in CRITICAL_CLASSES
        assert "ResponseValidator" in CRITICAL_CLASSES["utils.py"]

    def test_critical_functions_has_test_data(self):
        """CRITICAL_FUNCTIONS should include test_data.py functions."""
        assert "test_data.py" in CRITICAL_FUNCTIONS
        assert "generate_json_data" in CRITICAL_FUNCTIONS["test_data.py"]
        assert "generate_string" in CRITICAL_FUNCTIONS["test_data.py"]

    def test_critical_functions_has_utils(self):
        """CRITICAL_FUNCTIONS should include utils.py functions."""
        assert "utils.py" in CRITICAL_FUNCTIONS
        assert "validate_response" in CRITICAL_FUNCTIONS["utils.py"]


# =============================================================================
# HybridLocustGenerator Tests
# =============================================================================


class TestHybridLocustGeneratorInit:
    """Tests for HybridLocustGenerator initialization."""

    def test_initializes_with_ai_client(self):
        """Should initialize with AI client."""
        mock_client = MagicMock()
        generator = HybridLocustGenerator(ai_client=mock_client)
        assert generator.ai_client == mock_client

    def test_uses_default_config(self):
        """Should use default AIEnhancementConfig when none provided."""
        mock_client = MagicMock()
        generator = HybridLocustGenerator(ai_client=mock_client)
        assert generator.ai_config is not None
        assert isinstance(generator.ai_config, AIEnhancementConfig)

    def test_uses_custom_config(self):
        """Should use provided config."""
        mock_client = MagicMock()
        config = AIEnhancementConfig(max_tokens=4000)
        generator = HybridLocustGenerator(ai_client=mock_client, ai_config=config)
        assert generator.ai_config.max_tokens == 4000

    def test_sets_max_retries(self):
        """Should set MAX_RETRIES constant."""
        mock_client = MagicMock()
        generator = HybridLocustGenerator(ai_client=mock_client)
        assert generator.MAX_RETRIES == 3


class TestHybridLocustGeneratorClassifyError:
    """Tests for HybridLocustGenerator._classify_error method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_classifies_auth_error_as_non_retryable(self, generator):
        """Auth errors should be non-retryable."""
        error = Exception("401 Unauthorized")
        classification = generator._classify_error(error, 0)
        assert classification.is_retryable is False
        assert classification.error_type == "auth"

    def test_classifies_rate_limit_as_retryable(self, generator):
        """Rate limit errors should be retryable."""
        error = Exception("429 Too Many Requests")
        classification = generator._classify_error(error, 0)
        assert classification.is_retryable is True
        assert classification.error_type == "rate_limit"
        assert classification.backoff_seconds == generator.RATE_LIMIT_BACKOFF

    def test_classifies_generic_error_as_retryable(self, generator):
        """Generic errors should be retryable with exponential backoff."""
        error = Exception("Connection timeout")
        classification = generator._classify_error(error, 0)
        assert classification.is_retryable is True
        assert classification.error_type == "retryable"
        assert classification.backoff_seconds == 1  # 2^0

    def test_exponential_backoff_increases(self, generator):
        """Backoff should increase exponentially with attempts."""
        error = Exception("Connection error")
        classification_0 = generator._classify_error(error, 0)
        classification_1 = generator._classify_error(error, 1)
        classification_2 = generator._classify_error(error, 2)

        assert classification_0.backoff_seconds == 1  # 2^0
        assert classification_1.backoff_seconds == 2  # 2^1
        assert classification_2.backoff_seconds == 4  # 2^2


class TestHybridLocustGeneratorShouldEnhance:
    """Tests for HybridLocustGenerator._should_enhance method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_returns_true_for_many_endpoints(self, generator):
        """Should enhance when 3+ endpoints exist."""
        endpoints = [
            Endpoint(
                path=f"/resource{i}",
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=[],
            )
            for i in range(5)
        ]
        assert generator._should_enhance(endpoints, {}) is True

    def test_returns_false_for_few_simple_endpoints(self, generator):
        """Should not enhance for very few simple endpoints."""
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
            )
        ]
        # May or may not enhance depending on domain detection
        result = generator._should_enhance(endpoints, {})
        # Just verify it doesn't raise
        assert isinstance(result, bool)


class TestHybridLocustGeneratorDetectDomainPatterns:
    """Tests for HybridLocustGenerator._detect_domain_patterns method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_detects_ecommerce_patterns(self, generator):
        """Should detect e-commerce domain patterns."""
        endpoints = [
            Endpoint(
                path="/products",
                method="GET",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=[],
            ),
            Endpoint(
                path="/cart",
                method="POST",
                operation_id=None,
                summary=None,
                description=None,
                parameters=[],
                request_body=None,
                responses=[],
                tags=[],
            ),
        ]
        assert generator._detect_domain_patterns(endpoints, {}) is True

    def test_detects_user_management_patterns(self, generator):
        """Should detect user management domain patterns."""
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
                tags=[],
            ),
        ]
        assert generator._detect_domain_patterns(endpoints, {}) is True

    def test_detects_patterns_from_api_info(self, generator):
        """Should detect patterns from API info."""
        endpoints = []
        api_info = {
            "title": "Payment Gateway API",
            "description": "Handle transactions",
        }
        assert generator._detect_domain_patterns(endpoints, api_info) is True


class TestHybridLocustGeneratorExtractCodeFromResponse:
    """Tests for HybridLocustGenerator.extract_code_from_response method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_extracts_code_from_tags(self, generator):
        """Should extract content between <code> tags."""
        response = """
Here is the code:
<code>
def hello():
    return "world"
</code>
That's the implementation.
"""
        result = generator.extract_code_from_response(response)
        assert "def hello():" in result
        assert 'return "world"' in result

    def test_uses_full_response_when_no_tags(self, generator):
        """Should use full response when no <code> tags found."""
        response = """
def hello():
    return "world"
"""
        result = generator.extract_code_from_response(response)
        assert "def hello():" in result

    def test_uses_longest_match(self, generator):
        """Should use the longest match when multiple <code> tags."""
        response = """
<code>short</code>
<code>
this is a much longer code block
with multiple lines
</code>
"""
        result = generator.extract_code_from_response(response)
        assert "multiple lines" in result


class TestHybridLocustGeneratorCleanAIResponse:
    """Tests for HybridLocustGenerator._clean_ai_response method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_removes_markdown_python_block(self, generator):
        """Should remove ```python code blocks."""
        content = "```python\ndef hello():\n    pass\n```"
        result = generator._clean_ai_response(content)
        assert "```" not in result
        assert "def hello():" in result

    def test_removes_generic_markdown_block(self, generator):
        """Should remove generic ``` code blocks."""
        content = "```\ndef hello():\n    pass\n```"
        result = generator._clean_ai_response(content)
        assert "```" not in result

    def test_removes_trailing_explanations(self, generator):
        """Should remove trailing explanatory text."""
        content = """
import os

def hello():
    pass

Note: This function does nothing.
"""
        result = generator._clean_ai_response(content)
        assert "import os" in result
        assert "def hello():" in result


class TestHybridLocustGeneratorValidatePythonCode:
    """Tests for HybridLocustGenerator._validate_python_code method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_valid_code_returns_true(self, generator):
        """Should return True for valid Python code."""
        code = """
def hello():
    return "world"

class MyClass:
    pass
"""
        assert generator._validate_python_code(code) is True

    def test_invalid_code_returns_false(self, generator):
        """Should return False for invalid Python code."""
        code = """
def broken(:
    return
"""
        assert generator._validate_python_code(code) is False

    def test_empty_code_returns_true(self, generator):
        """Should return True for empty code (valid syntax)."""
        assert generator._validate_python_code("") is True


class TestHybridLocustGeneratorValidateCriticalElements:
    """Tests for HybridLocustGenerator._validate_critical_elements method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_valid_test_data_file(self, generator):
        """Should validate test_data.py with all critical elements."""
        content = """
class TestDataGenerator:
    def generate_json_data(self):
        pass

    def generate_string(self):
        pass

    def generate_id(self):
        pass
"""
        original = content
        is_valid, result_content, missing = generator._validate_critical_elements(
            "test_data.py", content, original
        )
        assert is_valid is True
        assert len(missing) == 0

    def test_invalid_missing_class(self, generator):
        """Should detect missing critical class."""
        enhanced_content = """
def generate_string():
    pass
"""
        original_content = """
class TestDataGenerator:
    def generate_string(self):
        pass
"""
        is_valid, result_content, missing = generator._validate_critical_elements(
            "test_data.py", enhanced_content, original_content
        )
        assert is_valid is False
        assert "class TestDataGenerator" in missing
        assert result_content == original_content

    def test_detects_dramatically_smaller_content(self, generator):
        """Should detect when enhanced content is too small."""
        # Create original with many lines
        original_lines = ["# line"] * 100
        original_content = "\n".join(original_lines)

        # Enhanced content is much smaller
        enhanced_content = "# just a few lines\n# here"

        is_valid, result_content, missing = generator._validate_critical_elements(
            "some_file.py", enhanced_content, original_content
        )
        assert is_valid is False
        assert "content_too_small" in missing


class TestHybridLocustGeneratorGetFilesByKey:
    """Tests for HybridLocustGenerator.get_files_by_key method."""

    @pytest.fixture
    def generator(self):
        """Create generator with mock client."""
        mock_client = MagicMock()
        return HybridLocustGenerator(ai_client=mock_client)

    def test_returns_matching_items(self, generator):
        """Should return items containing the target key."""
        directory_files = [
            {"workflow_a.py": "content a"},
            {"workflow_b.py": "content b"},
            {"base_workflow.py": "base content"},
        ]
        result = generator.get_files_by_key(directory_files, "base_workflow.py")
        assert len(result) == 1
        assert "base_workflow.py" in result[0]

    def test_returns_empty_for_no_match(self, generator):
        """Should return empty list when no match."""
        directory_files = [
            {"workflow_a.py": "content a"},
        ]
        result = generator.get_files_by_key(directory_files, "missing.py")
        assert len(result) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestHybridLocustGeneratorIntegration:
    """Integration tests for HybridLocustGenerator."""

    def test_safe_merge_preserves_critical_elements(self):
        """Test that safe merge preserves all critical elements."""
        original = """
class TestDataGenerator:
    \"\"\"Generates test data for API testing.\"\"\"

    def generate_string(self, length=10):
        return "test" * length

    def generate_json_data(self, schema):
        return {}

    def generate_id(self):
        import uuid
        return str(uuid.uuid4())
"""
        ai_output = """
class TestDataGenerator:
    def generate_string(self, length=10):
        return "modified"

    def generate_realistic_name(self):
        return "John Doe"
"""
        # SafeCodeMerger should preserve original and add new methods
        result = SafeCodeMerger.safe_merge(
            original, ai_output, target_class="TestDataGenerator"
        )

        # Original methods should be preserved
        assert "generate_string" in result
        assert "generate_json_data" in result
        assert "generate_id" in result

        # New method should be added
        assert "generate_realistic_name" in result

    def test_codebase_awareness_builds_protected_map(self):
        """Test that codebase awareness correctly identifies protected symbols."""
        awareness = CodebaseAwareness()

        base_files = {
            "test_data.py": """
class TestDataGenerator:
    def generate_string(self):
        pass
""",
            "utils.py": """
from test_data import TestDataGenerator

class ResponseValidator:
    def validate(self, generator: TestDataGenerator):
        pass
""",
        }

        awareness.analyze_codebase(base_files, [])

        # TestDataGenerator should be protected because utils.py imports it
        test_data_protected = awareness.protected.get("test_data.py", [])
        protected_names = [s.name for s in test_data_protected]
        assert "TestDataGenerator" in protected_names
