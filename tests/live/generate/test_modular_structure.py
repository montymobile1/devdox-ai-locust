"""Live tests for SOLID-based modular output structure."""
import os
import ast
import pytest
from pathlib import Path


class TestModularOutputStructure:
    """Tests for the modular output directory structure."""

    def test_data_package_exists(self, output_dir):
        """Test data package is created."""
        data_dir = Path(output_dir) / "data"
        if not data_dir.exists():
            pytest.skip("Modular output not generated")

        assert (data_dir / "__init__.py").exists()
        assert (data_dir / "base_generator.py").exists()

    def test_scenarios_package_exists(self, output_dir):
        """Test scenarios package is created."""
        scenarios_dir = Path(output_dir) / "scenarios"
        if not scenarios_dir.exists():
            pytest.skip("Modular output not generated")

        assert (scenarios_dir / "__init__.py").exists()
        assert (scenarios_dir / "base_scenario.py").exists()

    def test_auth_package_exists(self, output_dir):
        """Test auth package is created."""
        auth_dir = Path(output_dir) / "auth"
        if not auth_dir.exists():
            pytest.skip("Modular output not generated")

        assert (auth_dir / "__init__.py").exists()

    def test_workflows_package_exists(self, output_dir):
        """Test workflows package is created."""
        workflows_dir = Path(output_dir) / "workflows"
        if not workflows_dir.exists():
            pytest.skip("Modular output not generated")

        assert (workflows_dir / "__init__.py").exists()


class TestModularDataGenerators:
    """Tests for data generator files."""

    def test_base_generator_has_utilities(self, output_dir):
        """Test base generator has required utility methods."""
        file_path = Path(output_dir) / "data" / "base_generator.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        assert "class BaseDataGenerator" in content
        assert "def generate_id" in content
        assert "def generate_string" in content
        assert "def generate_email" in content

    def test_valid_data_generator_syntax(self, output_dir):
        """Test valid data generator has valid Python syntax."""
        file_path = Path(output_dir) / "data" / "valid_data.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        # Should parse without error
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in valid_data.py: {e}")

    def test_invalid_data_generator_syntax(self, output_dir):
        """Test invalid data generator has valid Python syntax."""
        file_path = Path(output_dir) / "data" / "invalid_data.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in invalid_data.py: {e}")

    def test_security_payloads_has_patterns(self, output_dir):
        """Test security payloads has attack patterns."""
        file_path = Path(output_dir) / "data" / "security_payloads.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        assert "SQL_INJECTIONS" in content
        assert "XSS_PAYLOADS" in content
        assert "def sql_injection" in content
        assert "def xss_payload" in content


class TestModularScenarios:
    """Tests for scenario files."""

    def test_base_scenario_has_helpers(self, output_dir):
        """Test base scenario has required helper methods."""
        file_path = Path(output_dir) / "scenarios" / "base_scenario.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        assert "class BaseScenario" in content
        assert "def make_request" in content
        assert "def expect_error" in content
        assert "def store_id" in content
        assert "def get_id" in content

    def test_positive_tasks_syntax(self, output_dir):
        """Test positive tasks has valid Python syntax."""
        file_path = Path(output_dir) / "scenarios" / "positive_tasks.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in positive_tasks.py: {e}")

    def test_negative_tasks_syntax(self, output_dir):
        """Test negative tasks has valid Python syntax."""
        file_path = Path(output_dir) / "scenarios" / "negative_tasks.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in negative_tasks.py: {e}")

    def test_security_tasks_has_patterns(self, output_dir):
        """Test security tasks has security test methods."""
        file_path = Path(output_dir) / "scenarios" / "security_tasks.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        assert "class SecurityTasks" in content
        assert "sql_injection" in content or "@task" in content


class TestModularWorkflows:
    """Tests for workflow files."""

    def test_main_workflow_composes_scenarios(self, output_dir):
        """Test main workflow imports and composes scenarios."""
        file_path = Path(output_dir) / "workflows" / "main_workflow.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        assert "class MainWorkflow" in content
        assert "PositiveTasks" in content
        assert "NegativeTasks" in content
        assert "tasks = {" in content or "tasks=" in content


class TestModularSingleResponsibility:
    """Tests verifying Single Responsibility Principle."""

    def test_valid_data_only_has_valid_generators(self, output_dir):
        """Test valid_data.py only contains valid data generators."""
        file_path = Path(output_dir) / "data" / "valid_data.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        # Should NOT contain invalid/security patterns
        assert "SQL_INJECTION" not in content
        assert "XSS_PAYLOAD" not in content
        assert "missing_required" not in content

    def test_invalid_data_only_has_invalid_generators(self, output_dir):
        """Test invalid_data.py only contains invalid data generators."""
        file_path = Path(output_dir) / "data" / "invalid_data.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        # Should NOT contain security patterns
        assert "SQL_INJECTION" not in content
        assert "XSS_PAYLOAD" not in content

    def test_positive_tasks_expect_success(self, output_dir):
        """Test positive tasks expect success status codes."""
        file_path = Path(output_dir) / "scenarios" / "positive_tasks.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        # Should primarily use make_request (expects success)
        # Should NOT primarily use expect_error
        if "expect_error" in content:
            # If expect_error appears, it should be less frequent than make_request
            assert content.count("make_request") >= content.count("expect_error")

    def test_negative_tasks_expect_errors(self, output_dir):
        """Test negative tasks expect error status codes."""
        file_path = Path(output_dir) / "scenarios" / "negative_tasks.py"
        if not file_path.exists():
            pytest.skip("Modular output not generated")

        content = file_path.read_text()

        # Should use expect_error or check for error statuses
        assert "expect_error" in content or "400" in content or "422" in content
