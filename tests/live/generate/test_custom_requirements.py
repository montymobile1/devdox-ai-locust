"""
Custom requirements tests for the `generate` command.

Tests the --custom-requirement option for specialized test generation.
"""

import pytest

from .conftest import run_generate_command
from .test_validation import OutputAnalyzer


class TestCustomRequirements:
    """Test custom requirement handling."""

    def test_with_performance_requirement(self, api_key, swagger_url, output_dir):
        """Test generation with performance-focused requirement."""
        custom_req = "Focus on high-traffic performance testing with concurrent users"

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=custom_req,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"
        assert (output_dir / "locustfile.py").exists()

    def test_with_stress_test_requirement(self, api_key, swagger_url, output_dir):
        """Test generation with stress testing requirement."""
        custom_req = "Generate stress tests that push the API to its limits"

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=custom_req,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_with_validation_requirement(self, api_key, swagger_url, output_dir):
        """Test generation with validation-focused requirement."""
        custom_req = "Add comprehensive response validation for all API endpoints"

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=custom_req,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_with_empty_requirement(self, api_key, swagger_url, output_dir):
        """Test generation with empty requirement (should work normally)."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement="",
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_with_none_requirement(self, api_key, swagger_url, output_dir):
        """Test generation without requirement (None)."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=None,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"


class TestCustomRequirementValidation:
    """Validate that custom requirements produce valid output."""

    def test_custom_requirement_valid_syntax(self, api_key, swagger_url, output_dir):
        """Custom requirements should still produce valid Python."""
        custom_req = "Add extra logging and monitoring for debugging purposes"

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=custom_req,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        syntax_issues = {
            f: errs for f, errs in issues.items()
            if any("Invalid Python" in e for e in errs)
        }
        assert not syntax_issues, f"Invalid syntax: {syntax_issues}"

    def test_custom_requirement_no_duplicates(self, api_key, swagger_url, output_dir):
        """Custom requirements should not cause class duplication."""
        custom_req = "Add custom authentication handling and retry logic"

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=custom_req,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        violations = analyzer.check_singleton_classes()

        assert not violations, f"Singleton violations: {violations}"


class TestMultipleRequirements:
    """Test complex requirement scenarios."""

    @pytest.mark.slow
    def test_combined_requirements(self, api_key, swagger_url, output_dir):
        """Test generation with complex combined requirements."""
        custom_req = (
            "Focus on: 1) High performance testing with 1000+ concurrent users, "
            "2) Comprehensive response validation, "
            "3) Detailed logging for debugging, "
            "4) Error handling with retry logic"
        )

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=custom_req,
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

        # Validate output
        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        critical_issues = {
            f: errs for f, errs in issues.items()
            if any("Invalid Python" in e or "Duplicate" in e for e in errs)
        }
        assert not critical_issues, f"Critical issues: {critical_issues}"
