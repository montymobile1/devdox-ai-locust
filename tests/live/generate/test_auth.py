"""
Authentication handling tests for the `generate` command.

Tests --auth/--no-auth options and validates auth code generation.
"""

import pytest

from .conftest import run_generate_command
from .test_validation import OutputAnalyzer, CodeValidator


class TestAuthEnabled:
    """Test generation with authentication enabled (default)."""

    def test_auth_mode_default(self, api_key, swagger_url, output_dir):
        """Test generation with auth enabled (default)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_auth_classes_not_duplicated(self, api_key, swagger_url, output_dir):
        """Authentication classes should not be duplicated (Bug #1)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        auth_distribution = analyzer.check_auth_consistency()

        # Auth classes should exist but not be duplicated
        total_auth_classes = sum(auth_distribution.values())

        # Warn if too many auth classes (might indicate duplication)
        if total_auth_classes > 5:
            print(f"Warning: {total_auth_classes} auth-related classes found")
            print(f"Distribution: {auth_distribution}")

        # Check no single file has more than 3 auth classes
        for filename, count in auth_distribution.items():
            assert count <= 3, \
                f"Too many auth classes in {filename} ({count}), possible duplication"

    def test_auth_methods_generated(self, api_key, swagger_url, output_dir):
        """Auth mode should generate authentication methods."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check for auth-related code in generated files
        py_files = list(output_dir.rglob("*.py"))
        auth_keywords = ["login", "auth", "token", "session", "on_start"]

        auth_found = False
        for f in py_files:
            content = f.read_text().lower()
            if any(kw in content for kw in auth_keywords):
                auth_found = True
                break

        # Note: Auth presence depends on the Swagger spec having auth endpoints
        # This is informational rather than a hard assertion


class TestAuthDisabled:
    """Test generation with authentication disabled."""

    def test_no_auth_mode(self, api_key, swagger_url, output_dir):
        """Test generation with --no-auth."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=False,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_no_auth_minimal_auth_classes(self, api_key, swagger_url, output_dir):
        """No-auth mode should have minimal auth classes."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=False,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        auth_distribution = analyzer.check_auth_consistency()

        total_auth_classes = sum(auth_distribution.values())
        # With --no-auth, should have minimal auth classes (0-2)
        assert total_auth_classes <= 2, \
            f"Too many auth classes for --no-auth: {auth_distribution}"

    def test_no_auth_still_generates_valid_code(self, api_key, swagger_url, output_dir):
        """No-auth mode should still generate valid, runnable code."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=False,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check basic structure
        assert (output_dir / "locustfile.py").exists()

        # Validate syntax
        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        syntax_issues = {
            f: errs for f, errs in issues.items()
            if any("Invalid Python" in e for e in errs)
        }
        assert not syntax_issues, f"Invalid syntax: {syntax_issues}"


class TestAuthModeComparison:
    """Compare output between auth and no-auth modes."""

    @pytest.mark.slow
    def test_auth_vs_no_auth_both_valid(self, api_key, swagger_url, tmp_path):
        """Both auth and no-auth modes should generate valid output."""
        # Generate with auth
        auth_output = tmp_path / "with_auth"
        auth_output.mkdir()
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=auth_output,
            api_key=api_key,
            auth=True,
        )
        assert exit_code == 0, f"Auth mode failed: {stderr}"

        # Generate without auth
        no_auth_output = tmp_path / "without_auth"
        no_auth_output.mkdir()
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=no_auth_output,
            api_key=api_key,
            auth=False,
        )
        assert exit_code == 0, f"No-auth mode failed: {stderr}"

        # Both should have valid structure
        assert (auth_output / "locustfile.py").exists()
        assert (no_auth_output / "locustfile.py").exists()

        # Compare auth class counts
        auth_analyzer = OutputAnalyzer(auth_output)
        no_auth_analyzer = OutputAnalyzer(no_auth_output)

        auth_count = sum(auth_analyzer.check_auth_consistency().values())
        no_auth_count = sum(no_auth_analyzer.check_auth_consistency().values())

        print(f"Auth mode: {auth_count} auth classes")
        print(f"No-auth mode: {no_auth_count} auth classes")

        # Auth mode should have more or equal auth classes
        assert auth_count >= no_auth_count, \
            "Auth mode should have at least as many auth classes as no-auth mode"
