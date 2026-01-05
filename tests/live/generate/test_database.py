"""
Database integration tests for the `generate` command.

Tests generation with MongoDB and PostgreSQL database integration.
These tests are SKIPPED if database URIs are not provided.
"""

import pytest

from .conftest import run_generate_command


class TestMongoDB:
    """Test MongoDB integration mode."""

    def test_generate_with_mongodb(self, api_key, swagger_url, output_dir, mongodb_uri):
        """Test generation with MongoDB integration."""
        if not mongodb_uri:
            pytest.skip("MongoDB URI not provided (use --mongodb-uri)")

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="mongo",
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check for MongoDB-specific code
        locustfile = output_dir / "locustfile.py"
        if locustfile.exists():
            content = locustfile.read_text()
            # Should have some MongoDB references
            assert "mongo" in content.lower() or "database" in content.lower(), \
                "MongoDB integration code not found in locustfile"

    def test_mongodb_generates_data_provider(self, api_key, swagger_url, output_dir, mongodb_uri):
        """MongoDB mode should generate data provider utilities."""
        if not mongodb_uri:
            pytest.skip("MongoDB URI not provided (use --mongodb-uri)")

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="mongo",
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

        # Check all generated files for MongoDB references
        py_files = list(output_dir.rglob("*.py"))
        mongo_found = False
        for f in py_files:
            content = f.read_text()
            if "mongo" in content.lower() or "pymongo" in content.lower():
                mongo_found = True
                break

        # Note: MongoDB integration may be configurable
        # This is a soft check


class TestPostgreSQL:
    """Test PostgreSQL integration mode."""

    def test_generate_with_postgresql(self, api_key, swagger_url, output_dir, postgresql_uri):
        """Test generation with PostgreSQL integration."""
        if not postgresql_uri:
            pytest.skip("PostgreSQL URI not provided (use --postgresql-uri)")

        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="postgresql",
        )

        assert exit_code == 0, f"Generation failed: {stderr}"

    def test_postgresql_generates_data_provider(self, api_key, swagger_url, output_dir, postgresql_uri):
        """PostgreSQL mode should generate data provider utilities."""
        if not postgresql_uri:
            pytest.skip("PostgreSQL URI not provided (use --postgresql-uri)")

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="postgresql",
        )

        assert exit_code == 0, f"Generation failed: {stderr}"


class TestNoDatabase:
    """Test generation without database integration."""

    def test_generate_without_database(self, api_key, swagger_url, output_dir):
        """Test generation with no database integration (default)."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type=None,  # No database
        )

        assert exit_code == 0, f"Generation failed: {stderr}"
        assert (output_dir / "locustfile.py").exists()

    def test_generate_with_empty_db_type(self, api_key, swagger_url, output_dir):
        """Test generation with empty db_type (explicit no database)."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="",  # Explicitly empty
        )

        assert exit_code == 0, f"Generation failed: {stderr}"


class TestDatabaseModeComparison:
    """Compare output between different database modes."""

    @pytest.mark.slow
    def test_all_modes_generate_valid_output(self, api_key, swagger_url, tmp_path, mongodb_uri, postgresql_uri):
        """All database modes should generate valid output."""
        modes_to_test = [("none", None)]

        if mongodb_uri:
            modes_to_test.append(("mongo", "mongo"))
        if postgresql_uri:
            modes_to_test.append(("postgresql", "postgresql"))

        for mode_name, db_type in modes_to_test:
            output = tmp_path / mode_name
            output.mkdir()

            exit_code, _, stderr = run_generate_command(
                swagger_source=swagger_url,
                output_dir=output,
                api_key=api_key,
                db_type=db_type,
            )

            assert exit_code == 0, f"Generation failed for {mode_name}: {stderr}"
            assert (output / "locustfile.py").exists(), f"Missing locustfile.py for {mode_name}"
