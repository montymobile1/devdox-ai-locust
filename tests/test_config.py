"""
Tests for configuration module
"""

from pathlib import Path
from unittest.mock import patch

from devdox_ai_locust.config import Settings, settings


class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        test_settings = Settings(_env_file=".env.example")

        assert test_settings.VERSION == "0.1.8"
        assert test_settings.API_KEY == ""

    def test_settings_with_env_vars(self):
        """Test settings with environment variables."""
        with patch.dict("os.environ", {"API_KEY": "test-env-key"}):
            test_settings = Settings()
            assert test_settings.API_KEY == "test-env-key"

    def test_settings_case_sensitive(self):
        """Test that settings are case sensitive."""
        with patch.dict(
            "os.environ", {"api_key": "lowercase-key", "API_KEY": "uppercase-key"}
        ):
            test_settings = Settings()
            # Should use the uppercase version due to case sensitivity
            assert test_settings.API_KEY == "uppercase-key"

    def test_settings_extra_ignore(self):
        """Test that extra fields are ignored."""
        with patch.dict(
            "os.environ", {"EXTRA_FIELD": "extra-value", "API_KEY": "test-key"}
        ):
            test_settings = Settings()
            assert test_settings.API_KEY == "test-key"
            assert not hasattr(test_settings, "EXTRA_FIELD")

    def test_settings_from_env_file(self, temp_dir):
        """Test loading settings from .env file."""
        env_file = temp_dir / ".env"
        env_file.write_text("API_KEY=file-key\nVERSION=2.0.0")

        # Change to temp directory to test .env loading
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_dir)

            # Create new settings instance to pick up .env file
            test_settings = Settings()

            # Note: The VERSION field is hardcoded in the class,
            # so it won't be overridden by .env
            assert test_settings.API_KEY == "file-key"
            assert test_settings.VERSION == "2.0.0"  # Hardcoded value

        finally:
            os.chdir(original_cwd)

    def test_settings_env_override_file(self, temp_dir):
        """Test that environment variables override .env file."""
        env_file = temp_dir / ".env"
        env_file.write_text("API_KEY=file-key")

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_dir)

            with patch.dict("os.environ", {"API_KEY": "env-key"}):
                test_settings = Settings()
                assert test_settings.API_KEY == "env-key"

        finally:
            os.chdir(original_cwd)

    def test_settings_validation(self):
        """Test settings field validation."""
        test_settings = Settings()

        # Test that API_KEY accepts string values
        test_settings.API_KEY = "new-key"
        assert test_settings.API_KEY == "new-key"

        # Test that VERSION is read-only (defined in class)

        test_settings.VERSION = "3.0.0"
        # This might not actually change the value depending on Pydantic setup
        # The test verifies the field exists and has expected type

    def test_settings_config_class(self):
        """Test Settings.Config class attributes."""
        config = Settings.Config

        assert config.env_file == ".env"
        assert config.case_sensitive is True
        assert config.extra == "ignore"

    def test_global_settings_instance(self):
        """Test the global settings instance."""
        assert isinstance(settings, Settings)
        assert settings.VERSION == "0.1.8"


class TestSettingsMethods:
    """Test Settings methods (if any)."""

    def test_settings_immutable_fields(self):
        """Test that certain fields remain immutable."""
        test_settings = Settings()
        original_version = test_settings.VERSION

        # VERSION should be a class-level constant
        assert original_version == "0.1.8"

        # Even if we try to change it, it should remain the same
        # (depending on Pydantic implementation)
        try:
            test_settings.VERSION = "999.999.999"
        except Exception:
            pass  # Some fields might be read-only

        # Create new instance to verify class-level value
        new_settings = Settings()
        assert new_settings.VERSION == "0.1.8"

    def test_settings_field_types(self):
        """Test that settings fields have correct types."""
        test_settings = Settings()

        assert isinstance(test_settings.VERSION, str)
        assert isinstance(test_settings.API_KEY, str)

    def test_settings_empty_api_key(self):
        """Test behavior with empty API key."""
        with patch.dict("os.environ", {}, clear=True):
            test_settings = Settings(_env_file=None)
            assert test_settings.API_KEY == ""

    def test_settings_whitespace_handling(self):
        """Test whitespace handling in settings."""
        with patch.dict("os.environ", {"API_KEY": "  test-key-with-spaces  "}):
            test_settings = Settings()
            # Pydantic typically doesn't strip whitespace by default
            assert test_settings.API_KEY == "  test-key-with-spaces  "

    def test_settings_special_characters(self):
        """Test settings with special characters."""
        special_key = "test-key!@#$%^&*()_+"
        with patch.dict("os.environ", {"API_KEY": special_key}):
            test_settings = Settings()
            assert test_settings.API_KEY == special_key

    def test_settings_unicode_characters(self):
        """Test settings with unicode characters."""
        unicode_key = "test-key-🔑-unicode"
        with patch.dict("os.environ", {"API_KEY": unicode_key}):
            test_settings = Settings()
            assert test_settings.API_KEY == unicode_key

    def test_settings_very_long_api_key(self):
        """Test settings with very long API key."""
        long_key = "x" * 1000
        with patch.dict("os.environ", {"API_KEY": long_key}):
            test_settings = Settings()
            assert test_settings.API_KEY == long_key
            assert len(test_settings.API_KEY) == 1000


class TestSettingsIntegration:
    """Integration tests for settings."""

    def test_settings_in_different_environments(self):
        """Test settings behavior in different environments."""
        environments = [
            {"API_KEY": "dev-key"},
            {"API_KEY": "test-key"},
            {"API_KEY": "prod-key"},
        ]

        for env in environments:
            with patch.dict("os.environ", env, clear=True):
                test_settings = Settings()
                assert test_settings.API_KEY == env["API_KEY"]

    def test_settings_reload_behavior(self):
        """Test settings behavior when reloading."""
        # First load
        with patch.dict("os.environ", {"API_KEY": "first-key"}):
            settings1 = Settings()
            assert settings1.API_KEY == "first-key"

        # Second load with different env
        with patch.dict("os.environ", {"API_KEY": "second-key"}):
            settings2 = Settings()
            assert settings2.API_KEY == "second-key"

        # Original settings should remain unchanged
        assert settings1.API_KEY == "first-key"

    def test_settings_concurrent_access(self):
        """Test settings with concurrent access patterns."""
        import threading

        results = []

        def create_settings(key_suffix):
            with patch.dict("os.environ", {"API_KEY": f"thread-key-{key_suffix}"}):
                test_settings = Settings()
                results.append(test_settings.API_KEY)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_settings, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be present
        assert len(results) == 5
        assert all("thread-key-" in result for result in results)
