"""
Comprehensive tests for config.py module.

Tests cover:
- Settings class instantiation
- Default values
- Environment variable loading
- Config class settings
- Settings singleton instance
"""

import os
from unittest.mock import patch

from devdox_ai_locust.config import Settings, settings


# =============================================================================
# Settings Class Default Values Tests
# =============================================================================


class TestSettingsDefaultValues:
    """Tests for default values in Settings class."""

    def test_version_has_default(self):
        """VERSION should have a default value."""
        s = Settings()
        assert s.VERSION is not None
        assert isinstance(s.VERSION, str)

    def test_version_format(self):
        """VERSION should follow semver format."""
        s = Settings()
        # Should be in format X.Y.Z
        parts = s.VERSION.split(".")
        assert len(parts) >= 2  # At least major.minor
        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_api_key_defaults_empty(self):
        """API_KEY should default to empty string."""
        # Clear any env var
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(_env_file=None)
            assert s.API_KEY == ""

    def test_api_key_is_string(self):
        """API_KEY should be a string type."""
        s = Settings()
        assert isinstance(s.API_KEY, str)


# =============================================================================
# Settings Class Environment Variable Tests
# =============================================================================


class TestSettingsEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_loads_api_key_from_env(self):
        """Should load API_KEY from environment variable."""
        with patch.dict(os.environ, {"API_KEY": "test-api-key-123"}):
            s = Settings(_env_file=None)
            assert s.API_KEY == "test-api-key-123"

    def test_loads_version_from_env(self):
        """Should load VERSION from environment variable."""
        with patch.dict(os.environ, {"VERSION": "2.0.0"}):
            s = Settings(_env_file=None)
            assert s.VERSION == "2.0.0"

    def test_env_vars_case_sensitive(self):
        """Environment variables should be case sensitive."""
        with patch.dict(os.environ, {"api_key": "lowercase-key"}, clear=True):
            s = Settings(_env_file=None)
            # Should NOT load lowercase api_key, keeps default
            assert s.API_KEY == ""

    def test_ignores_extra_env_vars(self):
        """Should ignore extra environment variables (extra='ignore')."""
        with patch.dict(os.environ, {"UNKNOWN_SETTING": "value"}):
            # Should not raise any errors
            s = Settings(_env_file=None)
            assert not hasattr(s, "UNKNOWN_SETTING")


# =============================================================================
# Settings Config Class Tests
# =============================================================================


class TestSettingsConfig:
    """Tests for Settings.Config inner class."""

    def test_env_file_setting(self):
        """Config should specify .env as env_file."""
        assert Settings.model_config.get("env_file") == ".env"

    def test_case_sensitive_setting(self):
        """Config should be case sensitive."""
        assert Settings.model_config.get("case_sensitive") is True

    def test_extra_ignore_setting(self):
        """Config should ignore extra fields."""
        assert Settings.model_config.get("extra") == "ignore"


# =============================================================================
# Settings Singleton Instance Tests
# =============================================================================


class TestSettingsSingleton:
    """Tests for the settings singleton instance."""

    def test_settings_is_settings_instance(self):
        """settings should be an instance of Settings."""
        assert isinstance(settings, Settings)

    def test_settings_has_version(self):
        """settings singleton should have VERSION attribute."""
        assert hasattr(settings, "VERSION")
        assert settings.VERSION is not None

    def test_settings_has_api_key(self):
        """settings singleton should have API_KEY attribute."""
        assert hasattr(settings, "API_KEY")
        # API_KEY should be a string (might be empty or from env)
        assert isinstance(settings.API_KEY, str)


# =============================================================================
# Settings Immutability Tests
# =============================================================================


class TestSettingsImmutability:
    """Tests for settings behavior regarding modifications."""

    def test_can_access_version(self):
        """Should be able to access VERSION."""
        s = Settings()
        _ = s.VERSION  # Should not raise

    def test_can_access_api_key(self):
        """Should be able to access API_KEY."""
        s = Settings()
        _ = s.API_KEY  # Should not raise


# =============================================================================
# Settings Serialization Tests
# =============================================================================


class TestSettingsSerialization:
    """Tests for Settings serialization."""

    def test_model_dump(self):
        """Should serialize to dictionary."""
        s = Settings()
        data = s.model_dump()
        assert "VERSION" in data
        assert "API_KEY" in data

    def test_model_dump_contains_correct_types(self):
        """Serialized data should have correct types."""
        s = Settings()
        data = s.model_dump()
        assert isinstance(data["VERSION"], str)
        assert isinstance(data["API_KEY"], str)


# =============================================================================
# Settings Version Tests
# =============================================================================


class TestSettingsVersion:
    """Tests specifically for VERSION field."""

    def test_version_not_empty(self):
        """VERSION should not be empty."""
        s = Settings()
        assert s.VERSION != ""

    def test_version_starts_with_digit(self):
        """VERSION should start with a digit (semantic versioning)."""
        s = Settings()
        assert s.VERSION[0].isdigit()

    def test_version_is_current(self):
        """VERSION should match expected value."""
        s = Settings()
        # Check it's a valid version string
        assert "." in s.VERSION


# =============================================================================
# Settings Edge Cases Tests
# =============================================================================


class TestSettingsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_multiple_instances_have_same_defaults(self):
        """Multiple Settings instances should have same default values."""
        s1 = Settings()
        s2 = Settings()
        assert s1.VERSION == s2.VERSION

    def test_settings_with_empty_env(self):
        """Should work with completely empty environment."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(_env_file=None)
            assert s.VERSION is not None
            assert s.API_KEY == ""

    def test_api_key_with_special_characters(self):
        """Should handle API keys with special characters."""
        with patch.dict(
            os.environ, {"API_KEY": "sk-test_key!@#$%^&*()_+-=[]{}|;':\",./<>?"}
        ):
            s = Settings(_env_file=None)
            assert "sk-test_key" in s.API_KEY

    def test_api_key_with_spaces(self):
        """Should preserve spaces in API key if present."""
        with patch.dict(os.environ, {"API_KEY": "key with spaces"}):
            s = Settings(_env_file=None)
            assert s.API_KEY == "key with spaces"

    def test_long_api_key(self):
        """Should handle long API keys."""
        long_key = "sk-" + "a" * 1000
        with patch.dict(os.environ, {"API_KEY": long_key}):
            s = Settings(_env_file=None)
            assert len(s.API_KEY) > 1000


# =============================================================================
# Integration Tests
# =============================================================================


class TestSettingsIntegration:
    """Integration tests for Settings with environment."""

    def test_settings_from_env_dict(self):
        """Should correctly load all settings from environment."""
        env_vars = {
            "VERSION": "3.0.0",
            "API_KEY": "integration-test-key",
        }
        with patch.dict(os.environ, env_vars):
            s = Settings(_env_file=None)
            assert s.VERSION == "3.0.0"
            assert s.API_KEY == "integration-test-key"

    def test_partial_env_override(self):
        """Should allow partial override via environment."""
        with patch.dict(os.environ, {"API_KEY": "only-api-key"}):
            s = Settings(_env_file=None)
            assert s.API_KEY == "only-api-key"
            # VERSION should still have default
            assert s.VERSION is not None
