"""
Configuration settings for the DevDox AI Locust
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.1.8"

    API_KEY: str = ""  # Fallback for backward compatibility

    class Config:
        """Pydantic config class."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Initialize settings instance
settings = Settings()
