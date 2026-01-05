"""
Configuration settings for the DevDox AI Locust
"""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.1.9"

    API_KEY: str = ""  # Fallback for backward compatibility

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


# Initialize settings instance
settings = Settings()
