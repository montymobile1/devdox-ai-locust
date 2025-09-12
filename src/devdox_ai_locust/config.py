"""
Configuration settings for the DevDox AI Locust
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic_settings import BaseSettings


search_path = "vault,public"

class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.1.0"

    API_KEY: str = ""  # Fallback for backward compatibility
    
    @property
    def api_key(self) -> str:
        """Get the API key, preferring TOGETHER_API_KEY over API_KEY."""
        return self.API_KEY

    class Config:
        """Pydantic config class."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Initialize settings instance
settings = Settings()


