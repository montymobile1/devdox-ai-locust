"""
Configuration settings for the DevDox AI Locust
"""

from dataclasses import dataclass

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.1.9"

    API_KEY: str = ""  # Fallback for backward compatibility

    class Config:
        """Pydantic config class."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Initialize settings instance
settings = Settings()


@dataclass
class AIEnhancementConfig:
    """Configuration for AI-based test generation."""

    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 120
