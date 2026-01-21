"""AI Enhancement Configuration"""

from dataclasses import dataclass


@dataclass
class AIEnhancementConfig:
    """Configuration for AI-based test generation"""

    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 120
