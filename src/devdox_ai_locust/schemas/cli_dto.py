"""Pydantic DTOs for CLI commands and internal processing context."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from devdox_ai_locust.utils.constants import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_LLM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_TIME,
    DEFAULT_SCHEMA_TIMEOUT,
    DEFAULT_SPAWN_RATE,
    DEFAULT_USERS,
)


class GenerateParams(BaseModel):
    """DTO for the ``generate`` CLI command parameters."""

    swagger_url: str
    output: str = DEFAULT_OUTPUT_DIR
    users: int = DEFAULT_USERS
    spawn_rate: float = DEFAULT_SPAWN_RATE
    run_time: str = DEFAULT_RUN_TIME
    host: Optional[str] = None
    auth: bool = True
    db_type: str = ""
    dry_run: bool = False
    custom_requirement: Optional[str] = None
    together_api_key: Optional[str] = None
    timeout: int = DEFAULT_LLM_TIMEOUT
    schema_timeout: int = DEFAULT_SCHEMA_TIMEOUT
    max_llm_workers: int = DEFAULT_MAX_LLM_WORKERS
    no_llm: Optional[str] = None
    debug: bool = False
    verbose: bool = False

    @property
    def llm_enabled(self) -> bool:
        """True when LLM is active (--no-llm was NOT passed)."""
        return self.no_llm is None

    @property
    def replay_dir(self) -> Optional[str]:
        """Non-empty path when replay mode is active."""
        return self.no_llm if self.no_llm else None


class RunParams(BaseModel):
    """DTO for the ``run`` CLI command parameters."""

    test_file: str
    users: int = DEFAULT_USERS
    spawn_rate: float = DEFAULT_SPAWN_RATE
    run_time: str = DEFAULT_RUN_TIME
    host: str
    headless: bool = False
    verbose: bool = False


class EndpointProcessingContext(BaseModel):
    """Immutable context shared across all endpoint processing calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_gen: Any  # ScenarioWorkflowGenerator
    template_gen: Any  # LocustTestGenerator
    workflows_dir: Path
    base_workflow_content: str
    test_data_content: str
    auth_endpoints: Optional[List[Any]] = None
    all_endpoints: List[Any]
    custom_requirement: Optional[str] = None
    db_type: str = ""
    pre_llm_templates: Dict[Tuple[int, str], str]
    endpoint_to_tag: Dict[int, str]
    no_llm: Optional[str] = None

    @property
    def llm_enabled(self) -> bool:
        """True when LLM is active."""
        return self.no_llm is None

    @property
    def replay_dir(self) -> Optional[str]:
        """Non-empty path when replay mode is active."""
        return self.no_llm if self.no_llm else None
