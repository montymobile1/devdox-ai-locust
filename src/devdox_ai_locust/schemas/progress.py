"""
Progress Tracking Models

Provides progress callbacks and status tracking for generation operations.
Uses Pydantic for validation and serialization.
"""

from enum import Enum
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field


class ProgressPhase(str, Enum):
    """Phases of the generation process"""
    INITIALIZING = "initializing"
    PARSING_SCHEMA = "parsing_schema"
    GENERATING_TEMPLATES = "generating_templates"
    ANALYZING_CODEBASE = "analyzing_codebase"
    ENHANCING_LOCUSTFILE = "enhancing_locustfile"
    ENHANCING_TEST_DATA = "enhancing_test_data"
    ENHANCING_VALIDATION = "enhancing_validation"
    ENHANCING_DOMAIN_FLOWS = "enhancing_domain_flows"
    ENHANCING_WORKFLOWS = "enhancing_workflows"
    MERGING_CODE = "merging_code"
    VALIDATING_OUTPUT = "validating_output"
    WRITING_FILES = "writing_files"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


class ProgressStatus(BaseModel):
    """Status update for a generation phase"""
    phase: ProgressPhase
    message: str
    detail: Optional[str] = None
    current: int = 0
    total: int = 0
    is_ai_call: bool = False  # True if this phase involves an AI API call

    @property
    def percentage(self) -> float:
        """Get completion percentage"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    def format_progress(self) -> str:
        """Format progress for display"""
        if self.total > 0:
            return f"{self.message} ({self.current}/{self.total})"
        return self.message


# Type alias for progress callbacks
ProgressCallback = Callable[[ProgressStatus], Awaitable[None]]


class GenerationProgress(BaseModel):
    """Tracks overall generation progress"""
    total_phases: int = 13  # Total number of phases
    completed_phases: int = 0
    current_phase: ProgressPhase = ProgressPhase.INITIALIZING
    phase_messages: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    # Human-readable phase descriptions
    PHASE_DESCRIPTIONS: dict[ProgressPhase, str] = {
        ProgressPhase.INITIALIZING: "Initializing generator",
        ProgressPhase.PARSING_SCHEMA: "Parsing API schema",
        ProgressPhase.GENERATING_TEMPLATES: "Generating base templates",
        ProgressPhase.ANALYZING_CODEBASE: "Analyzing codebase dependencies",
        ProgressPhase.ENHANCING_LOCUSTFILE: "Enhancing main locustfile",
        ProgressPhase.ENHANCING_TEST_DATA: "Enhancing test data generator",
        ProgressPhase.ENHANCING_VALIDATION: "Enhancing validation utilities",
        ProgressPhase.ENHANCING_DOMAIN_FLOWS: "Generating domain-specific flows",
        ProgressPhase.ENHANCING_WORKFLOWS: "Enhancing workflow files",
        ProgressPhase.MERGING_CODE: "Safely merging AI enhancements",
        ProgressPhase.VALIDATING_OUTPUT: "Validating generated code",
        ProgressPhase.WRITING_FILES: "Writing output files",
        ProgressPhase.FINALIZING: "Finalizing generation",
        ProgressPhase.COMPLETE: "Generation complete",
        ProgressPhase.FAILED: "Generation failed",
    }

    def get_phase_description(self, phase: ProgressPhase) -> str:
        """Get human-readable description for a phase"""
        return self.PHASE_DESCRIPTIONS.get(phase, str(phase.value))

    @property
    def overall_percentage(self) -> float:
        """Get overall completion percentage"""
        return (self.completed_phases / self.total_phases) * 100
