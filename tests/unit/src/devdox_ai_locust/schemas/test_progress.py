"""
Comprehensive tests for progress.py module.

Tests cover:
- ProgressPhase enum
- ProgressStatus model
- GenerationProgress model
"""

from devdox_ai_locust.schemas.progress import (
    ProgressPhase,
    ProgressStatus,
    GenerationProgress,
)


# =============================================================================
# ProgressPhase Enum Tests
# =============================================================================


class TestProgressPhase:
    """Tests for ProgressPhase enum."""

    def test_all_phases_defined(self):
        """Should define all expected phases."""
        expected_phases = [
            "INITIALIZING",
            "PARSING_SCHEMA",
            "GENERATING_TEMPLATES",
            "ANALYZING_CODEBASE",
            "ENHANCING_LOCUSTFILE",
            "ENHANCING_TEST_DATA",
            "ENHANCING_VALIDATION",
            "ENHANCING_DOMAIN_FLOWS",
            "ENHANCING_WORKFLOWS",
            "MERGING_CODE",
            "VALIDATING_OUTPUT",
            "WRITING_FILES",
            "FINALIZING",
            "COMPLETE",
            "FAILED",
        ]
        for phase in expected_phases:
            assert hasattr(ProgressPhase, phase)

    def test_phase_count(self):
        """Should have 15 phases."""
        assert len(ProgressPhase) == 15

    def test_initializing_value(self):
        """INITIALIZING should have correct value."""
        assert ProgressPhase.INITIALIZING.value == "initializing"

    def test_complete_value(self):
        """COMPLETE should have correct value."""
        assert ProgressPhase.COMPLETE.value == "complete"

    def test_failed_value(self):
        """FAILED should have correct value."""
        assert ProgressPhase.FAILED.value == "failed"

    def test_phase_is_string_enum(self):
        """Phases should be string enums."""
        assert isinstance(ProgressPhase.INITIALIZING.value, str)

    def test_can_create_from_value(self):
        """Should be able to create from string value."""
        assert ProgressPhase("initializing") == ProgressPhase.INITIALIZING
        assert ProgressPhase("complete") == ProgressPhase.COMPLETE


# =============================================================================
# ProgressStatus Model Tests
# =============================================================================


class TestProgressStatusBasic:
    """Basic tests for ProgressStatus model."""

    def test_creates_with_required_fields(self):
        """Should create with phase and message."""
        status = ProgressStatus(
            phase=ProgressPhase.INITIALIZING,
            message="Starting up",
        )
        assert status.phase == ProgressPhase.INITIALIZING
        assert status.message == "Starting up"

    def test_default_values(self):
        """Should have correct default values."""
        status = ProgressStatus(
            phase=ProgressPhase.PARSING_SCHEMA,
            message="Parsing",
        )
        assert status.detail is None
        assert status.current == 0
        assert status.total == 0
        assert status.is_ai_call is False

    def test_all_fields(self):
        """Should accept all fields."""
        status = ProgressStatus(
            phase=ProgressPhase.ENHANCING_WORKFLOWS,
            message="Enhancing workflows",
            detail="Processing workflow 3 of 5",
            current=3,
            total=5,
            is_ai_call=True,
        )
        assert status.detail == "Processing workflow 3 of 5"
        assert status.current == 3
        assert status.total == 5
        assert status.is_ai_call is True


class TestProgressStatusPercentage:
    """Tests for ProgressStatus.percentage property."""

    def test_percentage_calculation(self):
        """Should calculate percentage correctly."""
        status = ProgressStatus(
            phase=ProgressPhase.ENHANCING_WORKFLOWS,
            message="Processing",
            current=3,
            total=10,
        )
        assert status.percentage == 30.0

    def test_percentage_zero_total(self):
        """Should return 0 when total is 0."""
        status = ProgressStatus(
            phase=ProgressPhase.INITIALIZING,
            message="Starting",
            current=5,
            total=0,
        )
        assert status.percentage == 0.0

    def test_percentage_complete(self):
        """Should return 100 when current equals total."""
        status = ProgressStatus(
            phase=ProgressPhase.COMPLETE,
            message="Done",
            current=10,
            total=10,
        )
        assert status.percentage == 100.0


class TestProgressStatusFormatProgress:
    """Tests for ProgressStatus.format_progress method."""

    def test_format_with_progress(self):
        """Should format with current/total when total > 0."""
        status = ProgressStatus(
            phase=ProgressPhase.ENHANCING_WORKFLOWS,
            message="Processing files",
            current=3,
            total=5,
        )
        result = status.format_progress()
        assert result == "Processing files (3/5)"

    def test_format_without_progress(self):
        """Should return just message when total is 0."""
        status = ProgressStatus(
            phase=ProgressPhase.INITIALIZING,
            message="Starting up",
            current=0,
            total=0,
        )
        result = status.format_progress()
        assert result == "Starting up"


# =============================================================================
# GenerationProgress Model Tests
# =============================================================================


class TestGenerationProgressBasic:
    """Basic tests for GenerationProgress model."""

    def test_default_values(self):
        """Should have correct default values."""
        progress = GenerationProgress()
        assert progress.total_phases == 13
        assert progress.completed_phases == 0
        assert progress.current_phase == ProgressPhase.INITIALIZING
        assert progress.phase_messages == {}
        assert progress.errors == []

    def test_custom_values(self):
        """Should accept custom values."""
        progress = GenerationProgress(
            completed_phases=5,
            current_phase=ProgressPhase.ENHANCING_TEST_DATA,
            errors=["Error 1"],
        )
        assert progress.completed_phases == 5
        assert progress.current_phase == ProgressPhase.ENHANCING_TEST_DATA
        assert len(progress.errors) == 1


class TestGenerationProgressPhaseDescriptions:
    """Tests for PHASE_DESCRIPTIONS and get_phase_description."""

    def test_has_all_phase_descriptions(self):
        """Should have descriptions for all phases."""
        progress = GenerationProgress()
        for phase in ProgressPhase:
            assert phase in progress.PHASE_DESCRIPTIONS

    def test_get_phase_description(self):
        """Should return correct description for phase."""
        progress = GenerationProgress()
        desc = progress.get_phase_description(ProgressPhase.INITIALIZING)
        assert desc == "Initializing generator"

    def test_get_phase_description_complete(self):
        """Should return correct description for COMPLETE."""
        progress = GenerationProgress()
        desc = progress.get_phase_description(ProgressPhase.COMPLETE)
        assert desc == "Generation complete"

    def test_get_phase_description_failed(self):
        """Should return correct description for FAILED."""
        progress = GenerationProgress()
        desc = progress.get_phase_description(ProgressPhase.FAILED)
        assert desc == "Generation failed"


class TestGenerationProgressPercentage:
    """Tests for GenerationProgress.overall_percentage property."""

    def test_initial_percentage(self):
        """Should be 0% initially."""
        progress = GenerationProgress()
        assert progress.overall_percentage == 0.0

    def test_partial_percentage(self):
        """Should calculate correct percentage."""
        progress = GenerationProgress(
            total_phases=10,
            completed_phases=5,
        )
        assert progress.overall_percentage == 50.0

    def test_complete_percentage(self):
        """Should be 100% when all phases complete."""
        progress = GenerationProgress(
            total_phases=13,
            completed_phases=13,
        )
        assert progress.overall_percentage == 100.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestProgressIntegration:
    """Integration tests for progress tracking."""

    def test_progress_workflow(self):
        """Test a typical progress workflow."""
        # Start progress
        progress = GenerationProgress()
        assert progress.current_phase == ProgressPhase.INITIALIZING

        # Move through phases
        progress.current_phase = ProgressPhase.PARSING_SCHEMA
        progress.completed_phases += 1
        assert progress.overall_percentage > 0

        # Create status updates
        status = ProgressStatus(
            phase=ProgressPhase.PARSING_SCHEMA,
            message="Parsing API schema",
            current=0,
            total=1,
        )
        assert status.format_progress() == "Parsing API schema (0/1)"

        # Complete
        progress.current_phase = ProgressPhase.COMPLETE
        progress.completed_phases = progress.total_phases
        assert progress.overall_percentage == 100.0

    def test_error_workflow(self):
        """Test error handling in progress."""
        progress = GenerationProgress()
        progress.current_phase = ProgressPhase.ENHANCING_LOCUSTFILE
        progress.errors.append("AI service timeout")
        progress.current_phase = ProgressPhase.FAILED

        assert progress.current_phase == ProgressPhase.FAILED
        assert len(progress.errors) == 1
        assert "timeout" in progress.errors[0]
