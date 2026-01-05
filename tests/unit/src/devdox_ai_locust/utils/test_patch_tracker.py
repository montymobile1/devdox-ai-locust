"""
Comprehensive tests for patch_tracker.py module.

Tests cover:
- PatchTracker class
- PatchTrackerContext async context manager
- Backwards compatibility functions
"""

import pytest

from devdox_ai_locust.utils.patch_tracker import (
    PatchTracker,
    PatchTrackerContext,
    capture_pre_llm_state,
    capture_post_llm_state,
)
from devdox_ai_locust.utils.metadata_manager import (
    MetadataManager,
)


# =============================================================================
# PatchTracker Initialization Tests
# =============================================================================


class TestPatchTrackerInit:
    """Tests for PatchTracker initialization."""

    def test_init_basic(self, tmp_path):
        """Should initialize with output directory."""
        tracker = PatchTracker(tmp_path)
        assert tracker.output_dir == tmp_path
        assert tracker.metadata_manager is None

    def test_init_with_metadata_manager(self, tmp_path):
        """Should initialize with metadata manager."""
        manager = MetadataManager(tmp_path)
        tracker = PatchTracker(tmp_path, metadata_manager=manager)
        assert tracker.metadata_manager is manager

    def test_init_defaults(self, tmp_path):
        """Should have correct default state."""
        tracker = PatchTracker(tmp_path)
        assert tracker.template_files == {}
        assert tracker.enhanced_files == {}
        assert tracker._start_time is None
        assert tracker._session_started is False

    def test_from_metadata_manager(self, tmp_path):
        """Should create from metadata manager."""
        manager = MetadataManager(tmp_path)
        tracker = PatchTracker.from_metadata_manager(manager)

        assert tracker.output_dir == manager.output_dir
        assert tracker.metadata_manager is manager


# =============================================================================
# PatchTracker Session Management Tests
# =============================================================================


class TestPatchTrackerSession:
    """Tests for PatchTracker session management."""

    def test_start_session(self, tmp_path):
        """Should start a tracking session."""
        tracker = PatchTracker(tmp_path)
        tracker.start_session()

        assert tracker._session_started is True
        assert tracker._start_time is not None
        assert tracker.template_files == {}
        assert tracker.enhanced_files == {}

    def test_start_session_resets_state(self, tmp_path):
        """Should reset state on new session."""
        tracker = PatchTracker(tmp_path)
        tracker.template_files = {"old.py": "content"}
        tracker.enhanced_files = {"old.py": "enhanced"}

        tracker.start_session()
        assert tracker.template_files == {}
        assert tracker.enhanced_files == {}

    def test_finalize(self, tmp_path):
        """Should finalize the session."""
        tracker = PatchTracker(tmp_path)
        tracker.start_session()
        tracker.template_files = {"test.py": "content"}

        tracker.finalize()

        assert tracker._session_started is False
        assert tracker._start_time is None
        assert tracker.template_files == {}
        assert tracker.enhanced_files == {}


# =============================================================================
# PatchTracker Capture Template State Tests
# =============================================================================


class TestPatchTrackerCaptureTemplateState:
    """Tests for PatchTracker.capture_template_state method."""

    def test_captures_files(self, tmp_path):
        """Should capture template files."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        files = {"locustfile.py": "content1", "test_data.py": "content2"}
        tracker.capture_template_state(files)

        assert tracker.template_files == files

    def test_captures_workflow_files(self, tmp_path):
        """Should capture workflow files with prefix."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        files = {"locustfile.py": "main"}
        workflow_files = [{"base.py": "base content"}, {"user.py": "user content"}]

        tracker.capture_template_state(files, workflow_files)

        assert "locustfile.py" in tracker.template_files
        assert "workflows/base.py" in tracker.template_files
        assert "workflows/user.py" in tracker.template_files

    def test_starts_session_if_needed(self, tmp_path):
        """Should start session if not started."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        assert tracker._session_started is False
        tracker.capture_template_state({"test.py": "content"})
        assert tracker._session_started is True

    def test_creates_template_patch(self, tmp_path):
        """Should create template_generation patch."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        files = {"test.py": "print('hello')"}
        entry = tracker.capture_template_state(files)

        assert entry is not None
        assert entry.milestone == "template_generation"
        assert len(manager.session_info.patches) == 1


# =============================================================================
# PatchTracker Capture Enhanced State Tests
# =============================================================================


class TestPatchTrackerCaptureEnhancedState:
    """Tests for PatchTracker.capture_enhanced_state method."""

    def test_captures_enhanced_files(self, tmp_path):
        """Should capture enhanced files."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        # First capture template
        tracker.capture_template_state({"test.py": "original"})

        # Then capture enhanced
        enhanced = {"test.py": "enhanced content"}
        tracker.capture_enhanced_state(enhanced)

        assert tracker.enhanced_files == enhanced

    def test_captures_workflow_files(self, tmp_path):
        """Should capture enhanced workflow files."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.start_session()
        tracker.template_files = {"workflows/base.py": "original"}

        tracker.capture_enhanced_state({}, workflow_files=[{"base.py": "enhanced"}])

        assert "workflows/base.py" in tracker.enhanced_files

    def test_requires_active_session(self, tmp_path):
        """Should raise error if no active session."""
        tracker = PatchTracker(tmp_path)

        with pytest.raises(RuntimeError, match="No active session"):
            tracker.capture_enhanced_state({"test.py": "content"})

    def test_creates_llm_enhancement_patch(self, tmp_path):
        """Should create llm_enhancement patch."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.capture_template_state({"test.py": "original"})
        entry = tracker.capture_enhanced_state({"test.py": "enhanced"})

        assert entry is not None
        assert entry.milestone == "llm_enhancement"

    def test_includes_ai_model_metadata(self, tmp_path):
        """Should include AI model in metadata."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.capture_template_state({"test.py": "original"})
        entry = tracker.capture_enhanced_state(
            {"test.py": "enhanced"}, ai_model="gpt-4"
        )

        assert entry.metadata.get("ai_model") == "gpt-4"


# =============================================================================
# PatchTracker Capture Validation State Tests
# =============================================================================


class TestPatchTrackerCaptureValidationState:
    """Tests for PatchTracker.capture_validation_state method."""

    def test_captures_validation_state(self, tmp_path):
        """Should capture validation state."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        # Setup initial state
        tracker.start_session()
        tracker.template_files = {"test.py": "original"}
        tracker.enhanced_files = {"test.py": "enhanced"}

        validated = {"test.py": "validated content"}
        entry = tracker.capture_validation_state(validated)

        assert entry is not None
        assert entry.milestone == "validation"

    def test_uses_enhanced_as_previous(self, tmp_path):
        """Should use enhanced files as previous state."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.start_session()
        tracker.template_files = {"test.py": "template"}
        tracker.enhanced_files = {"test.py": "enhanced"}

        # The diff should be from enhanced to validated
        entry = tracker.capture_validation_state({"test.py": "validated"})
        assert entry is not None

    def test_uses_template_if_no_enhanced(self, tmp_path):
        """Should use template files if no enhanced."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.start_session()
        tracker.template_files = {"test.py": "template"}
        # enhanced_files is empty

        entry = tracker.capture_validation_state({"test.py": "validated"})
        assert entry is not None

    def test_includes_validation_metadata(self, tmp_path):
        """Should include validation results in metadata."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.start_session()
        tracker.template_files = {"test.py": "original"}

        validation_results = {"syntax_valid": True, "tests_passed": 5}
        entry = tracker.capture_validation_state(
            {"test.py": "validated"}, validation_results=validation_results
        )

        assert entry.metadata.get("validation") == validation_results

    def test_requires_active_session(self, tmp_path):
        """Should raise error if no active session."""
        tracker = PatchTracker(tmp_path)

        with pytest.raises(RuntimeError, match="No active session"):
            tracker.capture_validation_state({"test.py": "content"})


# =============================================================================
# PatchTracker Diff Generation Tests
# =============================================================================


class TestPatchTrackerDiffGeneration:
    """Tests for PatchTracker._generate_unified_diff method."""

    def test_generates_diff_for_new_file(self, tmp_path):
        """Should generate diff for new file."""
        tracker = PatchTracker(tmp_path)
        before = {}
        after = {"test.py": "print('hello')"}

        diff = tracker._generate_unified_diff(before, after)

        assert "--- a/test.py" in diff
        assert "+++ b/test.py" in diff
        assert "+print('hello')" in diff

    def test_generates_diff_for_changed_file(self, tmp_path):
        """Should generate diff for modified file."""
        tracker = PatchTracker(tmp_path)
        before = {"test.py": "print('old')"}
        after = {"test.py": "print('new')"}

        diff = tracker._generate_unified_diff(before, after)

        assert "-print('old')" in diff
        assert "+print('new')" in diff

    def test_generates_diff_for_deleted_file(self, tmp_path):
        """Should generate diff for deleted file."""
        tracker = PatchTracker(tmp_path)
        before = {"test.py": "content"}
        after = {}

        diff = tracker._generate_unified_diff(before, after)

        assert "--- a/test.py" in diff
        assert "-content" in diff

    def test_skips_unchanged_files(self, tmp_path):
        """Should skip unchanged files."""
        tracker = PatchTracker(tmp_path)
        before = {"same.py": "unchanged", "changed.py": "old"}
        after = {"same.py": "unchanged", "changed.py": "new"}

        diff = tracker._generate_unified_diff(before, after)

        assert "same.py" not in diff
        assert "changed.py" in diff

    def test_handles_multiple_files(self, tmp_path):
        """Should handle multiple files."""
        tracker = PatchTracker(tmp_path)
        before = {"a.py": "a content", "b.py": "b content"}
        after = {"a.py": "a modified", "b.py": "b modified"}

        diff = tracker._generate_unified_diff(before, after)

        assert "a.py" in diff
        assert "b.py" in diff

    def test_respects_context_lines(self, tmp_path):
        """Should include context lines."""
        tracker = PatchTracker(tmp_path)
        before = {"test.py": "line1\nline2\nline3\nline4\nline5"}
        after = {"test.py": "line1\nline2\nMODIFIED\nline4\nline5"}

        diff = tracker._generate_unified_diff(before, after, context_lines=1)
        # Should include context around the change
        assert "line2" in diff
        assert "line4" in diff


# =============================================================================
# PatchTracker Stats Calculation Tests
# =============================================================================


class TestPatchTrackerStatsCalculation:
    """Tests for PatchTracker._calculate_stats method."""

    def test_calculates_files_changed(self, tmp_path):
        """Should count changed files."""
        tracker = PatchTracker(tmp_path)
        before = {"a.py": "a", "b.py": "b", "c.py": "c"}
        after = {"a.py": "a modified", "b.py": "b modified", "c.py": "c"}

        stats = tracker._calculate_stats(before, after)
        assert stats.files_changed == 2

    def test_calculates_additions(self, tmp_path):
        """Should count additions."""
        tracker = PatchTracker(tmp_path)
        before = {"test.py": "line1"}
        after = {"test.py": "line1\nline2\nline3"}

        stats = tracker._calculate_stats(before, after)
        assert stats.additions == 2

    def test_calculates_deletions(self, tmp_path):
        """Should count deletions."""
        tracker = PatchTracker(tmp_path)
        before = {"test.py": "line1\nline2\nline3"}
        after = {"test.py": "line1"}

        stats = tracker._calculate_stats(before, after)
        assert stats.deletions == 2

    def test_handles_new_files(self, tmp_path):
        """Should handle new files."""
        tracker = PatchTracker(tmp_path)
        before = {}
        after = {"new.py": "line1\nline2"}

        stats = tracker._calculate_stats(before, after)
        assert stats.files_changed == 1
        assert stats.additions == 2
        assert stats.deletions == 0

    def test_handles_deleted_files(self, tmp_path):
        """Should handle deleted files."""
        tracker = PatchTracker(tmp_path)
        before = {"old.py": "line1\nline2"}
        after = {}

        stats = tracker._calculate_stats(before, after)
        assert stats.files_changed == 1
        assert stats.additions == 0
        assert stats.deletions == 2


# =============================================================================
# PatchTracker Get Summary Tests
# =============================================================================


class TestPatchTrackerGetSummary:
    """Tests for PatchTracker.get_summary method."""

    def test_summary_without_manager(self, tmp_path):
        """Should return minimal summary without manager."""
        tracker = PatchTracker(tmp_path)
        summary = tracker.get_summary()
        assert summary == {"patches": 0}

    def test_summary_with_patches(self, tmp_path):
        """Should return full summary with patches."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.capture_template_state({"test.py": "content"})
        tracker.capture_enhanced_state({"test.py": "enhanced"})

        summary = tracker.get_summary()

        assert "session_id" in summary
        assert summary["total_patches"] == 2
        assert "template_generation" in summary["patches_by_milestone"]
        assert "llm_enhancement" in summary["patches_by_milestone"]


# =============================================================================
# PatchTracker No Changes Tests
# =============================================================================


class TestPatchTrackerNoChanges:
    """Tests for handling no changes."""

    def test_returns_none_when_no_changes(self, tmp_path):
        """Should return None when files are identical."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.capture_template_state({"test.py": "content"})
        # Capture same content - no changes
        entry = tracker.capture_enhanced_state({"test.py": "content"})

        # Should return None because no actual changes
        assert entry is None

    def test_no_metadata_manager_warning(self, tmp_path):
        """Should return None without metadata manager."""
        tracker = PatchTracker(tmp_path)
        tracker.start_session()
        tracker.template_files = {}

        result = tracker._create_milestone_patch(
            milestone="test",
            files_before={},
            files_after={"test.py": "content"},
        )
        assert result is None


# =============================================================================
# PatchTrackerContext Tests
# =============================================================================


class TestPatchTrackerContext:
    """Tests for PatchTrackerContext async context manager."""

    def test_creates_tracker(self, tmp_path):
        """Should create tracker on enter."""
        import anyio

        async def run_test():
            manager = MetadataManager(tmp_path)
            manager.initialize_session()

            async with PatchTrackerContext(manager) as tracker:
                assert tracker is not None
                assert tracker._session_started is True

        anyio.run(run_test)

    def test_finalizes_on_exit(self, tmp_path):
        """Should finalize tracker on exit."""
        import anyio

        async def run_test():
            manager = MetadataManager(tmp_path)
            manager.initialize_session()

            tracker_ref = None
            async with PatchTrackerContext(manager) as tracker:
                tracker_ref = tracker
                tracker.capture_template_state({"test.py": "content"})

            assert tracker_ref._session_started is False

        anyio.run(run_test)

    def test_disabled_returns_none(self, tmp_path):
        """Should return None when disabled."""
        import anyio

        async def run_test():
            manager = MetadataManager(tmp_path)
            manager.initialize_session()

            async with PatchTrackerContext(manager, enabled=False) as tracker:
                assert tracker is None

        anyio.run(run_test)

    def test_handles_exception(self, tmp_path):
        """Should finalize even on exception."""
        import anyio

        async def run_test():
            manager = MetadataManager(tmp_path)
            manager.initialize_session()

            tracker_ref = None
            try:
                async with PatchTrackerContext(manager) as tracker:
                    tracker_ref = tracker
                    raise ValueError("Test error")
            except ValueError:
                pass

            assert tracker_ref._session_started is False

        anyio.run(run_test)


# =============================================================================
# Backwards Compatibility Functions Tests
# =============================================================================


class TestBackwardsCompatibility:
    """Tests for backwards compatibility functions."""

    def test_capture_pre_llm_state(self, tmp_path):
        """capture_pre_llm_state should call capture_template_state."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        files = {"test.py": "content"}
        entry = capture_pre_llm_state(tracker, files)

        assert entry is not None
        assert entry.milestone == "template_generation"

    def test_capture_pre_llm_state_with_workflows(self, tmp_path):
        """capture_pre_llm_state should handle workflow files."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        files = {"main.py": "main"}
        workflow_files = [{"workflow.py": "workflow"}]
        capture_pre_llm_state(tracker, files, workflow_files)

        assert "workflows/workflow.py" in tracker.template_files

    def test_capture_post_llm_state(self, tmp_path):
        """capture_post_llm_state should call capture_enhanced_state."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        # First capture template
        tracker.capture_template_state({"test.py": "original"})

        # Then capture enhanced via backwards-compat function
        files = {"test.py": "enhanced"}
        entry = capture_post_llm_state(tracker, files)

        assert entry is not None
        assert entry.milestone == "llm_enhancement"

    def test_capture_post_llm_state_with_workflows(self, tmp_path):
        """capture_post_llm_state should handle workflow files."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        tracker.capture_template_state({"test.py": "original"})

        workflow_files = [{"enhanced.py": "enhanced workflow"}]
        capture_post_llm_state(tracker, {"test.py": "enhanced"}, workflow_files)

        assert "workflows/enhanced.py" in tracker.enhanced_files


# =============================================================================
# Integration Tests
# =============================================================================


class TestPatchTrackerIntegration:
    """Integration tests for full patch tracking workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete tracking workflow."""
        # Setup
        manager = MetadataManager(tmp_path)
        manager.initialize_session(api_info={"title": "Test API"})
        tracker = PatchTracker.from_metadata_manager(manager)

        # Template generation
        template_files = {
            "locustfile.py": "# Template locustfile\nimport locust\n",
            "test_data.py": "# Template test data\n",
        }
        workflow_files = [
            {"base_workflow.py": "# Base workflow\n"},
        ]
        entry1 = tracker.capture_template_state(template_files, workflow_files)
        assert entry1.milestone == "template_generation"

        # LLM enhancement
        enhanced_files = {
            "locustfile.py": "# Enhanced locustfile\nimport locust\n\nclass User:\n    pass\n",
            "test_data.py": "# Enhanced test data\ndata = {}\n",
        }
        enhanced_workflow = [
            {"base_workflow.py": "# Enhanced workflow\nclass BaseFlow:\n    pass\n"},
        ]
        entry2 = tracker.capture_enhanced_state(
            enhanced_files, enhanced_workflow, ai_model="gpt-4"
        )
        assert entry2.milestone == "llm_enhancement"

        # Validation
        validated_files = {
            "locustfile.py": "# Validated locustfile\nimport locust\n\nclass User:\n    @task\n    def test(self): pass\n",
            "test_data.py": "# Validated test data\ndata = {'key': 'value'}\n",
        }
        entry3 = tracker.capture_validation_state(
            validated_files, validation_results={"syntax_valid": True}
        )
        assert entry3.milestone == "validation"

        # Finalize
        tracker.finalize()
        manager.finalize_session()

        # Verify patches were created
        assert len(manager.session_info.patches) == 3
        assert manager.metadata_path.exists()

        # Verify patch files exist
        for entry in manager.session_info.patches:
            patch_content = manager.get_patch(entry.id)
            assert patch_content is not None

    def test_workflow_with_no_llm_changes(self, tmp_path):
        """Test workflow where LLM makes no changes."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        tracker = PatchTracker.from_metadata_manager(manager)

        # Template
        files = {"test.py": "content"}
        tracker.capture_template_state(files)

        # LLM returns same content
        entry = tracker.capture_enhanced_state(files)
        assert entry is None  # No changes

        # Only template patch should exist
        assert len(manager.session_info.patches) == 1
