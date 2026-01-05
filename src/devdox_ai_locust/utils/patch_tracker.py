"""
Patch Tracking System for DevDox AI Locust

PostgreSQL WAL-Inspired Design (v3.0)
=====================================

Tracks code generation milestones by creating sequential patch files.
Each milestone (template generation, LLM enhancement, etc.) gets its own patch.

Structure:
    .devdox_ai_locust/
    ├── metadata.json       # Central metadata
    ├── manifest.json       # WAL manifest - maps patches to milestones
    └── wal/                # Sequential patches
        ├── 000001_a1b2c3d4.patch   # template_generation
        ├── 000002_e5f6g7h8.patch   # llm_enhancement
        └── ...

Key Concepts:
    - Milestones are types of code changes (template_generation, llm_enhancement, etc.)
    - Each patch is named with sequence number + short UUID
    - manifest.json tracks what milestone each patch represents
    - Extensible for future milestone types
"""

import logging
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .metadata_manager import MetadataManager, PatchEntry, PatchStats

logger = logging.getLogger(__name__)


class PatchTracker:
    """
    Tracks code generation milestones by creating sequential patches.

    Integrates with MetadataManager to store patches in the WAL directory.
    Each milestone (template generation, LLM enhancement, etc.) creates a patch.

    Usage:
        metadata_manager = MetadataManager(output_dir)
        metadata_manager.initialize_session(api_info)

        tracker = PatchTracker.from_metadata_manager(metadata_manager)

        # After template generation
        tracker.capture_template_state(files_dict)

        # After AI enhancement
        tracker.capture_enhanced_state(enhanced_files_dict)

        # Finalize
        tracker.finalize()
    """

    def __init__(
        self,
        output_dir: Path,
        metadata_manager: Optional["MetadataManager"] = None
    ):
        """
        Initialize patch tracker.

        Args:
            output_dir: Base output directory
            metadata_manager: MetadataManager instance (recommended)
        """
        self.output_dir = Path(output_dir)
        self.metadata_manager = metadata_manager

        # State tracking
        self.template_files: Dict[str, str] = {}
        self.enhanced_files: Dict[str, str] = {}
        self._start_time: Optional[datetime] = None
        self._session_started = False

    @classmethod
    def from_metadata_manager(cls, metadata_manager: "MetadataManager") -> "PatchTracker":
        """Create a PatchTracker integrated with a MetadataManager."""
        return cls(
            output_dir=metadata_manager.output_dir,
            metadata_manager=metadata_manager
        )

    def start_session(self) -> None:
        """Start tracking a new session."""
        self._start_time = datetime.now()
        self.template_files = {}
        self.enhanced_files = {}
        self._session_started = True
        logger.debug("Started patch tracking session")

    def capture_template_state(
        self,
        files: Dict[str, str],
        workflow_files: Optional[List[Dict[str, str]]] = None
    ) -> Optional["PatchEntry"]:
        """
        Capture files after template generation (before AI enhancement).

        Creates a template_generation milestone patch.

        Args:
            files: Main files dict (filename -> content)
            workflow_files: List of workflow file dicts

        Returns:
            PatchEntry if patch was created
        """
        if not self._session_started:
            self.start_session()

        # Capture main files
        for filename, content in files.items():
            self.template_files[filename] = content

        # Capture workflow/directory files
        if workflow_files:
            for file_dict in workflow_files:
                for filename, content in file_dict.items():
                    self.template_files[f"workflows/{filename}"] = content

        logger.debug(f"Captured template state: {len(self.template_files)} files")

        # Create the template generation patch
        return self._create_milestone_patch(
            milestone="template_generation",
            files_before={},
            files_after=self.template_files,
            description="Initial template-based generation",
        )

    def capture_enhanced_state(
        self,
        files: Dict[str, str],
        workflow_files: Optional[List[Dict[str, str]]] = None,
        ai_model: Optional[str] = None,
    ) -> Optional["PatchEntry"]:
        """
        Capture files after AI enhancement.

        Creates an llm_enhancement milestone patch showing the diff.

        Args:
            files: Enhanced files dict (filename -> content)
            workflow_files: List of workflow file dicts
            ai_model: Name of the AI model used

        Returns:
            PatchEntry if patch was created
        """
        if not self._session_started:
            raise RuntimeError("No active session. Call start_session() first.")

        # Capture main files
        for filename, content in files.items():
            self.enhanced_files[filename] = content

        # Capture workflow/directory files
        if workflow_files:
            for file_dict in workflow_files:
                for filename, content in file_dict.items():
                    self.enhanced_files[f"workflows/{filename}"] = content

        logger.debug(f"Captured enhanced state: {len(self.enhanced_files)} files")

        # Create the LLM enhancement patch
        metadata = {}
        if ai_model:
            metadata["ai_model"] = ai_model

        return self._create_milestone_patch(
            milestone="llm_enhancement",
            files_before=self.template_files,
            files_after=self.enhanced_files,
            description="AI enhancement of test files",
            extra_metadata=metadata,
        )

    def capture_validation_state(
        self,
        files: Dict[str, str],
        validation_results: Optional[Dict[str, Any]] = None,
    ) -> Optional["PatchEntry"]:
        """
        Capture files after validation/fixes.

        Creates a validation milestone patch.

        Args:
            files: Files after validation
            validation_results: Optional validation metadata

        Returns:
            PatchEntry if patch was created
        """
        if not self._session_started:
            raise RuntimeError("No active session.")

        # Get the previous state (enhanced or template)
        previous = self.enhanced_files if self.enhanced_files else self.template_files

        metadata = {}
        if validation_results:
            metadata["validation"] = validation_results

        return self._create_milestone_patch(
            milestone="validation",
            files_before=previous,
            files_after=files,
            description="Validation and fixes",
            extra_metadata=metadata,
        )

    def _create_milestone_patch(
        self,
        milestone: str,
        files_before: Dict[str, str],
        files_after: Dict[str, str],
        description: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional["PatchEntry"]:
        """
        Create a patch for a milestone.

        Args:
            milestone: Type of milestone
            files_before: Files before this milestone
            files_after: Files after this milestone
            description: Human-readable description
            extra_metadata: Additional metadata

        Returns:
            PatchEntry if patch was created, None if no changes
        """
        if not self.metadata_manager:
            logger.warning("No metadata manager - patch not saved")
            return None

        # Generate unified diff
        diff_content = self._generate_unified_diff(files_before, files_after)

        if not diff_content.strip():
            logger.debug(f"No changes for milestone: {milestone}")
            return None

        # Calculate stats
        from .metadata_manager import PatchStats
        stats = self._calculate_stats(files_before, files_after)

        # Create the patch
        entry = self.metadata_manager.create_patch(
            milestone=milestone,
            content=diff_content,
            description=description,
            stats=stats,
            metadata=extra_metadata,
        )

        logger.info(f"Created patch {entry.id} for milestone: {milestone}")
        return entry

    def _generate_unified_diff(
        self,
        files_before: Dict[str, str],
        files_after: Dict[str, str],
        context_lines: int = 3
    ) -> str:
        """Generate unified diff between two file states."""
        diff_lines = []

        # Get all unique filenames
        all_files = sorted(set(files_before.keys()) | set(files_after.keys()))

        for filename in all_files:
            before_content = files_before.get(filename, "")
            after_content = files_after.get(filename, "")

            if before_content == after_content:
                continue  # Skip unchanged files

            before_lines = before_content.splitlines(keepends=True)
            after_lines = after_content.splitlines(keepends=True)

            # Ensure lines end with newline for proper diff
            if before_lines and not before_lines[-1].endswith('\n'):
                before_lines[-1] += '\n'
            if after_lines and not after_lines[-1].endswith('\n'):
                after_lines[-1] += '\n'

            diff = difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                n=context_lines
            )

            diff_lines.extend(diff)

        return "".join(diff_lines)

    def _calculate_stats(
        self,
        files_before: Dict[str, str],
        files_after: Dict[str, str],
    ) -> "PatchStats":
        """Calculate patch statistics."""
        from .metadata_manager import PatchStats

        files_changed = 0
        additions = 0
        deletions = 0

        all_files = set(files_before.keys()) | set(files_after.keys())

        for filename in all_files:
            before = files_before.get(filename, "")
            after = files_after.get(filename, "")

            if before != after:
                files_changed += 1

                before_lines = set(before.splitlines())
                after_lines = set(after.splitlines())

                additions += len(after_lines - before_lines)
                deletions += len(before_lines - after_lines)

        return PatchStats(
            files_changed=files_changed,
            additions=additions,
            deletions=deletions,
        )

    def finalize(self) -> None:
        """
        Finalize the tracking session.

        Note: Does NOT call metadata_manager.finalize_session().
        The caller should finalize the metadata manager separately.
        """
        # Reset state
        self.template_files = {}
        self.enhanced_files = {}
        self._start_time = None
        self._session_started = False

        logger.info("Finalized patch tracking session")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.metadata_manager:
            return {"patches": 0}

        manifest = self.metadata_manager.manifest
        patches_by_milestone = {}

        for patch in manifest.patches:
            milestone = patch.milestone
            if milestone not in patches_by_milestone:
                patches_by_milestone[milestone] = []
            patches_by_milestone[milestone].append(patch.id)

        return {
            "session_id": manifest.session_id,
            "total_patches": len(manifest.patches),
            "patches_by_milestone": patches_by_milestone,
        }


class PatchTrackerContext:
    """
    Context manager for patch tracking.

    Usage:
        metadata_manager = MetadataManager(output_dir)
        metadata_manager.initialize_session(api_info)

        async with PatchTrackerContext(metadata_manager) as tracker:
            tracker.capture_template_state(base_files, workflow_files)
            enhanced_files = await enhance_with_ai(base_files)
            tracker.capture_enhanced_state(enhanced_files)
    """

    def __init__(
        self,
        metadata_manager: "MetadataManager",
        enabled: bool = True,
    ):
        """
        Initialize context.

        Args:
            metadata_manager: MetadataManager instance
            enabled: Whether tracking is enabled
        """
        self.metadata_manager = metadata_manager
        self.enabled = enabled
        self.tracker: Optional[PatchTracker] = None

    async def __aenter__(self) -> Optional[PatchTracker]:
        if not self.enabled:
            return None

        self.tracker = PatchTracker.from_metadata_manager(self.metadata_manager)
        self.tracker.start_session()
        return self.tracker

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.tracker and self.tracker._session_started:
            self.tracker.finalize()


# Backwards compatibility aliases
def capture_pre_llm_state(tracker: PatchTracker, files: Dict[str, str], workflow_files=None):
    """Backwards compatible wrapper for capture_template_state."""
    return tracker.capture_template_state(files, workflow_files)


def capture_post_llm_state(tracker: PatchTracker, files: Dict[str, str], workflow_files=None):
    """Backwards compatible wrapper for capture_enhanced_state."""
    return tracker.capture_enhanced_state(files, workflow_files)
