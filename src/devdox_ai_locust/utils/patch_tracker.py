"""
Patch Tracking System for DevDox AI Locust

Tracks code generation by saving patch files before and after AI LLM operations.
Similar to PostgreSQL's WAL (Write-Ahead Logging) concept - creates timestamped
directories with pre/post LLM patches for analysis and comparison.

This module integrates with the central MetadataManager to store patch sessions
under the organized .devdox_ai_locust/ directory structure.

Structure (v2.0):
    .devdox_ai_locust/
    ├── metadata.json                          # Central metadata (MetadataManager)
    └── ai_enhancement/
        └── patches/
            └── {session_id}/                  # e.g., 2025-12-28_21-47-06
                ├── pre_llm.patch              # Template output before AI
                ├── post_llm.patch             # Changes made by AI
                ├── summary.json               # Patch statistics
                └── session.json               # Session-specific metadata
"""

import json
import logging
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Protocol, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
import asyncio

if TYPE_CHECKING:
    from .metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


@dataclass
class PatchSessionMetadata:
    """Metadata for a single patch session (stored in session.json)"""
    session_id: str
    timestamp: str
    api_title: str
    api_version: str
    endpoints_count: int
    files_generated: List[str] = field(default_factory=list)
    ai_model: Optional[str] = None
    enhancements_applied: List[str] = field(default_factory=list)
    generation_time_seconds: float = 0.0
    pre_llm_files_count: int = 0
    post_llm_files_count: int = 0
    files_changed: int = 0
    error: Optional[str] = None


class PatchStorageProtocol(Protocol):
    """Protocol for patch storage backends (allows dependency injection)"""

    def save_patch(self, session_id: str, patch_name: str, content: str) -> Path:
        """Save a patch file"""
        ...

    def save_session_metadata(self, session_id: str, metadata: PatchSessionMetadata) -> Path:
        """Save session metadata file"""
        ...

    def get_session_dir(self, session_id: str) -> Path:
        """Get the session directory path"""
        ...


class FileSystemPatchStorage:
    """File system based patch storage"""

    def __init__(self, patches_dir: Path):
        """
        Initialize patch storage.

        Args:
            patches_dir: The patches directory (typically .devdox_ai_locust/patches/)
        """
        self.patches_dir = Path(patches_dir)

    def _ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists"""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_session_dir(self, session_id: str) -> Path:
        """Get the session directory path"""
        return self._ensure_dir(self.patches_dir / session_id)

    def save_patch(self, session_id: str, patch_name: str, content: str) -> Path:
        """Save a patch file"""
        session_dir = self.get_session_dir(session_id)
        patch_file = session_dir / f"{patch_name}.patch"
        patch_file.write_text(content, encoding="utf-8")
        logger.debug(f"Saved patch: {patch_file}")
        return patch_file

    def save_session_metadata(self, session_id: str, metadata: PatchSessionMetadata) -> Path:
        """Save session-specific metadata file (session.json)"""
        session_dir = self.get_session_dir(session_id)
        session_file = session_dir / "session.json"
        session_file.write_text(
            json.dumps(asdict(metadata), indent=2, default=str),
            encoding="utf-8"
        )
        logger.debug(f"Saved session metadata: {session_file}")
        return session_file


class PatchTracker:
    """
    Tracks code generation by creating unified diffs before/after AI operations.

    Integrates with MetadataManager to store patches under .devdox_ai_locust/patches/
    and register sessions with the central metadata.

    Usage:
        # Standalone usage (creates its own patches directory)
        tracker = PatchTracker(output_dir)
        tracker.start_session(api_info)

        # Or integrated with MetadataManager
        metadata_manager = MetadataManager(output_dir)
        tracker = PatchTracker.from_metadata_manager(metadata_manager)
        tracker.start_session(api_info)

        # Before AI enhancement
        tracker.capture_pre_llm_state(files_dict)

        # ... AI enhancement happens ...

        # After AI enhancement
        tracker.capture_post_llm_state(enhanced_files_dict)

        tracker.finalize_session(metadata)
    """

    def __init__(
        self,
        output_dir: Path,
        storage: Optional[PatchStorageProtocol] = None,
        metadata_manager: Optional["MetadataManager"] = None
    ):
        self.output_dir = Path(output_dir)
        self.metadata_manager = metadata_manager

        # Determine patches directory
        if metadata_manager:
            patches_dir = metadata_manager.patches_dir
        else:
            patches_dir = self.output_dir / ".devdox_ai_locust" / "patches"
            patches_dir.mkdir(parents=True, exist_ok=True)

        self.storage = storage or FileSystemPatchStorage(patches_dir)
        self.session_id: Optional[str] = None
        self.pre_llm_files: Dict[str, str] = {}
        self.post_llm_files: Dict[str, str] = {}
        self._start_time: Optional[datetime] = None

    @classmethod
    def from_metadata_manager(cls, metadata_manager: "MetadataManager") -> "PatchTracker":
        """Create a PatchTracker integrated with a MetadataManager"""
        return cls(
            output_dir=metadata_manager.output_dir,
            metadata_manager=metadata_manager
        )

    def _generate_session_id(self) -> str:
        """Generate PostgreSQL-style timestamped session ID"""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def start_session(self, api_info: Optional[Dict[str, Any]] = None) -> str:
        """Start a new patch tracking session"""
        self.session_id = self._generate_session_id()
        self._start_time = datetime.now()
        self.pre_llm_files = {}
        self.post_llm_files = {}

        logger.info(f"Started patch tracking session: {self.session_id}")
        return self.session_id

    def capture_pre_llm_state(
        self,
        files: Dict[str, str],
        directory_files: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        Capture the state of generated files BEFORE AI LLM enhancement.

        Args:
            files: Main files dict (filename -> content)
            directory_files: List of workflow file dicts
        """
        if not self.session_id:
            self.start_session()

        # Capture main files
        for filename, content in files.items():
            self.pre_llm_files[filename] = content

        # Capture workflow/directory files
        if directory_files:
            for file_dict in directory_files:
                for filename, content in file_dict.items():
                    self.pre_llm_files[f"workflows/{filename}"] = content

        logger.debug(f"Captured pre-LLM state: {len(self.pre_llm_files)} files")

    def capture_post_llm_state(
        self,
        files: Dict[str, str],
        directory_files: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        Capture the state of generated files AFTER AI LLM enhancement.

        Args:
            files: Main files dict (filename -> content)
            directory_files: List of workflow file dicts
        """
        if not self.session_id:
            raise RuntimeError("No active session. Call start_session() first.")

        # Capture main files
        for filename, content in files.items():
            self.post_llm_files[filename] = content

        # Capture workflow/directory files
        if directory_files:
            for file_dict in directory_files:
                for filename, content in file_dict.items():
                    self.post_llm_files[f"workflows/{filename}"] = content

        logger.debug(f"Captured post-LLM state: {len(self.post_llm_files)} files")

    def _generate_unified_diff(
        self,
        files_before: Dict[str, str],
        files_after: Dict[str, str],
        context_lines: int = 3
    ) -> str:
        """Generate unified diff between two file states"""
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

    def save_patches(self) -> Dict[str, Path]:
        """Save pre-LLM and post-LLM patches"""
        if not self.session_id:
            raise RuntimeError("No active session.")

        saved_paths = {}

        # Pre-LLM patch: diff from empty to pre-LLM state (shows template output)
        pre_llm_diff = self._generate_unified_diff({}, self.pre_llm_files)
        if pre_llm_diff:
            saved_paths["pre_llm"] = self.storage.save_patch(
                self.session_id, "pre_llm", pre_llm_diff
            )

        # Post-LLM patch: diff from pre-LLM to post-LLM (shows AI changes)
        post_llm_diff = self._generate_unified_diff(
            self.pre_llm_files, self.post_llm_files
        )
        if post_llm_diff:
            saved_paths["post_llm"] = self.storage.save_patch(
                self.session_id, "post_llm", post_llm_diff
            )

        return saved_paths

    def finalize_session(
        self,
        api_info: Optional[Dict[str, Any]] = None,
        endpoints_count: int = 0,
        ai_model: Optional[str] = None,
        enhancements_applied: Optional[List[str]] = None,
        error: Optional[str] = None
    ) -> PatchSessionMetadata:
        """Finalize the session and save metadata"""
        if not self.session_id:
            raise RuntimeError("No active session.")

        # Calculate generation time
        generation_time = 0.0
        if self._start_time:
            generation_time = (datetime.now() - self._start_time).total_seconds()

        # Calculate files changed
        files_changed = self._count_changed_files()

        # Build session metadata
        metadata = PatchSessionMetadata(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            api_title=api_info.get("title", "Unknown") if api_info else "Unknown",
            api_version=api_info.get("version", "Unknown") if api_info else "Unknown",
            endpoints_count=endpoints_count,
            files_generated=list(self.post_llm_files.keys() or self.pre_llm_files.keys()),
            ai_model=ai_model,
            enhancements_applied=enhancements_applied or [],
            generation_time_seconds=generation_time,
            pre_llm_files_count=len(self.pre_llm_files),
            post_llm_files_count=len(self.post_llm_files),
            files_changed=files_changed,
            error=error
        )

        # Save patches and session metadata
        self.save_patches()
        self.storage.save_session_metadata(self.session_id, metadata)

        # Register with central metadata manager if available
        if self.metadata_manager:
            self.metadata_manager.register_patch_session(self.session_id)

        logger.info(f"Finalized patch session: {self.session_id}")

        # Reset state
        session_id = self.session_id
        self.session_id = None
        self.pre_llm_files = {}
        self.post_llm_files = {}
        self._start_time = None

        return metadata

    def _count_changed_files(self) -> int:
        """Count number of files that were changed by LLM"""
        changed = 0
        all_files = set(self.pre_llm_files.keys()) | set(self.post_llm_files.keys())
        for filename in all_files:
            before = self.pre_llm_files.get(filename, "")
            after = self.post_llm_files.get(filename, "")
            if before != after:
                changed += 1
        return changed

    def get_session_path(self) -> Optional[Path]:
        """Get the current session's directory path"""
        if not self.session_id:
            return None
        return self.storage.get_session_dir(self.session_id)


class PatchTrackerContext:
    """
    Context manager for patch tracking.

    Usage:
        # Standalone usage
        async with PatchTrackerContext(output_dir, api_info) as tracker:
            tracker.capture_pre_llm_state(base_files, workflow_files)
            enhanced_files = await enhance_with_ai(base_files)
            tracker.capture_post_llm_state(enhanced_files, enhanced_workflow_files)

        # With MetadataManager integration
        metadata_manager = MetadataManager(output_dir)
        async with PatchTrackerContext(output_dir, api_info, metadata_manager=metadata_manager) as tracker:
            ...
    """

    def __init__(
        self,
        output_dir: Path,
        api_info: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        metadata_manager: Optional["MetadataManager"] = None
    ):
        self.output_dir = output_dir
        self.api_info = api_info
        self.enabled = enabled
        self.metadata_manager = metadata_manager
        self.tracker: Optional[PatchTracker] = None

    async def __aenter__(self) -> Optional[PatchTracker]:
        if not self.enabled:
            return None

        if self.metadata_manager:
            self.tracker = PatchTracker.from_metadata_manager(self.metadata_manager)
        else:
            self.tracker = PatchTracker(self.output_dir)

        self.tracker.start_session(self.api_info)
        return self.tracker

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.tracker and self.tracker.session_id:
            error_msg = str(exc_val) if exc_val else None
            self.tracker.finalize_session(
                api_info=self.api_info,
                error=error_msg
            )
