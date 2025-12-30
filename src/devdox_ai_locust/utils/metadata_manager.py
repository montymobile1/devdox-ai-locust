"""
Central Metadata Manager for DevDox AI Locust

Simplified Structure (v3.0)
===========================

Directory Structure:
    .devdox_ai_locust/
    ├── metadata.json                    # Central metadata (API info, config)
    └── {session_id}/                    # e.g., 2025-12-29_10-39-24
        ├── session.json                 # Session milestones and patches info
        └── .patches/                    # Sequential patch files
            ├── 000001_a1b2c3d4.patch
            └── 000002_e5f6g7h8.patch

Key Concepts:
    - Each generation session gets a datetime-stamped directory
    - session.json tracks milestones (what each patch represents)
    - .patches/ contains sequential UUID-named patch files
    - Milestones can be: template_generation, llm_enhancement, validation, etc.
    - Extensible for future milestones (user_edit, refactor, etc.)

Example session.json:
    {
        "version": "3.0",
        "session_id": "2025-12-29_10-39-24",
        "created_at": "2025-12-29T10:39:24Z",
        "patches": [
            {
                "id": "000001_a1b2c3d4",
                "sequence": 1,
                "milestone": "template_generation",
                "description": "Initial template-based generation",
                "created_at": "2025-12-29T10:39:24Z",
                "stats": {"files_changed": 8, "additions": 450, "deletions": 0}
            },
            {
                "id": "000002_e5f6g7h8",
                "sequence": 2,
                "milestone": "llm_enhancement",
                "description": "AI enhancement of test files",
                "created_at": "2025-12-29T10:40:15Z",
                "stats": {"files_changed": 3, "additions": 120, "deletions": 15}
            }
        ]
    }
"""

import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

METADATA_VERSION = "3.0"
METADATA_FILENAME = "metadata.json"
SESSION_FILENAME = "session.json"
PATCHES_DIR = ".patches"
DEVDOX_DIR_NAME = ".devdox_ai_locust"


class Milestone(str, Enum):
    """Types of code change milestones"""
    TEMPLATE_GENERATION = "template_generation"
    LLM_ENHANCEMENT = "llm_enhancement"
    VALIDATION = "validation"
    USER_EDIT = "user_edit"
    REFACTOR = "refactor"
    MERGE = "merge"


class PatchStats(BaseModel):
    """Statistics for a patch"""
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0


class PatchEntry(BaseModel):
    """Entry in the session's patch list"""
    id: str  # e.g., "000001_a1b2c3d4"
    sequence: int
    milestone: str
    description: str = ""
    created_at: str
    stats: PatchStats = Field(default_factory=PatchStats)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    """Session information stored in session.json"""
    version: str = METADATA_VERSION
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    patches: List[PatchEntry] = Field(default_factory=list)

    def get_next_sequence(self) -> int:
        """Get next patch sequence number"""
        if not self.patches:
            return 1
        return max(p.sequence for p in self.patches) + 1

    def add_patch(
        self,
        milestone: str,
        description: str = "",
        stats: Optional[PatchStats] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatchEntry:
        """Add a new patch entry"""
        sequence = self.get_next_sequence()
        short_uuid = uuid.uuid4().hex[:8]
        patch_id = f"{sequence:06d}_{short_uuid}"

        entry = PatchEntry(
            id=patch_id,
            sequence=sequence,
            milestone=milestone,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
            stats=stats or PatchStats(),
            metadata=metadata or {},
        )
        self.patches.append(entry)
        return entry


class APIMetadata(BaseModel):
    """Metadata about the API being tested"""
    title: str = "Unknown"
    version: str = "Unknown"
    base_url: str = ""
    description: str = ""
    endpoints_count: int = 0
    swagger_source: str = ""
    source_type: str = ""


class GenerationConfig(BaseModel):
    """Configuration used for generation"""
    host: str = ""
    auth_enabled: bool = True
    db_type: str = ""
    ai_model: str = ""
    custom_requirement: str = ""


class FileEntry(BaseModel):
    """Metadata for a single generated file"""
    category: str = "main"  # main, workflow, config, data, docs
    size_bytes: int = 0
    lines: int = 0
    description: str = ""


class OutputSummary(BaseModel):
    """Summary statistics for output"""
    total_files: int = 0
    total_bytes: int = 0
    by_category: Dict[str, int] = Field(default_factory=dict)


class OutputInfo(BaseModel):
    """
    Information about generated output with file tree structure.

    Structure:
        {
            "root": "locust_tests_iter_7",
            "files": {
                "locustfile.py": {"category": "main", "size_bytes": 1234, ...},
                "workflows/base_workflow.py": {"category": "workflow", ...}
            },
            "summary": {"total_files": 14, "total_bytes": 15000, ...}
        }
    """
    root: str = ""
    files: Dict[str, FileEntry] = Field(default_factory=dict)
    summary: OutputSummary = Field(default_factory=OutputSummary)


class CentralMetadata(BaseModel):
    """
    Central metadata structure for DevDox AI Locust.
    Stored in .devdox_ai_locust/metadata.json
    """
    version: str = METADATA_VERSION
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    api: APIMetadata = Field(default_factory=APIMetadata)
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputInfo = Field(default_factory=OutputInfo)


class MetadataManager:
    """
    Manages the .devdox_ai_locust/ directory structure.

    Structure:
    - metadata.json: Central config and API info
    - {session_id}/session.json: Session milestones info
    - {session_id}/.patches/: Directory of sequential patch files
    """

    def __init__(self, output_dir: Path):
        """
        Initialize metadata manager.

        Args:
            output_dir: Directory where .devdox_ai_locust will be created
        """
        self.output_dir = Path(output_dir)
        self.devdox_dir = self.output_dir / DEVDOX_DIR_NAME
        self.metadata_path = self.devdox_dir / METADATA_FILENAME

        self.session_id = ""
        self._session_dir: Optional[Path] = None
        self._patches_dir: Optional[Path] = None
        self._session_path: Optional[Path] = None

        self.metadata = CentralMetadata()
        self.session_info = SessionInfo()
        self._start_time: Optional[datetime] = None

    # =========================================================================
    # Directory Setup
    # =========================================================================

    def _ensure_directories(self) -> None:
        """Create directory structure if needed"""
        self.devdox_dir.mkdir(parents=True, exist_ok=True)
        if self._session_dir:
            self._session_dir.mkdir(parents=True, exist_ok=True)
        if self._patches_dir:
            self._patches_dir.mkdir(parents=True, exist_ok=True)

    @property
    def patches_dir(self) -> Path:
        """Get the patches directory for the current session"""
        if self._patches_dir:
            return self._patches_dir
        # Fallback for backwards compatibility
        return self.devdox_dir / "patches"

    @property
    def manifest(self) -> SessionInfo:
        """Backwards compatibility alias for session_info"""
        return self.session_info

    # =========================================================================
    # Session Management
    # =========================================================================

    def initialize_session(
        self,
        api_info: Optional[Dict[str, Any]] = None,
        swagger_source: str = "",
        source_type: str = "",
    ) -> str:
        """
        Initialize a new generation session.

        Args:
            api_info: API metadata from swagger
            swagger_source: Source of swagger (URL or file path)
            source_type: "url" or "file"

        Returns:
            Session ID
        """
        self._start_time = datetime.now(timezone.utc)

        # Generate session ID (datetime stamp)
        self.session_id = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")

        # Setup session directories
        self._session_dir = self.devdox_dir / self.session_id
        self._patches_dir = self._session_dir / PATCHES_DIR
        self._session_path = self._session_dir / SESSION_FILENAME

        self._ensure_directories()

        # Initialize metadata
        self.metadata = CentralMetadata(
            session_id=self.session_id,
            created_at=self._start_time.isoformat(),
            updated_at=self._start_time.isoformat(),
        )

        # Set API info
        if api_info:
            self.metadata.api = APIMetadata(
                title=api_info.get("title", "Unknown"),
                version=api_info.get("version", "Unknown"),
                base_url=api_info.get("base_url", ""),
                description=api_info.get("description", ""),
                swagger_source=swagger_source,
                source_type=source_type,
            )

        # Initialize session info
        self.session_info = SessionInfo(
            session_id=self.session_id,
            created_at=self._start_time.isoformat(),
            updated_at=self._start_time.isoformat(),
        )

        # Load existing session if present (for incremental updates)
        self._load_session()

        logger.info(f"Initialized session: {self.session_id}")
        return self.session_id

    def finalize_session(self) -> None:
        """Finalize the session and save all metadata"""
        now = datetime.now(timezone.utc).isoformat()

        # Update timestamps
        self.metadata.updated_at = now
        self.session_info.updated_at = now

        # Calculate output summary
        self._calculate_output_summary()

        # Save everything
        self._save_metadata()
        self._save_session()

        logger.info(f"Finalized session: {self.session_id}")

    # =========================================================================
    # Patch Management
    # =========================================================================

    def create_patch(
        self,
        milestone: str,
        content: str,
        description: str = "",
        stats: Optional[PatchStats] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatchEntry:
        """
        Create a new patch in the session.

        Args:
            milestone: Type of milestone (template_generation, llm_enhancement, etc.)
            content: Patch content (unified diff format)
            description: Human-readable description
            stats: Patch statistics
            metadata: Additional metadata

        Returns:
            PatchEntry for the created patch
        """
        self._ensure_directories()

        # Add to session info
        entry = self.session_info.add_patch(
            milestone=milestone,
            description=description,
            stats=stats,
            metadata=metadata,
        )

        # Write patch file
        if self._patches_dir:
            patch_path = self._patches_dir / f"{entry.id}.patch"
            patch_path.write_text(content, encoding="utf-8")

        # Save session info
        self._save_session()

        logger.debug(f"Created patch: {entry.id} ({milestone})")
        return entry

    def get_patch(self, patch_id: str) -> Optional[str]:
        """
        Read a patch by ID.

        Args:
            patch_id: Patch ID (e.g., "000001_a1b2c3d4")

        Returns:
            Patch content or None if not found
        """
        if not self._patches_dir:
            return None
        patch_path = self._patches_dir / f"{patch_id}.patch"
        if patch_path.exists():
            return patch_path.read_text(encoding="utf-8")
        return None

    def get_patches_by_milestone(self, milestone: str) -> List[PatchEntry]:
        """Get all patches for a specific milestone type"""
        return [p for p in self.session_info.patches if p.milestone == milestone]

    def get_latest_patch(self) -> Optional[PatchEntry]:
        """Get the most recent patch entry"""
        if self.session_info.patches:
            return max(self.session_info.patches, key=lambda p: p.sequence)
        return None

    # =========================================================================
    # Configuration Updates
    # =========================================================================

    def update_api_endpoints_count(self, count: int) -> None:
        """Update the endpoints count"""
        self.metadata.api.endpoints_count = count

    def update_generation_config(
        self,
        host: Optional[str] = None,
        auth_enabled: Optional[bool] = None,
        db_type: Optional[str] = None,
        ai_model: Optional[str] = None,
        custom_requirement: Optional[str] = None,
    ) -> None:
        """Update generation configuration"""
        if host is not None:
            self.metadata.config.host = host
        if auth_enabled is not None:
            self.metadata.config.auth_enabled = auth_enabled
        if db_type is not None:
            self.metadata.config.db_type = db_type
        if ai_model is not None:
            self.metadata.config.ai_model = ai_model
        if custom_requirement is not None:
            self.metadata.config.custom_requirement = custom_requirement

    def register_file(
        self,
        path: str,
        content: str,
        category: str = "main",
        description: str = "",
    ) -> None:
        """
        Register a single generated file with metadata.

        Args:
            path: Relative path from output root (e.g., "workflows/base.py")
            content: File content (used to calculate size and lines)
            category: File category (main, workflow, config, data, docs)
            description: Optional description of the file
        """
        entry = FileEntry(
            category=category,
            size_bytes=len(content.encode("utf-8")),
            lines=content.count("\n") + (1 if content and not content.endswith("\n") else 0),
            description=description,
        )
        self.metadata.output.files[path] = entry

    def register_files(
        self,
        files: Dict[str, str],
        category: str = "main",
    ) -> None:
        """
        Register multiple files at once.

        Args:
            files: Dict of {path: content}
            category: Category for all files
        """
        for path, content in files.items():
            self.register_file(path, content, category=category)

    def set_output_root(self, root: str) -> None:
        """Set the output root directory name"""
        self.metadata.output.root = root

    def _calculate_output_summary(self) -> None:
        """Calculate output summary statistics"""
        files = self.metadata.output.files
        total_bytes = sum(f.size_bytes for f in files.values())
        by_category: Dict[str, int] = {}

        for entry in files.values():
            by_category[entry.category] = by_category.get(entry.category, 0) + 1

        self.metadata.output.summary = OutputSummary(
            total_files=len(files),
            total_bytes=total_bytes,
            by_category=by_category,
        )

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save_metadata(self) -> None:
        """Save central metadata to disk"""
        self._ensure_directories()
        self.metadata_path.write_text(
            json.dumps(self.metadata.model_dump(), indent=2),
            encoding="utf-8",
        )

    def _save_session(self) -> None:
        """Save session info to disk"""
        if not self._session_path:
            return
        self._ensure_directories()
        self._session_path.write_text(
            json.dumps(self.session_info.model_dump(), indent=2),
            encoding="utf-8",
        )

    def _load_session(self) -> None:
        """Load existing session if present"""
        if not self._session_path or not self._session_path.exists():
            return
        try:
            data = json.loads(self._session_path.read_text(encoding="utf-8"))
            # Preserve existing patches
            if "patches" in data:
                for patch_data in data["patches"]:
                    entry = PatchEntry(**patch_data)
                    self.session_info.patches.append(entry)
        except Exception as e:
            logger.warning(f"Failed to load existing session: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_structure_info(self) -> str:
        """Get a formatted string showing the directory structure"""
        patch_count = len(self.session_info.patches)
        return f"""
.devdox_ai_locust/
├── metadata.json                    # Central metadata (API info, config)
└── {self.session_id}/               # Session directory
    ├── session.json                 # Milestones ({patch_count} patches)
    └── .patches/                    # Patch files
"""

    def to_dict(self) -> Dict[str, Any]:
        """Export all data as dictionary"""
        return {
            "metadata": self.metadata.model_dump(),
            "session": self.session_info.model_dump(),
        }
