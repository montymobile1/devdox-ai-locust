"""
Central Metadata Manager for DevDox AI Locust

PostgreSQL WAL-Inspired Structure (v3.0)
========================================

Simple, flat, extensible design inspired by PostgreSQL's Write-Ahead Log.

Directory Structure:
    .devdox_ai_locust/
    ├── metadata.json       # Central metadata (API info, config)
    ├── manifest.json       # WAL manifest - maps patches to milestones
    └── wal/                # Write-Ahead Log - sequential patches
        ├── 000001_a1b2c3d4.patch
        ├── 000002_e5f6g7h8.patch
        └── 000003_i9j0k1l2.patch

Key Concepts:
    - Each code change milestone gets a sequential patch file
    - Patches are named: {sequence}_{short_uuid}.patch
    - manifest.json tracks what each patch represents
    - Milestones can be: template_generation, llm_enhancement, validation, etc.
    - Extensible for future milestones (user_edit, refactor, etc.)

Example manifest.json:
    {
        "version": "3.0",
        "session_id": "2025-12-29_10-39-24",
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
MANIFEST_FILENAME = "manifest.json"
WAL_DIR = "wal"
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
    """Entry in the WAL manifest"""
    id: str  # e.g., "000001_a1b2c3d4"
    sequence: int
    milestone: str
    description: str = ""
    created_at: str
    stats: PatchStats = Field(default_factory=PatchStats)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WALManifest(BaseModel):
    """Write-Ahead Log manifest"""
    version: str = METADATA_VERSION
    session_id: str = ""
    created_at: str = ""
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


class OutputInfo(BaseModel):
    """Information about generated output"""
    directory: str = ""
    main_files: List[str] = Field(default_factory=list)
    workflow_files: List[str] = Field(default_factory=list)
    total_files: int = 0


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
    Manages the .devdox_ai_locust/ directory with PostgreSQL WAL-inspired structure.

    Simple, flat, extensible:
    - metadata.json: Central config and API info
    - manifest.json: WAL manifest mapping patches to milestones
    - wal/: Directory of sequential patch files
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
        self.manifest_path = self.devdox_dir / MANIFEST_FILENAME
        self.wal_dir = self.devdox_dir / WAL_DIR

        self.session_id = ""
        self.metadata = CentralMetadata()
        self.manifest = WALManifest()
        self._start_time: Optional[datetime] = None

    # =========================================================================
    # Directory Setup
    # =========================================================================

    def _ensure_directories(self) -> None:
        """Create directory structure if needed"""
        self.devdox_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(exist_ok=True)

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
        self._ensure_directories()
        self._start_time = datetime.now(timezone.utc)

        # Generate session ID
        self.session_id = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")

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

        # Initialize manifest
        self.manifest = WALManifest(
            session_id=self.session_id,
            created_at=self._start_time.isoformat(),
        )

        # Load existing manifest if present (for incremental updates)
        self._load_manifest()

        logger.info(f"Initialized session: {self.session_id}")
        return self.session_id

    def finalize_session(self) -> None:
        """Finalize the session and save all metadata"""
        # Update timestamp
        self.metadata.updated_at = datetime.now(timezone.utc).isoformat()

        # Calculate totals
        self.metadata.output.total_files = (
            len(self.metadata.output.main_files) +
            len(self.metadata.output.workflow_files)
        )

        # Save everything
        self._save_metadata()
        self._save_manifest()

        logger.info(f"Finalized session: {self.session_id}")

    # =========================================================================
    # Patch Management (WAL)
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
        Create a new patch in the WAL.

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

        # Add to manifest
        entry = self.manifest.add_patch(
            milestone=milestone,
            description=description,
            stats=stats,
            metadata=metadata,
        )

        # Write patch file
        patch_path = self.wal_dir / f"{entry.id}.patch"
        patch_path.write_text(content, encoding="utf-8")

        # Save manifest
        self._save_manifest()

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
        patch_path = self.wal_dir / f"{patch_id}.patch"
        if patch_path.exists():
            return patch_path.read_text(encoding="utf-8")
        return None

    def get_patches_by_milestone(self, milestone: str) -> List[PatchEntry]:
        """Get all patches for a specific milestone type"""
        return [p for p in self.manifest.patches if p.milestone == milestone]

    def get_latest_patch(self) -> Optional[PatchEntry]:
        """Get the most recent patch entry"""
        if self.manifest.patches:
            return max(self.manifest.patches, key=lambda p: p.sequence)
        return None

    @property
    def patches_dir(self) -> Path:
        """Get the WAL directory (for backwards compatibility)"""
        return self.wal_dir

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

    def register_files(
        self,
        main_files: Optional[List[str]] = None,
        workflow_files: Optional[List[str]] = None,
    ) -> None:
        """Register generated files"""
        if main_files:
            self.metadata.output.main_files = main_files
        if workflow_files:
            self.metadata.output.workflow_files = workflow_files
        self.metadata.output.directory = str(self.output_dir)

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

    def _save_manifest(self) -> None:
        """Save WAL manifest to disk"""
        self._ensure_directories()
        self.manifest_path.write_text(
            json.dumps(self.manifest.model_dump(), indent=2),
            encoding="utf-8",
        )

    def _load_manifest(self) -> None:
        """Load existing manifest if present"""
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                # Preserve existing patches
                if "patches" in data:
                    for patch_data in data["patches"]:
                        entry = PatchEntry(**patch_data)
                        self.manifest.patches.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load existing manifest: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_structure_info(self) -> str:
        """Get a formatted string showing the directory structure"""
        patch_count = len(self.manifest.patches)
        return f"""
.devdox_ai_locust/
├── metadata.json       # Central metadata (API info, config)
├── manifest.json       # WAL manifest ({patch_count} patches)
└── wal/                # Write-Ahead Log
    └── *.patch         # Sequential patch files
"""

    def to_dict(self) -> Dict[str, Any]:
        """Export all data as dictionary"""
        return {
            "metadata": self.metadata.model_dump(),
            "manifest": self.manifest.model_dump(),
        }
