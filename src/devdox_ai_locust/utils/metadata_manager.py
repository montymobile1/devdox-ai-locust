"""
Central Metadata Manager for DevDox AI Locust

Manages the .devdox_ai_locust/ directory with a clear, organized structure.
This is the single source of truth for all generated test information.

Directory Structure:
    .devdox_ai_locust/
    ├── metadata.json                    # Central metadata file (main index)
    │
    ├── generation/                      # Generation-related data
    │   └── sessions/                    # Historical session data
    │       └── {session_id}/
    │           └── config.json          # Session-specific config
    │
    ├── ai_enhancement/                  # AI enhancement tracking
    │   ├── patches/                     # Pre/post LLM code patches
    │   │   └── {session_id}/
    │   │       ├── pre_llm.patch        # Code before AI enhancement
    │   │       ├── post_llm.patch       # Code after AI enhancement
    │   │       └── summary.json         # Patch statistics
    │   │
    │   └── constraints/                 # AI sandbox constraints used
    │       └── {session_id}/
    │           ├── test_data.txt        # Constraints for test_data.py
    │           └── utils.txt            # Constraints for utils.py
    │
    ├── codebase_analysis/               # CodebaseAwareness outputs
    │   └── {session_id}/
    │       ├── dependencies.json        # File dependency map
    │       ├── protected_symbols.json   # Protected symbols per file
    │       └── exports.json             # Exports per file
    │
    └── logs/                            # Generation logs
        └── {session_id}.log
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Protocol, Set
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

METADATA_VERSION = "2.0"
METADATA_FILENAME = "metadata.json"
DEVDOX_DIR_NAME = ".devdox_ai_locust"


class SubDirectory(str, Enum):
    """Subdirectories within .devdox_ai_locust/"""
    GENERATION = "generation"
    GENERATION_SESSIONS = "generation/sessions"
    AI_ENHANCEMENT = "ai_enhancement"
    AI_PATCHES = "ai_enhancement/patches"
    AI_CONSTRAINTS = "ai_enhancement/constraints"
    CODEBASE_ANALYSIS = "codebase_analysis"
    LOGS = "logs"


class APIMetadata(BaseModel):
    """Metadata about the API being tested"""
    title: str = "Unknown"
    version: str = "Unknown"
    base_url: str = ""
    description: str = ""
    endpoints_count: int = 0
    swagger_source: str = ""
    source_type: str = ""  # "url" or "file"


class GenerationMetadata(BaseModel):
    """Metadata about the generation session"""
    session_id: str = ""
    created_at: str = ""
    host: str = ""
    auth_enabled: bool = True
    db_type: str = ""
    custom_requirement: str = ""
    processing_time_seconds: float = 0.0
    ai_model: str = ""
    enhancements_applied: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class OutputMetadata(BaseModel):
    """Metadata about generated output files"""
    directory: str = ""
    main_files: List[str] = Field(default_factory=list)
    workflow_files: List[str] = Field(default_factory=list)
    total_files: int = 0


class PatchSessionInfo(BaseModel):
    """Information about a patch session"""
    session_id: str
    created_at: str
    pre_llm_files: int = 0
    post_llm_files: int = 0
    files_changed: int = 0
    files_added: int = 0
    files_removed: int = 0


class AIEnhancementMetadata(BaseModel):
    """Metadata for AI enhancement feature"""
    enabled: bool = True
    total_sessions: int = 0
    latest_session: str = ""
    sessions: List[str] = Field(default_factory=list)


class CodebaseAnalysisMetadata(BaseModel):
    """Metadata for codebase analysis feature"""
    enabled: bool = True
    latest_session: str = ""
    total_protected_symbols: int = 0


class FeaturesMetadata(BaseModel):
    """Container for all extensible features"""
    ai_enhancement: AIEnhancementMetadata = Field(default_factory=AIEnhancementMetadata)
    codebase_analysis: CodebaseAnalysisMetadata = Field(default_factory=CodebaseAnalysisMetadata)


class CentralMetadata(BaseModel):
    """
    Central metadata structure for DevDox AI Locust.

    This is the main data structure stored in .devdox_ai_locust/metadata.json.
    It provides a high-level overview; detailed data is in subdirectories.
    """
    version: str = METADATA_VERSION
    created_at: str = ""
    updated_at: str = ""
    api: APIMetadata = Field(default_factory=APIMetadata)
    generation: GenerationMetadata = Field(default_factory=GenerationMetadata)
    output: OutputMetadata = Field(default_factory=OutputMetadata)
    features: FeaturesMetadata = Field(default_factory=FeaturesMetadata)


class MetadataStorageProtocol(Protocol):
    """Protocol for metadata storage backends (allows dependency injection)"""

    def load(self) -> Optional[CentralMetadata]:
        """Load metadata from storage"""
        ...

    def save(self, metadata: CentralMetadata) -> Path:
        """Save metadata to storage"""
        ...

    def exists(self) -> bool:
        """Check if metadata exists"""
        ...

    def get_devdox_dir(self) -> Path:
        """Get the .devdox_ai_locust directory path"""
        ...


class FileSystemMetadataStorage:
    """File system based metadata storage"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.devdox_dir = self.output_dir / DEVDOX_DIR_NAME
        self.metadata_path = self.devdox_dir / METADATA_FILENAME

    def _ensure_dir(self) -> Path:
        """Ensure .devdox_ai_locust directory exists"""
        self.devdox_dir.mkdir(parents=True, exist_ok=True)
        return self.devdox_dir

    def get_devdox_dir(self) -> Path:
        """Get the .devdox_ai_locust directory path"""
        return self._ensure_dir()

    def exists(self) -> bool:
        """Check if metadata file exists"""
        return self.metadata_path.exists()

    def load(self) -> Optional[CentralMetadata]:
        """Load metadata from file"""
        if not self.exists():
            return None

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct the dataclass structure
            return self._dict_to_metadata(data)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None

    def save(self, metadata: CentralMetadata) -> Path:
        """Save metadata to file"""
        self._ensure_dir()

        # Update the updated_at timestamp
        metadata.updated_at = datetime.now(timezone.utc).isoformat()

        # Convert to dict and save
        data = self._metadata_to_dict(metadata)

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Saved metadata to: {self.metadata_path}")
        return self.metadata_path

    def _metadata_to_dict(self, metadata: CentralMetadata) -> Dict[str, Any]:
        """Convert CentralMetadata to dictionary"""
        return metadata.model_dump()

    def _dict_to_metadata(self, data: Dict[str, Any]) -> CentralMetadata:
        """Convert dictionary to CentralMetadata"""
        # Handle nested dataclasses
        api_data = data.get('api', {})
        api = APIMetadata(**api_data) if api_data else APIMetadata()

        gen_data = data.get('generation', {})
        generation = GenerationMetadata(**gen_data) if gen_data else GenerationMetadata()

        output_data = data.get('output', {})
        output = OutputMetadata(**output_data) if output_data else OutputMetadata()

        features_data = data.get('features', {})

        # Handle AI enhancement metadata
        ai_data = features_data.get('ai_enhancement', {})
        ai_enhancement = AIEnhancementMetadata(**ai_data) if ai_data else AIEnhancementMetadata()

        # Handle codebase analysis metadata
        codebase_data = features_data.get('codebase_analysis', {})
        codebase_analysis = CodebaseAnalysisMetadata(**codebase_data) if codebase_data else CodebaseAnalysisMetadata()

        features = FeaturesMetadata(
            ai_enhancement=ai_enhancement,
            codebase_analysis=codebase_analysis
        )

        return CentralMetadata(
            version=data.get('version', METADATA_VERSION),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            api=api,
            generation=generation,
            output=output,
            features=features
        )


class MetadataManager:
    """
    Central manager for .devdox_ai_locust/ metadata.

    This is the main interface for the generate command and other features
    to interact with the metadata system.

    Usage:
        manager = MetadataManager(output_dir)
        manager.initialize_session(api_info, swagger_source)

        # During generation
        manager.update_generation_info(host=host, auth=auth)

        # After generation
        manager.register_files(main_files, workflow_files)
        manager.finalize_session(processing_time, ai_model, enhancements)
    """

    def __init__(
        self,
        output_dir: Path,
        storage: Optional[MetadataStorageProtocol] = None
    ):
        self.output_dir = Path(output_dir)
        self.storage = storage or FileSystemMetadataStorage(self.output_dir)
        self._metadata: Optional[CentralMetadata] = None
        self._session_start_time: Optional[datetime] = None
        self._current_session_id: Optional[str] = None

    @property
    def metadata(self) -> CentralMetadata:
        """Get current metadata, loading from storage if needed"""
        if self._metadata is None:
            self._metadata = self.storage.load() or CentralMetadata()
        return self._metadata

    @property
    def devdox_dir(self) -> Path:
        """Get the .devdox_ai_locust directory path"""
        return self.storage.get_devdox_dir()

    @property
    def session_id(self) -> str:
        """Get the current session ID"""
        return self._current_session_id or self.generate_session_id()

    # =========================================================================
    # Directory Access Methods
    # =========================================================================

    def get_subdir(self, subdir: SubDirectory, session_id: Optional[str] = None) -> Path:
        """
        Get a subdirectory path within .devdox_ai_locust/

        Args:
            subdir: The subdirectory type
            session_id: Optional session ID for session-specific subdirs

        Returns:
            Path to the subdirectory (created if needed)
        """
        path = self.devdox_dir / subdir.value
        if session_id:
            path = path / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_patches_dir(self, session_id: Optional[str] = None) -> Path:
        """Get the AI patches directory for a session"""
        return self.get_subdir(SubDirectory.AI_PATCHES, session_id or self.session_id)

    def get_constraints_dir(self, session_id: Optional[str] = None) -> Path:
        """Get the AI constraints directory for a session"""
        return self.get_subdir(SubDirectory.AI_CONSTRAINTS, session_id or self.session_id)

    def get_codebase_analysis_dir(self, session_id: Optional[str] = None) -> Path:
        """Get the codebase analysis directory for a session"""
        return self.get_subdir(SubDirectory.CODEBASE_ANALYSIS, session_id or self.session_id)

    def get_generation_session_dir(self, session_id: Optional[str] = None) -> Path:
        """Get the generation session directory"""
        return self.get_subdir(SubDirectory.GENERATION_SESSIONS, session_id or self.session_id)

    def get_logs_dir(self) -> Path:
        """Get the logs directory"""
        return self.get_subdir(SubDirectory.LOGS)

    # Backwards compatibility
    @property
    def patches_dir(self) -> Path:
        """Get the patches subdirectory path (backwards compatible)"""
        return self.get_patches_dir()

    # =========================================================================
    # Session Management
    # =========================================================================

    def generate_session_id(self) -> str:
        """Generate a timestamped session ID"""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def initialize_session(
        self,
        api_info: Optional[Dict[str, Any]] = None,
        swagger_source: str = "",
        source_type: str = ""
    ) -> str:
        """
        Initialize a new generation session.

        Args:
            api_info: API information from OpenAPI parser
            swagger_source: URL or file path of the swagger source
            source_type: "url" or "file"

        Returns:
            Session ID
        """
        self._current_session_id = self.generate_session_id()
        self._session_start_time = datetime.now(timezone.utc)

        # Load existing metadata or create new
        self._metadata = self.storage.load() or CentralMetadata()

        # Set creation timestamp if new
        if not self._metadata.created_at:
            self._metadata.created_at = self._session_start_time.isoformat()

        # Update API metadata
        if api_info:
            self._metadata.api = APIMetadata(
                title=api_info.get('title', 'Unknown'),
                version=api_info.get('version', 'Unknown'),
                base_url=api_info.get('base_url', ''),
                description=api_info.get('description', ''),
                endpoints_count=api_info.get('endpoints_count', 0),
                swagger_source=swagger_source,
                source_type=source_type
            )

        # Initialize generation metadata
        self._metadata.generation = GenerationMetadata(
            session_id=self._current_session_id,
            created_at=self._session_start_time.isoformat()
        )

        # Set output directory
        self._metadata.output.directory = str(self.output_dir.absolute())

        logger.info(f"Initialized metadata session: {self._current_session_id}")
        return self._current_session_id

    def update_generation_config(
        self,
        host: Optional[str] = None,
        auth_enabled: Optional[bool] = None,
        db_type: Optional[str] = None,
        custom_requirement: Optional[str] = None,
        ai_model: Optional[str] = None
    ) -> None:
        """Update generation configuration in metadata"""
        gen = self.metadata.generation

        if host is not None:
            gen.host = host
        if auth_enabled is not None:
            gen.auth_enabled = auth_enabled
        if db_type is not None:
            gen.db_type = db_type
        if custom_requirement is not None:
            gen.custom_requirement = custom_requirement
        if ai_model is not None:
            gen.ai_model = ai_model

    def update_api_endpoints_count(self, count: int) -> None:
        """Update the endpoints count"""
        self.metadata.api.endpoints_count = count

    def register_files(
        self,
        main_files: Optional[List[str]] = None,
        workflow_files: Optional[List[str]] = None
    ) -> None:
        """Register generated files in metadata"""
        output = self.metadata.output

        if main_files:
            output.main_files = main_files
        if workflow_files:
            output.workflow_files = workflow_files

        output.total_files = len(output.main_files) + len(output.workflow_files)

    # =========================================================================
    # AI Enhancement Tracking
    # =========================================================================

    def save_patch_summary(
        self,
        pre_llm_files: int,
        post_llm_files: int,
        files_changed: int,
        files_added: int = 0,
        files_removed: int = 0,
        patches: Optional[Dict[str, str]] = None
    ) -> Path:
        """
        Save patch summary to the patches directory.

        Args:
            pre_llm_files: Number of files before LLM
            post_llm_files: Number of files after LLM
            files_changed: Number of files changed
            files_added: Number of files added
            files_removed: Number of files removed
            patches: Dict of filename -> unified diff

        Returns:
            Path to the summary file
        """
        patches_dir = self.get_patches_dir()

        summary = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pre_llm_files_count": pre_llm_files,
            "post_llm_files_count": post_llm_files,
            "files_changed": files_changed,
            "files_added": files_added,
            "files_removed": files_removed,
        }

        # Save summary
        summary_path = patches_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Save patches if provided
        if patches:
            patches_file = patches_dir / "patches.json"
            with open(patches_file, 'w', encoding='utf-8') as f:
                json.dump(patches, f, indent=2)

        # Update metadata
        ai_meta = self.metadata.features.ai_enhancement
        ai_meta.enabled = True
        ai_meta.total_sessions += 1
        ai_meta.latest_session = self.session_id
        if self.session_id not in ai_meta.sessions:
            ai_meta.sessions.append(self.session_id)

        logger.info(f"Saved patch summary to: {patches_dir}")
        return summary_path

    def save_constraints(
        self,
        filename: str,
        constraints: str
    ) -> Path:
        """
        Save AI constraints used for a file.

        Args:
            filename: The file the constraints were for (e.g., "test_data.py")
            constraints: The constraint text sent to AI

        Returns:
            Path to the saved constraints file
        """
        constraints_dir = self.get_constraints_dir()

        # Clean filename for saving
        clean_name = filename.replace('.py', '').replace('/', '_')
        constraints_path = constraints_dir / f"{clean_name}.constraints.txt"

        with open(constraints_path, 'w', encoding='utf-8') as f:
            f.write(f"# Constraints for {filename}\n")
            f.write(f"# Session: {self.session_id}\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("\n")
            f.write(constraints)

        logger.debug(f"Saved constraints for {filename} to: {constraints_path}")
        return constraints_path

    # =========================================================================
    # Codebase Analysis Storage
    # =========================================================================

    def save_codebase_analysis(
        self,
        exports: Dict[str, List[str]],
        imports: Dict[str, Dict[str, List[str]]],
        protected_symbols: Dict[str, List[Dict[str, Any]]]
    ) -> Path:
        """
        Save codebase analysis results.

        Args:
            exports: Dict of filename -> list of exported symbols
            imports: Dict of filename -> {source_file: imported_symbols}
            protected_symbols: Dict of filename -> list of protected symbol info

        Returns:
            Path to the analysis directory
        """
        analysis_dir = self.get_codebase_analysis_dir()

        # Save exports
        exports_path = analysis_dir / "exports.json"
        with open(exports_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": {k: list(v) if isinstance(v, set) else v for k, v in exports.items()}
            }, f, indent=2)

        # Save dependencies (imports)
        deps_path = analysis_dir / "dependencies.json"
        with open(deps_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "imports": {
                    k: {sk: list(sv) if isinstance(sv, set) else sv for sk, sv in v.items()}
                    for k, v in imports.items()
                }
            }, f, indent=2)

        # Save protected symbols
        protected_path = analysis_dir / "protected_symbols.json"
        total_protected = sum(len(v) for v in protected_symbols.values())
        with open(protected_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_protected_symbols": total_protected,
                "by_file": protected_symbols
            }, f, indent=2)

        # Update metadata
        codebase_meta = self.metadata.features.codebase_analysis
        codebase_meta.enabled = True
        codebase_meta.latest_session = self.session_id
        codebase_meta.total_protected_symbols = total_protected

        logger.info(f"Saved codebase analysis to: {analysis_dir}")
        return analysis_dir

    # =========================================================================
    # Backwards Compatibility
    # =========================================================================

    def register_patch_session(self, patch_session_id: str) -> None:
        """Register a patch tracking session (backwards compatible)"""
        ai_meta = self.metadata.features.ai_enhancement
        ai_meta.enabled = True

        if patch_session_id not in ai_meta.sessions:
            ai_meta.sessions.append(patch_session_id)

        ai_meta.latest_session = patch_session_id
        ai_meta.total_sessions = len(ai_meta.sessions)

    # =========================================================================
    # Enhancement and Error Tracking
    # =========================================================================

    def add_enhancement(self, enhancement: str) -> None:
        """Add an enhancement to the list"""
        if enhancement not in self.metadata.generation.enhancements_applied:
            self.metadata.generation.enhancements_applied.append(enhancement)

    def add_error(self, error: str) -> None:
        """Add an error to the list"""
        self.metadata.generation.errors.append(error)

    # =========================================================================
    # Session Finalization
    # =========================================================================

    def finalize_session(
        self,
        enhancements_applied: Optional[List[str]] = None,
        errors: Optional[List[str]] = None
    ) -> CentralMetadata:
        """
        Finalize the session and save metadata.

        Args:
            enhancements_applied: List of enhancements that were applied
            errors: List of errors that occurred

        Returns:
            The finalized CentralMetadata
        """
        # Calculate processing time
        if self._session_start_time:
            processing_time = (
                datetime.now(timezone.utc) - self._session_start_time
            ).total_seconds()
            self.metadata.generation.processing_time_seconds = processing_time

        # Update enhancements and errors
        if enhancements_applied:
            self.metadata.generation.enhancements_applied = enhancements_applied
        if errors:
            self.metadata.generation.errors = errors

        # Save to storage
        self.storage.save(self.metadata)

        logger.info(f"Finalized metadata session: {self.metadata.generation.session_id}")
        return self.metadata

    def get_file_paths(self) -> Dict[str, List[Path]]:
        """Get full paths to all registered files"""
        output_dir = Path(self.metadata.output.directory)

        return {
            'main_files': [
                output_dir / f for f in self.metadata.output.main_files
            ],
            'workflow_files': [
                output_dir / f for f in self.metadata.output.workflow_files
            ]
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export metadata as dictionary"""
        return self.metadata.model_dump()

    # =========================================================================
    # Directory Structure Info
    # =========================================================================

    def get_structure_info(self) -> str:
        """Get a formatted string showing the directory structure"""
        return f"""
.devdox_ai_locust/
├── metadata.json                         # Central metadata
├── generation/
│   └── sessions/{self.session_id}/       # Current session
├── ai_enhancement/
│   ├── patches/{self.session_id}/        # Code patches (pre/post LLM)
│   └── constraints/{self.session_id}/    # AI sandbox constraints
├── codebase_analysis/{self.session_id}/  # Dependency analysis
└── logs/                                 # Generation logs
"""
