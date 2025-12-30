"""
Central Metadata Manager for DevDox AI Locust

Manages the .devdox_ai_locust/ directory and central metadata.json file.
This is the single source of truth for all generated test information and
is extensible for features like patch tracking, test history, and more.

Structure:
    .devdox_ai_locust/
    ├── metadata.json           # Central metadata file
    └── patches/                # Patch tracking subdirectory
        └── 2025-12-28_21-47-06/
            ├── pre_llm.patch
            ├── post_llm.patch
            └── session.json
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

METADATA_VERSION = "1.0"
METADATA_FILENAME = "metadata.json"
DEVDOX_DIR_NAME = ".devdox_ai_locust"


@dataclass
class APIMetadata:
    """Metadata about the API being tested"""
    title: str = "Unknown"
    version: str = "Unknown"
    base_url: str = ""
    description: str = ""
    endpoints_count: int = 0
    swagger_source: str = ""
    source_type: str = ""  # "url" or "file"


@dataclass
class GenerationMetadata:
    """Metadata about the generation session"""
    session_id: str = ""
    created_at: str = ""
    host: str = ""
    auth_enabled: bool = True
    db_type: str = ""
    custom_requirement: str = ""
    processing_time_seconds: float = 0.0
    ai_model: str = ""
    enhancements_applied: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class OutputMetadata:
    """Metadata about generated output files"""
    directory: str = ""
    main_files: List[str] = field(default_factory=list)
    workflow_files: List[str] = field(default_factory=list)
    total_files: int = 0


@dataclass
class PatchFeatureMetadata:
    """Metadata for the patch tracking feature"""
    enabled: bool = True
    sessions: List[str] = field(default_factory=list)
    latest_session: str = ""


@dataclass
class FeaturesMetadata:
    """Container for all extensible features"""
    patches: PatchFeatureMetadata = field(default_factory=PatchFeatureMetadata)
    # Future features can be added here:
    # test_history: TestHistoryMetadata = field(default_factory=TestHistoryMetadata)
    # analytics: AnalyticsMetadata = field(default_factory=AnalyticsMetadata)


@dataclass
class CentralMetadata:
    """
    Central metadata structure for DevDox AI Locust.

    This is the main data structure stored in .devdox_ai_locust/metadata.json.
    It's designed to be extensible - new features can add their own sections
    under the `features` field.
    """
    version: str = METADATA_VERSION
    created_at: str = ""
    updated_at: str = ""
    api: APIMetadata = field(default_factory=APIMetadata)
    generation: GenerationMetadata = field(default_factory=GenerationMetadata)
    output: OutputMetadata = field(default_factory=OutputMetadata)
    features: FeaturesMetadata = field(default_factory=FeaturesMetadata)


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
        return asdict(metadata)

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
        patches_data = features_data.get('patches', {})
        patches = PatchFeatureMetadata(**patches_data) if patches_data else PatchFeatureMetadata()
        features = FeaturesMetadata(patches=patches)

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
    def patches_dir(self) -> Path:
        """Get the patches subdirectory path"""
        patches_path = self.devdox_dir / "patches"
        patches_path.mkdir(parents=True, exist_ok=True)
        return patches_path

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
        session_id = self.generate_session_id()
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
            session_id=session_id,
            created_at=self._session_start_time.isoformat()
        )

        # Set output directory
        self._metadata.output.directory = str(self.output_dir.absolute())

        logger.info(f"Initialized metadata session: {session_id}")
        return session_id

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

    def register_patch_session(self, patch_session_id: str) -> None:
        """Register a patch tracking session"""
        patches = self.metadata.features.patches
        patches.enabled = True

        if patch_session_id not in patches.sessions:
            patches.sessions.append(patch_session_id)

        patches.latest_session = patch_session_id

    def add_enhancement(self, enhancement: str) -> None:
        """Add an enhancement to the list"""
        if enhancement not in self.metadata.generation.enhancements_applied:
            self.metadata.generation.enhancements_applied.append(enhancement)

    def add_error(self, error: str) -> None:
        """Add an error to the list"""
        self.metadata.generation.errors.append(error)

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
        return asdict(self.metadata)
