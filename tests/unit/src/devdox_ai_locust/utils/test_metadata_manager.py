"""
Comprehensive tests for metadata_manager.py module.

Tests cover:
- Milestone enum
- PatchStats, PatchEntry, SessionInfo models
- APIMetadata, GenerationConfig, FileNode, CentralMetadata models
- MetadataManager class
"""

import json
from datetime import datetime, timezone

from devdox_ai_locust.utils.metadata_manager import (
    Milestone,
    PatchStats,
    PatchEntry,
    SessionInfo,
    APIMetadata,
    GenerationConfig,
    FileNode,
    CentralMetadata,
    MetadataManager,
    METADATA_VERSION,
    METADATA_FILENAME,
    SESSION_FILENAME,
    PATCHES_DIR,
    DEVDOX_DIR_NAME,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_metadata_version(self):
        """Should have correct metadata version."""
        assert METADATA_VERSION == "3.0"

    def test_filename_constants(self):
        """Should have correct filename constants."""
        assert METADATA_FILENAME == "metadata.json"
        assert SESSION_FILENAME == "session.json"
        assert PATCHES_DIR == ".patches"
        assert DEVDOX_DIR_NAME == ".devdox_ai_locust"


# =============================================================================
# Milestone Enum Tests
# =============================================================================


class TestMilestone:
    """Tests for Milestone enum."""

    def test_all_milestones_defined(self):
        """Should define all expected milestones."""
        expected = [
            "TEMPLATE_GENERATION",
            "LLM_ENHANCEMENT",
            "VALIDATION",
            "USER_EDIT",
            "REFACTOR",
            "MERGE",
        ]
        for milestone in expected:
            assert hasattr(Milestone, milestone)

    def test_milestone_count(self):
        """Should have 6 milestones."""
        assert len(Milestone) == 6

    def test_milestone_values(self):
        """Should have correct string values."""
        assert Milestone.TEMPLATE_GENERATION.value == "template_generation"
        assert Milestone.LLM_ENHANCEMENT.value == "llm_enhancement"
        assert Milestone.VALIDATION.value == "validation"
        assert Milestone.USER_EDIT.value == "user_edit"
        assert Milestone.REFACTOR.value == "refactor"
        assert Milestone.MERGE.value == "merge"

    def test_is_string_enum(self):
        """Milestones should be string enums."""
        assert isinstance(Milestone.TEMPLATE_GENERATION.value, str)

    def test_can_create_from_value(self):
        """Should be able to create from string value."""
        assert Milestone("template_generation") == Milestone.TEMPLATE_GENERATION
        assert Milestone("llm_enhancement") == Milestone.LLM_ENHANCEMENT


# =============================================================================
# PatchStats Model Tests
# =============================================================================


class TestPatchStats:
    """Tests for PatchStats model."""

    def test_default_values(self):
        """Should have correct default values."""
        stats = PatchStats()
        assert stats.files_changed == 0
        assert stats.additions == 0
        assert stats.deletions == 0

    def test_custom_values(self):
        """Should accept custom values."""
        stats = PatchStats(files_changed=5, additions=100, deletions=20)
        assert stats.files_changed == 5
        assert stats.additions == 100
        assert stats.deletions == 20

    def test_model_dump(self):
        """Should serialize correctly."""
        stats = PatchStats(files_changed=3, additions=50, deletions=10)
        data = stats.model_dump()
        assert data == {"files_changed": 3, "additions": 50, "deletions": 10}


# =============================================================================
# PatchEntry Model Tests
# =============================================================================


class TestPatchEntry:
    """Tests for PatchEntry model."""

    def test_creates_with_required_fields(self):
        """Should create with required fields."""
        entry = PatchEntry(
            id="000001_abcd1234",
            sequence=1,
            milestone="template_generation",
            created_at="2025-01-01T00:00:00Z",
        )
        assert entry.id == "000001_abcd1234"
        assert entry.sequence == 1
        assert entry.milestone == "template_generation"
        assert entry.created_at == "2025-01-01T00:00:00Z"

    def test_default_values(self):
        """Should have correct default values."""
        entry = PatchEntry(
            id="000001_abcd1234",
            sequence=1,
            milestone="template_generation",
            created_at="2025-01-01T00:00:00Z",
        )
        assert entry.description == ""
        assert isinstance(entry.stats, PatchStats)
        assert entry.metadata == {}

    def test_all_fields(self):
        """Should accept all fields."""
        stats = PatchStats(files_changed=5)
        entry = PatchEntry(
            id="000002_efgh5678",
            sequence=2,
            milestone="llm_enhancement",
            description="AI enhanced",
            created_at="2025-01-01T01:00:00Z",
            stats=stats,
            metadata={"ai_model": "gpt-4"},
        )
        assert entry.description == "AI enhanced"
        assert entry.stats.files_changed == 5
        assert entry.metadata == {"ai_model": "gpt-4"}


# =============================================================================
# SessionInfo Model Tests
# =============================================================================


class TestSessionInfoBasic:
    """Basic tests for SessionInfo model."""

    def test_default_values(self):
        """Should have correct default values."""
        session = SessionInfo()
        assert session.version == METADATA_VERSION
        assert session.session_id == ""
        assert session.created_at == ""
        assert session.updated_at == ""
        assert session.patches == []

    def test_custom_values(self):
        """Should accept custom values."""
        session = SessionInfo(
            session_id="2025-01-01_10-00-00",
            created_at="2025-01-01T10:00:00Z",
        )
        assert session.session_id == "2025-01-01_10-00-00"
        assert session.created_at == "2025-01-01T10:00:00Z"


class TestSessionInfoGetNextSequence:
    """Tests for SessionInfo.get_next_sequence method."""

    def test_first_sequence(self):
        """Should return 1 for first sequence."""
        session = SessionInfo()
        assert session.get_next_sequence() == 1

    def test_next_sequence_after_patches(self):
        """Should return next sequence after existing patches."""
        session = SessionInfo()
        session.patches = [
            PatchEntry(id="000001_a", sequence=1, milestone="test", created_at=""),
            PatchEntry(id="000002_b", sequence=2, milestone="test", created_at=""),
        ]
        assert session.get_next_sequence() == 3

    def test_handles_non_sequential_patches(self):
        """Should handle non-sequential patches by using max."""
        session = SessionInfo()
        session.patches = [
            PatchEntry(id="000001_a", sequence=1, milestone="test", created_at=""),
            PatchEntry(id="000005_b", sequence=5, milestone="test", created_at=""),
        ]
        assert session.get_next_sequence() == 6


class TestSessionInfoAddPatch:
    """Tests for SessionInfo.add_patch method."""

    def test_add_patch_basic(self):
        """Should add patch with basic fields."""
        session = SessionInfo()
        entry = session.add_patch(
            milestone="template_generation",
            description="Initial generation",
        )
        assert entry.sequence == 1
        assert entry.milestone == "template_generation"
        assert entry.description == "Initial generation"
        assert len(session.patches) == 1

    def test_add_multiple_patches(self):
        """Should add multiple patches with incrementing sequence."""
        session = SessionInfo()
        entry1 = session.add_patch(milestone="template_generation")
        entry2 = session.add_patch(milestone="llm_enhancement")

        assert entry1.sequence == 1
        assert entry2.sequence == 2
        assert len(session.patches) == 2

    def test_add_patch_with_stats(self):
        """Should add patch with stats."""
        session = SessionInfo()
        stats = PatchStats(files_changed=5, additions=100, deletions=10)
        entry = session.add_patch(milestone="test", stats=stats)

        assert entry.stats.files_changed == 5
        assert entry.stats.additions == 100
        assert entry.stats.deletions == 10

    def test_add_patch_with_metadata(self):
        """Should add patch with metadata."""
        session = SessionInfo()
        entry = session.add_patch(
            milestone="llm_enhancement",
            metadata={"ai_model": "gpt-4"},
        )
        assert entry.metadata == {"ai_model": "gpt-4"}

    def test_patch_id_format(self):
        """Patch ID should be in correct format."""
        session = SessionInfo()
        entry = session.add_patch(milestone="test")

        # Format: 000001_xxxxxxxx (sequence_shortuuid)
        assert "_" in entry.id
        parts = entry.id.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 6  # Zero-padded sequence
        assert len(parts[1]) == 8  # Short UUID

    def test_patch_created_at_set(self):
        """Should set created_at timestamp."""
        session = SessionInfo()
        entry = session.add_patch(milestone="test")
        assert entry.created_at != ""
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(entry.created_at.replace("Z", "+00:00"))


# =============================================================================
# APIMetadata Model Tests
# =============================================================================


class TestAPIMetadata:
    """Tests for APIMetadata model."""

    def test_default_values(self):
        """Should have correct default values."""
        api = APIMetadata()
        assert api.title == "Unknown"
        assert api.version == "Unknown"
        assert api.base_url == ""
        assert api.description == ""
        assert api.endpoints_count == 0
        assert api.swagger_source == ""
        assert api.source_type == ""

    def test_custom_values(self):
        """Should accept custom values."""
        api = APIMetadata(
            title="Pet Store API",
            version="1.0.0",
            base_url="https://api.petstore.com",
            description="A pet store API",
            endpoints_count=25,
            swagger_source="https://api.petstore.com/swagger.json",
            source_type="url",
        )
        assert api.title == "Pet Store API"
        assert api.version == "1.0.0"
        assert api.base_url == "https://api.petstore.com"
        assert api.endpoints_count == 25


# =============================================================================
# GenerationConfig Model Tests
# =============================================================================


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_default_values(self):
        """Should have correct default values."""
        config = GenerationConfig()
        assert config.host == ""
        assert config.auth_enabled is True
        assert config.db_type == ""
        assert config.ai_model == ""
        assert config.custom_requirement == ""

    def test_custom_values(self):
        """Should accept custom values."""
        config = GenerationConfig(
            host="localhost:8000",
            auth_enabled=False,
            db_type="postgresql",
            ai_model="gpt-4",
            custom_requirement="Focus on edge cases",
        )
        assert config.host == "localhost:8000"
        assert config.auth_enabled is False
        assert config.db_type == "postgresql"
        assert config.ai_model == "gpt-4"


# =============================================================================
# FileNode Model Tests
# =============================================================================


class TestFileNode:
    """Tests for FileNode model."""

    def test_default_values(self):
        """Should have correct default values."""
        node = FileNode()
        assert node.size == 0
        assert node.lines == 0

    def test_custom_values(self):
        """Should accept custom values."""
        node = FileNode(size=1024, lines=50)
        assert node.size == 1024
        assert node.lines == 50


# =============================================================================
# CentralMetadata Model Tests
# =============================================================================


class TestCentralMetadata:
    """Tests for CentralMetadata model."""

    def test_default_values(self):
        """Should have correct default values."""
        metadata = CentralMetadata()
        assert metadata.version == METADATA_VERSION
        assert metadata.session_id == ""
        assert metadata.created_at == ""
        assert metadata.updated_at == ""
        assert isinstance(metadata.api, APIMetadata)
        assert isinstance(metadata.config, GenerationConfig)
        assert metadata.files == {}

    def test_custom_values(self):
        """Should accept custom values."""
        api = APIMetadata(title="Test API")
        metadata = CentralMetadata(
            session_id="2025-01-01_10-00-00",
            api=api,
        )
        assert metadata.session_id == "2025-01-01_10-00-00"
        assert metadata.api.title == "Test API"

    def test_files_dict(self):
        """Should handle files dictionary."""
        metadata = CentralMetadata()
        metadata.files["locustfile.py"] = FileNode(size=1000, lines=50)
        assert metadata.files["locustfile.py"].size == 1000


# =============================================================================
# MetadataManager Tests
# =============================================================================


class TestMetadataManagerInit:
    """Tests for MetadataManager initialization."""

    def test_init_sets_paths(self, tmp_path):
        """Should set correct paths on init."""
        manager = MetadataManager(tmp_path)
        assert manager.output_dir == tmp_path
        assert manager.devdox_dir == tmp_path / DEVDOX_DIR_NAME
        assert manager.metadata_path == tmp_path / DEVDOX_DIR_NAME / METADATA_FILENAME

    def test_init_defaults(self, tmp_path):
        """Should have correct default values."""
        manager = MetadataManager(tmp_path)
        assert manager.session_id == ""
        assert manager._session_dir is None
        assert manager._patches_dir is None
        assert isinstance(manager.metadata, CentralMetadata)
        assert isinstance(manager.session_info, SessionInfo)


class TestMetadataManagerProperties:
    """Tests for MetadataManager properties."""

    def test_patches_dir_with_session(self, tmp_path):
        """patches_dir should return session patches directory."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        assert manager.patches_dir == manager._patches_dir

    def test_patches_dir_fallback(self, tmp_path):
        """patches_dir should return fallback when no session."""
        manager = MetadataManager(tmp_path)
        assert manager.patches_dir == manager.devdox_dir / "patches"

    def test_manifest_alias(self, tmp_path):
        """manifest should be alias for session_info."""
        manager = MetadataManager(tmp_path)
        assert manager.manifest is manager.session_info


class TestMetadataManagerInitializeSession:
    """Tests for MetadataManager.initialize_session method."""

    def test_creates_session_id(self, tmp_path):
        """Should create a session ID."""
        manager = MetadataManager(tmp_path)
        session_id = manager.initialize_session()
        assert session_id != ""
        assert manager.session_id == session_id

    def test_session_id_format(self, tmp_path):
        """Session ID should be in datetime format."""
        manager = MetadataManager(tmp_path)
        session_id = manager.initialize_session()
        # Format: YYYY-MM-DD_HH-MM-SS
        parts = session_id.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 10  # Date
        assert len(parts[1]) == 8  # Time

    def test_creates_directories(self, tmp_path):
        """Should create directory structure."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        assert manager.devdox_dir.exists()
        assert manager._session_dir.exists()
        assert manager._patches_dir.exists()

    def test_sets_api_info(self, tmp_path):
        """Should set API info from dict."""
        manager = MetadataManager(tmp_path)
        api_info = {
            "title": "Test API",
            "version": "2.0.0",
            "base_url": "https://api.test.com",
            "description": "Test description",
        }
        manager.initialize_session(
            api_info=api_info,
            swagger_source="https://api.test.com/swagger.json",
            source_type="url",
        )
        assert manager.metadata.api.title == "Test API"
        assert manager.metadata.api.version == "2.0.0"
        assert (
            manager.metadata.api.swagger_source == "https://api.test.com/swagger.json"
        )
        assert manager.metadata.api.source_type == "url"

    def test_initializes_session_info(self, tmp_path):
        """Should initialize session info."""
        manager = MetadataManager(tmp_path)
        session_id = manager.initialize_session()
        assert manager.session_info.session_id == session_id
        assert manager.session_info.created_at != ""


class TestMetadataManagerFinalizeSession:
    """Tests for MetadataManager.finalize_session method."""

    def test_updates_timestamps(self, tmp_path):
        """Should update timestamps."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.finalize_session()
        # updated_at should be updated (may be same if very fast)
        assert manager.metadata.updated_at != ""
        assert manager.session_info.updated_at != ""

    def test_saves_files(self, tmp_path):
        """Should save metadata and session files."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()
        manager.finalize_session()

        assert manager.metadata_path.exists()
        assert manager._session_path.exists()

    def test_saved_metadata_content(self, tmp_path):
        """Saved metadata should have correct content."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session(
            api_info={"title": "Test API"},
            swagger_source="test.json",
            source_type="file",
        )
        manager.finalize_session()

        content = json.loads(manager.metadata_path.read_text())
        assert content["api"]["title"] == "Test API"
        assert content["version"] == METADATA_VERSION


class TestMetadataManagerCreatePatch:
    """Tests for MetadataManager.create_patch method."""

    def test_creates_patch_entry(self, tmp_path):
        """Should create patch entry in session."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        entry = manager.create_patch(
            milestone="template_generation",
            content="--- a/test.py\n+++ b/test.py\n",
            description="Test patch",
        )

        assert entry.milestone == "template_generation"
        assert entry.description == "Test patch"
        assert len(manager.session_info.patches) == 1

    def test_writes_patch_file(self, tmp_path):
        """Should write patch content to file."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        content = "--- a/test.py\n+++ b/test.py\n+print('hello')"
        entry = manager.create_patch(
            milestone="test",
            content=content,
        )

        patch_path = manager._patches_dir / f"{entry.id}.patch"
        assert patch_path.exists()
        assert patch_path.read_text() == content

    def test_saves_session_after_patch(self, tmp_path):
        """Should save session info after creating patch."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.create_patch(milestone="test", content="patch content")

        assert manager._session_path.exists()
        content = json.loads(manager._session_path.read_text())
        assert len(content["patches"]) == 1


class TestMetadataManagerGetPatch:
    """Tests for MetadataManager.get_patch method."""

    def test_returns_patch_content(self, tmp_path):
        """Should return patch content."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        content = "--- a/test.py\n+++ b/test.py\n"
        entry = manager.create_patch(milestone="test", content=content)

        result = manager.get_patch(entry.id)
        assert result == content

    def test_returns_none_for_missing(self, tmp_path):
        """Should return None for non-existent patch."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        result = manager.get_patch("999999_nonexistent")
        assert result is None

    def test_returns_none_when_no_session(self, tmp_path):
        """Should return None when no session active."""
        manager = MetadataManager(tmp_path)
        result = manager.get_patch("000001_test")
        assert result is None


class TestMetadataManagerGetPatches:
    """Tests for MetadataManager patch retrieval methods."""

    def test_get_patches_by_milestone(self, tmp_path):
        """Should get patches by milestone type."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.create_patch(milestone="template_generation", content="patch1")
        manager.create_patch(milestone="llm_enhancement", content="patch2")
        manager.create_patch(milestone="template_generation", content="patch3")

        template_patches = manager.get_patches_by_milestone("template_generation")
        assert len(template_patches) == 2

        llm_patches = manager.get_patches_by_milestone("llm_enhancement")
        assert len(llm_patches) == 1

    def test_get_latest_patch(self, tmp_path):
        """Should get most recent patch."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.create_patch(milestone="test1", content="patch1")
        manager.create_patch(milestone="test2", content="patch2")
        entry3 = manager.create_patch(milestone="test3", content="patch3")

        latest = manager.get_latest_patch()
        assert latest.id == entry3.id
        assert latest.sequence == 3

    def test_get_latest_patch_empty(self, tmp_path):
        """Should return None when no patches."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        assert manager.get_latest_patch() is None


class TestMetadataManagerConfigUpdates:
    """Tests for MetadataManager configuration update methods."""

    def test_update_api_endpoints_count(self, tmp_path):
        """Should update endpoints count."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.update_api_endpoints_count(25)
        assert manager.metadata.api.endpoints_count == 25

    def test_update_generation_config(self, tmp_path):
        """Should update generation config."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.update_generation_config(
            host="localhost:8000",
            auth_enabled=False,
            db_type="postgresql",
            ai_model="gpt-4",
            custom_requirement="Test requirement",
        )

        assert manager.metadata.config.host == "localhost:8000"
        assert manager.metadata.config.auth_enabled is False
        assert manager.metadata.config.db_type == "postgresql"
        assert manager.metadata.config.ai_model == "gpt-4"
        assert manager.metadata.config.custom_requirement == "Test requirement"

    def test_update_generation_config_partial(self, tmp_path):
        """Should allow partial config updates."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.update_generation_config(host="test.com")
        assert manager.metadata.config.host == "test.com"
        # Other values should remain default
        assert manager.metadata.config.auth_enabled is True


class TestMetadataManagerRegisterFile:
    """Tests for MetadataManager.register_file method."""

    def test_registers_file(self, tmp_path):
        """Should register file in metadata."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        content = "print('hello')\nprint('world')\n"
        manager.register_file("test.py", content)

        assert "test.py" in manager.metadata.files
        file_node = manager.metadata.files["test.py"]
        assert file_node.size == len(content.encode("utf-8"))
        assert file_node.lines == 2

    def test_registers_nested_path(self, tmp_path):
        """Should handle nested file paths."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        manager.register_file("workflows/base.py", "content")
        assert "workflows/base.py" in manager.metadata.files

    def test_counts_lines_correctly(self, tmp_path):
        """Should count lines correctly."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        # Content without trailing newline
        manager.register_file("a.py", "line1\nline2")
        assert manager.metadata.files["a.py"].lines == 2

        # Content with trailing newline
        manager.register_file("b.py", "line1\nline2\n")
        assert manager.metadata.files["b.py"].lines == 2


class TestMetadataManagerUtility:
    """Tests for MetadataManager utility methods."""

    def test_get_structure_info(self, tmp_path):
        """Should return formatted structure info."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session()

        info = manager.get_structure_info()
        assert ".devdox_ai_locust/" in info
        assert "metadata.json" in info
        assert "session.json" in info
        assert manager.session_id in info

    def test_to_dict(self, tmp_path):
        """Should export all data as dict."""
        manager = MetadataManager(tmp_path)
        manager.initialize_session(api_info={"title": "Test API"})
        manager.create_patch(milestone="test", content="patch")

        data = manager.to_dict()
        assert "metadata" in data
        assert "session" in data
        assert data["metadata"]["api"]["title"] == "Test API"
        assert len(data["session"]["patches"]) == 1


class TestMetadataManagerLoadSession:
    """Tests for MetadataManager._load_session method."""

    def test_loads_existing_session(self, tmp_path):
        """Should load existing session data."""
        # Create a session and save it
        manager1 = MetadataManager(tmp_path)
        session_id = manager1.initialize_session()
        manager1.create_patch(milestone="test1", content="patch1")
        manager1.finalize_session()

        # Create new manager and reinitialize same session
        manager2 = MetadataManager(tmp_path)
        # Simulate reinitializing with same session
        manager2._start_time = datetime.now(timezone.utc)
        manager2.session_id = session_id
        manager2._session_dir = manager2.devdox_dir / session_id
        manager2._patches_dir = manager2._session_dir / PATCHES_DIR
        manager2._session_path = manager2._session_dir / SESSION_FILENAME
        manager2.session_info = SessionInfo(session_id=session_id)
        manager2._load_session()

        # Should have loaded the existing patch
        assert len(manager2.session_info.patches) == 1
