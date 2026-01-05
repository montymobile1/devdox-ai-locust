"""
Metadata and patch tracking tests for the `generate` command.

Tests the .devdox_ai_locust/ directory structure, metadata.json,
session tracking, and patch generation (Branch 2 features).
"""

import re
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .conftest import run_generate_command


class MetadataAnalyzer:
    """Analyzer for .devdox_ai_locust directory structure."""

    DEVDOX_DIR = ".devdox_ai_locust"
    METADATA_FILENAME = "metadata.json"
    SESSION_FILENAME = "session.json"
    PATCHES_DIR = ".patches"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.devdox_dir = output_dir / self.DEVDOX_DIR

    def exists(self) -> bool:
        """Check if .devdox_ai_locust directory exists."""
        return self.devdox_dir.exists()

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Read metadata.json if it exists."""
        metadata_path = self.devdox_dir / self.METADATA_FILENAME
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        return None

    def get_sessions(self) -> List[str]:
        """Get list of session directories."""
        if not self.devdox_dir.exists():
            return []
        return [
            d.name for d in self.devdox_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read session.json for a specific session."""
        session_path = self.devdox_dir / session_id / self.SESSION_FILENAME
        if session_path.exists():
            return json.loads(session_path.read_text())
        return None

    def get_patches(self, session_id: str) -> List[str]:
        """Get list of patch files in a session."""
        patches_dir = self.devdox_dir / session_id / self.PATCHES_DIR
        if patches_dir.exists():
            return [p.name for p in patches_dir.glob("*.patch")]
        return []

    def get_patch_content(self, session_id: str, patch_id: str) -> Optional[str]:
        """Read a patch file's content."""
        patch_path = self.devdox_dir / session_id / self.PATCHES_DIR / f"{patch_id}.patch"
        if patch_path.exists():
            return patch_path.read_text()
        return None

    def validate_metadata_structure(self) -> Tuple[bool, List[str]]:
        """Validate metadata.json structure."""
        issues = []
        metadata = self.get_metadata()

        if not metadata:
            issues.append("metadata.json not found")
            return False, issues

        # Required top-level fields
        required_fields = ["version", "session_id", "created_at", "api", "config"]
        for field in required_fields:
            if field not in metadata:
                issues.append(f"Missing field: {field}")

        # API subfields
        if "api" in metadata:
            api = metadata["api"]
            api_fields = ["title", "swagger_source"]
            for field in api_fields:
                if field not in api:
                    issues.append(f"Missing api.{field}")

        # Config subfields
        if "config" in metadata:
            config = metadata["config"]
            config_fields = ["host", "auth_enabled"]
            for field in config_fields:
                if field not in config:
                    issues.append(f"Missing config.{field}")

        return len(issues) == 0, issues

    def validate_session_structure(self, session_id: str) -> Tuple[bool, List[str]]:
        """Validate session.json structure."""
        issues = []
        session = self.get_session_info(session_id)

        if not session:
            issues.append("session.json not found")
            return False, issues

        required_fields = ["version", "session_id", "created_at", "patches"]
        for field in required_fields:
            if field not in session:
                issues.append(f"Missing field: {field}")

        # Validate patches array
        if "patches" in session:
            for i, patch in enumerate(session["patches"]):
                patch_fields = ["id", "sequence", "milestone", "created_at"]
                for field in patch_fields:
                    if field not in patch:
                        issues.append(f"Patch {i} missing field: {field}")

        return len(issues) == 0, issues


class TestMetadataDirectoryStructure:
    """Test .devdox_ai_locust directory creation."""

    def test_devdox_directory_created(self, api_key, swagger_url, output_dir):
        """Verify .devdox_ai_locust directory is created."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        assert analyzer.exists(), ".devdox_ai_locust directory not created"

    def test_metadata_json_exists(self, api_key, swagger_url, output_dir):
        """Verify metadata.json is created."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        metadata = analyzer.get_metadata()

        assert metadata is not None, "metadata.json not created"


class TestMetadataContent:
    """Test metadata.json content and structure."""

    def test_metadata_json_valid_structure(self, api_key, swagger_url, output_dir):
        """Verify metadata.json has correct structure."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        is_valid, issues = analyzer.validate_metadata_structure()

        assert is_valid, f"Invalid metadata.json structure: {issues}"

    def test_api_metadata_captured(self, api_key, swagger_url, output_dir):
        """Verify API metadata is captured correctly."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        metadata = analyzer.get_metadata()

        if metadata and "api" in metadata:
            api = metadata["api"]

            # Should capture swagger source
            assert api.get("swagger_source"), "swagger_source not captured"

            # Should have a title
            assert api.get("title"), "API title not captured"

    def test_config_captured(self, api_key, swagger_url, output_dir):
        """Verify generation config is captured."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        metadata = analyzer.get_metadata()

        if metadata and "config" in metadata:
            config = metadata["config"]
            assert "auth_enabled" in config, "auth_enabled not captured"

    def test_file_tree_tracked(self, api_key, swagger_url, output_dir):
        """Verify file tree is tracked in metadata."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        metadata = analyzer.get_metadata()

        if metadata and "files" in metadata:
            files = metadata["files"]

            # Should track at least the main files
            assert len(files) > 0, "No files tracked in metadata"

            # Files should have size and lines info
            for filename, info in files.items():
                if isinstance(info, dict):
                    assert "size" in info or "lines" in info, \
                        f"File {filename} missing size/lines info"


class TestSessionManagement:
    """Test session directory and tracking."""

    def test_session_directory_created(self, api_key, swagger_url, output_dir):
        """Verify session directory is created with proper structure."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        assert len(sessions) > 0, "No session directories created"

    def test_session_json_valid(self, api_key, swagger_url, output_dir):
        """Verify session.json has correct structure."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            is_valid, issues = analyzer.validate_session_structure(session_id)

            assert is_valid, f"Invalid session structure: {issues}"

    def test_session_id_format(self, api_key, swagger_url, output_dir):
        """Verify session ID follows datetime format."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            # Should match datetime pattern: YYYY-MM-DD_HH-MM-SS
            pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
            assert re.match(pattern, session_id), \
                f"Session ID '{session_id}' doesn't match expected format"


class TestPatchTracking:
    """Test patch file creation and tracking."""

    def test_patches_created(self, api_key, swagger_url, output_dir):
        """Verify patch files are created for generation milestones."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            patches = analyzer.get_patches(session_id)

            # Should have at least one patch (template_generation)
            assert len(patches) > 0, "No patch files created"

    def test_patch_naming_convention(self, api_key, swagger_url, output_dir):
        """Verify patch files follow naming convention."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            patches = analyzer.get_patches(session_id)

            for patch in patches:
                # Should match pattern: NNNNNN_xxxxxxxx.patch
                assert re.match(r"\d{6}_[a-f0-9]{8}\.patch", patch), \
                    f"Invalid patch filename: {patch}"

    def test_milestone_tracking(self, api_key, swagger_url, output_dir):
        """Verify milestones are properly tracked in session.json."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            session = analyzer.get_session_info(session_id)

            if session and "patches" in session:
                patches = session["patches"]

                # Verify milestone types are valid
                valid_milestones = {
                    "template_generation",
                    "llm_enhancement",
                    "validation",
                    "user_edit",
                    "refactor",
                    "merge",
                }

                milestones = [p.get("milestone") for p in patches]
                for milestone in milestones:
                    assert milestone in valid_milestones, \
                        f"Unknown milestone type: {milestone}"

    def test_patch_sequence_numbers(self, api_key, swagger_url, output_dir):
        """Verify patch sequence numbers are correct."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            session = analyzer.get_session_info(session_id)

            if session and "patches" in session:
                patches = session["patches"]

                # Verify sequences are consecutive
                sequences = [p.get("sequence") for p in patches]
                expected = list(range(1, len(sequences) + 1))
                assert sequences == expected, \
                    f"Non-consecutive sequences: {sequences}"


class TestPatchContent:
    """Test patch file content."""

    def test_patch_content_valid_diff(self, api_key, swagger_url, output_dir):
        """Verify patch files contain valid unified diff format."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            session = analyzer.get_session_info(session_id)

            if session and "patches" in session:
                for patch_info in session["patches"]:
                    patch_id = patch_info["id"]
                    content = analyzer.get_patch_content(session_id, patch_id)

                    if content and content.strip():
                        # Unified diff should contain --- and +++
                        # or be empty for no-change patches
                        if "---" in content or "+++" in content:
                            assert "---" in content and "+++" in content, \
                                f"Patch {patch_id} has malformed diff format"

    def test_patch_stats_present(self, api_key, swagger_url, output_dir):
        """Verify patch entries have stats."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = MetadataAnalyzer(output_dir)
        sessions = analyzer.get_sessions()

        if sessions:
            session_id = sessions[0]
            session = analyzer.get_session_info(session_id)

            if session and "patches" in session:
                for patch_info in session["patches"]:
                    if "stats" in patch_info:
                        stats = patch_info["stats"]
                        # Should have stats fields
                        assert "files_changed" in stats or "additions" in stats, \
                            f"Patch {patch_info['id']} missing stats"
