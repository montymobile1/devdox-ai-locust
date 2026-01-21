"""
Tests for the PatchGenerator diagnostic module.
"""

import pytest
from pathlib import Path

from devdox_ai_locust.utils.patch_generator import PatchGenerator


class TestPatchGeneratorInit:
    """Tests for PatchGenerator initialization."""

    def test_init_with_path(self, temp_dir):
        """Test initialization with a Path object."""
        pg = PatchGenerator(temp_dir)
        assert pg.output_dir == temp_dir
        assert pg.session_dir is None
        assert pg.prompts == []

    def test_init_with_string(self, temp_dir):
        """Test initialization with a string path."""
        pg = PatchGenerator(str(temp_dir))
        assert pg.output_dir == temp_dir


class TestPatchGeneratorSession:
    """Tests for session management."""

    def test_start_session_creates_directory(self, temp_dir):
        """Test that start_session creates the expected directory structure."""
        pg = PatchGenerator(temp_dir)
        session_dir = pg.start_session()

        assert session_dir.exists()
        assert ".devdox-ai-locust" in str(session_dir)
        assert "generate" in str(session_dir)
        assert "patches" in str(session_dir)

    def test_start_session_uses_timestamp(self, temp_dir):
        """Test that session directory includes a timestamp."""
        pg = PatchGenerator(temp_dir)

        session_dir = pg.start_session()

        # Verify the timestamp format is in the path (YYYYMMDD_HHMMSS)
        import re
        timestamp_pattern = r"\d{8}_\d{6}"
        assert re.search(timestamp_pattern, str(session_dir)) is not None

    def test_start_session_clears_prompts(self, temp_dir):
        """Test that start_session clears previous prompts."""
        pg = PatchGenerator(temp_dir)
        pg.prompts = [{"file": "test.py", "prompt": "test"}]

        pg.start_session()

        assert pg.prompts == []


class TestPatchGeneratorPromptLogging:
    """Tests for prompt logging functionality."""

    def test_log_prompt_stores_prompt(self, temp_dir):
        """Test that log_prompt stores prompts correctly."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pg.log_prompt("locustfile.py", "Generate a test")

        assert len(pg.prompts) == 1
        assert pg.prompts[0]["file"] == "locustfile.py"
        assert pg.prompts[0]["prompt"] == "Generate a test"
        assert "timestamp" in pg.prompts[0]

    def test_log_multiple_prompts(self, temp_dir):
        """Test logging multiple prompts."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pg.log_prompt("file1.py", "Prompt 1")
        pg.log_prompt("file2.py", "Prompt 2")

        assert len(pg.prompts) == 2

    def test_save_prompts_log_creates_file(self, temp_dir):
        """Test that save_prompts_log creates a prompts.log file."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()
        pg.log_prompt("test.py", "Test prompt content")

        log_path = pg.save_prompts_log()

        assert log_path is not None
        assert log_path.exists()
        assert log_path.name == "prompts.log"

        content = log_path.read_text()
        assert "test.py" in content
        assert "Test prompt content" in content

    def test_save_prompts_log_no_session(self, temp_dir):
        """Test save_prompts_log returns None when no session started."""
        pg = PatchGenerator(temp_dir)

        result = pg.save_prompts_log()

        assert result is None

    def test_save_prompts_log_no_prompts(self, temp_dir):
        """Test save_prompts_log returns None when no prompts logged."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        result = pg.save_prompts_log()

        assert result is None


class TestPatchGeneratorPreLLM:
    """Tests for pre-LLM patch generation."""

    def test_save_pre_llm_patch_creates_file(self, temp_dir):
        """Test that save_pre_llm_patch creates the patch file."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        base_files = {"locustfile.py": "# Main locust file\nfrom locust import HttpUser"}
        directory_files = []

        patch_path = pg.save_pre_llm_patch(base_files, directory_files)

        assert patch_path is not None
        assert patch_path.exists()
        assert patch_path.name == "pre_llm.patch"

    def test_save_pre_llm_patch_content(self, temp_dir):
        """Test the content of pre-LLM patch."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        base_files = {"locustfile.py": "from locust import HttpUser\n"}
        directory_files = []

        patch_path = pg.save_pre_llm_patch(base_files, directory_files)
        content = patch_path.read_text()

        assert "PRE_LLM" in content
        assert "locustfile.py" in content
        assert "from locust import HttpUser" in content

    def test_save_pre_llm_patch_with_workflows(self, temp_dir):
        """Test pre-LLM patch with workflow files."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        base_files = {"locustfile.py": "# main"}
        directory_files = [
            {"users_workflow.py": "# users workflow content"}
        ]

        patch_path = pg.save_pre_llm_patch(base_files, directory_files)
        content = patch_path.read_text()

        assert "workflows/users_workflow.py" in content

    def test_save_pre_llm_patch_no_session(self, temp_dir):
        """Test that save_pre_llm_patch returns None when no session."""
        pg = PatchGenerator(temp_dir)

        result = pg.save_pre_llm_patch({}, [])

        assert result is None


class TestPatchGeneratorPostLLM:
    """Tests for post-LLM patch generation."""

    def test_save_post_llm_patch_creates_file(self, temp_dir):
        """Test that save_post_llm_patch creates the patch file."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pre_files = {"locustfile.py": "# original"}
        post_files = {"locustfile.py": "# enhanced"}

        patch_path = pg.save_post_llm_patch(
            pre_files, post_files, [], []
        )

        assert patch_path is not None
        assert patch_path.exists()
        assert patch_path.name == "post_llm.patch"

    def test_save_post_llm_patch_shows_diff(self, temp_dir):
        """Test that post-LLM patch shows the diff between pre and post."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pre_files = {"locustfile.py": "# original\n"}
        post_files = {"locustfile.py": "# enhanced by LLM\n"}

        patch_path = pg.save_post_llm_patch(
            pre_files, post_files, [], []
        )
        content = patch_path.read_text()

        assert "POST_LLM" in content
        assert "-# original" in content or "- # original" in content
        assert "+# enhanced" in content or "+ # enhanced" in content

    def test_save_post_llm_patch_unchanged_files(self, temp_dir):
        """Test that unchanged files are not included in diff."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pre_files = {"unchanged.py": "same content"}
        post_files = {"unchanged.py": "same content"}

        patch_path = pg.save_post_llm_patch(
            pre_files, post_files, [], []
        )
        content = patch_path.read_text()

        # Unchanged file should not appear in diff body
        # (only header should be present)
        assert "unchanged.py" not in content or content.count("unchanged.py") == 0

    def test_save_post_llm_patch_new_files(self, temp_dir):
        """Test handling of new files added by LLM."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pre_files = {}
        post_files = {"new_file.py": "# new content"}

        patch_path = pg.save_post_llm_patch(
            pre_files, post_files, [], []
        )
        content = patch_path.read_text()

        assert "new_file.py" in content

    def test_save_post_llm_patch_with_workflows(self, temp_dir):
        """Test post-LLM patch with workflow file changes."""
        pg = PatchGenerator(temp_dir)
        pg.start_session()

        pre_workflows = [{"users_workflow.py": "# original"}]
        post_workflows = [{"users_workflow.py": "# enhanced"}]

        patch_path = pg.save_post_llm_patch(
            {}, {}, pre_workflows, post_workflows
        )
        content = patch_path.read_text()

        assert "workflows/users_workflow.py" in content

    def test_save_post_llm_patch_no_session(self, temp_dir):
        """Test that save_post_llm_patch returns None when no session."""
        pg = PatchGenerator(temp_dir)

        result = pg.save_post_llm_patch({}, {}, [], [])

        assert result is None


class TestPatchGeneratorHelpers:
    """Tests for helper methods."""

    def test_flatten_directory_files(self, temp_dir):
        """Test _flatten_directory_files helper."""
        pg = PatchGenerator(temp_dir)

        directory_files = [
            {"file1.py": "content1"},
            {"file2.py": "content2"},
        ]

        result = pg._flatten_directory_files(directory_files)

        assert result == {"file1.py": "content1", "file2.py": "content2"}

    def test_flatten_directory_files_skips_non_strings(self, temp_dir):
        """Test that non-string values are skipped."""
        pg = PatchGenerator(temp_dir)

        directory_files = [
            {"file.py": "content"},
            {"metadata": {"key": "value"}},  # Should be skipped
        ]

        result = pg._flatten_directory_files(directory_files)

        assert result == {"file.py": "content"}

    def test_create_file_patch_new_file(self, temp_dir):
        """Test _create_file_patch for a new file."""
        pg = PatchGenerator(temp_dir)

        patch = pg._create_file_patch("test.py", "", "new content\n", "test")

        assert "a/test.py" in patch
        assert "b/test.py" in patch
        assert "+new content" in patch

    def test_create_file_patch_modified_file(self, temp_dir):
        """Test _create_file_patch for a modified file."""
        pg = PatchGenerator(temp_dir)

        patch = pg._create_file_patch(
            "test.py",
            "old content\n",
            "new content\n",
            "test"
        )

        assert "-old content" in patch
        assert "+new content" in patch


class TestPatchGeneratorIntegration:
    """Integration tests for the full workflow."""

    def test_full_diagnostic_workflow(self, temp_dir):
        """Test the complete diagnostic workflow."""
        pg = PatchGenerator(temp_dir)

        # Start session
        session_dir = pg.start_session()
        assert session_dir.exists()

        # Pre-LLM: Template generated files
        base_files = {
            "locustfile.py": "from locust import HttpUser\n\nclass APIUser(HttpUser):\n    pass\n"
        }
        directory_files = [
            {"users_workflow.py": "# Users workflow\nclass UsersWorkflow:\n    pass\n"}
        ]

        pre_patch = pg.save_pre_llm_patch(base_files, directory_files)
        assert pre_patch.exists()

        # Log prompts
        pg.log_prompt("locustfile.py", "Enhance this locustfile with better tasks")
        pg.log_prompt("users_workflow.py", "Add realistic user scenarios")

        # Post-LLM: AI enhanced files
        enhanced_files = {
            "locustfile.py": "from locust import HttpUser, task, between\nimport random\n\nclass APIUser(HttpUser):\n    wait_time = between(1, 3)\n    \n    @task\n    def test_users(self):\n        self.client.get('/users')\n"
        }
        enhanced_directory = [
            {"users_workflow.py": "# Enhanced Users workflow\nclass UsersWorkflow:\n    def create_user(self):\n        pass\n"}
        ]

        post_patch = pg.save_post_llm_patch(
            base_files, enhanced_files,
            directory_files, enhanced_directory
        )
        assert post_patch.exists()

        # Save prompts log
        prompts_log = pg.save_prompts_log()
        assert prompts_log.exists()

        # Verify all files exist in the session directory
        assert (session_dir / "pre_llm.patch").exists()
        assert (session_dir / "post_llm.patch").exists()
        assert (session_dir / "prompts.log").exists()

        # Verify content
        pre_content = pre_patch.read_text()
        post_content = post_patch.read_text()
        prompts_content = prompts_log.read_text()

        assert "locustfile.py" in pre_content
        assert "import random" in post_content  # LLM addition
        assert "Enhance this locustfile" in prompts_content
