"""
Diagnostic patch generator for comparing pre-LLM and post-LLM outputs.

This module provides observability into the code generation process by capturing
file states at two stages:
1. Pre-LLM: After template generation, before AI enhancement
2. Post-LLM: After AI enhancement

This helps diagnose whether bugs originate from templates or LLM enhancements.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import difflib

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    Generates git-style patch files for diagnostic purposes.

    Captures file states at two stages:
    1. Pre-LLM: After template generation, before AI enhancement
    2. Post-LLM: After AI enhancement

    Usage:
        patch_gen = PatchGenerator(output_dir)
        patch_gen.start_session()
        patch_gen.save_pre_llm_patch(base_files, directory_files)
        # ... AI enhancement happens ...
        patch_gen.save_post_llm_patch(base_files, enhanced_files, ...)
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the patch generator.

        Args:
            output_dir: The output directory where generated tests are saved.
                       The .devdox-ai-locust directory will be created inside this.
        """
        self.output_dir = Path(output_dir)
        self.session_dir: Optional[Path] = None
        self.prompts: List[Dict[str, str]] = []

    def start_session(self) -> Path:
        """
        Create a new diagnostic session directory with timestamp.

        Returns:
            Path to the session's patches directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = (
            self.output_dir / ".devdox-ai-locust" / timestamp / "generate" / "patches"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = []
        logger.info(f"Diagnostics session started: {self.session_dir}")
        return self.session_dir

    def log_prompt(self, file_name: str, prompt: str) -> None:
        """
        Log a prompt sent to the LLM.

        Args:
            file_name: Name of the file being enhanced.
            prompt: The full prompt sent to the LLM.
        """
        self.prompts.append(
            {
                "file": file_name,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def save_prompts_log(self) -> Optional[Path]:
        """
        Save all logged prompts to prompts.log file.

        Returns:
            Path to the prompts.log file, or None if no session started.
        """
        if not self.session_dir:
            logger.warning("No session started, cannot save prompts log")
            return None

        if not self.prompts:
            logger.info("No prompts logged, skipping prompts.log")
            return None

        prompts_path = self.session_dir / "prompts.log"

        content_lines = []
        for i, entry in enumerate(self.prompts, start=1):
            content_lines.append(f"{'=' * 80}")
            content_lines.append(f"PROMPT #{i}")
            content_lines.append(f"File: {entry['file']}")
            content_lines.append(f"Timestamp: {entry['timestamp']}")
            content_lines.append(f"{'=' * 80}")
            content_lines.append(entry["prompt"])
            content_lines.append("")
            content_lines.append("")

        prompts_path.write_text("\n".join(content_lines), encoding="utf-8")
        logger.info(f"Saved {len(self.prompts)} prompts to {prompts_path}")
        return prompts_path

    def save_pre_llm_patch(
        self, base_files: Dict[str, str], directory_files: List[Dict[str, Any]]
    ) -> Optional[Path]:
        """
        Save patch representing template-generated files (pre-LLM).

        This captures the state of files after template generation but before
        any AI enhancement is applied.

        Args:
            base_files: Dictionary mapping filename to content for main files.
            directory_files: List of dictionaries containing workflow files.

        Returns:
            Path to the pre_llm.patch file, or None if no session started.
        """
        if not self.session_dir:
            logger.warning("No session started, cannot save pre-LLM patch")
            return None

        patch_content = self._generate_creation_patch(
            base_files, directory_files, label="pre_llm"
        )

        patch_path = self.session_dir / "pre_llm.patch"
        patch_path.write_text(patch_content, encoding="utf-8")
        logger.info(f"Saved pre-LLM patch to {patch_path}")
        return patch_path

    def save_post_llm_patch(
        self,
        pre_files: Dict[str, str],
        post_files: Dict[str, str],
        pre_directory_files: List[Dict[str, Any]],
        post_directory_files: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """
        Save patch showing changes made by LLM (post-LLM vs pre-LLM).

        This captures the diff between template-generated files and AI-enhanced
        files, making it easy to see exactly what the LLM changed.

        Args:
            pre_files: Dictionary of files before LLM enhancement.
            post_files: Dictionary of files after LLM enhancement.
            pre_directory_files: Workflow files before enhancement.
            post_directory_files: Workflow files after enhancement.

        Returns:
            Path to the post_llm.patch file, or None if no session started.
        """
        if not self.session_dir:
            logger.warning("No session started, cannot save post-LLM patch")
            return None

        patch_content = self._generate_diff_patch(
            pre_files,
            post_files,
            pre_directory_files,
            post_directory_files,
            label="post_llm",
        )

        patch_path = self.session_dir / "post_llm.patch"
        patch_path.write_text(patch_content, encoding="utf-8")
        logger.info(f"Saved post-LLM patch to {patch_path}")
        return patch_path

    def _generate_creation_patch(
        self,
        base_files: Dict[str, str],
        directory_files: List[Dict[str, Any]],
        label: str,
    ) -> str:
        """
        Generate a patch showing file creation (from nothing to content).

        Args:
            base_files: Dictionary mapping filename to content.
            directory_files: List of workflow file dictionaries.
            label: Label for the patch (e.g., "pre_llm").

        Returns:
            Unified diff patch content as string.
        """
        patches = []
        patches.append(f"# {label.upper()} Patch - Template Generated Files")
        patches.append(f"# Generated at: {datetime.now().isoformat()}")
        patches.append("")

        # Process main files
        for filename, content in sorted(base_files.items()):
            patch = self._create_file_patch(filename, "", content, label)
            patches.append(patch)

        # Process directory files (workflows)
        for dir_file in directory_files:
            for filename, content in dir_file.items():
                if isinstance(content, str):
                    workflow_path = f"workflows/{filename}"
                    patch = self._create_file_patch(workflow_path, "", content, label)
                    patches.append(patch)

        return "\n".join(patches)

    def _generate_diff_patch(
        self,
        pre_files: Dict[str, str],
        post_files: Dict[str, str],
        pre_directory_files: List[Dict[str, Any]],
        post_directory_files: List[Dict[str, Any]],
        label: str,
    ) -> str:
        """
        Generate a patch showing differences between pre and post LLM.

        Args:
            pre_files: Files before enhancement.
            post_files: Files after enhancement.
            pre_directory_files: Workflow files before enhancement.
            post_directory_files: Workflow files after enhancement.
            label: Label for the patch.

        Returns:
            Unified diff patch content as string.
        """
        patches = []
        patches.append(f"# {label.upper()} Patch - LLM Enhanced Files")
        patches.append(f"# Generated at: {datetime.now().isoformat()}")
        patches.append("# Shows changes made by LLM enhancement")
        patches.append("")

        # Process main files
        all_filenames = set(pre_files.keys()) | set(post_files.keys())
        for filename in sorted(all_filenames):
            pre_content = pre_files.get(filename, "")
            post_content = post_files.get(filename, "")

            if pre_content != post_content:
                patch = self._create_file_patch(
                    filename, pre_content, post_content, label
                )
                patches.append(patch)

        # Process directory files (workflows)
        pre_workflows = self._flatten_directory_files(pre_directory_files)
        post_workflows = self._flatten_directory_files(post_directory_files)

        all_workflow_names = set(pre_workflows.keys()) | set(post_workflows.keys())
        for filename in sorted(all_workflow_names):
            pre_content = pre_workflows.get(filename, "")
            post_content = post_workflows.get(filename, "")

            if pre_content != post_content:
                workflow_path = f"workflows/{filename}"
                patch = self._create_file_patch(
                    workflow_path, pre_content, post_content, label
                )
                patches.append(patch)

        return "\n".join(patches)

    def _flatten_directory_files(
        self, directory_files: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Flatten list of directory file dicts into a single dict.

        Args:
            directory_files: List of dictionaries containing workflow files.

        Returns:
            Flattened dictionary mapping filename to content.
        """
        result = {}
        for dir_file in directory_files:
            for filename, content in dir_file.items():
                if isinstance(content, str):
                    result[filename] = content
        return result

    def _create_file_patch(
        self, filename: str, old_content: str, new_content: str, label: str
    ) -> str:
        """
        Create a unified diff patch for a single file.

        Args:
            filename: Name of the file.
            old_content: Content before (empty string for new files).
            new_content: Content after.
            label: Label for the patch.

        Returns:
            Unified diff patch for this file.
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Ensure lines end with newline for proper diff formatting
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )

        return "".join(diff)
