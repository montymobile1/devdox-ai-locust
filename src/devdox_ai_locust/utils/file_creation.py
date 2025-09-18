import asyncio
import shutil
from pathlib import Path
from typing import List, Optional
import uuid
import os
import re
import logging

logger = logging.getLogger(__name__)


class FileCreationConfig:
    """Configuration for file creation process"""

    ALLOWED_EXTENSIONS = {
        ".py",
        ".md",
        ".txt",
        ".sh",
        ".yml",
        ".yaml",
        ".json",
        ".example",
    }
    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    EXECUTABLE_EXTENSIONS = {".sh"}


class SafeFileCreator:
    """Handles safe file creation with separated concerns"""

    def __init__(self, config: Optional[FileCreationConfig] = None):
        self.config = config or FileCreationConfig()

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent security issues"""

        # Remove directory components
        clean_name = os.path.basename(filename).lower()

        # Remove dangerous characters
        clean_name = re.sub(r'[<>:"/\\|?*]', "", clean_name)
        # Replace spaces with underscores
        clean_name = clean_name.replace("- ", "_")

        # Ensure reasonable length
        if len(clean_name) > 255:
            name_part, ext = os.path.splitext(clean_name)
            clean_name = name_part[:250] + ext

        # Prevent hidden files and ensure not empty
        safe_dotfiles = {".env.example", ".gitignore", ".env.template"}
        if not clean_name or (
            clean_name.startswith(".") and clean_name not in safe_dotfiles
        ):
            clean_name = f"generated_{uuid.uuid4().hex[:8]}.py"

        return clean_name

    def validate_file(self, filename: str, content: str) -> tuple[bool, str, str]:
        """Validate file and return (is_valid, clean_filename, processed_content)"""
        clean_filename = self._sanitize_filename(filename)
        file_extension = Path(clean_filename).suffix.lower()

        if file_extension not in self.config.ALLOWED_EXTENSIONS:
            logger.warning(f"Skipping file with disallowed extension: {filename}")
            return False, clean_filename, content

        # Handle oversized files
        if len(content.encode("utf-8")) > self.config.MAX_FILE_SIZE:
            logger.warning(f"File too large, truncating: {filename}")
            content = content[: self.config.MAX_FILE_SIZE // 2]

        return True, clean_filename, content

    async def create_temp_file(
        self, filename: str, content: str, temp_dir: Path
    ) -> dict:
        """Create a single file in temp directory"""
        temp_file_path = temp_dir / filename

        await asyncio.to_thread(temp_file_path.write_text, content, encoding="utf-8")

        # Set permissions
        file_extension = Path(filename).suffix.lower()
        permissions = (
            0o755 if file_extension in self.config.EXECUTABLE_EXTENSIONS else 0o644
        )
        temp_file_path.chmod(permissions)

        return {
            "filename": filename,
            "temp_path": temp_file_path,
            "size": len(content.encode("utf-8")),
            "type": file_extension.lstrip("."),
        }

    async def move_files_atomically(
        self, file_infos: List[dict], output_path: Path
    ) -> List[dict]:
        """Move all files from temp to final location atomically"""
        successfully_moved = []

        for file_info in file_infos:
            final_path = output_path / file_info["filename"]

            try:
                await asyncio.to_thread(
                    shutil.move, str(file_info["temp_path"]), str(final_path)
                )
                file_info["final_path"] = final_path
                file_info["path"] = final_path
                successfully_moved.append(file_info)
                logger.info(f"Created: {file_info['filename']}")

            except Exception as e:
                logger.error(f"Failed to move file {file_info['filename']}: {e}")

        return successfully_moved
