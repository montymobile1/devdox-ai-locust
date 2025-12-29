"""
File System Implementations

Production and testing implementations of the FileSystem protocol.
"""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from devdox_ai_locust.abstractions.file_system import (
    FileSystem,
    WriteResult,
)

logger = logging.getLogger(__name__)


class LocalFileSystem:
    """
    Production file system using real disk operations.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize local file system.

        Args:
            base_path: Optional base path for relative operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._temp_dir: Optional[Path] = None

    def _resolve(self, path: Path) -> Path:
        """Resolve path relative to base."""
        if path.is_absolute():
            return path
        return self.base_path / path

    async def write_text(self, path: Path, content: str) -> WriteResult:
        """Write text to file."""
        resolved = self._resolve(path)
        try:
            # Ensure parent directory exists
            await self.mkdir(resolved.parent)

            # Write file in thread to not block
            await asyncio.to_thread(
                resolved.write_text,
                content,
                encoding="utf-8",
            )

            return WriteResult(
                path=resolved,
                success=True,
                bytes_written=len(content.encode("utf-8")),
            )
        except Exception as e:
            logger.error(f"Failed to write {resolved}: {e}")
            return WriteResult(path=resolved, success=False, error=str(e))

    async def read_text(self, path: Path) -> Optional[str]:
        """Read text from file."""
        resolved = self._resolve(path)
        try:
            return await asyncio.to_thread(
                resolved.read_text,
                encoding="utf-8",
            )
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to read {resolved}: {e}")
            return None

    async def mkdir(self, path: Path, parents: bool = True, exist_ok: bool = True) -> bool:
        """Create directory."""
        resolved = self._resolve(path)
        try:
            await asyncio.to_thread(
                resolved.mkdir,
                parents=parents,
                exist_ok=exist_ok,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {resolved}: {e}")
            return False

    async def exists(self, path: Path) -> bool:
        """Check if path exists."""
        resolved = self._resolve(path)
        return await asyncio.to_thread(resolved.exists)

    async def is_file(self, path: Path) -> bool:
        """Check if path is file."""
        resolved = self._resolve(path)
        return await asyncio.to_thread(resolved.is_file)

    async def is_dir(self, path: Path) -> bool:
        """Check if path is directory."""
        resolved = self._resolve(path)
        return await asyncio.to_thread(resolved.is_dir)

    async def delete(self, path: Path, recursive: bool = False) -> bool:
        """Delete file or directory."""
        resolved = self._resolve(path)
        try:
            if await self.is_dir(resolved):
                if recursive:
                    await asyncio.to_thread(shutil.rmtree, resolved)
                else:
                    await asyncio.to_thread(resolved.rmdir)
            else:
                await asyncio.to_thread(resolved.unlink)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {resolved}: {e}")
            return False

    async def move(self, src: Path, dst: Path) -> bool:
        """Move file or directory."""
        src_resolved = self._resolve(src)
        dst_resolved = self._resolve(dst)
        try:
            await asyncio.to_thread(shutil.move, src_resolved, dst_resolved)
            return True
        except Exception as e:
            logger.error(f"Failed to move {src_resolved} to {dst_resolved}: {e}")
            return False

    async def copy(self, src: Path, dst: Path) -> bool:
        """Copy file or directory."""
        src_resolved = self._resolve(src)
        dst_resolved = self._resolve(dst)
        try:
            if await self.is_dir(src_resolved):
                await asyncio.to_thread(shutil.copytree, src_resolved, dst_resolved)
            else:
                await asyncio.to_thread(shutil.copy2, src_resolved, dst_resolved)
            return True
        except Exception as e:
            logger.error(f"Failed to copy {src_resolved} to {dst_resolved}: {e}")
            return False

    async def list_dir(self, path: Path) -> List[Path]:
        """List directory contents."""
        resolved = self._resolve(path)
        try:
            return list(await asyncio.to_thread(lambda: list(resolved.iterdir())))
        except Exception as e:
            logger.error(f"Failed to list {resolved}: {e}")
            return []

    def get_temp_dir(self) -> Path:
        """Get or create temporary directory."""
        if self._temp_dir is None or not self._temp_dir.exists():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="devdox_"))
        return self._temp_dir


class InMemoryFileSystem:
    """
    In-memory file system for testing.

    All operations are performed in memory without touching the disk.
    """

    def __init__(self):
        """Initialize in-memory file system."""
        self.files: Dict[str, str] = {}
        self.directories: set[str] = set()
        self._temp_counter = 0

    def _normalize(self, path: Path) -> str:
        """Normalize path to string key."""
        return str(path).replace("\\", "/")

    async def write_text(self, path: Path, content: str) -> WriteResult:
        """Write text to in-memory storage."""
        key = self._normalize(path)
        self.files[key] = content

        # Add parent directories
        parent = path.parent
        while str(parent) not in (".", "", "/"):
            self.directories.add(self._normalize(parent))
            parent = parent.parent

        return WriteResult(
            path=path,
            success=True,
            bytes_written=len(content.encode("utf-8")),
        )

    async def read_text(self, path: Path) -> Optional[str]:
        """Read text from in-memory storage."""
        key = self._normalize(path)
        return self.files.get(key)

    async def mkdir(self, path: Path, parents: bool = True, exist_ok: bool = True) -> bool:
        """Create directory in memory."""
        key = self._normalize(path)
        self.directories.add(key)

        if parents:
            parent = path.parent
            while str(parent) not in (".", "", "/"):
                self.directories.add(self._normalize(parent))
                parent = parent.parent

        return True

    async def exists(self, path: Path) -> bool:
        """Check if path exists in memory."""
        key = self._normalize(path)
        return key in self.files or key in self.directories

    async def is_file(self, path: Path) -> bool:
        """Check if path is a file in memory."""
        key = self._normalize(path)
        return key in self.files

    async def is_dir(self, path: Path) -> bool:
        """Check if path is a directory in memory."""
        key = self._normalize(path)
        return key in self.directories

    async def delete(self, path: Path, recursive: bool = False) -> bool:
        """Delete from in-memory storage."""
        key = self._normalize(path)

        if key in self.files:
            del self.files[key]
            return True

        if key in self.directories:
            if recursive:
                # Delete all files in directory
                prefix = key + "/"
                to_delete = [k for k in self.files if k.startswith(prefix)]
                for k in to_delete:
                    del self.files[k]

                # Delete subdirectories
                to_delete = [d for d in self.directories if d.startswith(prefix)]
                for d in to_delete:
                    self.directories.discard(d)

            self.directories.discard(key)
            return True

        return False

    async def move(self, src: Path, dst: Path) -> bool:
        """Move file in memory."""
        src_key = self._normalize(src)
        dst_key = self._normalize(dst)

        if src_key in self.files:
            self.files[dst_key] = self.files[src_key]
            del self.files[src_key]
            return True

        return False

    async def copy(self, src: Path, dst: Path) -> bool:
        """Copy file in memory."""
        src_key = self._normalize(src)
        dst_key = self._normalize(dst)

        if src_key in self.files:
            self.files[dst_key] = self.files[src_key]
            return True

        return False

    async def list_dir(self, path: Path) -> List[Path]:
        """List directory contents in memory."""
        key = self._normalize(path)
        prefix = key + "/" if key else ""

        results = set()

        # Find files in directory
        for file_path in self.files:
            if file_path.startswith(prefix):
                remainder = file_path[len(prefix):]
                if "/" not in remainder:
                    results.add(Path(file_path))

        # Find subdirectories
        for dir_path in self.directories:
            if dir_path.startswith(prefix):
                remainder = dir_path[len(prefix):]
                if "/" not in remainder and remainder:
                    results.add(Path(dir_path))

        return list(results)

    def get_temp_dir(self) -> Path:
        """Get a fake temp directory path."""
        self._temp_counter += 1
        path = Path(f"/tmp/devdox_test_{self._temp_counter}")
        self.directories.add(self._normalize(path))
        return path

    def get_all_files(self) -> Dict[str, str]:
        """Get all files (for test assertions)."""
        return self.files.copy()

    def clear(self) -> None:
        """Clear all files and directories."""
        self.files.clear()
        self.directories.clear()
