"""
File System Protocol

Defines the contract for file system operations.
Allows intercepting disk operations for testing,
dry-run modes, and remote file systems.
"""

from typing import Protocol, Optional, List, Dict, Any, runtime_checkable
from pathlib import Path
from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a file"""
    path: Path
    exists: bool
    is_file: bool = True
    is_dir: bool = False
    size: int = 0
    content: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class WriteResult(BaseModel):
    """Result of a write operation"""
    path: Path
    success: bool
    bytes_written: int = 0
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@runtime_checkable
class FileSystem(Protocol):
    """
    Protocol for file system operations.

    Implementations:
        - LocalFileSystem: Real file system operations
        - InMemoryFileSystem: All operations in memory (for testing)
        - DryRunFileSystem: Logs operations without executing
        - RecordingFileSystem: Records operations for verification

    Example:
        class LocalFileSystem:
            async def write_text(self, path: Path, content: str) -> WriteResult:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding='utf-8')
                    return WriteResult(path=path, success=True, bytes_written=len(content))
                except Exception as e:
                    return WriteResult(path=path, success=False, error=str(e))
    """

    async def write_text(self, path: Path, content: str) -> WriteResult:
        """
        Write text content to a file.

        Args:
            path: File path to write to
            content: Text content to write

        Returns:
            WriteResult with success status
        """
        ...

    async def read_text(self, path: Path) -> Optional[str]:
        """
        Read text content from a file.

        Args:
            path: File path to read from

        Returns:
            File content if exists, None otherwise
        """
        ...

    async def mkdir(self, path: Path, parents: bool = True, exist_ok: bool = True) -> bool:
        """
        Create a directory.

        Args:
            path: Directory path to create
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Returns:
            True if created or exists, False on error
        """
        ...

    async def exists(self, path: Path) -> bool:
        """
        Check if a path exists.

        Args:
            path: Path to check

        Returns:
            True if path exists
        """
        ...

    async def is_file(self, path: Path) -> bool:
        """
        Check if path is a file.

        Args:
            path: Path to check

        Returns:
            True if path is a file
        """
        ...

    async def is_dir(self, path: Path) -> bool:
        """
        Check if path is a directory.

        Args:
            path: Path to check

        Returns:
            True if path is a directory
        """
        ...

    async def delete(self, path: Path, recursive: bool = False) -> bool:
        """
        Delete a file or directory.

        Args:
            path: Path to delete
            recursive: If True, delete directories recursively

        Returns:
            True if deleted, False if not found or error
        """
        ...

    async def move(self, src: Path, dst: Path) -> bool:
        """
        Move a file or directory.

        Args:
            src: Source path
            dst: Destination path

        Returns:
            True if moved successfully
        """
        ...

    async def copy(self, src: Path, dst: Path) -> bool:
        """
        Copy a file or directory.

        Args:
            src: Source path
            dst: Destination path

        Returns:
            True if copied successfully
        """
        ...

    async def list_dir(self, path: Path) -> List[Path]:
        """
        List contents of a directory.

        Args:
            path: Directory path

        Returns:
            List of paths in directory
        """
        ...

    def get_temp_dir(self) -> Path:
        """
        Get a temporary directory path.

        Returns:
            Path to temp directory
        """
        ...


class FileSystemError(Exception):
    """Base exception for file system errors"""
    pass


class FileNotFoundError(FileSystemError):
    """File not found"""

    def __init__(self, path: Path):
        super().__init__(f"File not found: {path}")
        self.path = path


class PermissionError(FileSystemError):
    """Permission denied"""

    def __init__(self, path: Path, operation: str):
        super().__init__(f"Permission denied for {operation} on {path}")
        self.path = path
        self.operation = operation
