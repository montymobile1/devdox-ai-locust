"""
Tests for file_creation utility module
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import os
import uuid

from devdox_ai_locust.utils.file_creation import (
    FileCreationConfig, 
    SafeFileCreator
)


class TestFileCreationConfig:
    """Test FileCreationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FileCreationConfig()
        
        assert config.ALLOWED_EXTENSIONS == {
            ".py", ".md", ".txt", ".sh", ".yml", 
            ".yaml", ".json", ".example"
        }
        assert config.MAX_FILE_SIZE == 1024 * 1024  # 1MB
        assert config.EXECUTABLE_EXTENSIONS == {".sh"}

    def test_config_modification(self):
        """Test configuration modification."""
        config = FileCreationConfig()
        
        # Test modifying allowed extensions
        config.ALLOWED_EXTENSIONS.add(".xml")
        assert ".xml" in config.ALLOWED_EXTENSIONS
        
        # Test modifying max file size
        config.MAX_FILE_SIZE = 2048 * 1024  # 2MB
        assert config.MAX_FILE_SIZE == 2048 * 1024


class TestSafeFileCreator:
    """Test SafeFileCreator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        creator = SafeFileCreator()
        
        assert isinstance(creator.config, FileCreationConfig)
        assert creator.config.MAX_FILE_SIZE == 1024 * 1024

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = FileCreationConfig()
        custom_config.MAX_FILE_SIZE = 512 * 1024
        
        creator = SafeFileCreator(custom_config)
        
        assert creator.config.MAX_FILE_SIZE == 512 * 1024

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        creator = SafeFileCreator()
        
        # Test normal filename
        result = creator._sanitize_filename("test_file.py")
        assert result == "test_file.py"
        
        # Test filename with path
        result = creator._sanitize_filename("/path/to/test_file.py")
        assert result == "test_file.py"
        
        # Test filename with dangerous characters
        result = creator._sanitize_filename("test<>file.py")
        assert result == "testfile.py"

    def test_sanitize_filename_long_name(self):
        """Test sanitization of very long filenames."""
        creator = SafeFileCreator()
        
        long_name = "x" * 300 + ".py"
        result = creator._sanitize_filename(long_name)
        
        assert len(result) <= 255
        assert result.endswith(".py")

    def test_sanitize_filename_hidden_files(self):
        """Test sanitization of hidden files."""
        creator = SafeFileCreator()
        
        # Test dangerous hidden file
        result = creator._sanitize_filename(".hidden_file")
        assert result.startswith("generated_")
        assert result.endswith(".py")
        
        # Test safe hidden files
        result = creator._sanitize_filename(".env.example")
        assert result == ".env.example"
        
        result = creator._sanitize_filename(".gitignore")
        assert result == ".gitignore"

    def test_sanitize_filename_empty(self):
        """Test sanitization of empty filename."""
        creator = SafeFileCreator()
        
        result = creator._sanitize_filename("")
        assert result.startswith("generated_")
        assert result.endswith(".py")

    def test_validate_file_allowed_extension(self):
        """Test file validation with allowed extensions."""
        creator = SafeFileCreator()
        
        valid_files = [
            ("test.py", "print('hello')"),
            ("readme.md", "# README"),
            ("config.json", '{"key": "value"}'),
            ("script.sh", "#!/bin/bash"),
            ("data.yaml", "key: value"),
            ("example.txt", "sample text"),
            (".env.example", "API_KEY=test")
        ]
        
        for filename, content in valid_files:
            is_valid, clean_name, processed_content = creator.validate_file(filename, content)
            assert is_valid
            assert clean_name == filename.lower()
            assert processed_content == content

    def test_validate_file_disallowed_extension(self):
        """Test file validation with disallowed extensions."""
        creator = SafeFileCreator()
        
        invalid_files = [
            ("malware.exe", "binary content"),
            ("script.bat", "malicious script"),
            ("document.docx", "document content"),
            ("image.png", "binary image data")
        ]
        
        for filename, content in invalid_files:
            is_valid, clean_name, processed_content = creator.validate_file(filename, content)
            assert not is_valid

    def test_validate_file_oversized(self):
        """Test file validation with oversized content."""
        creator = SafeFileCreator()
        
        # Create content larger than max size
        large_content = "x" * (creator.config.MAX_FILE_SIZE + 1000)
        
        is_valid, clean_name, processed_content = creator.validate_file("large.py", large_content)
        
        assert is_valid
        assert len(processed_content) < len(large_content)
        assert len(processed_content.encode("utf-8")) <= creator.config.MAX_FILE_SIZE

    @pytest.mark.asyncio
    async def test_create_temp_file(self, temp_dir):
        """Test temporary file creation."""
        creator = SafeFileCreator()
        
        content = "print('hello world')"
        filename = "test.py"
        
        file_info = await creator.create_temp_file(filename, content, temp_dir)
        
        assert file_info["filename"] == filename
        assert file_info["temp_path"] == temp_dir / filename
        assert file_info["size"] == len(content.encode("utf-8"))
        assert file_info["type"] == "py"
        
        # Check that file was actually created
        assert (temp_dir / filename).exists()
        assert (temp_dir / filename).read_text() == content

    @pytest.mark.asyncio
    async def test_create_temp_file_executable(self, temp_dir):
        """Test creating executable temporary file."""
        creator = SafeFileCreator()
        
        content = "#!/bin/bash\necho 'hello'"
        filename = "script.sh"
        
        file_info = await creator.create_temp_file(filename, content, temp_dir)
        
        assert file_info["type"] == "sh"
        
        # Check permissions
        file_path = temp_dir / filename
        assert file_path.exists()
        
        # Check if file has executable permissions
        file_stat = file_path.stat()
        assert file_stat.st_mode & 0o111  # Check execute bits

    @pytest.mark.asyncio
    async def test_create_temp_file_unicode(self, temp_dir):
        """Test creating temporary file with Unicode content."""
        creator = SafeFileCreator()
        
        content = "print('Hello 世界! 🌍')"
        filename = "unicode.py"
        
        file_info = await creator.create_temp_file(filename, content, temp_dir)
        
        assert file_info["size"] == len(content.encode("utf-8"))
        
        # Verify content is correctly written and readable
        file_path = temp_dir / filename
        assert file_path.read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_move_files_atomically_success(self, temp_dir):
        """Test successful atomic file movement."""
        creator = SafeFileCreator()
        
        # Create source and destination directories
        source_dir = temp_dir / "source"
        dest_dir = temp_dir / "dest"
        source_dir.mkdir()
        dest_dir.mkdir()
        
        # Create test files
        test_files = [
            {"filename": "test1.py", "content": "print('test1')"},
            {"filename": "test2.py", "content": "print('test2')"},
        ]
        
        file_infos = []
        for file_data in test_files:
            temp_path = source_dir / file_data["filename"]
            temp_path.write_text(file_data["content"])
            
            file_infos.append({
                "filename": file_data["filename"],
                "temp_path": temp_path,
                "size": len(file_data["content"]),
                "type": "py"
            })
        
        # Move files
        result = await creator.move_files_atomically(file_infos, dest_dir)
        
        assert len(result) == 2
        
        # Check that files were moved
        for file_info in result:
            dest_path = dest_dir / file_info["filename"]
            assert dest_path.exists()
            assert "final_path" in file_info
            assert "path" in file_info
            
            # Source should no longer exist
            assert not file_info["temp_path"].exists()

    @pytest.mark.asyncio
    async def test_move_files_atomically_partial_failure(self, temp_dir):
        """Test atomic file movement with partial failure."""
        creator = SafeFileCreator()
        
        source_dir = temp_dir / "source"
        dest_dir = temp_dir / "dest"
        source_dir.mkdir()
        dest_dir.mkdir()
        
        # Create one valid file and one that will fail
        valid_file = source_dir / "valid.py"
        valid_file.write_text("print('valid')")
        
        # Create destination file that will cause conflict
        conflict_file = dest_dir / "conflict.py"
        conflict_file.write_text("existing content")
        
        file_infos = [
            {
                "filename": "valid.py",
                "temp_path": valid_file,
                "size": 100,
                "type": "py"
            },
            {
                "filename": "conflict.py",
                "temp_path": source_dir / "nonexistent.py",  # This will fail
                "size": 100,
                "type": "py"
            }
        ]
        
        result = await creator.move_files_atomically(file_infos, dest_dir)
        
        # Should only return successfully moved files
        assert len(result) == 1
        assert result[0]["filename"] == "valid.py"

    @pytest.mark.asyncio
    async def test_move_files_atomically_empty_list(self, temp_dir):
        """Test atomic file movement with empty file list."""
        creator = SafeFileCreator()
        
        dest_dir = temp_dir / "dest"
        dest_dir.mkdir()
        
        result = await creator.move_files_atomically([], dest_dir)
        
        assert result == []



    def test_validate_file_edge_cases(self):
        """Test file validation edge cases."""
        creator = SafeFileCreator()
        
        # Test empty content
        is_valid, clean_name, processed_content = creator.validate_file("empty.py", "")
        assert is_valid
        assert processed_content == ""
        
        # Test whitespace-only content
        is_valid, clean_name, processed_content = creator.validate_file("whitespace.py", "   \n\t   ")
        assert is_valid
        assert processed_content == "   \n\t   "
        
        # Test content with null bytes
        content_with_null = "print('hello')\x00print('world')"
        is_valid, clean_name, processed_content = creator.validate_file("null.py", content_with_null)
        assert is_valid
        assert processed_content == content_with_null


class TestSafeFileCreatorIntegration:
    """Integration tests for SafeFileCreator."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_dir):
        """Test complete file creation workflow."""
        creator = SafeFileCreator()
        
        # Test data
        files_to_create = {
            "main.py": "print('main application')",
            "config.json": '{"debug": true, "port": 8000}',
            "README.md": "# Test Project\n\nThis is a test.",
            "setup.sh": "#!/bin/bash\necho 'Setting up...'"
        }
        
        # Create temporary directory for staging
        staging_dir = temp_dir / "staging"
        final_dir = temp_dir / "final"
        staging_dir.mkdir()
        final_dir.mkdir()
        
        file_infos = []
        
        # Validate and create temp files
        for filename, content in files_to_create.items():
            is_valid, clean_name, processed_content = creator.validate_file(filename, content)
            assert is_valid
            
            file_info = await creator.create_temp_file(clean_name, processed_content, staging_dir)
            file_infos.append(file_info)
        
        # Move files to final location
        result = await creator.move_files_atomically(file_infos, final_dir)
        
        # Verify all files were created successfully
        assert len(result) == len(files_to_create)
        
        for filename, expected_content in files_to_create.items():
            final_path = final_dir / filename.lower()
            assert final_path.exists()
            
            if filename.endswith('.sh'):
                # Check executable permissions
                assert final_path.stat().st_mode & 0o111
            
            assert final_path.read_text() == expected_content

    @pytest.mark.asyncio
    async def test_concurrent_file_creation(self, temp_dir):
        """Test concurrent file creation operations."""
        creator = SafeFileCreator()
        
        staging_dir = temp_dir / "staging"
        staging_dir.mkdir()
        
        # Create multiple files concurrently
        async def create_file(i):
            filename = f"file_{i}.py"
            content = f"print('file {i}')"
            
            is_valid, clean_name, processed_content = creator.validate_file(filename, content)
            if is_valid:
                return await creator.create_temp_file(clean_name, processed_content, staging_dir)
            return None
        
        tasks = [create_file(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All files should be created successfully
        valid_results = [r for r in results if r is not None]
        assert len(valid_results) == 10
        
        # Check that all files exist
        for i in range(10):
            assert (staging_dir / f"file_{i}.py").exists()

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_dir):
        """Test error handling in file operations."""
        creator = SafeFileCreator()
        
        # Test with read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            with pytest.raises(PermissionError):
                await creator.create_temp_file("test.py", "print('test')", readonly_dir)
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_custom_config_integration(self):
        """Test integration with custom configuration."""
        # Create custom config with different settings
        custom_config = FileCreationConfig()
        custom_config.ALLOWED_EXTENSIONS = {".py", ".txt"}
        custom_config.MAX_FILE_SIZE = 100  # Very small
        custom_config.EXECUTABLE_EXTENSIONS = {".py"}  # Make Python files executable
        
        creator = SafeFileCreator(custom_config)
        
        # Test allowed extension
        is_valid, _, _ = creator.validate_file("test.py", "print('test')")
        assert is_valid
        
        # Test disallowed extension
        is_valid, _, _ = creator.validate_file("test.json", '{"key": "value"}')
        assert not is_valid
        
        # Test oversized file
        large_content = "x" * 200
        is_valid, _, processed_content = creator.validate_file("large.py", large_content)
        assert is_valid
        assert len(processed_content) < len(large_content)
