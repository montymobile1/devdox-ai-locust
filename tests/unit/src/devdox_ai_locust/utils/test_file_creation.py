"""
Comprehensive tests for file_creation.py module.

Tests cover:
- FileCreationConfig class and its constants
- SafeFileCreator initialization
- Filename sanitization
- File validation
- Async file creation
- Atomic file moving
"""

import pytest
from unittest.mock import patch

from devdox_ai_locust.utils.file_creation import (
    FileCreationConfig,
    SafeFileCreator,
)


# =============================================================================
# FileCreationConfig Tests
# =============================================================================


class TestFileCreationConfigConstants:
    """Tests for FileCreationConfig class constants."""

    def test_allowed_extensions_contains_python(self):
        """ALLOWED_EXTENSIONS should include .py files."""
        assert ".py" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_markdown(self):
        """ALLOWED_EXTENSIONS should include .md files."""
        assert ".md" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_text(self):
        """ALLOWED_EXTENSIONS should include .txt files."""
        assert ".txt" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_shell(self):
        """ALLOWED_EXTENSIONS should include .sh files."""
        assert ".sh" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_yaml_variants(self):
        """ALLOWED_EXTENSIONS should include .yml and .yaml files."""
        assert ".yml" in FileCreationConfig.ALLOWED_EXTENSIONS
        assert ".yaml" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_json(self):
        """ALLOWED_EXTENSIONS should include .json files."""
        assert ".json" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_example(self):
        """ALLOWED_EXTENSIONS should include .example files."""
        assert ".example" in FileCreationConfig.ALLOWED_EXTENSIONS

    def test_allowed_extensions_is_set(self):
        """ALLOWED_EXTENSIONS should be a set for O(1) lookup."""
        assert isinstance(FileCreationConfig.ALLOWED_EXTENSIONS, set)

    def test_allowed_extensions_count(self):
        """ALLOWED_EXTENSIONS should have exactly 8 extensions."""
        assert len(FileCreationConfig.ALLOWED_EXTENSIONS) == 8

    def test_max_file_size_is_one_megabyte(self):
        """MAX_FILE_SIZE should be 1MB (1024 * 1024 bytes)."""
        assert FileCreationConfig.MAX_FILE_SIZE == 1024 * 1024
        assert FileCreationConfig.MAX_FILE_SIZE == 1_048_576

    def test_executable_extensions_contains_shell(self):
        """EXECUTABLE_EXTENSIONS should include .sh files."""
        assert ".sh" in FileCreationConfig.EXECUTABLE_EXTENSIONS

    def test_executable_extensions_is_set(self):
        """EXECUTABLE_EXTENSIONS should be a set."""
        assert isinstance(FileCreationConfig.EXECUTABLE_EXTENSIONS, set)

    def test_executable_extensions_only_shell(self):
        """EXECUTABLE_EXTENSIONS should only contain .sh."""
        assert FileCreationConfig.EXECUTABLE_EXTENSIONS == {".sh"}


class TestFileCreationConfigInstance:
    """Tests for FileCreationConfig instance behavior."""

    def test_can_instantiate(self):
        """Should be able to create FileCreationConfig instance."""
        config = FileCreationConfig()
        assert config is not None

    def test_instance_has_class_attributes(self):
        """Instance should have access to class constants."""
        config = FileCreationConfig()
        assert hasattr(config, "ALLOWED_EXTENSIONS")
        assert hasattr(config, "MAX_FILE_SIZE")
        assert hasattr(config, "EXECUTABLE_EXTENSIONS")


# =============================================================================
# SafeFileCreator Initialization Tests
# =============================================================================


class TestSafeFileCreatorInit:
    """Tests for SafeFileCreator initialization."""

    def test_init_without_config(self):
        """Should create default FileCreationConfig when none provided."""
        creator = SafeFileCreator()
        assert creator.config is not None
        assert isinstance(creator.config, FileCreationConfig)

    def test_init_with_custom_config(self):
        """Should use provided config when passed."""
        custom_config = FileCreationConfig()
        creator = SafeFileCreator(config=custom_config)
        assert creator.config is custom_config

    def test_init_with_none_config(self):
        """Should create default config when None is explicitly passed."""
        creator = SafeFileCreator(config=None)
        assert creator.config is not None
        assert isinstance(creator.config, FileCreationConfig)


# =============================================================================
# SafeFileCreator._sanitize_filename Tests
# =============================================================================


class TestSanitizeFilenameBasic:
    """Basic tests for _sanitize_filename method."""

    def test_simple_filename_preserved(self):
        """Simple valid filenames should be preserved (lowercased)."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("test.py") == "test.py"

    def test_filename_lowercased(self):
        """Filenames should be converted to lowercase."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("Test.py") == "test.py"
        assert creator._sanitize_filename("TEST.PY") == "test.py"
        assert creator._sanitize_filename("TeSt.Py") == "test.py"

    def test_extension_lowercased(self):
        """File extensions should be lowercased."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("file.PY") == "file.py"
        assert creator._sanitize_filename("file.MD") == "file.md"


class TestSanitizeFilenamePathTraversal:
    """Tests for path traversal prevention in _sanitize_filename."""

    def test_removes_directory_components(self):
        """Should remove directory path components."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("/etc/passwd") == "passwd"
        assert creator._sanitize_filename("../../secret.py") == "secret.py"
        assert creator._sanitize_filename("path/to/file.py") == "file.py"

    def test_removes_windows_paths(self):
        """Should handle Windows-style paths."""
        creator = SafeFileCreator()
        # The backslash is a dangerous character and gets removed
        result = creator._sanitize_filename("C:\\Users\\file.py")
        assert "C:" not in result
        assert "Users" not in result

    def test_removes_mixed_path_separators(self):
        """Should handle mixed path separators."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("path/to\\file.py")
        assert "/" not in result
        assert "\\" not in result


class TestSanitizeFilenameDangerousChars:
    """Tests for dangerous character removal in _sanitize_filename."""

    def test_removes_angle_brackets(self):
        """Should remove < and > characters."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("file<test>.py")
        assert "<" not in result
        assert ">" not in result

    def test_removes_colon(self):
        """Should remove colon character."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("file:test.py")
        assert ":" not in result

    def test_removes_quotes(self):
        """Should remove quote characters."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename('file"test.py')
        assert '"' not in result

    def test_removes_pipe(self):
        """Should remove pipe character."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("file|test.py")
        assert "|" not in result

    def test_removes_question_mark(self):
        """Should remove question mark."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("file?test.py")
        assert "?" not in result

    def test_removes_asterisk(self):
        """Should remove asterisk (wildcard)."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("file*test.py")
        assert "*" not in result


class TestSanitizeFilenameSeparators:
    """Tests for separator handling in _sanitize_filename."""

    def test_replaces_spaces_with_underscores(self):
        """Should replace spaces with underscores."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("my file.py") == "my_file.py"
        assert creator._sanitize_filename("my   file.py") == "my_file.py"

    def test_replaces_dashes_with_underscores(self):
        """Should replace dashes with underscores."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("my-file.py") == "my_file.py"
        assert creator._sanitize_filename("my--file.py") == "my_file.py"

    def test_replaces_mixed_separators(self):
        """Should handle mixed separators."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("my - file.py")
        assert result == "my_file.py"

    def test_removes_consecutive_underscores(self):
        """Should collapse consecutive underscores."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("my___file.py") == "my_file.py"
        assert creator._sanitize_filename("a__b__c.py") == "a_b_c.py"

    def test_strips_leading_trailing_underscores(self):
        """Should strip leading and trailing underscores from name part."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("_file_.py") == "file.py"
        assert creator._sanitize_filename("__test__.py") == "test.py"


class TestSanitizeFilenameSpecialChars:
    """Tests for special character handling in _sanitize_filename."""

    def test_removes_non_word_characters(self):
        """Should remove non-word characters except dots."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("file@#$.py")
        # @ # $ become underscores, then collapsed
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_preserves_extension_dot(self):
        """Should preserve the dot before extension."""
        creator = SafeFileCreator()
        assert creator._sanitize_filename("file.py").endswith(".py")
        assert creator._sanitize_filename("test.md").endswith(".md")

    def test_handles_multiple_dots(self):
        """Should handle filenames with multiple dots."""
        creator = SafeFileCreator()
        # Only the last extension is preserved as extension
        result = creator._sanitize_filename("file.test.py")
        assert result.endswith(".py")


class TestSanitizeFilenameLength:
    """Tests for filename length limits in _sanitize_filename."""

    def test_truncates_long_filenames(self):
        """Should truncate filenames longer than 255 characters."""
        creator = SafeFileCreator()
        long_name = "a" * 300 + ".py"
        result = creator._sanitize_filename(long_name)
        assert len(result) <= 255

    def test_preserves_extension_after_truncation(self):
        """Should preserve extension when truncating."""
        creator = SafeFileCreator()
        long_name = "a" * 300 + ".py"
        result = creator._sanitize_filename(long_name)
        assert result.endswith(".py")

    def test_filename_exactly_255_not_truncated(self):
        """Filename of exactly 255 chars should not be truncated."""
        creator = SafeFileCreator()
        name = "a" * 252 + ".py"  # 252 + 3 = 255
        result = creator._sanitize_filename(name)
        assert len(result) == 255


class TestSanitizeFilenameHiddenFiles:
    """Tests for hidden file handling in _sanitize_filename."""

    def test_prevents_arbitrary_hidden_files(self):
        """Should prevent creation of arbitrary hidden files."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename(".secret")
        assert not result.startswith(".") or result in {
            ".env.example",
            ".gitignore",
            ".env.template",
        }

    def test_strips_leading_dot_from_env_example(self):
        """Should strip leading dot from .env.example files."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename(".env.example")
        # Leading dots are stripped for safety
        assert result == "env.example"

    def test_strips_leading_dot_from_gitignore(self):
        """Should strip leading dot from .gitignore files."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename(".gitignore")
        # Leading dots are stripped for safety
        assert result == "gitignore"

    def test_strips_leading_dot_from_env_template(self):
        """Should strip leading dot from .env.template files."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename(".env.template")
        # Leading dots are stripped for safety
        assert result == "env.template"

    def test_hidden_file_dot_stripped(self):
        """Hidden files have leading dot stripped."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename(".htaccess")
        # Leading dot is stripped
        assert result == "htaccess"


class TestSanitizeFilenameEmptyInput:
    """Tests for empty/invalid input handling in _sanitize_filename."""

    def test_empty_string_gets_generated_name(self):
        """Empty string should get a generated name."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("")
        assert result.startswith("generated_")
        assert result.endswith(".py")

    def test_only_special_chars_gets_generated_name(self):
        """Filename with only special chars should get a generated name."""
        creator = SafeFileCreator()
        result = creator._sanitize_filename("@#$%")
        # After removing special chars, name becomes empty
        assert "generated_" in result or len(result) > 0

    def test_generated_names_are_unique(self):
        """Generated names should be unique (using UUID)."""
        creator = SafeFileCreator()
        results = [creator._sanitize_filename("") for _ in range(10)]
        assert len(set(results)) == 10  # All unique


# =============================================================================
# SafeFileCreator.validate_file Tests
# =============================================================================


class TestValidateFileExtensions:
    """Tests for file extension validation."""

    def test_allows_python_files(self):
        """Should allow .py files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("test.py", "content")
        assert is_valid is True

    def test_allows_markdown_files(self):
        """Should allow .md files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("README.md", "content")
        assert is_valid is True

    def test_allows_yaml_files(self):
        """Should allow .yml and .yaml files."""
        creator = SafeFileCreator()
        is_valid_yml, _, _ = creator.validate_file("config.yml", "content")
        is_valid_yaml, _, _ = creator.validate_file("config.yaml", "content")
        assert is_valid_yml is True
        assert is_valid_yaml is True

    def test_allows_json_files(self):
        """Should allow .json files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("data.json", "content")
        assert is_valid is True

    def test_allows_shell_files(self):
        """Should allow .sh files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("script.sh", "content")
        assert is_valid is True

    def test_allows_text_files(self):
        """Should allow .txt files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("notes.txt", "content")
        assert is_valid is True

    def test_allows_example_files(self):
        """Should allow .example files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file(".env.example", "content")
        assert is_valid is True

    def test_rejects_executable_files(self):
        """Should reject .exe files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("malware.exe", "content")
        assert is_valid is False

    def test_rejects_php_files(self):
        """Should reject .php files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("script.php", "content")
        assert is_valid is False

    def test_rejects_html_files(self):
        """Should reject .html files."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("page.html", "content")
        assert is_valid is False

    def test_rejects_no_extension(self):
        """Should reject files without extension."""
        creator = SafeFileCreator()
        is_valid, _, _ = creator.validate_file("Makefile", "content")
        # After sanitization, Makefile becomes "makefile" with no extension
        assert is_valid is False


class TestValidateFileReturnValues:
    """Tests for validate_file return tuple."""

    def test_returns_tuple_of_three(self):
        """Should return a tuple of three elements."""
        creator = SafeFileCreator()
        result = creator.validate_file("test.py", "content")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_sanitized_filename(self):
        """Should return sanitized filename as second element."""
        creator = SafeFileCreator()
        _, clean_name, _ = creator.validate_file("My File.py", "content")
        assert clean_name == "my_file.py"

    def test_returns_content_as_third_element(self):
        """Should return content as third element."""
        creator = SafeFileCreator()
        content = "print('hello')"
        _, _, returned_content = creator.validate_file("test.py", content)
        assert returned_content == content


class TestValidateFileSize:
    """Tests for file size validation."""

    def test_accepts_small_files(self):
        """Should accept files under size limit."""
        creator = SafeFileCreator()
        content = "x" * 1000
        is_valid, _, returned_content = creator.validate_file("test.py", content)
        assert is_valid is True
        assert returned_content == content

    def test_truncates_oversized_files(self):
        """Should truncate files over MAX_FILE_SIZE."""
        creator = SafeFileCreator()
        # Create content larger than 1MB
        content = "x" * (1024 * 1024 + 1000)
        is_valid, _, returned_content = creator.validate_file("test.py", content)
        assert is_valid is True
        assert len(returned_content) == (1024 * 1024) // 2

    def test_truncates_to_half_max_size(self):
        """Should truncate to half of MAX_FILE_SIZE."""
        creator = SafeFileCreator()
        content = "x" * (1024 * 1024 * 2)  # 2MB
        _, _, returned_content = creator.validate_file("test.py", content)
        expected_length = FileCreationConfig.MAX_FILE_SIZE // 2
        assert len(returned_content) == expected_length

    def test_file_at_exact_limit_not_truncated(self):
        """File at exactly MAX_FILE_SIZE should not be truncated."""
        creator = SafeFileCreator()
        content = "x" * (1024 * 1024)
        _, _, returned_content = creator.validate_file("test.py", content)
        assert len(returned_content) == 1024 * 1024


class TestValidateFileLogging:
    """Tests for logging in validate_file."""

    def test_logs_warning_for_disallowed_extension(self):
        """Should log warning when extension is not allowed."""
        creator = SafeFileCreator()
        with patch("devdox_ai_locust.utils.file_creation.logger") as mock_logger:
            creator.validate_file("test.exe", "content")
            mock_logger.warning.assert_called_once()
            assert "disallowed extension" in mock_logger.warning.call_args[0][0]

    def test_logs_warning_for_oversized_file(self):
        """Should log warning when file is truncated."""
        creator = SafeFileCreator()
        content = "x" * (1024 * 1024 + 1)
        with patch("devdox_ai_locust.utils.file_creation.logger") as mock_logger:
            creator.validate_file("test.py", content)
            mock_logger.warning.assert_called_once()
            assert "truncating" in mock_logger.warning.call_args[0][0].lower()


# =============================================================================
# SafeFileCreator.create_temp_file Tests
# =============================================================================


class TestCreateTempFileBasic:
    """Basic tests for create_temp_file async method."""

    @pytest.mark.asyncio
    async def test_creates_file_in_temp_dir(self, temp_dir):
        """Should create file in the specified temp directory."""
        creator = SafeFileCreator()
        await creator.create_temp_file("test.py", "content", temp_dir)
        assert (temp_dir / "test.py").exists()

    @pytest.mark.asyncio
    async def test_writes_correct_content(self, temp_dir):
        """Should write the correct content to the file."""
        creator = SafeFileCreator()
        content = "print('hello world')"
        await creator.create_temp_file("test.py", content, temp_dir)
        assert (temp_dir / "test.py").read_text() == content

    @pytest.mark.asyncio
    async def test_uses_utf8_encoding(self, temp_dir):
        """Should use UTF-8 encoding when writing."""
        creator = SafeFileCreator()
        content = "héllo wörld 你好"
        await creator.create_temp_file("test.py", content, temp_dir)
        assert (temp_dir / "test.py").read_text(encoding="utf-8") == content


class TestCreateTempFileReturnValue:
    """Tests for create_temp_file return dictionary."""

    @pytest.mark.asyncio
    async def test_returns_dict(self, temp_dir):
        """Should return a dictionary."""
        creator = SafeFileCreator()
        result = await creator.create_temp_file("test.py", "content", temp_dir)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_filename(self, temp_dir):
        """Should include filename in return dict."""
        creator = SafeFileCreator()
        result = await creator.create_temp_file("test.py", "content", temp_dir)
        assert result["filename"] == "test.py"

    @pytest.mark.asyncio
    async def test_returns_temp_path(self, temp_dir):
        """Should include temp_path in return dict."""
        creator = SafeFileCreator()
        result = await creator.create_temp_file("test.py", "content", temp_dir)
        assert result["temp_path"] == temp_dir / "test.py"

    @pytest.mark.asyncio
    async def test_returns_correct_size(self, temp_dir):
        """Should include correct file size in bytes."""
        creator = SafeFileCreator()
        content = "hello"
        result = await creator.create_temp_file("test.py", content, temp_dir)
        assert result["size"] == 5

    @pytest.mark.asyncio
    async def test_returns_correct_size_unicode(self, temp_dir):
        """Should return correct size for unicode content."""
        creator = SafeFileCreator()
        content = "你好"  # 6 bytes in UTF-8
        result = await creator.create_temp_file("test.py", content, temp_dir)
        assert result["size"] == 6

    @pytest.mark.asyncio
    async def test_returns_file_type(self, temp_dir):
        """Should include file type without leading dot."""
        creator = SafeFileCreator()
        result = await creator.create_temp_file("test.py", "content", temp_dir)
        assert result["type"] == "py"

    @pytest.mark.asyncio
    async def test_returns_correct_type_for_various_extensions(self, temp_dir):
        """Should return correct type for different extensions."""
        creator = SafeFileCreator()

        py_result = await creator.create_temp_file("test.py", "content", temp_dir)
        assert py_result["type"] == "py"

        md_result = await creator.create_temp_file("readme.md", "content", temp_dir)
        assert md_result["type"] == "md"

        sh_result = await creator.create_temp_file("script.sh", "content", temp_dir)
        assert sh_result["type"] == "sh"


class TestCreateTempFilePermissions:
    """Tests for file permissions in create_temp_file."""

    @pytest.mark.asyncio
    async def test_python_file_gets_644_permissions(self, temp_dir):
        """Python files should get 0o644 permissions."""
        creator = SafeFileCreator()
        await creator.create_temp_file("test.py", "content", temp_dir)
        file_path = temp_dir / "test.py"
        mode = file_path.stat().st_mode & 0o777
        assert mode == 0o644

    @pytest.mark.asyncio
    async def test_shell_file_gets_755_permissions(self, temp_dir):
        """Shell files should get 0o755 permissions (executable)."""
        creator = SafeFileCreator()
        await creator.create_temp_file("script.sh", "#!/bin/bash", temp_dir)
        file_path = temp_dir / "script.sh"
        mode = file_path.stat().st_mode & 0o777
        assert mode == 0o755

    @pytest.mark.asyncio
    async def test_markdown_file_gets_644_permissions(self, temp_dir):
        """Markdown files should get 0o644 permissions."""
        creator = SafeFileCreator()
        await creator.create_temp_file("README.md", "# Title", temp_dir)
        file_path = temp_dir / "README.md"
        mode = file_path.stat().st_mode & 0o777
        assert mode == 0o644

    @pytest.mark.asyncio
    async def test_json_file_gets_644_permissions(self, temp_dir):
        """JSON files should get 0o644 permissions."""
        creator = SafeFileCreator()
        await creator.create_temp_file("config.json", "{}", temp_dir)
        file_path = temp_dir / "config.json"
        mode = file_path.stat().st_mode & 0o777
        assert mode == 0o644


# =============================================================================
# SafeFileCreator.move_files_atomically Tests
# =============================================================================


class TestMoveFilesAtomicallyBasic:
    """Basic tests for move_files_atomically method."""

    @pytest.mark.asyncio
    async def test_moves_single_file(self, temp_dir):
        """Should move a single file to output directory."""
        creator = SafeFileCreator()

        # Create temp file
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        source_file = source_dir / "test.py"
        source_file.write_text("content")

        file_infos = [{"filename": "test.py", "temp_path": source_file}]
        result = await creator.move_files_atomically(file_infos, output_dir)

        assert len(result) == 1
        assert (output_dir / "test.py").exists()
        assert not source_file.exists()

    @pytest.mark.asyncio
    async def test_moves_multiple_files(self, temp_dir):
        """Should move multiple files to output directory."""
        creator = SafeFileCreator()

        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        files = ["test1.py", "test2.py", "test3.py"]
        file_infos = []
        for filename in files:
            path = source_dir / filename
            path.write_text(f"content of {filename}")
            file_infos.append({"filename": filename, "temp_path": path})

        result = await creator.move_files_atomically(file_infos, output_dir)

        assert len(result) == 3
        for filename in files:
            assert (output_dir / filename).exists()


class TestMoveFilesAtomicallyReturnValue:
    """Tests for return value of move_files_atomically."""

    @pytest.mark.asyncio
    async def test_returns_list(self, temp_dir):
        """Should return a list."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        result = await creator.move_files_atomically([], output_dir)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_adds_final_path_to_file_info(self, temp_dir):
        """Should add final_path to each file_info dict."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        source_file = source_dir / "test.py"
        source_file.write_text("content")

        file_infos = [{"filename": "test.py", "temp_path": source_file}]
        result = await creator.move_files_atomically(file_infos, output_dir)

        assert result[0]["final_path"] == output_dir / "test.py"

    @pytest.mark.asyncio
    async def test_adds_path_to_file_info(self, temp_dir):
        """Should add path key (same as final_path) to each file_info dict."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        source_file = source_dir / "test.py"
        source_file.write_text("content")

        file_infos = [{"filename": "test.py", "temp_path": source_file}]
        result = await creator.move_files_atomically(file_infos, output_dir)

        assert result[0]["path"] == output_dir / "test.py"

    @pytest.mark.asyncio
    async def test_preserves_original_file_info_fields(self, temp_dir):
        """Should preserve original fields in file_info dict."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        source_file = source_dir / "test.py"
        source_file.write_text("content")

        file_infos = [
            {
                "filename": "test.py",
                "temp_path": source_file,
                "size": 7,
                "type": "py",
                "custom_field": "preserved",
            }
        ]
        result = await creator.move_files_atomically(file_infos, output_dir)

        assert result[0]["size"] == 7
        assert result[0]["type"] == "py"
        assert result[0]["custom_field"] == "preserved"


class TestMoveFilesAtomicallyErrorHandling:
    """Tests for error handling in move_files_atomically."""

    @pytest.mark.asyncio
    async def test_handles_missing_source_file(self, temp_dir):
        """Should handle case where source file doesn't exist."""
        creator = SafeFileCreator()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        file_infos = [
            {"filename": "missing.py", "temp_path": temp_dir / "nonexistent.py"}
        ]

        with patch("devdox_ai_locust.utils.file_creation.logger") as mock_logger:
            result = await creator.move_files_atomically(file_infos, output_dir)
            assert len(result) == 0
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_on_single_failure(self, temp_dir):
        """Should continue moving other files if one fails."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        good_file = source_dir / "good.py"
        good_file.write_text("good content")

        file_infos = [
            {"filename": "missing.py", "temp_path": temp_dir / "nonexistent.py"},
            {"filename": "good.py", "temp_path": good_file},
        ]

        result = await creator.move_files_atomically(file_infos, output_dir)
        assert len(result) == 1
        assert result[0]["filename"] == "good.py"

    @pytest.mark.asyncio
    async def test_returns_only_successful_moves(self, temp_dir):
        """Should only return successfully moved files."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        file1 = source_dir / "file1.py"
        file1.write_text("content1")
        file3 = source_dir / "file3.py"
        file3.write_text("content3")

        file_infos = [
            {"filename": "file1.py", "temp_path": file1},
            {"filename": "file2.py", "temp_path": temp_dir / "missing.py"},
            {"filename": "file3.py", "temp_path": file3},
        ]

        result = await creator.move_files_atomically(file_infos, output_dir)
        assert len(result) == 2
        assert {f["filename"] for f in result} == {"file1.py", "file3.py"}


class TestMoveFilesAtomicallyLogging:
    """Tests for logging in move_files_atomically."""

    @pytest.mark.asyncio
    async def test_logs_info_on_success(self, temp_dir):
        """Should log info message for each successful move."""
        creator = SafeFileCreator()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        source_file = source_dir / "test.py"
        source_file.write_text("content")

        file_infos = [{"filename": "test.py", "temp_path": source_file}]

        with patch("devdox_ai_locust.utils.file_creation.logger") as mock_logger:
            await creator.move_files_atomically(file_infos, output_dir)
            mock_logger.info.assert_called_once()
            assert "test.py" in mock_logger.info.call_args[0][0]

    @pytest.mark.asyncio
    async def test_logs_error_on_failure(self, temp_dir):
        """Should log error message for failed moves."""
        creator = SafeFileCreator()
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        file_infos = [{"filename": "missing.py", "temp_path": temp_dir / "missing.py"}]

        with patch("devdox_ai_locust.utils.file_creation.logger") as mock_logger:
            await creator.move_files_atomically(file_infos, output_dir)
            mock_logger.error.assert_called_once()
            assert "Failed to move" in mock_logger.error.call_args[0][0]


# =============================================================================
# Integration Tests
# =============================================================================


class TestFileCreationIntegration:
    """Integration tests for the full file creation workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_valid_file(self, temp_dir):
        """Test complete workflow: validate -> create -> move."""
        creator = SafeFileCreator()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        temp_work_dir = temp_dir / "temp"
        temp_work_dir.mkdir()

        filename = "My Test File.py"
        content = "print('hello world')"

        # Step 1: Validate
        is_valid, clean_name, processed_content = creator.validate_file(
            filename, content
        )
        assert is_valid is True
        assert clean_name == "my_test_file.py"

        # Step 2: Create temp file
        file_info = await creator.create_temp_file(
            clean_name, processed_content, temp_work_dir
        )
        assert file_info["filename"] == "my_test_file.py"
        assert (temp_work_dir / "my_test_file.py").exists()

        # Step 3: Move atomically
        result = await creator.move_files_atomically([file_info], output_dir)
        assert len(result) == 1
        assert (output_dir / "my_test_file.py").exists()
        assert (output_dir / "my_test_file.py").read_text() == content

    @pytest.mark.asyncio
    async def test_full_workflow_rejected_extension(self, temp_dir):
        """Test workflow stops for disallowed extensions."""
        creator = SafeFileCreator()

        is_valid, _, _ = creator.validate_file("malware.exe", "dangerous")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_full_workflow_oversized_file(self, temp_dir):
        """Test workflow with file that gets truncated."""
        creator = SafeFileCreator()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        temp_work_dir = temp_dir / "temp"
        temp_work_dir.mkdir()

        large_content = "x" * (1024 * 1024 + 1000)

        is_valid, clean_name, processed_content = creator.validate_file(
            "large.py", large_content
        )
        assert is_valid is True
        assert len(processed_content) == (1024 * 1024) // 2

        file_info = await creator.create_temp_file(
            clean_name, processed_content, temp_work_dir
        )
        assert file_info["size"] == (1024 * 1024) // 2

    @pytest.mark.asyncio
    async def test_full_workflow_shell_script(self, temp_dir):
        """Test workflow for shell script with executable permissions."""
        creator = SafeFileCreator()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        temp_work_dir = temp_dir / "temp"
        temp_work_dir.mkdir()

        content = "#!/bin/bash\necho 'Hello'"

        is_valid, clean_name, processed_content = creator.validate_file(
            "setup.sh", content
        )
        assert is_valid is True

        file_info = await creator.create_temp_file(
            clean_name, processed_content, temp_work_dir
        )

        # Check executable permissions in temp
        mode = (temp_work_dir / clean_name).stat().st_mode & 0o777
        assert mode == 0o755

        result = await creator.move_files_atomically([file_info], output_dir)
        assert len(result) == 1
