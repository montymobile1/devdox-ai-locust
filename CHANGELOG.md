# Changelog

All notable changes to this project will be documented in this file.


## [1.0.3] - 2025-10-14

### Added
- Asynchronous API calls using `AsyncTogether` for improved performance
- Enhanced timeout handling with configurable retry mechanisms
- Detailed code block extraction validation with multiple fallback scenarios
- Comprehensive error management throughout the AI generation pipeline

### Changed
- Migrated from `Together` to `AsyncTogether` for non-blocking operations
- Improved `<code>` extraction logic with better error messages
- Enhanced retry logic for transient API errors

### Fixed
- Edge cases in code block extraction causing generation failures
- Timeout handling during long-running AI generations
- Validation of AI-generated code before template insertion

### Performance
- 2-3x faster generation for large APIs with multiple endpoints
- Better handling of concurrent API calls

## [1.0.2] - 2025-09-23

### Added
- Added Jinja2 template support for code generation (fixes a missing-template bug introduced in v1.0.2).
