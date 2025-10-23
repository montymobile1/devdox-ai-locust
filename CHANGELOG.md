# Changelog

All notable changes to this project will be documented in this file.

## [0.1.4] - 2025-10-23
## 🆕 What's New in 0.1.4

### MongoDB Integration

- Added a new data provider class: MongoDataProvider
- Connects Locust test data generation directly to MongoDB
- Enables realistic test data retrieval for entities like users, products, orders, affiliates, etc.
- Supports **real data** from the database and **synthetic fallback generation** when MongoDB is disabled or unavailable

#### 2. **New MongoDataProvider Methods**
| Method                                                   | Description |
|----------------------------------------------------------|--------------|
| `get_document(collection_name)`                      | Retrieves a single realistic document from MongoDB or fallback generator |
| `get_multiple_documents(collection_name, count=10, query=None)` | Retrieves multiple documents or generates them in batches |
| `clear_cache()`                                          | Clears in-memory cached data for all collections |
| `get_stats()`                                            | Returns usage and cache statistics for debugging and optimization |

#### 3. **Smart Fallbacks**
If MongoDB is disabled (`enable_mongodb = false` in `db_config.py`),  
the system automatically switches to **synthetic data generation** using the LLM-based `TestDataGenerator`.


## [0.1.3.post1] - 2025-10-14

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
