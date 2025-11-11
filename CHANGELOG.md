# Changelog

All notable changes to this project will be documented in this file.

## [0.1.7] - 2025-11-11
## 🆕 What's New in 0.1.7

Capitalize group names in generated class names for improved readability and consistency across generated code.

Added MongoDB-related environment variables examples to .env configuration, simplifying setup for MongoDB integrations.



## [0.1.6] - 2025-11-06
## 🆕 What's New in 0.1.6

 Add sonarqube badge to README.md


## [0.1.5] - 2025-11-06
## 🆕 What's New in 0.1.5
### 💥 GitHub Actions Integration

You can now use **DevDox AI Locust ** directly in your GitHub workflows!  
The new reusable **Docker-based GitHub Action** lets you automatically generate and upload Locust test scripts for your APIs.

**Example Workflow:**

```yaml
name: "Swagger Test Generator"
on: [push]

jobs:
  generate-locust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate Locust tests
        uses: montymobile1/devdox-ai-locust@0.1.5
        with:
          swagger_url: "https://petstore3.swagger.io/api/v3/openapi.json"
          output: "generated_tests"
          users: "15"
          spawn_rate: "3"
          run_time: "10m"
          together_api_key: ${{ secrets.TOGETHER_API_KEY }}

      - name: Upload generated tests
        uses: actions/upload-artifact@v4
        with:
          name: locust-tests
          path: generated_tests
```

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
