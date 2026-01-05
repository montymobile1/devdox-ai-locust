# Live Integration Tests

Live tests exercise the complete code generation pipeline with real API calls.
They are **skipped by default** to prevent accidental API usage in CI/CD.

## Directory Structure

```
tests/live/
├── conftest.py              # Shared fixtures (api_key, output_dir)
├── README.md                # This file
└── generate/                # Tests for `generate` command
    ├── conftest.py          # Generate-specific fixtures
    ├── test_basic.py        # Basic generation tests
    ├── test_validation.py   # Output validation tests
    ├── test_extensive.py    # Extensive deep validation
    ├── test_database.py     # Database integration (optional)
    ├── test_auth.py         # Authentication tests
    └── test_custom_requirements.py
```

## Quick Start

### Option 1: Using `.env.test` File (Recommended)

The safest way to configure live tests is using a `.env.test` file:

```bash
# Copy the example file
cp .env.test.example .env.test

# Edit with your values
vim .env.test

# Run tests (configuration loaded automatically)
pytest tests/live --run-live
```

### Option 2: Using Environment Variables

```bash
export TOGETHER_API_KEY=your_api_key
export SWAGGER_URL=https://petstore.swagger.io/v2/swagger.json
pytest tests/live --run-live
```

### Option 3: Using Command Line Arguments

```bash
pytest tests/live --run-live \
    --api-key YOUR_API_KEY \
    --swagger-url https://petstore.swagger.io/v2/swagger.json
```

### Configuration Priority

Values are loaded in this order (first found wins):
1. **Command-line arguments** (`--api-key`, `--swagger-url`, etc.)
2. **Environment variables** (`TOGETHER_API_KEY`, `SWAGGER_URL`, etc.)
3. **`.env.test` file** (auto-loaded from project root)

## Configuration Options

### Required

| Option | Environment Variable | `.env.test` Key | Description |
|--------|---------------------|-----------------|-------------|
| `--api-key` | `TOGETHER_API_KEY` | `TOGETHER_API_KEY` | Together AI API key |
| `--swagger-url` | `SWAGGER_URL` | `SWAGGER_URL` | Swagger/OpenAPI URL |

### Optional

| Option | Environment Variable | `.env.test` Key | Description |
|--------|---------------------|-----------------|-------------|
| `--swagger-file` | `SWAGGER_FILE` | `SWAGGER_FILE` | Local Swagger file path |
| `--output-dir` | `TEST_OUTPUT_DIR` | `TEST_OUTPUT_DIR` | Output directory |
| `--keep-output` | `KEEP_TEST_OUTPUT` | `KEEP_TEST_OUTPUT` | Keep output files |
| `--mongodb-uri` | `MONGODB_URI` | `MONGODB_URI` | MongoDB URI for DB tests |
| `--postgresql-uri` | `POSTGRESQL_URI` | `POSTGRESQL_URI` | PostgreSQL URI for DB tests |
| `--target-host` | `TARGET_HOST` | `TARGET_HOST` | Target host for tests |
| `--env-test` | - | - | Custom `.env.test` path |

## `.env.test` File Example

```bash
# Required
TOGETHER_API_KEY=your_api_key_here
SWAGGER_URL=https://petstore.swagger.io/v2/swagger.json

# Optional
# SWAGGER_FILE=/path/to/local/swagger.json
# TARGET_HOST=http://localhost:8000
# TEST_OUTPUT_DIR=./test_output
# KEEP_TEST_OUTPUT=true
# MONGODB_URI=mongodb://localhost:27017/testdb
# POSTGRESQL_URI=postgresql://user:pass@localhost:5432/testdb
```

## Test Categories

### 1. Basic Generation (`test_basic.py`)

Tests different input formats and output configurations:

| Test Class | Description |
|------------|-------------|
| `TestInputFormats` | URL and file input sources |
| `TestOutputConfiguration` | Output directory and file generation |
| `TestVerboseMode` | Verbose and quiet output modes |
| `TestLocustConfiguration` | Users, spawn-rate, run-time options |
| `TestHostConfiguration` | Target host URL configuration |
| `TestDryRun` | Dry-run mode |

### 2. Output Validation (`test_validation.py`)

Validates generated code quality:

| Test Class | Description |
|------------|-------------|
| `TestSyntaxValidation` | Valid Python syntax |
| `TestCodeDuplication` | No duplicate classes (Bug #1, #3) |
| `TestHardcodedValues` | No hardcoded endpoints (Bug #2) or secrets |
| `TestLocustPatterns` | Proper Locust patterns |
| `TestImportValidation` | No circular imports, resolvable imports |
| `TestFileStructure` | Required files exist |
| `TestComprehensiveValidation` | Full validation suite |

### 3. Database Integration (`test_database.py`)

Tests database modes (**skipped if URIs not provided**):

| Test Class | Description |
|------------|-------------|
| `TestMongoDB` | MongoDB integration mode |
| `TestPostgreSQL` | PostgreSQL integration mode |
| `TestNoDatabase` | No database mode (default) |
| `TestDatabaseModeComparison` | Compare all modes |

### 4. Authentication (`test_auth.py`)

Tests auth handling:

| Test Class | Description |
|------------|-------------|
| `TestAuthEnabled` | Auth enabled mode (default) |
| `TestAuthDisabled` | --no-auth mode |
| `TestAuthModeComparison` | Compare auth modes |

### 5. Custom Requirements (`test_custom_requirements.py`)

Tests --custom-requirement option:

| Test Class | Description |
|------------|-------------|
| `TestCustomRequirements` | Various requirement strings |
| `TestCustomRequirementValidation` | Output validity with requirements |
| `TestMultipleRequirements` | Complex combined requirements |

### 6. Extensive Validation (`test_extensive.py`)

Deep validation that ensures options actually affect output:

| Test Class | Description |
|------------|-------------|
| `TestExtensiveValidation` | Full validation report with stats |
| `TestAuthOptionValidation` | Verify --auth/--no-auth affects output |
| `TestDatabaseOptionValidation` | Verify --db-type affects output |
| `TestCustomRequirementValidation` | Verify --custom-requirement affects output |
| `TestGeneratedCodeUsability` | Code is importable and has tasks |

**Run extensive tests:**
```bash
pytest tests/live/generate/test_extensive.py --run-live -m extensive
```

## Running Specific Tests

```bash
# Run all tests for generate command
pytest tests/live/generate --run-live --api-key KEY --swagger-url URL

# Run only basic generation tests
pytest tests/live/generate/test_basic.py --run-live \
    --api-key KEY --swagger-url URL

# Run only validation tests
pytest tests/live/generate/test_validation.py --run-live \
    --api-key KEY --swagger-url URL

# Run only database tests (requires DB URIs)
pytest tests/live/generate/test_database.py --run-live \
    --api-key KEY --swagger-url URL \
    --mongodb-uri mongodb://localhost:27017

# Run only auth tests
pytest tests/live/generate/test_auth.py --run-live \
    --api-key KEY --swagger-url URL

# Run comprehensive validation only
pytest tests/live/generate/test_validation.py::TestComprehensiveValidation \
    --run-live --api-key KEY --swagger-url URL

# Skip slow tests
pytest tests/live --run-live -m "not slow" --api-key KEY --swagger-url URL
```

## Validation Checks

The live tests perform these validations on generated code:

### Code Integrity
- ✅ Valid Python syntax (AST parseable)
- ✅ No duplicate class definitions (Bug #1)
- ✅ No class redefinitions after imports (Bug #3)
- ✅ No hardcoded endpoint arrays (Bug #2)

### Import Validation
- ✅ All imports resolvable within generated output
- ✅ No circular imports between files

### Locust-Specific
- ✅ Proper HttpUser base class usage
- ✅ Proper `catch_response=True` in with-blocks

### Authentication
- ✅ Auth classes in single location only
- ✅ No duplicate auth implementations

### File Structure
- ✅ Required files exist (locustfile.py, config.py)
- ✅ Optional files properly structured

### Security
- ✅ No hardcoded secrets/credentials
- ✅ No hardcoded API keys or tokens

## Optional Test Skipping

Tests automatically skip when optional resources aren't available:

```python
# Database tests skip without URIs
if not mongodb_uri:
    pytest.skip("MongoDB URI not provided (use --mongodb-uri)")

# File input tests skip without file
if not swagger_file:
    pytest.skip("No swagger file provided (use --swagger-file)")
```

## Debugging Failed Tests

### Keep Output for Inspection

```bash
pytest tests/live --run-live --keep-output \
    --output-dir ./debug_output \
    --api-key KEY --swagger-url URL

# Then inspect the generated files
ls -la ./debug_output/
cat ./debug_output/locustfile.py
```

### Run with Verbose Pytest

```bash
pytest tests/live --run-live -v -s \
    --api-key KEY --swagger-url URL
```

### Check Specific Validation

```bash
# Run only syntax validation
pytest tests/live/generate/test_validation.py::TestSyntaxValidation \
    --run-live --api-key KEY --swagger-url URL -v
```

## CI/CD Integration

### GitHub Actions - Skip Live Tests (Default)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests (excludes live)
        run: |
          pytest tests/ --ignore=tests/live
```

### GitHub Actions - Run Live Tests (Manual)

```yaml
jobs:
  live-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'  # Manual trigger only
    steps:
      - uses: actions/checkout@v4
      - name: Run live tests
        env:
          TOGETHER_API_KEY: ${{ secrets.TOGETHER_API_KEY }}
          SWAGGER_URL: ${{ secrets.SWAGGER_URL }}
        run: |
          pip install -e ".[dev]"
          pytest tests/live --run-live -v
```

## Troubleshooting

### "No API key provided"
Set `TOGETHER_API_KEY` environment variable or use `--api-key` flag.

### "No Swagger URL provided"
Set `SWAGGER_URL` environment variable or use `--swagger-url` flag.

### Database tests skipped
Provide `--mongodb-uri` or `--postgresql-uri` to run database tests.

### Timeout errors
The default timeout is 5 minutes. For slow APIs, tests may timeout.

### "Invalid Python syntax" errors
Check if the Swagger spec has unusual patterns that cause generation issues.
Use `--keep-output` to inspect the generated files.

## Adding New CLI Commands

To add tests for a new CLI command (e.g., `analyze`):

1. Create directory: `tests/live/analyze/`
2. Create `conftest.py` with command-specific fixtures
3. Create test files: `test_basic.py`, `test_validation.py`, etc.
4. Update this README

Example structure:
```
tests/live/
├── conftest.py           # Shared fixtures
├── generate/             # Generate command tests
│   └── ...
└── analyze/              # New command tests
    ├── conftest.py
    ├── test_basic.py
    └── test_output.py
```
