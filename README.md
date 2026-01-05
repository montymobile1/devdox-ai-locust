# DevDox AI Locust

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-722%20passed-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Transform your API docs into powerful load tests with AI magic** 🪄

[Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [CLI Guide](#-cli-reference) • [Architecture](#-architecture) • [Contributing](#-development)

</div>

---

## 🎯 What is DevDox AI Locust?

**DevDox AI Locust** is a CLI tool that reads your OpenAPI/Swagger specification and generates a complete, ready-to-run [Locust](https://locust.io) load testing suite. But here's the twist: it uses **AI** to make the tests actually *smart*.

Instead of just generating boilerplate HTTP calls, it creates:
- **Realistic test scenarios** with proper data flows
- **Positive, negative, and edge-case tests** (the full testing pyramid!)
- **Authentication-aware tests** that handle tokens properly
- **Clean, modular code** you can actually read and maintain

> **TL;DR**: Point it at your API spec, get production-ready load tests. ✨

---

## 🚀 Quick Start

### 1. Install

```bash
pip install devdox-ai-locust
```

### 2. Get Your AI Key

Grab a free API key from [Together AI](https://api.together.xyz/) - it takes 30 seconds.

```bash
export TOGETHER_API_KEY="your_key_here"
```

### 3. Generate Tests

```bash
devdox_ai_locust generate https://petstore3.swagger.io/api/v3/openapi.json \
  --output ./my-load-tests \
  --host https://petstore3.swagger.io
```

### 4. Run Your Tests

```bash
cd my-load-tests
pip install -r requirements.txt
locust -f locustfile.py
```

That's it! Open http://localhost:8089 and start load testing. 🎉

---

## ✨ What Makes It Special?

| Feature | What It Means For You |
|---------|----------------------|
| **🤖 AI-Powered** | Not just templates - AI understands your API and generates smart test logic |
| **🧪 Full Test Coverage** | Positive tests, negative tests, edge cases, security tests - all generated |
| **🔐 Auth-Aware** | Detects Bearer, API Key, OAuth from your spec and handles it automatically |
| **📦 SOLID Architecture** | Generated code follows best practices - small files, single responsibility |
| **🔄 Patch Tracking** | Every generation creates a diff - see exactly what changed (like git for tests!) |
| **⚡ Fast Generation** | Parallel AI calls - generates 40+ files in under 30 seconds |

---

## 🧩 How It Works

Here's the journey from API spec to running tests:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR API SPEC                                │
│                    (OpenAPI/Swagger JSON/YAML)                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PARSING PHASE                                 │
│  • Fetch spec from URL or read from file                            │
│  • Parse endpoints, schemas, security schemes                        │
│  • Group endpoints by API tag (users, products, etc.)               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TEMPLATE GENERATION                              │
│  • Generate static files (config, utils, auth)                       │
│  • Create project structure with SOLID principles                    │
│  • Set up base classes for scenarios and workflows                   │
│                                                                      │
│  📁 This creates ~18 files in milliseconds                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AI ENHANCEMENT                                  │
│  • Send focused prompts to Together AI (Qwen 2.5 Coder)             │
│  • Generate test methods for each API group in parallel              │
│  • Create valid/invalid data generators                              │
│  • Validate generated code (AST parsing)                             │
│  • Auto-fix common issues (dead code, indentation)                   │
│                                                                      │
│  🤖 This adds smart test logic to ~12 files                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                        │
│  📁 Complete Locust test suite (30-40 files)                        │
│  📋 Metadata tracking (.devdox_ai_locust/)                          │
│  📄 Patch files showing all changes                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### The Secret Sauce: Parallel AI Calls 🚀

Instead of sending one giant prompt to the AI (which would hit token limits and be slow), we:

1. **Group endpoints by API tag** - Users API, Products API, etc.
2. **Send parallel requests** - Each API group gets its own focused prompt
3. **Validate everything** - AST parsing catches syntax errors
4. **Auto-fix issues** - Dead code, indentation problems get fixed automatically

This means generating tests for a 50-endpoint API takes ~30 seconds, not 5 minutes.

---

## 📖 CLI Reference

### The `generate` Command

This is where the magic happens. Here's everything you can customize:

```bash
devdox_ai_locust generate [OPTIONS] SWAGGER_SOURCE
```

#### Required Argument

| Argument | Description |
|----------|-------------|
| `SWAGGER_SOURCE` | URL or file path to your OpenAPI/Swagger spec |

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Where to put generated files | `output` |
| `--host` | `-H` | Target host for tests (overrides spec) | Auto-detected |
| `--users` | `-u` | Simulated users for Locust | `10` |
| `--spawn-rate` | `-r` | Users spawned per second | `2.0` |
| `--run-time` | `-t` | Test duration (`30s`, `5m`, `1h`) | `5m` |
| `--auth/--no-auth` | | Include authentication tests | `--auth` |
| `--db-type` | | Database integration (`mongo`, `postgresql`) | none |
| `--custom-requirement` | | Custom instructions for AI | none |
| `--together-api-key` | | Your Together AI key | `$TOGETHER_API_KEY` |
| `--retry-on-invalid` | | Retry AI calls if code invalid | `0` |
| `--dry-run` | | Preview without creating files | `false` |
| `-v, --verbose` | | Show detailed output | `false` |
| `-d, --debug` | | Enable debug logging | `false` |

#### Real-World Examples

```bash
# Basic - just generate from a URL
devdox_ai_locust generate https://api.mycompany.com/openapi.json

# Full control - production setup
devdox_ai_locust generate https://api.mycompany.com/openapi.json \
  --output ./load-tests \
  --host https://staging.mycompany.com \
  --users 100 \
  --spawn-rate 10 \
  --run-time 30m \
  --db-type postgresql

# Focus on specific scenarios
devdox_ai_locust generate ./openapi.json \
  --custom-requirement "Focus on payment endpoints, test currency edge cases, include rate limiting tests"

# Local development
devdox_ai_locust generate ./swagger.yaml \
  --host http://localhost:3000 \
  --no-auth \
  -v
```

### The `run` Command

Run your generated tests without remembering Locust CLI syntax:

```bash
devdox_ai_locust run ./tests/locustfile.py \
  --host https://api.example.com \
  --users 50 \
  --spawn-rate 5 \
  --run-time 10m \
  --headless  # No web UI, just run
```

---

## 📁 What Gets Generated?

Here's the project structure you'll get. Each file has a **single responsibility** (we're serious about SOLID):

```
my-load-tests/
│
├── 📄 locustfile.py           # Entry point - Locust reads this
├── 📄 config.py               # All settings, env vars, hosts
├── 📄 utils.py                # Helper functions
├── 📄 requirements.txt        # pip install -r requirements.txt
├── 📄 README.md               # How to run (auto-generated)
├── 📄 .env.example            # Environment template
│
├── 📁 auth/                   # Authentication handling
│   ├── __init__.py
│   └── authenticator.py       # Login, token refresh, logout
│
├── 📁 data/                   # Test data generators
│   ├── __init__.py
│   ├── base_generator.py      # Faker integration
│   ├── valid_data.py          # AI-generated: complete_data(), minimal_data()
│   ├── invalid_data.py        # AI-generated: missing_required(), wrong_types()
│   └── security_payloads.py   # SQL injection, XSS patterns
│
├── 📁 scenarios/              # Test cases (the actual tests!)
│   ├── __init__.py
│   ├── base_scenario.py       # Base TaskSet class
│   ├── common_security.py     # Auth bypass, injection tests
│   │
│   ├── 📁 users/              # One folder per API group
│   │   ├── __init__.py
│   │   ├── positive.py        # Happy path: create user works
│   │   ├── negative.py        # Errors: invalid email fails
│   │   └── edge_cases.py      # Boundaries: max length names
│   │
│   ├── 📁 products/
│   │   ├── __init__.py
│   │   ├── positive.py
│   │   ├── negative.py
│   │   └── edge_cases.py
│   └── ...
│
├── 📁 workflows/              # Multi-step flows
│   ├── __init__.py
│   ├── main_workflow.py       # Orchestrates everything
│   ├── users_workflow.py      # CRUD: create → read → update → delete
│   └── products_workflow.py
│
└── 📁 .devdox_ai_locust/      # Metadata (git-like tracking)
    ├── metadata.json          # API info, config snapshot
    └── 2025-01-04_10-30-00/   # Session folder
        ├── session.json       # What happened this run
        └── .patches/          # Diff files
            ├── 000001_abc.patch  # Template generation
            └── 000002_def.patch  # AI enhancements
```

### Understanding the Scenario Types

| Type | What It Tests | Example |
|------|---------------|---------|
| **Positive** | Happy paths - things that should work | Create user with valid data → 201 |
| **Negative** | Error handling - things that should fail gracefully | Create user with invalid email → 422 |
| **Edge Cases** | Boundaries - things at the limits | Username with 1000 characters → ? |
| **Security** | Attack patterns - things that should be blocked | SQL injection in username → 400 |

---

## 🏗️ Architecture

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│                        (cli.py)                             │
│  • Parse commands                                           │
│  • Display progress                                         │
│  • Handle errors                                            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    ModularGenerator                         │
│               (modular_generator.py)                        │
│  • Orchestrate generation                                   │
│  • Manage templates + AI calls                              │
│  • Validate output                                          │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
┌────────────────┐  ┌─────────────────┐  ┌────────────────┐
│   Templates    │  │  AI Integration │  │  Validation    │
│   (Jinja2)     │  │  (Together AI)  │  │  (AST Parser)  │
│                │  │                 │  │                │
│ • Static files │  │ • Scenario gen  │  │ • Syntax check │
│ • Config       │  │ • Data gen      │  │ • Auto-fix     │
│ • Utils        │  │ • Workflow gen  │  │ • Format       │
└────────────────┘  └─────────────────┘  └────────────────┘
```

### Key Components

| Component | File | What It Does |
|-----------|------|--------------|
| **CLI** | `cli.py` | The command-line interface. Parses args, shows pretty progress, calls the generator. |
| **ModularGenerator** | `modular_generator.py` | The brain. Orchestrates templates + AI + validation. |
| **HybridLocustGenerator** | `hybrid_loctus_generator.py` | Makes AI API calls, extracts code from responses. |
| **CodeValidator** | `validation.py` | Parses generated Python with AST, catches syntax errors. |
| **CodeFixer** | `validation.py` | Auto-fixes common issues (unreachable code, indentation). |
| **OpenAPIParser** | `utils/open_ai_parser.py` | Parses OpenAPI specs, extracts endpoints and schemas. |
| **MetadataManager** | `utils/metadata_manager.py` | Tracks API info, config, generated files in JSON. |
| **PatchTracker** | `utils/patch_tracker.py` | Creates diff files for each generation phase. |

### Why SOLID Matters Here

**Single Responsibility**: Each generated file does ONE thing.
- `valid_data.py` only generates valid data
- `users/positive.py` only tests happy paths for users API
- `authenticator.py` only handles auth

**Why?**
1. **Smaller AI prompts** = better quality output
2. **Easier to debug** = when a test fails, you know where to look
3. **Maintainable** = you can modify one file without breaking others

---

## 🔄 Patch Tracking (The Cool Git-Like Feature)

Every time you generate tests, we create **patch files** showing exactly what changed:

```
.devdox_ai_locust/
├── metadata.json                     # Central config
└── 2025-01-04_10-30-00/             # One folder per run
    ├── session.json                  # Summary of this run
    └── .patches/
        ├── 000001_a1b2c3d4.patch    # "template_generation" - static files
        └── 000002_e5f6g7h8.patch    # "llm_enhancement" - AI additions
```

**What's in a patch?** Standard unified diff format:

```diff
--- a/scenarios/users/positive.py
+++ b/scenarios/users/positive.py
@@ -0,0 +1,25 @@
+@task(5)
+def test_create_user(self):
+    """Test creating a user with valid data."""
+    data = self.valid_data.complete_data("users")
+    with self.client.post("/api/v1/users/", json=data, catch_response=True) as response:
+        if response.status_code == 201:
+            response.success()
```

**Why track patches?**
- **Reproducibility**: See exactly what AI generated
- **Debugging**: Compare different runs
- **Auditing**: Know what changed and when

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
TOGETHER_API_KEY=your_api_key_here

# Optional
LOG_LEVEL=INFO  # DEBUG for verbose logging
```

### Generated `.env.example`

Each generated project includes an `.env.example`:

```bash
# Target API
API_BASE_URL=http://localhost:8000
API_TITLE=My API

# Locust settings
LOCUST_USERS=50
LOCUST_SPAWN_RATE=5
LOCUST_RUN_TIME=10m

# Auth (if your API requires it)
AUTH_USERNAME=test@example.com
AUTH_PASSWORD=password123
AUTH_TOKEN=  # Or use a token directly
```

---

## 🧪 Development

### Setting Up

```bash
# Clone
git clone https://github.com/montymobile1/devdox-ai-locust.git
cd devdox-ai-locust

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev,test,ai]"

# Pre-commit hooks (for consistent formatting)
pre-commit install
```

### Running Tests

```bash
# All tests (722 of them!)
python -m pytest tests/unit/ -v

# With coverage
coverage run -m pytest tests/unit/
coverage report -m

# Just CLI tests
python -m pytest tests/unit/src/devdox_ai_locust/test_cli.py -v
```

### Code Quality

```bash
black src/ tests/       # Format code
isort src/ tests/       # Sort imports
mypy src/               # Type checking
ruff check src/         # Linting
```

### Project Layout

```
src/devdox_ai_locust/
├── cli.py                     # CLI entry point
├── config.py                  # Settings/config
├── modular_generator.py       # Main generator (SOLID)
├── hybrid_loctus_generator.py # AI integration
├── validation.py              # Code validation
├── templates/                 # Jinja2 templates
├── prompt/                    # AI prompt templates
├── schemas/                   # Pydantic models
└── utils/                     # Utilities
    ├── open_ai_parser.py      # OpenAPI parsing
    ├── swagger_utils.py       # Spec fetching
    ├── metadata_manager.py    # Metadata tracking
    └── patch_tracker.py       # WAL-style patches
```

---

## 🐛 Troubleshooting

### "No module named 'devdox_ai_locust'"

```bash
pip install -e .
python -c "import devdox_ai_locust; print('OK')"
```

### "Together AI API key is required"

```bash
export TOGETHER_API_KEY="your_key"
# or
devdox_ai_locust generate ... --together-api-key "your_key"
```

### "Failed to fetch API schema"

```bash
# Check the URL works
curl https://api.example.com/openapi.json

# For local files, use full path
devdox_ai_locust generate /full/path/to/spec.json
```

### Generated code has errors

```bash
# Enable auto-retry
devdox_ai_locust generate ... --retry-on-invalid 2

# See what's happening
devdox_ai_locust -d generate ...
```

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE)

---

## 🙏 Credits

- **[Locust](https://locust.io/)** - The incredible load testing framework
- **[Together AI](https://together.ai/)** - AI model hosting
- **[Rich](https://rich.readthedocs.io/)** - Beautiful terminal output
- **[Faker](https://faker.readthedocs.io/)** - Test data generation
- **[Pydantic](https://pydantic.dev/)** - Data validation
- **[Jinja2](https://jinja.palletsprojects.com/)** - Templating

---

<div align="center">

**Made with ❤️ by the DevDox team**

Got questions? [Open an issue](https://github.com/montymobile1/devdox-ai-locust/issues) • Found a bug? [Submit a PR](https://github.com/montymobile1/devdox-ai-locust/pulls)

*Now go break some APIs!* 🚀

</div>
