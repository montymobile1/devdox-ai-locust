# DevDox AI Locust

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=montymobile1_devdox-ai-locust&metric=alert_status)](https://sonarcloud.io/dashboard?id=montymobile1_devdox-ai-locust)

> **AI-powered Locust load test generator from API documentation**

DevDox AI Locust automatically generates comprehensive Locust load testing scripts from your API documentation (OpenAPI/Swagger specs). Using advanced AI capabilities, it creates realistic test scenarios, handles complex authentication flows, and generates production-ready performance tests.

## [0.1.8] - 2025-11-18
### 🆕 What's New in 0.1.8

**🔐 Smart Authentication Integration:**
- Seamless integration with existing `base_workflow.py` authentication patterns
- Automatic detection and reuse of existing authentication methods
- Zero duplication of authentication logic across test files
- Smart 401 response handling with automatic re-authentication
- Configurable authentication retry mechanisms

**📊 Comprehensive Test Scenario Framework:**
- **Weighted scenario distribution** for realistic load patterns:
  - 60% Positive scenarios (happy path, valid data)
  - 25% Negative scenarios (validation errors, missing fields)
  - 10% Edge cases (boundary testing, unicode characters)
  - 4% State-dependent scenarios (concurrent operations, duplicates)
  - 1% Security scenarios (injection attacks, vulnerability detection)
- **Enhanced test data generators** supporting all scenario types
- **Security vulnerability detection** with immediate alerting for critical issues


## ✨ Features

- 🤖 **AI-Enhanced Generation**: Uses Together AI to create intelligent, realistic load test scenarios
- 📚 **OpenAPI/Swagger Support**: Automatically parses OpenAPI 3.0 and Swagger 2.0 specifications  
- 🔧 **Production-Ready Code**: Generates clean, maintainable Locust test files
- 🎯 **Smart Test Scenarios**: Creates CRUD workflows, authentication flows, and error handling
- 📊 **Realistic Data**: Generates appropriate test data using Faker
- 🛠️ **Template-Based**: Highly customizable Jinja2 templates for different testing needs
- 🔄 **Hybrid Approach**: Combines rule-based generation with AI enhancement
- 📈 **Comprehensive Coverage**: Handles various HTTP methods, content types, and response scenarios
- ⚡ **Asynchronous Processing**: Fast, non-blocking test generation with async/await

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when published)
pip install devdox-ai-locust

# Or install from source
git clone https://github.com/montymobile1/devdox-ai-locust.git
cd devdox-ai-locust
pip install -e .
```

### Prerequisites

1. **Python 3.12+** - Required for modern async/await syntax and type hints
2. **Together AI API Key** - For AI-enhanced generation capabilities

### Configuration

1. Get your Together AI API key from [Together AI](https://api.together.xyz/)
2. Set up your environment:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
echo "API_KEY=your_together_ai_api_key_here" > .env
```

### Basic Usage

```bash
# Generate from OpenAPI URL
devdox_ai_locust generate --openapi-url https://api.example.com/openapi.json --output ./tests

# Generate with custom configuration
devdox_ai_locust generate \
   https://petstore3.swagger.io/api/v3/openapi.json \
  --output ./petstore-tests \
  --together-api-key your_api_key \
  
 # Generate with db integration 
devdox_ai_locust generate \
  https://petstore3.swagger.io/api/v3/openapi.json \
  --output ./petstore-tests \
  --db-type mongo \
```


## 🚀 Installation with Inputs

Add this step to your GitHub Actions workflow:

```yaml
- name: DevDox Locust Test Generator
  uses: montymobile1/devdox-ai-locust@v0.1.6
  with:
    swagger_url: "https://portal-api.devdox.ai/openapi.json"
    output: "generated_tests"
    users: "15"
    spawn_rate: "3"
    run_time: "10m"
    together_api_key: ${{ secrets.TOGETHER_API_KEY }}


## 📖 Documentation

### Command Line Interface

```bash
devdox_ai_locust generate [OPTIONS] SWAGGER_URL
```

| Option | Short | Type | Description | Default |
|--------|--------|------|-------------|---------|
| `--output` | `-o` | Path | Output directory for generated tests | `output` |
| `--users` | `-u` | Integer | Number of simulated users | `10` |
| `--spawn-rate` | `-r` | Float | User spawn rate (users/second) | `2` |
| `--run-time` | `-t` | String | Test duration (e.g., 5m, 1h) | `5m` |
| `--host` | `-H` | String | Target host URL | None |
| `--auth/--no-auth` | | Boolean | Include authentication | `True` |
| `--dry-run` | | Flag | Generate without running | `False` |
| `--custom-requirement` | | String | Custom AI instructions | None |
| `--together-api-key` | | String | Together AI API key | From env |




### Generated Test Structure

```
locust_tests/
├── locust.py              # Main Locust test file
├── config.py              # Test configuration
├── test_data.py           # Test data generators
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # Test documentation
├── workflows/
│   ├── user_workflow.py   # User-related test scenarios
│   ├── auth_workflow.py   # Authentication workflows
│   └── crud_workflow.py   # CRUD operation workflows
└── data/
    ├── users.json         # Sample user data
    └── test_cases.json    # Predefined test cases
```

## 🔧 Advanced Usage

### Custom Templates

Create custom Jinja2 templates for specialized test generation:

```python
# custom_template.py.j2
from locust import HttpUser, task, between

class {{ class_name }}User(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup method called before tasks"""
        self.token = self.login()
    
    {% for endpoint in endpoints %}
    @task({{ endpoint.weight | default(1) }})
    def {{ endpoint.method_name }}(self):
        """{{ endpoint.description }}"""
        # Custom task implementation
        {{ endpoint.ai_generated_code | indent(8) }}
    {% endfor %}
```

### Programmatic Usage

```python
from devdox_ai_locust import HybridLocustGenerator
from devdox_ai_locust.config import Settings

# Initialize generator
config = Settings()
generator = HybridLocustGenerator(config)

# Generate from OpenAPI spec
async def generate_tests():
    files, workflows = await generator.generate_from_url(
        "https://api.example.com/openapi.json",
        output_dir="./tests",
        ai_enhanced=True
    )
    
    print(f"Generated {len(files)} test files")
    print(f"Created {len(workflows)} workflows")

# Run generation
import asyncio
asyncio.run(generate_tests())
```


## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/montymobile1/devdox-ai-locust.git
cd devdox-ai-locust

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,test,ai]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash

# Use the test runner script
coverage run -m pytest tests/
```



### Project Structure

```
src/devdox_ai_locust/
├── __init__.py
├── cli.py                      # Command line interface
├── config.py                   # Configuration management
├── locust_generator.py         # Basic test generation
├── hybrid_loctus_generator.py  # AI-enhanced generation
├── templates/                  # Jinja2 templates
│   ├── locust.py.j2
│   ├── config.py.j2
│   └── workflow.py.j2
├── schemas/                    # Data models and validation
│   ├── processing_result.py
│   └── endpoint_schema.py
├── utils/                      # Utility modules
│   ├── swagger_utils.py
│   ├── open_ai_parser.py
│   └── file_creation.py
└── prompt/                     # AI prompt templates
    ├── system_prompts.py
    └── enhancement_prompts.py
```

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** the test suite (`make test`)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to the branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Code Standards

- **Python 3.12+** with type hints
- **Black** for code formatting
- **isort** for import sorting
- **pytest** for testing
- **mypy** for type checking
- **Pre-commit hooks** for quality gates

## 📋 Requirements

### System Requirements

- **Python**: 3.12 or higher
- **Memory**: 512MB RAM minimum, 1GB recommended
- **Storage**: 100MB for installation, additional space for generated tests

### Dependencies

#### Core Dependencies
- `locust` - Load testing framework
- `click` - CLI framework
- `pyyaml` - YAML parsing
- `requests` - HTTP client
- `pydantic` - Data validation
- `jinja2` - Template engine
- `rich` - Rich text output
- `faker` - Test data generation

#### AI Dependencies
- `langchain-together` - Together AI integration
- `together` - Together AI Python client

#### Development Dependencies
- `pytest` - Testing framework
- `black` - Code formatter
- `mypy` - Type checker
- `pre-commit` - Git hooks

## 🔍 Examples

### Example 1: E-commerce API

```bash
# Generate tests for an e-commerce API
devdox_ai_locust generate \
   https://api.shop.example.com/v1/openapi.json \
  --output ./ecommerce-tests 

# The generator will create:
# - Product browsing scenarios
# - Shopping cart workflows  
# - Checkout and payment flows
# - User registration and login
# - Order management tests
```

## ❓ FAQ

**Q: What API specification formats are supported?**
A: We support OpenAPI 3.0+ and Swagger 2.0 specifications in JSON and YAML formats.

**Q: Do I need an AI API key?**
A: Yes, for AI-enhanced generation you need a Together AI API key. Basic generation works without AI.

**Q: Can I customize the generated test templates?**
A: Absolutely! You can provide custom Jinja2 templates using the `--template-dir` option.

**Q: How do I handle authentication in generated tests?**
A: The generator automatically detects authentication schemes from your OpenAPI spec and creates appropriate test flows.

**Q: Can I run tests against different environments?**
A: Yes, use environment variables or configuration files to specify different target hosts and settings.

**Q: What if my API has complex business logic?**
A: The AI enhancement feature analyzes your API patterns and generates realistic business workflows, not just individual endpoint tests.

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure proper installation
pip install -e .

# Check Python path
python -c "import devdox_ai_locust; print('OK')"
```

**API Key Issues**
```bash
# Verify API key is set
echo $API_KEY

# Test API connectivity
python -c "from together import Together; print(Together(api_key='your_key').models.list())"
```


**Template Errors**
```bash
# Validate your OpenAPI spec
# Use online validators like swagger.io/tools/swagger-editor/

# Check template syntax
# Ensure custom templates use valid Jinja2 syntax
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Locust** - The amazing load testing framework that powers our generated tests
- **Together AI** - Providing the AI capabilities for intelligent test generation  
- **OpenAPI Initiative** - For standardizing API documentation
- **Python Community** - For the excellent libraries and tools


---

**Made with ❤️ by the DevDox team**

Transform your API documentation into powerful load tests with the magic of AI! 🚀
