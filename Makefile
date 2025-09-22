.PHONY: help test test-unit test-integration test-cov test-cov-report test-fast test-slow lint format clean install dev-install pre-release build-check publish-test publish-prod release-full inspect-build bump-version

# Default target
help:
	@echo "Available targets:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  test-cov-report - Run tests with coverage and open HTML report"
	@echo "  test-fast      - Run fast tests only (exclude slow tests)"
	@echo "  test-slow      - Run slow tests only"
	@echo "  lint           - Run linting (flake8, mypy)"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean up generated files"
	@echo "  install        - Install package in development mode"
	@echo "  dev-install    - Install package with development dependencies"
	@echo ""
	@echo "Release targets:"
	@echo "  bump-version   - Show current version and bump instructions"
	@echo "  pre-release    - Run all quality checks before release"
	@echo "  build-check    - Build package and validate"
	@echo "  publish-test   - Publish to TestPyPI"
	@echo "  publish-prod   - Publish to production PyPI"
	@echo "  release-full   - Complete release workflow (test first)"
	@echo "  inspect-build  - Show contents of built packages"

# Test targets
test:
	pytest

test-unit:
	pytest -m "unit or not integration"

test-integration:
	pytest -m integration

test-cov:
	pytest --cov=src/devdox_ai_locust --cov-report=term-missing --cov-report=html

test-cov-report: test-cov
	@echo "Opening coverage report..."
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')"

test-fast:
	pytest -m "not slow"

test-slow:
	pytest -m slow

test-ai:
	pytest -m ai

# Code quality targets
lint:
	ruff check src tests
	mypy --explicit-package-bases src

format:
	black src tests
	ruff check --fix src tests  # Auto-fix what can be fixed
	ruff format src tests       # Format imports

# Utility targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,test,ai]"

# CI/CD targets
ci-test: lint test-cov
	@echo "All CI tests passed!"

# Docker targets (if needed)
docker-test:
	docker build -t devdox-ai-locust-test .
	docker run --rm devdox-ai-locust-test make test

# Documentation targets
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Release targets
build:
	python -m build

# Enhanced release workflow
bump-version:
	@echo "Current version: $(shell python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
	@echo "Update version in pyproject.toml manually, then run 'make release-full'"

# Pre-release validation
pre-release: clean format lint test-cov security-check
	@echo "✅ All pre-release checks passed!"
	@echo "Ready to build and publish"

# Build with validation
build-check: build
	twine check dist/*
	@echo "✅ Package built and validated successfully"
	@ls -la dist/

# Test publication
publish-test: build-check
	@echo "📦 Uploading to TestPyPI..."
	twine upload --repository testpypi dist/*
	@echo "✅ Published to TestPyPI: https://test.pypi.org/project/devdox-ai-locust/"

# Production publication
publish-prod: build-check
	@echo "🚀 Uploading to PyPI..."
	twine upload --repository pypi dist/*
	@echo "✅ Published to PyPI: https://pypi.org/project/devdox-ai-locust/"

# Full release workflow
release-full: pre-release publish-test
	@echo "✅ Package published to TestPyPI successfully!"
	@echo "Test the installation from TestPyPI, then run 'make publish-prod'"
	@echo ""
	@echo "Test command:"
	@echo "pip install --index-url https://test.pypi.org/simple/ devdox-ai-locust"

# Check what's in the built package
inspect-build:
	@echo "📦 Contents of wheel file:"
	python -m zipfile -l dist/*.whl
	@echo ""
	@echo "📦 Contents of source distribution:"
	tar -tzf dist/*.tar.gz

# Legacy targets (for compatibility)
release-test: publish-test

release: publish-prod

# Security check
security-check:
	safety check
	bandit -r src/

# Dependency check
deps-check:
	pip-audit

# Full check (everything)
check-all: format lint security-check test-cov
	@echo "All checks passed!"

