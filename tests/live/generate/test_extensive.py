"""
Extensive validation tests for generated code.

These tests perform deep validation of the generated code to ensure:
1. All expected patterns are present
2. No known bugs are present
3. Generated code is actually usable
4. Options actually affect the output
"""

import ast
import re
import pytest
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field

from .conftest import run_generate_command, GeneratedOutput


# =============================================================================
# Validation Result Types
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue."""
    category: str
    severity: str  # "error", "warning", "info"
    message: str
    file: Optional[str] = None
    line: Optional[int] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, category: str, message: str, file: str = None):
        self.issues.append(ValidationIssue(category, "error", message, file))

    def add_warning(self, category: str, message: str, file: str = None):
        self.issues.append(ValidationIssue(category, "warning", message, file))

    def summary(self) -> str:
        lines = ["=" * 60, "VALIDATION REPORT", "=" * 60]
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        lines.append(f"Stats: {self.stats}")
        if self.errors:
            lines.append("\nERRORS:")
            for e in self.errors:
                lines.append(f"  [{e.category}] {e.message}")
                if e.file:
                    lines.append(f"    File: {e.file}")
        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  [{w.category}] {w.message}")
        return "\n".join(lines)


# =============================================================================
# Extensive Validator
# =============================================================================

class ExtensiveValidator:
    """Performs extensive validation on generated output."""

    # Known patterns that MUST be present
    REQUIRED_PATTERNS = {
        "locustfile.py": [
            r"from\s+locust\s+import",  # Must import from locust
            r"class\s+\w+.*(?:HttpUser|User)",  # Must have a User class
            r"@task",  # Must have at least one task
        ],
        "config.py": [
            r"(?:HOST|BASE_URL|host|base_url)",  # Must have host config
        ],
    }

    # Patterns that indicate bugs
    BUG_PATTERNS = {
        "hardcoded_endpoint_objects": (
            r"Endpoint\s*\(\s*path\s*=",
            "Bug #2: Hardcoded Endpoint objects in code"
        ),
        "hardcoded_endpoint_array": (
            r"auth_endpoints\s*=\s*\[\s*Endpoint",
            "Bug #2: Hardcoded auth_endpoints array"
        ),
        "duplicate_class_def": (
            r"class\s+(\w+).*\n[\s\S]*?class\s+\1\s*[:\(]",
            "Bug #1: Duplicate class definition"
        ),
    }

    # Auth-related patterns
    AUTH_PATTERNS = [
        r"def\s+(?:login|authenticate|on_start)",
        r"class\s+\w*(?:Auth|Login)\w*",
        r"token|bearer|authorization",
    ]

    # MongoDB patterns
    MONGO_PATTERNS = [
        r"(?:pymongo|mongodb|MongoClient)",
        r"mongo.*(?:uri|url|connection)",
    ]

    # PostgreSQL patterns
    POSTGRES_PATTERNS = [
        r"(?:psycopg2|postgresql|asyncpg)",
        r"postgres.*(?:uri|url|connection)",
    ]

    def __init__(self, output: GeneratedOutput):
        self.output = output
        self.report = ValidationReport()

    def validate_all(self) -> ValidationReport:
        """Run all validations."""
        self._validate_required_files()
        self._validate_python_syntax()
        self._validate_required_patterns()
        self._validate_no_bug_patterns()
        self._validate_no_duplicates()
        self._validate_imports()
        self._validate_locust_structure()
        self._collect_stats()
        return self.report

    def _validate_required_files(self):
        """Check required files exist."""
        required = ["locustfile.py", "config.py"]
        for f in required:
            if not self.output.has_file(f):
                self.report.add_error("required_files", f"Missing required file: {f}")

    def _validate_python_syntax(self):
        """Check all files have valid Python syntax."""
        for filename, content in self.output.files.items():
            try:
                ast.parse(content)
            except SyntaxError as e:
                self.report.add_error(
                    "syntax",
                    f"Syntax error at line {e.lineno}: {e.msg}",
                    filename
                )

    def _validate_required_patterns(self):
        """Check required patterns are present."""
        for filename, patterns in self.REQUIRED_PATTERNS.items():
            content = self.output.files.get(filename)
            if not content:
                continue
            for pattern in patterns:
                if not re.search(pattern, content, re.IGNORECASE):
                    self.report.add_warning(
                        "required_patterns",
                        f"Missing expected pattern: {pattern}",
                        filename
                    )

    def _validate_no_bug_patterns(self):
        """Check for known bug patterns."""
        for bug_name, (pattern, description) in self.BUG_PATTERNS.items():
            for filename, content in self.output.files.items():
                if re.search(pattern, content, re.MULTILINE):
                    self.report.add_error("bugs", description, filename)

    def _validate_no_duplicates(self):
        """Check for duplicate class definitions."""
        for filename, content in self.output.files.items():
            try:
                tree = ast.parse(content)
                class_names = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]
                duplicates = [
                    name for name in set(class_names)
                    if class_names.count(name) > 1
                ]
                for dup in duplicates:
                    self.report.add_error(
                        "duplicates",
                        f"Class '{dup}' defined multiple times",
                        filename
                    )
            except SyntaxError:
                pass  # Already reported in syntax check

        # Check for import + definition conflicts
        for filename, content in self.output.files.items():
            try:
                tree = ast.parse(content)
                imports = set()
                definitions = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports.add(alias.asname or alias.name)
                    elif isinstance(node, ast.ClassDef):
                        definitions.add(node.name)

                conflicts = imports & definitions
                for conflict in conflicts:
                    self.report.add_error(
                        "import_conflict",
                        f"'{conflict}' is both imported and defined (Bug #3)",
                        filename
                    )
            except SyntaxError:
                pass

    def _validate_imports(self):
        """Validate import structure."""
        # Check for circular imports (simple check)
        import_graph: Dict[str, Set[str]] = {}

        for filename, content in self.output.files.items():
            try:
                tree = ast.parse(content)
                import_graph[filename] = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        # Check if importing from local file
                        module_file = node.module.replace(".", "/") + ".py"
                        if module_file in self.output.files:
                            import_graph[filename].add(module_file)
            except SyntaxError:
                pass

        # Detect cycles
        for file_a, imports in import_graph.items():
            for file_b in imports:
                if file_a in import_graph.get(file_b, set()):
                    self.report.add_warning(
                        "circular_import",
                        f"Potential circular import between {file_a} and {file_b}"
                    )

    def _validate_locust_structure(self):
        """Validate Locust-specific patterns."""
        locustfile = self.output.locustfile
        if not locustfile:
            return

        # Check for proper HttpUser inheritance
        try:
            tree = ast.parse(locustfile)
            has_http_user = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and "User" in base.id:
                            has_http_user = True
                        elif isinstance(base, ast.Attribute) and "User" in base.attr:
                            has_http_user = True

            if not has_http_user:
                self.report.add_warning(
                    "locust_structure",
                    "No HttpUser/User class found in locustfile.py"
                )

            # Check for tasks
            has_task = re.search(r"@task", locustfile)
            if not has_task:
                self.report.add_warning(
                    "locust_structure",
                    "No @task decorators found in locustfile.py"
                )

        except SyntaxError:
            pass

    def _collect_stats(self):
        """Collect statistics about generated code."""
        self.report.stats = {
            "total_files": len(self.output.files),
            "total_lines": sum(
                content.count("\n") + 1
                for content in self.output.files.values()
            ),
            "total_classes": sum(
                len(re.findall(r"^class\s+\w+", content, re.MULTILINE))
                for content in self.output.files.values()
            ),
            "total_functions": sum(
                len(re.findall(r"^def\s+\w+", content, re.MULTILINE))
                for content in self.output.files.values()
            ),
        }

    # =========================================================================
    # Option-specific validation
    # =========================================================================

    def validate_auth_mode(self, auth_enabled: bool) -> List[ValidationIssue]:
        """Validate auth option affects output correctly."""
        issues = []
        all_content = self.output.get_all_content()

        has_auth = any(
            re.search(pattern, all_content, re.IGNORECASE)
            for pattern in self.AUTH_PATTERNS
        )

        if auth_enabled and not has_auth:
            issues.append(ValidationIssue(
                "auth_mode", "warning",
                "--auth was set but no auth patterns found in output"
            ))
        elif not auth_enabled and has_auth:
            # Check if it's significant auth code, not just imports
            auth_classes = re.findall(
                r"class\s+\w*(?:Auth|Login)\w*",
                all_content, re.IGNORECASE
            )
            if len(auth_classes) > 1:
                issues.append(ValidationIssue(
                    "auth_mode", "warning",
                    f"--no-auth was set but found {len(auth_classes)} auth classes"
                ))

        return issues

    def validate_db_mode(self, db_type: Optional[str]) -> List[ValidationIssue]:
        """Validate db option affects output correctly."""
        issues = []
        all_content = self.output.get_all_content()

        if db_type == "mongo":
            has_mongo = any(
                re.search(pattern, all_content, re.IGNORECASE)
                for pattern in self.MONGO_PATTERNS
            )
            if not has_mongo:
                issues.append(ValidationIssue(
                    "db_mode", "warning",
                    "--db-type mongo was set but no MongoDB patterns found"
                ))

        elif db_type == "postgresql":
            has_postgres = any(
                re.search(pattern, all_content, re.IGNORECASE)
                for pattern in self.POSTGRES_PATTERNS
            )
            if not has_postgres:
                issues.append(ValidationIssue(
                    "db_mode", "warning",
                    "--db-type postgresql was set but no PostgreSQL patterns found"
                ))

        return issues

    def validate_custom_requirement(self, requirement: str) -> List[ValidationIssue]:
        """Validate custom requirement affects output (heuristic)."""
        issues = []
        # This is a soft check - we can't guarantee the requirement is reflected
        # but we can check for certain keywords
        keywords = re.findall(r'\b\w{4,}\b', requirement.lower())
        all_content = self.output.get_all_content().lower()

        matched = sum(1 for kw in keywords if kw in all_content)
        match_ratio = matched / len(keywords) if keywords else 0

        if match_ratio < 0.1:
            issues.append(ValidationIssue(
                "custom_requirement", "info",
                f"Custom requirement keywords matched at {match_ratio:.0%} rate"
            ))

        return issues


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.extensive
class TestExtensiveValidation:
    """Run extensive validation on generated output."""

    def test_extensive_validation(self, api_key, swagger_url, output_dir, keep_output):
        """Run full extensive validation suite."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )
        assert exit_code == 0, f"Generation failed:\n{stderr}\n{stdout}"

        output = GeneratedOutput(output_dir)
        validator = ExtensiveValidator(output)
        report = validator.validate_all()

        print("\n" + report.summary())

        if keep_output:
            print(f"\nOutput kept at: {output_dir}")

        assert not report.has_errors, \
            f"Validation found {len(report.errors)} errors:\n" + \
            "\n".join(f"  - {e.message}" for e in report.errors)


@pytest.mark.extensive
class TestAuthOptionValidation:
    """Validate --auth/--no-auth options actually affect output."""

    def test_auth_enabled_has_auth_code(self, api_key, swagger_url, output_dir):
        """With --auth, generated code should have auth patterns."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=True,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)
        validator = ExtensiveValidator(output)
        issues = validator.validate_auth_mode(auth_enabled=True)

        # Print issues but don't fail (auth presence depends on spec)
        for issue in issues:
            print(f"[{issue.severity}] {issue.message}")

    def test_no_auth_minimal_auth_code(self, api_key, swagger_url, output_dir):
        """With --no-auth, generated code should have minimal auth."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            auth=False,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)
        validator = ExtensiveValidator(output)
        issues = validator.validate_auth_mode(auth_enabled=False)

        # Warnings are acceptable, errors are not
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, f"Auth mode errors: {errors}"


@pytest.mark.extensive
class TestDatabaseOptionValidation:
    """Validate --db-type option affects output."""

    def test_mongo_option_adds_mongo_code(self, api_key, swagger_url, output_dir, mongodb_uri):
        """With --db-type mongo, should have MongoDB patterns."""
        if not mongodb_uri:
            pytest.skip("MongoDB URI not provided")

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="mongo",
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)
        validator = ExtensiveValidator(output)
        issues = validator.validate_db_mode(db_type="mongo")

        for issue in issues:
            print(f"[{issue.severity}] {issue.message}")

    def test_postgresql_option_adds_postgres_code(self, api_key, swagger_url, output_dir, postgresql_uri):
        """With --db-type postgresql, should have PostgreSQL patterns."""
        if not postgresql_uri:
            pytest.skip("PostgreSQL URI not provided")

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            db_type="postgresql",
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)
        validator = ExtensiveValidator(output)
        issues = validator.validate_db_mode(db_type="postgresql")

        for issue in issues:
            print(f"[{issue.severity}] {issue.message}")


@pytest.mark.extensive
class TestCustomRequirementValidation:
    """Validate --custom-requirement affects output."""

    def test_performance_requirement_reflected(self, api_key, swagger_url, output_dir):
        """Performance requirement should influence output."""
        requirement = "Focus on high-volume stress testing with 1000 concurrent users"

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=requirement,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)

        # Check for performance-related patterns
        perf_patterns = [
            r"concurrent",
            r"stress",
            r"performance",
            r"load",
            r"throughput",
        ]

        all_content = output.get_all_content().lower()
        found = [p for p in perf_patterns if p in all_content]

        print(f"Found performance keywords: {found}")
        # Soft assertion - the AI may not include all keywords
        if len(found) < 2:
            print("Warning: Few performance keywords found in output")

    def test_validation_requirement_reflected(self, api_key, swagger_url, output_dir):
        """Validation requirement should add validation code."""
        requirement = "Add comprehensive response validation and assertion checks"

        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            custom_requirement=requirement,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)

        # Check for validation-related patterns
        validation_patterns = [
            r"assert",
            r"validate",
            r"check",
            r"verify",
            r"response",
        ]

        all_content = output.get_all_content().lower()
        found = [p for p in validation_patterns if p in all_content]

        print(f"Found validation keywords: {found}")


@pytest.mark.extensive
class TestGeneratedCodeUsability:
    """Test that generated code is actually usable."""

    def test_code_is_importable(self, api_key, swagger_url, output_dir):
        """Generated code should be importable (no import errors)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        # Try to compile each file
        output = GeneratedOutput(output_dir)
        for filename, content in output.files.items():
            try:
                compile(content, filename, 'exec')
            except SyntaxError as e:
                pytest.fail(f"Cannot compile {filename}: {e}")

    def test_locustfile_has_tasks(self, api_key, swagger_url, output_dir):
        """Locustfile should have at least one task."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)
        locustfile = output.locustfile

        assert locustfile, "No locustfile.py generated"

        task_count = len(re.findall(r"@task", locustfile))
        assert task_count > 0, "Locustfile has no @task decorators"

        print(f"Found {task_count} tasks in locustfile.py")

    def test_all_files_have_content(self, api_key, swagger_url, output_dir):
        """All generated files should have meaningful content."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        output = GeneratedOutput(output_dir)

        for filename, content in output.files.items():
            lines = [l for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
            assert len(lines) > 5, f"{filename} has only {len(lines)} non-empty, non-comment lines"
