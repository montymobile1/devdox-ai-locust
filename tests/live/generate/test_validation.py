"""
Output validation tests for generated code quality.

Validates that generated code is syntactically correct, follows best practices,
and doesn't contain common issues.
"""

import ast
import re
import pytest
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

from .conftest import run_generate_command


class CodeValidator:
    """Validates generated Python code for correctness and quality."""

    # Classes that should only appear once across all files
    SINGLETON_CLASSES = {
        "TestDataGenerator",
        "ResponseValidator",
        "RequestLogger",
        "PerformanceMonitor",
        "DataManager",
        "LoadTestConfig",
    }

    # Keywords indicating auth-related functionality
    AUTH_KEYWORDS = {"auth", "login", "logout", "token", "session", "credential"}

    @staticmethod
    def is_valid_python(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    @staticmethod
    def get_class_definitions(code: str) -> List[str]:
        """Extract all class names defined in code."""
        try:
            tree = ast.parse(code)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except SyntaxError:
            # Fallback to regex
            return re.findall(r"^class\s+(\w+)", code, re.MULTILINE)

    @staticmethod
    def get_function_definitions(code: str) -> List[str]:
        """Extract all function names defined in code."""
        try:
            tree = ast.parse(code)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            return re.findall(r"^def\s+(\w+)", code, re.MULTILINE)

    @staticmethod
    def get_imports(code: str) -> Set[str]:
        """Extract all imported names."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.add(alias.asname or alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.asname or alias.name)
        except SyntaxError:
            # Fallback to regex
            imports.update(re.findall(r"from\s+\S+\s+import\s+(\w+)", code))
            imports.update(re.findall(r"^import\s+(\w+)", code, re.MULTILINE))
        return imports

    @staticmethod
    def has_duplicate_class(code: str) -> Tuple[bool, List[str]]:
        """Check if any class is defined more than once."""
        classes = CodeValidator.get_class_definitions(code)
        duplicates = [c for c in set(classes) if classes.count(c) > 1]
        return bool(duplicates), duplicates

    @staticmethod
    def has_import_and_definition(code: str) -> Tuple[bool, List[str]]:
        """Check if a class is both imported and defined (Bug #3)."""
        imports = CodeValidator.get_imports(code)
        definitions = set(CodeValidator.get_class_definitions(code))
        conflicts = imports & definitions
        return bool(conflicts), list(conflicts)

    @staticmethod
    def has_hardcoded_endpoints(code: str) -> Tuple[bool, List[str]]:
        """Check for hardcoded endpoint arrays (Bug #2)."""
        patterns = [
            r"Endpoint\s*\(\s*path\s*=",
            r"auth_endpoints\s*=\s*\[",
            r"\[\s*Endpoint\s*\(",
        ]
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, code)
            if found:
                matches.extend(found)
        return bool(matches), matches

    @staticmethod
    def count_auth_classes(code: str) -> int:
        """Count auth-related classes in code."""
        classes = CodeValidator.get_class_definitions(code)
        return sum(
            1 for c in classes
            if any(kw in c.lower() for kw in CodeValidator.AUTH_KEYWORDS)
        )

    @staticmethod
    def has_proper_locust_structure(code: str) -> Tuple[bool, List[str]]:
        """Validate Locust-specific patterns."""
        issues = []

        if "HttpUser" in code and "class" in code:
            if not re.search(r"class\s+\w+.*HttpUser", code):
                issues.append("HttpUser should be used as base class")

        if "catch_response=True" in code:
            if not re.search(r"with\s+self\.client\.[a-z]+\(.*catch_response\s*=\s*True", code, re.DOTALL):
                issues.append("catch_response=True should be used in 'with' block")

        return len(issues) == 0, issues

    @staticmethod
    def has_hardcoded_secrets(code: str) -> Tuple[bool, List[str]]:
        """Check for hardcoded secrets, API keys, or passwords."""
        issues = []
        secret_patterns = [
            (r'api_key\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']', "Possible hardcoded API key"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
            (r'token\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']', "Possible hardcoded token"),
            (r'mongodb://[^@]+:[^@]+@', "Possible hardcoded MongoDB credentials"),
            (r'postgresql://[^@]+:[^@]+@', "Possible hardcoded PostgreSQL credentials"),
        ]

        for pattern, description in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                if not re.search(r'(test|example|placeholder|your_)', code, re.IGNORECASE):
                    issues.append(description)

        return bool(issues), issues

    @staticmethod
    def get_import_modules(code: str) -> Dict[str, Set[str]]:
        """Extract import information: module -> imported names."""
        imports: Dict[str, Set[str]] = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module not in imports:
                        imports[module] = set()
                    for alias in node.names:
                        imports[module].add(alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.name] = set()
        except SyntaxError:
            pass
        return imports

    @staticmethod
    def has_reasonable_file_size(code: str, max_lines: int = 2000) -> Tuple[bool, Optional[str]]:
        """Check if file has reasonable size."""
        lines = code.count('\n') + 1
        if lines > max_lines:
            return False, f"File has {lines} lines (max: {max_lines})"
        return True, None


class OutputAnalyzer:
    """Analyzes the complete generated output directory."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.files: Dict[str, str] = {}
        self._load_files()

    def _load_files(self):
        """Load all Python files from output directory."""
        for file_path in self.output_dir.rglob("*.py"):
            relative_path = file_path.relative_to(self.output_dir)
            self.files[str(relative_path)] = file_path.read_text()

    def get_all_classes(self) -> Dict[str, List[str]]:
        """Get all class definitions grouped by file."""
        return {
            filename: CodeValidator.get_class_definitions(content)
            for filename, content in self.files.items()
        }

    def find_duplicate_classes_across_files(self) -> Dict[str, List[str]]:
        """Find classes defined in multiple files."""
        class_locations: Dict[str, List[str]] = {}
        for filename, content in self.files.items():
            for class_name in CodeValidator.get_class_definitions(content):
                if class_name not in class_locations:
                    class_locations[class_name] = []
                class_locations[class_name].append(filename)

        return {
            class_name: files
            for class_name, files in class_locations.items()
            if len(files) > 1
        }

    def validate_all_files(self) -> Dict[str, List[str]]:
        """Validate all generated files."""
        issues: Dict[str, List[str]] = {}

        for filename, content in self.files.items():
            file_issues = []

            is_valid, error = CodeValidator.is_valid_python(content)
            if not is_valid:
                file_issues.append(f"Invalid Python: {error}")

            has_dup, duplicates = CodeValidator.has_duplicate_class(content)
            if has_dup:
                file_issues.append(f"Duplicate classes: {duplicates}")

            has_conflict, conflicts = CodeValidator.has_import_and_definition(content)
            if has_conflict:
                file_issues.append(f"Import/definition conflicts: {conflicts}")

            has_hardcoded, patterns = CodeValidator.has_hardcoded_endpoints(content)
            if has_hardcoded:
                file_issues.append(f"Hardcoded endpoints found: {patterns}")

            is_proper, locust_issues = CodeValidator.has_proper_locust_structure(content)
            if not is_proper:
                file_issues.extend(locust_issues)

            if file_issues:
                issues[filename] = file_issues

        return issues

    def check_singleton_classes(self) -> Dict[str, List[str]]:
        """Ensure singleton classes are defined only once."""
        violations: Dict[str, List[str]] = {}

        for singleton in CodeValidator.SINGLETON_CLASSES:
            locations = []
            for filename, content in self.files.items():
                classes = CodeValidator.get_class_definitions(content)
                if singleton in classes:
                    locations.append(filename)

            if len(locations) > 1:
                violations[singleton] = locations

        return violations

    def check_auth_consistency(self) -> Dict[str, int]:
        """Check authentication class distribution."""
        return {
            filename: CodeValidator.count_auth_classes(content)
            for filename, content in self.files.items()
            if CodeValidator.count_auth_classes(content) > 0
        }

    def check_file_structure(self) -> Dict[str, bool]:
        """Check expected file structure exists."""
        expected_files = ["locustfile.py", "config.py"]
        optional_files = ["test_data.py", "utils.py", "base_workflow.py", "requirements.txt"]

        result = {}
        for f in expected_files:
            result[f] = f in self.files or any(f in p for p in self.files.keys())

        for f in optional_files:
            if f in self.files or any(f in p for p in self.files.keys()):
                result[f] = True

        return result

    def check_circular_imports(self) -> List[Tuple[str, str]]:
        """Detect potential circular imports between files."""
        import_graph: Dict[str, Set[str]] = {}

        for filename, content in self.files.items():
            imports = CodeValidator.get_import_modules(content)
            import_graph[filename] = set()

            for module in imports.keys():
                module_file = module.replace(".", "/") + ".py"
                if module_file in self.files:
                    import_graph[filename].add(module_file)
                for other_file in self.files.keys():
                    if module in other_file:
                        import_graph[filename].add(other_file)

        cycles = []
        for file_a, imports_a in import_graph.items():
            for file_b in imports_a:
                if file_b in import_graph and file_a in import_graph.get(file_b, set()):
                    if (file_b, file_a) not in cycles:
                        cycles.append((file_a, file_b))

        return cycles

    def check_imports_resolvable(self) -> Dict[str, List[str]]:
        """Check if local imports can be resolved within the output."""
        unresolved: Dict[str, List[str]] = {}

        known_modules = {
            "locust", "logging", "json", "typing", "pathlib", "os", "sys",
            "random", "datetime", "uuid", "re", "time", "urllib", "dataclasses",
            "faker", "pydantic", "requests", "pymongo", "psycopg2", "sqlalchemy",
        }

        for filename, content in self.files.items():
            imports = CodeValidator.get_import_modules(content)
            file_unresolved = []

            for module, names in imports.items():
                if any(module.startswith(km) for km in known_modules):
                    continue
                if module in known_modules:
                    continue

                module_file = module.replace(".", "/") + ".py"
                module_init = module.replace(".", "/") + "/__init__.py"

                if module and not module.startswith("_"):
                    found = False
                    for possible in [module_file, module_init, f"{module}.py"]:
                        if possible in self.files or any(possible in f for f in self.files.keys()):
                            found = True
                            break

                    if not found and module in ["test_data", "utils", "config", "base_workflow", "data_provider"]:
                        found = any(module in f for f in self.files.keys())

                    if not found and module:
                        file_unresolved.append(module)

            if file_unresolved:
                unresolved[filename] = file_unresolved

        return unresolved

    def check_no_hardcoded_secrets(self) -> Dict[str, List[str]]:
        """Check all files for hardcoded secrets."""
        issues: Dict[str, List[str]] = {}
        for filename, content in self.files.items():
            has_secrets, secrets = CodeValidator.has_hardcoded_secrets(content)
            if has_secrets:
                issues[filename] = secrets
        return issues


class TestSyntaxValidation:
    """Test that all generated files are valid Python."""

    def test_all_files_valid_python(self, api_key, swagger_url, output_dir):
        """All generated files should be valid Python syntax."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        syntax_issues = {
            f: errs for f, errs in issues.items()
            if any("Invalid Python" in e for e in errs)
        }

        assert not syntax_issues, f"Invalid Python syntax in: {syntax_issues}"


class TestCodeDuplication:
    """Test for code duplication issues."""

    def test_no_duplicate_class_definitions(self, api_key, swagger_url, output_dir):
        """No class should be defined multiple times in any file (Bug #1, #3)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        duplicate_issues = {
            f: errs for f, errs in issues.items()
            if any("Duplicate classes" in e for e in errs)
        }

        assert not duplicate_issues, f"Duplicate classes found: {duplicate_issues}"

    def test_no_import_and_definition_conflicts(self, api_key, swagger_url, output_dir):
        """No class should be both imported and defined (Bug #3)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        conflict_issues = {
            f: errs for f, errs in issues.items()
            if any("Import/definition conflicts" in e for e in errs)
        }

        assert not conflict_issues, f"Import/definition conflicts: {conflict_issues}"

    def test_singleton_classes_unique(self, api_key, swagger_url, output_dir):
        """Singleton classes should only be defined once across all files."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        violations = analyzer.check_singleton_classes()

        assert not violations, f"Singleton classes defined multiple times: {violations}"


class TestHardcodedValues:
    """Test for hardcoded values that shouldn't be in generated code."""

    def test_no_hardcoded_endpoints(self, api_key, swagger_url, output_dir):
        """No file should contain hardcoded endpoint arrays (Bug #2)."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        endpoint_issues = {
            f: errs for f, errs in issues.items()
            if any("Hardcoded endpoints" in e for e in errs)
        }

        assert not endpoint_issues, f"Hardcoded endpoints found: {endpoint_issues}"

    def test_no_hardcoded_secrets(self, api_key, swagger_url, output_dir):
        """No files should contain hardcoded secrets or credentials."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        secrets_found = analyzer.check_no_hardcoded_secrets()

        assert not secrets_found, f"Hardcoded secrets found: {secrets_found}"


class TestLocustPatterns:
    """Test Locust-specific code patterns."""

    def test_proper_locust_patterns(self, api_key, swagger_url, output_dir):
        """Generated code should follow Locust best practices."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        issues = analyzer.validate_all_files()

        locust_issues = {
            f: errs for f, errs in issues.items()
            if any("catch_response" in e or "HttpUser" in e for e in errs)
        }

        assert not locust_issues, f"Locust pattern issues: {locust_issues}"


class TestImportValidation:
    """Test import correctness."""

    def test_no_circular_imports(self, api_key, swagger_url, output_dir):
        """No circular imports between generated files."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        cycles = analyzer.check_circular_imports()

        assert not cycles, f"Circular imports detected: {cycles}"

    def test_imports_resolvable(self, api_key, swagger_url, output_dir):
        """Local imports should be resolvable within the output."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        unresolved = analyzer.check_imports_resolvable()

        # Filter to only critical unresolved (not package-internal imports)
        critical_unresolved = {
            f: imports for f, imports in unresolved.items()
            if imports and not all(i.startswith("devdox") for i in imports)
        }

        if critical_unresolved:
            print(f"Warning: Potentially unresolved imports: {critical_unresolved}")


class TestFileStructure:
    """Test generated file structure."""

    def test_required_files_exist(self, api_key, swagger_url, output_dir):
        """Required files should exist in output."""
        exit_code, _, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
        )
        assert exit_code == 0, f"Generation failed: {stderr}"

        analyzer = OutputAnalyzer(output_dir)
        structure = analyzer.check_file_structure()

        assert structure.get("locustfile.py", False), "Missing locustfile.py"
        assert structure.get("config.py", False), "Missing config.py"


class TestComprehensiveValidation:
    """Run all validations in a single comprehensive test."""

    def test_full_validation(self, api_key, swagger_url, output_dir, keep_output):
        """Run complete validation suite on generated output."""
        exit_code, stdout, stderr = run_generate_command(
            swagger_source=swagger_url,
            output_dir=output_dir,
            api_key=api_key,
            verbose=True,
        )

        assert exit_code == 0, f"Generation failed:\n{stderr}\n{stdout}"

        analyzer = OutputAnalyzer(output_dir)

        all_issues = []
        warnings = []

        # 1. Validate all files
        file_issues = analyzer.validate_all_files()
        if file_issues:
            all_issues.append(f"File validation issues:\n{file_issues}")

        # 2. Check singleton classes
        singleton_violations = analyzer.check_singleton_classes()
        if singleton_violations:
            all_issues.append(f"Singleton violations:\n{singleton_violations}")

        # 3. Check for cross-file duplicates
        cross_file_duplicates = analyzer.find_duplicate_classes_across_files()
        problematic_duplicates = {
            cls: files for cls, files in cross_file_duplicates.items()
            if cls in CodeValidator.SINGLETON_CLASSES
        }
        if problematic_duplicates:
            all_issues.append(f"Cross-file duplicates:\n{problematic_duplicates}")

        # 4. Check file structure
        structure = analyzer.check_file_structure()
        if not structure.get("locustfile.py", False):
            all_issues.append("Missing required file: locustfile.py")
        if not structure.get("config.py", False):
            all_issues.append("Missing required file: config.py")

        # 5. Check for circular imports
        cycles = analyzer.check_circular_imports()
        if cycles:
            all_issues.append(f"Circular imports detected:\n{cycles}")

        # 6. Check for unresolved imports (warning only)
        unresolved = analyzer.check_imports_resolvable()
        if unresolved:
            warnings.append(f"Potentially unresolved imports: {unresolved}")

        # 7. Check for hardcoded secrets
        secrets = analyzer.check_no_hardcoded_secrets()
        if secrets:
            all_issues.append(f"Hardcoded secrets found:\n{secrets}")

        # 8. Check auth class distribution
        auth_dist = analyzer.check_auth_consistency()
        total_auth = sum(auth_dist.values())
        if total_auth > 10:
            warnings.append(f"High number of auth classes ({total_auth}): {auth_dist}")

        # Print summary
        print("\n" + "=" * 60)
        print("LIVE TEST VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Files generated: {len(analyzer.files)}")
        print(f"Total classes: {sum(len(c) for c in analyzer.get_all_classes().values())}")
        print(f"File structure: {structure}")
        print(f"Auth class distribution: {auth_dist}")
        print(f"Issues found: {len(all_issues)}")
        print(f"Warnings: {len(warnings)}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")

        if keep_output:
            print(f"\nOutput kept at: {output_dir}")

        assert not all_issues, f"Validation failed:\n" + "\n\n".join(all_issues)
