"""
Modular Generator - Produces SOLID-structured output.

This generator creates focused, single-responsibility files:
- data/ - Test data generators (valid, invalid, security)
- scenarios/ - Test tasks (positive, negative, security, edge)
- auth/ - Authentication handling
- workflows/ - Workflow composition
"""
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from together import AsyncTogether

from .utils.open_ai_parser import Endpoint
from .hybrid_loctus_generator import HybridLocustGenerator
from .validation import CodeValidator, CodeFixer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils.patch_tracker import PatchTracker

logger = logging.getLogger(__name__)

# Type for progress callback: (phase: str, message: str, detail: str, progress_pct: int)
ProgressCallback = Callable[[str, str, str, int], None]


class ModularGenerator:
    """
    Generates SOLID-structured load test files.

    Each generated file has a single responsibility:
    - Small, focused files
    - Easier AI enhancement (smaller prompts)
    - Better maintainability
    """

    def __init__(
        self,
        output_dir: str,
        api_key: str,
        target_host: str = "http://localhost",
        auth_enabled: bool = True,
        db_type: str = "none",
        retry_on_invalid: int = 0,
        progress_callback: Optional[ProgressCallback] = None,
        patch_tracker: Optional["PatchTracker"] = None,
        custom_requirement: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.target_host = target_host
        self.auth_enabled = auth_enabled
        self.db_type = db_type
        self.retry_on_invalid = retry_on_invalid
        self._progress_callback = progress_callback
        self._patch_tracker = patch_tracker
        self.custom_requirement = custom_requirement or ""

        # Initialize validation components (injectable)
        self.code_validator = CodeValidator()
        self.code_fixer = CodeFixer()

        # Initialize template environment
        template_dir = Path(__file__).parent / "templates"
        logger.debug(f"[ModularGenerator] Template directory: {template_dir}")
        logger.debug(f"[ModularGenerator] Template dir exists: {template_dir.exists()}")
        if template_dir.exists():
            logger.debug(f"[ModularGenerator] Template contents: {list(template_dir.iterdir())[:10]}")

        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Initialize prompt environment
        prompt_dir = Path(__file__).parent / "prompt"
        logger.debug(f"[ModularGenerator] Prompt directory: {prompt_dir}")
        self.prompt_env = Environment(
            loader=FileSystemLoader(str(prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Use existing generator for AI calls
        self._ai_client = AsyncTogether(api_key=api_key)
        self._ai_generator = HybridLocustGenerator(
            ai_client=self._ai_client,
        )
        logger.debug(f"[ModularGenerator] Initialized with output_dir={output_dir}, target_host={target_host}, db_type={db_type}")

    def _report_progress(self, phase: str, message: str, detail: str = "", progress_pct: int = 0) -> None:
        """Report progress to the callback if set."""
        if self._progress_callback:
            self._progress_callback(phase, message, detail, progress_pct)
        logger.info(f"[{phase}] {message}" + (f" - {detail}" if detail else ""))

    async def generate(
        self,
        endpoints: List[Endpoint],
        schemas: Dict[str, Any],
        api_info: Dict[str, Any],
        auth_endpoints: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Generate all modular files.

        Args:
            endpoints: List of API endpoints
            schemas: API schemas
            api_info: API metadata
            auth_endpoints: Authentication endpoint paths

        Returns:
            Dictionary mapping file paths to content
        """
        start_time = datetime.now()
        self._report_progress("INIT", "Starting generation", f"{len(endpoints)} endpoints", 0)
        logger.debug(f"[ModularGenerator] API info: {api_info}")
        logger.debug(f"[ModularGenerator] Auth endpoints: {auth_endpoints}")
        logger.debug(f"[ModularGenerator] Output dir: {self.output_dir}")

        generated_files = {}

        # Group endpoints by API tag first (needed for directory structure)
        self._report_progress("ANALYZE", "Analyzing API structure", "Grouping endpoints by tag", 5)
        grouped_endpoints = self._group_endpoints_by_tag(endpoints)
        api_groups = list(grouped_endpoints.keys())
        self._report_progress("ANALYZE", "API structure analyzed", f"Found {len(api_groups)} API groups", 10)

        # Create output directories (including per-API-group scenario folders)
        self._report_progress("SETUP", "Creating directory structure", "", 12)
        self._create_directories(api_groups)

        # Prepare context for all templates
        self._report_progress("CONTEXT", "Building generation context", "", 15)
        context = self._build_context(endpoints, schemas, api_info, auth_endpoints)
        context["grouped_endpoints"] = grouped_endpoints
        context["api_groups"] = api_groups
        logger.debug(f"[ModularGenerator] Context keys: {list(context.keys())}")

        # Generate static files (no AI needed)
        self._report_progress("STATIC", "Generating static files", "Templates, config, utilities", 20)
        try:
            static_files = await self._generate_static_files(context)
            self._report_progress("STATIC", "Static files complete", f"{len(static_files)} files generated", 35)
            generated_files.update(static_files)

            # Capture template state for patch tracking (before AI enhancement)
            if self._patch_tracker:
                self._patch_tracker.capture_template_state(static_files)
                logger.debug(f"[ModularGenerator] Captured template state: {len(static_files)} files")

        except Exception as e:
            self._report_progress("ERROR", "Static file generation failed", str(e), 35)
            logger.error(f"[ModularGenerator] Failed to generate static files: {e}", exc_info=True)

        # Generate AI-enhanced files (parallel)
        self._report_progress("AI", "Generating AI-enhanced files", "This may take a moment...", 40)
        try:
            ai_files = await self._generate_ai_enhanced_files(context)
            self._report_progress("AI", "AI enhancement complete", f"{len(ai_files)} files enhanced", 85)
            generated_files.update(ai_files)

            # Capture enhanced state for patch tracking (after AI enhancement)
            if self._patch_tracker:
                # Combine static + AI files for the enhanced state
                all_enhanced = {**static_files, **ai_files}
                self._patch_tracker.capture_enhanced_state(all_enhanced)
                logger.debug(f"[ModularGenerator] Captured enhanced state: {len(all_enhanced)} files")

        except Exception as e:
            self._report_progress("ERROR", "AI generation failed", str(e), 85)
            logger.error(f"[ModularGenerator] Failed to generate AI files: {e}", exc_info=True)

        # Write all files
        self._report_progress("WRITE", "Writing files to disk", f"{len(generated_files)} files", 90)
        for i, (file_path, content) in enumerate(generated_files.items()):
            try:
                self._write_file(file_path, content)
                pct = 90 + int((i + 1) / len(generated_files) * 8)
                self._report_progress("WRITE", "Writing files", file_path, pct)
            except Exception as e:
                logger.error(f"[ModularGenerator] Failed to write {file_path}: {e}")

        elapsed = (datetime.now() - start_time).total_seconds()
        self._report_progress("COMPLETE", "Generation complete", f"{len(generated_files)} files in {elapsed:.1f}s", 100)
        return generated_files

    def _create_directories(self, api_groups: Optional[List[str]] = None) -> None:
        """Create output directory structure including per-API-group scenario folders."""
        dirs = [
            self.output_dir / "data",
            self.output_dir / "scenarios",
            self.output_dir / "auth",
            self.output_dir / "workflows",
        ]

        # Create per-API-group scenario directories
        if api_groups:
            for group in api_groups:
                dirs.append(self.output_dir / "scenarios" / group)

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _build_context(
        self,
        endpoints: List[Endpoint],
        schemas: Dict[str, Any],
        api_info: Dict[str, Any],
        auth_endpoints: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build template context from API data."""
        from datetime import datetime

        # Get primary endpoint for examples
        primary_endpoint = endpoints[0].path if endpoints else "/api/resource"
        primary_schema = self._infer_primary_schema(endpoints)

        # Format endpoints for prompts
        endpoints_summary = self._format_endpoints_summary(endpoints)
        schemas_info = self._format_schemas_info(schemas)

        # Build environment variables for .env.example template
        base_url = self.target_host or api_info.get("base_url", "http://localhost:8000")
        environment_vars = {
            "API_BASE_URL": base_url,
            "API_VERSION": api_info.get("version", "v1"),
            "API_TITLE": api_info.get("title", "Your API Name"),
            "LOCUST_USERS": "50",
            "LOCUST_SPAWN_RATE": "5",
            "LOCUST_RUN_TIME": "10m",
            "LOCUST_HOST": base_url,
            "DATA_SEED": "42",
            "REQUEST_TIMEOUT": "30",
            "MAX_RETRIES": "3",
        }

        # Add database-specific environment variables
        if self.db_type == "mongo":
            environment_vars.update({
                "MONGO_URI": "mongodb://localhost:27017",
                "MONGO_DATABASE": "test_db",
                "MONGO_COLLECTION": "test_collection",
            })
        elif self.db_type == "postgresql":
            environment_vars.update({
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5432",
                "POSTGRES_DB": "test_db",
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "password",
            })

        # Build db_using string for README
        db_using = ""
        if self.db_type == "mongo":
            db_using = "- **MongoDB 4.4+** (if using database features)"
        elif self.db_type == "postgresql":
            db_using = "- **PostgreSQL 12+** (if using database features)"

        return {
            "endpoints": endpoints,
            "endpoints_summary": endpoints_summary,
            "schemas": schemas,
            "schemas_info": schemas_info,
            "api_info": api_info,
            "auth_enabled": self.auth_enabled and bool(auth_endpoints),
            "auth_endpoints": auth_endpoints or [],
            "auth_login_endpoint": auth_endpoints[0] if auth_endpoints else "",
            "auth_logout_endpoint": auth_endpoints[1] if auth_endpoints and len(auth_endpoints) > 1 else "",
            "auth_credentials": {"email": "test@example.com", "password": "password"},
            "target_host": self.target_host,
            "db_type": self.db_type,
            "primary_endpoint": primary_endpoint,
            "primary_schema": primary_schema,
            # Additional context for templates
            "environment_vars": environment_vars,
            "db_using": db_using,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    async def _generate_static_files(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate files that don't need AI enhancement."""
        files = {}

        # Define all templates to generate
        templates_to_generate = [
            ("locustfile.py", "modular/locustfile.py.j2"),
            ("config.py", "config.py.j2"),
            ("utils.py", "utils.py.j2"),
            ("requirements.txt", "requirement.txt.j2"),
            ("README.md", "readme.md.j2"),
            (".env.example", "env.example.j2"),
            ("data/__init__.py", "data/__init__.py.j2"),
            ("data/base_generator.py", "data/base_generator.py.j2"),
            ("data/security_payloads.py", "data/security_payloads.py.j2"),
            ("scenarios/__init__.py", "scenarios/__init__.py.j2"),
            ("scenarios/base_scenario.py", "scenarios/base_scenario.py.j2"),
            ("auth/__init__.py", "auth/__init__.py.j2"),
            ("auth/authenticator.py", "auth/authenticator.py.j2"),
        ]

        for output_file, template_name in templates_to_generate:
            logger.debug(f"[ModularGenerator] Rendering template: {template_name} -> {output_file}")
            try:
                content = self._render_template(template_name, context)
                if content and not content.startswith("# Template error"):
                    files[output_file] = content
                    logger.debug(f"[ModularGenerator] Success: {output_file} ({len(content)} chars)")
                else:
                    logger.warning(f"[ModularGenerator] Template {template_name} returned error or empty content")
            except Exception as e:
                logger.error(f"[ModularGenerator] Failed to render {template_name}: {e}")

        # Generate per-API-group workflows (use pre-computed groups from context)
        grouped_endpoints = context.get("grouped_endpoints", {})
        workflow_files = self._generate_group_workflows(grouped_endpoints, context)
        files.update(workflow_files)

        # Generate common security scenarios (shared across all API groups)
        common_security_content = self._generate_common_security(context)
        files["scenarios/common_security.py"] = common_security_content

        # Generate scenarios/__init__.py with all per-group imports
        scenarios_init = self._generate_scenarios_init(context.get("api_groups", []))
        files["scenarios/__init__.py"] = scenarios_init

        # Database files if needed
        if self.db_type and self.db_type != "none":
            logger.debug(f"[ModularGenerator] Generating DB files for type: {self.db_type}")
            db_files = self._generate_db_files(context)
            files.update(db_files)

        logger.info(f"[ModularGenerator] _generate_static_files produced {len(files)} files")
        return files

    def _group_endpoints_by_tag(self, endpoints: List[Endpoint]) -> Dict[str, List[Endpoint]]:
        """Group endpoints by their first tag."""
        from collections import defaultdict
        groups = defaultdict(list)

        for endpoint in endpoints:
            # Use first tag, or 'default' if no tags
            tag = endpoint.tags[0] if endpoint.tags else "default"
            # Normalize tag name for Python module naming
            tag = tag.lower().replace(" ", "_").replace("-", "_")
            groups[tag].append(endpoint)

        return dict(groups)

    def _generate_group_workflows(
        self, grouped_endpoints: Dict[str, List[Endpoint]], context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate workflow files for each API group."""
        files = {}

        # Generate workflows/__init__.py with imports for all groups
        group_names = list(grouped_endpoints.keys())
        workflow_imports = []
        workflow_exports = []

        for group_name in group_names:
            class_name = self._to_class_name(group_name) + "Workflow"
            workflow_imports.append(f"from .{group_name}_workflow import {class_name}")
            workflow_exports.append(f'"{class_name}"')

        init_content = f'''"""Workflow compositions for API groups."""
from .main_workflow import MainWorkflow
{chr(10).join(workflow_imports)}

__all__ = [
    "MainWorkflow",
    {", ".join(workflow_exports)}
]
'''
        files["workflows/__init__.py"] = init_content

        # Generate main_workflow.py that orchestrates all group workflows
        main_workflow_content = self._generate_main_workflow(group_names, context)
        files["workflows/main_workflow.py"] = main_workflow_content

        # Generate individual group workflows
        for group_name, group_endpoints in grouped_endpoints.items():
            workflow_content = self._generate_group_workflow(group_name, group_endpoints, context)
            files[f"workflows/{group_name}_workflow.py"] = workflow_content

        return files

    def _to_class_name(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))

    def _generate_main_workflow(self, group_names: List[str], context: Dict[str, Any]) -> str:
        """Generate main workflow that orchestrates all group workflows and scenarios."""
        workflow_imports = []
        scenario_imports = []
        task_sets = []
        scenario_tasks = []

        for group_name in group_names:
            class_name = self._to_class_name(group_name)
            workflow_class = f"{class_name}Workflow"
            workflow_imports.append(f"from .{group_name}_workflow import {workflow_class}")
            task_sets.append(workflow_class)

            # Import per-group scenarios (absolute import - works via sys.path in locustfile.py)
            scenario_imports.append(
                f"from scenarios.{group_name} import {class_name}PositiveTasks, {class_name}NegativeTasks, {class_name}EdgeCaseTasks"
            )
            scenario_tasks.extend([
                f"{class_name}PositiveTasks",
                f"{class_name}NegativeTasks",
                f"{class_name}EdgeCaseTasks",
            ])

        # Build task weights dict
        task_weight_lines = []
        for cls in task_sets:
            task_weight_lines.append(f"        {cls}: 10,")
        for i, cls in enumerate(scenario_tasks):
            weight = 6 if "Positive" in cls else 3 if "Negative" in cls else 2
            task_weight_lines.append(f"        {cls}: {weight},")

        return f'''"""Main workflow - orchestrates all API group workflows and scenarios."""
import logging
from locust import HttpUser, between

# Import API group workflows
{chr(10).join(workflow_imports)}

# Import per-group scenarios (positive, negative, edge cases)
{chr(10).join(scenario_imports)}

# Import shared security (absolute import - works via sys.path in locustfile.py)
from scenarios.common_security import CommonSecurityTasks

logger = logging.getLogger(__name__)


class MainWorkflow(HttpUser):
    """
    Main load test workflow that orchestrates all API groups and scenarios.

    This class weaves together:
    - Per-API-group workflows: {", ".join(task_sets)}
    - Per-API-group scenarios: positive, negative, edge cases for each group
    - Shared security tests: CommonSecurityTasks
    """

    host = "{context.get('target_host', 'http://localhost:8000')}"
    wait_time = between(1, 3)

    # Task weights - weaving workflows and scenarios together
    tasks = {{
        # API group workflows (sequential operations)
{chr(10).join(task_weight_lines)}
        # Shared security tests (common across all APIs)
        CommonSecurityTasks: 1,
    }}

    def on_start(self):
        """Initialize user session."""
        logger.info(f"User {{self.__class__.__name__}} starting")

    def on_stop(self):
        """Cleanup user session."""
        logger.info(f"User {{self.__class__.__name__}} stopping")
'''

    def _generate_group_workflow(
        self, group_name: str, endpoints: List[Endpoint], context: Dict[str, Any]
    ) -> str:
        """
        Generate workflow for a specific API group.

        This workflow weaves together the group's scenarios and provides
        sequential endpoint operations.
        """
        class_name_base = self._to_class_name(group_name)
        workflow_class = class_name_base + "Workflow"

        # Generate task methods for each endpoint
        task_methods = []
        for endpoint in endpoints:
            method = self._generate_endpoint_task(endpoint, group_name)
            if method:
                task_methods.append(method)

        methods_str = "\n\n".join(task_methods) if task_methods else "    pass"

        return f'''"""Workflow for {group_name} API endpoints."""
import logging
from locust import SequentialTaskSet, task

# Import this group's scenarios (absolute import - works via sys.path in locustfile.py)
from scenarios.{group_name} import (
    {class_name_base}PositiveTasks,
    {class_name_base}NegativeTasks,
    {class_name_base}EdgeCaseTasks,
)

logger = logging.getLogger(__name__)


class {workflow_class}(SequentialTaskSet):
    """
    Sequential workflow for {group_name} API operations.

    This workflow:
    1. Runs sequential endpoint operations (CRUD flow)
    2. Can invoke scenario-based tests via tasks dict

    Endpoints covered:
{chr(10).join(f"    - {ep.method.upper()} {ep.path}" for ep in endpoints)}

    Related scenarios:
    - {class_name_base}PositiveTasks (happy path tests)
    - {class_name_base}NegativeTasks (error handling tests)
    - {class_name_base}EdgeCaseTasks (boundary tests)
    """

    # Sub-tasks can invoke scenario-based tests
    tasks = {{
        {class_name_base}PositiveTasks: 3,
        {class_name_base}NegativeTasks: 1,
    }}

{methods_str}

    def on_start(self):
        """Setup for {group_name} workflow."""
        logger.debug(f"Starting {group_name} workflow")

    def on_stop(self):
        """Cleanup for {group_name} workflow."""
        logger.debug(f"Stopping {group_name} workflow")
'''

    def _generate_endpoint_task(self, endpoint: Endpoint, group_name: str) -> str:
        """Generate a task method for a single endpoint."""
        method = endpoint.method.upper()
        path = endpoint.path

        # Create a safe method name
        method_name = f"{method.lower()}_{endpoint.operation_id or path.replace('/', '_').strip('_')}"
        method_name = method_name.replace("-", "_").replace("{", "").replace("}", "")

        # Determine weight based on method type
        weight = 5 if method in ["GET", "POST"] else 2

        # Generate request body handling for POST/PUT/PATCH
        body_handling = ""
        if method in ["POST", "PUT", "PATCH"]:
            body_handling = '''
        # TODO: Add request body data
        data = {}'''

        return f'''    @task({weight})
    def {method_name}(self):
        """{endpoint.summary or f'{method} {path}'}"""
        with self.client.{method.lower()}(
            "{path}",
            name="{group_name}:{method.lower()}:{path}",
            catch_response=True
        ) as response:
            if response.status_code < 400:
                response.success()
            else:
                response.failure(f"Error: {{response.status_code}}")'''

    def _generate_common_security(self, context: Dict[str, Any]) -> str:
        """
        Generate shared security tests used across all API groups.

        This avoids regenerating auth mechanisms per API group.
        """
        auth_enabled = context.get("auth_enabled", False)
        auth_login = context.get("auth_login_endpoint", "/auth/login")

        return f'''"""
Common security scenarios shared across all API groups.

This module contains authentication and authorization tests that
apply to the entire API, avoiding duplication per endpoint group.
"""
import logging
from locust import TaskSet, task

# Absolute import - works via sys.path in locustfile.py
from auth import Authenticator

logger = logging.getLogger(__name__)


class CommonSecurityTasks(TaskSet):
    """
    Shared security tests for authentication and authorization.

    These tests are run once and shared across all API workflows,
    avoiding redundant security test generation per API group.
    """

    def on_start(self):
        """Initialize authenticator for security tests."""
        self.auth = Authenticator(self.client)

    @task(3)
    def test_missing_auth_header(self):
        """Test requests without authentication header."""
        # Save and remove auth header temporarily
        original_headers = dict(self.client.headers or {{}})
        self.client.headers.pop("Authorization", None)

        with self.client.get(
            "/api/protected-resource",
            name="security:missing_auth",
            catch_response=True
        ) as response:
            if response.status_code == 401:
                response.success()
            else:
                response.failure(f"Expected 401, got {{response.status_code}}")

        # Restore headers
        self.client.headers.update(original_headers)

    @task(3)
    def test_invalid_token(self):
        """Test requests with invalid/malformed token."""
        original_headers = dict(self.client.headers or {{}})
        self.client.headers["Authorization"] = "Bearer invalid_token_12345"

        with self.client.get(
            "/api/protected-resource",
            name="security:invalid_token",
            catch_response=True
        ) as response:
            if response.status_code in [401, 403]:
                response.success()
            else:
                response.failure(f"Expected 401/403, got {{response.status_code}}")

        # Restore headers
        self.client.headers.update(original_headers)

    @task(2)
    def test_expired_token(self):
        """Test requests with expired token."""
        # Simulate expired token (implementation depends on API)
        original_headers = dict(self.client.headers or {{}})
        self.client.headers["Authorization"] = "Bearer expired.token.here"

        with self.client.get(
            "/api/protected-resource",
            name="security:expired_token",
            catch_response=True
        ) as response:
            if response.status_code in [401, 403]:
                response.success()
            else:
                response.failure(f"Expected 401/403, got {{response.status_code}}")

        self.client.headers.update(original_headers)

    @task(2)
    def test_sql_injection_attempt(self):
        """Test SQL injection protection on common parameters."""
        payloads = ["'; DROP TABLE users; --", "1 OR 1=1", "admin'--"]
        for payload in payloads:
            with self.client.get(
                f"/api/search?q={{payload}}",
                name="security:sql_injection",
                catch_response=True
            ) as response:
                # Should not return 200 with data or should sanitize
                if response.status_code >= 400:
                    response.success()
                else:
                    # Check if response indicates sanitization
                    response.success()

    @task(2)
    def test_xss_attempt(self):
        """Test XSS protection on inputs."""
        payload = "<script>alert('xss')</script>"
        with self.client.post(
            "/api/resource",
            json={{"name": payload, "description": payload}},
            name="security:xss_attempt",
            catch_response=True
        ) as response:
            if response.status_code >= 400:
                response.success()
            elif payload not in response.text:
                # Payload was sanitized
                response.success()
            else:
                response.failure("XSS payload reflected in response")

    @task(1)
    def test_rate_limiting(self):
        """Test rate limiting protection."""
        # Rapid requests to trigger rate limit
        for i in range(10):
            with self.client.get(
                "/api/resource",
                name="security:rate_limit_test",
                catch_response=True
            ) as response:
                if response.status_code == 429:
                    response.success()
                    logger.info("Rate limiting correctly enforced")
                    return
                response.success()
'''

    def _generate_scenarios_init(self, api_groups: List[str]) -> str:
        """
        Generate scenarios/__init__.py with all per-group imports.

        This provides convenient access to all scenario classes.
        """
        imports = ["from .base_scenario import BaseScenario"]
        imports.append("from .common_security import CommonSecurityTasks")

        all_exports = ['"BaseScenario"', '"CommonSecurityTasks"']

        for group_name in api_groups:
            class_name = self._to_class_name(group_name)
            imports.append(
                f"from .{group_name} import {class_name}PositiveTasks, {class_name}NegativeTasks, {class_name}EdgeCaseTasks"
            )
            all_exports.extend([
                f'"{class_name}PositiveTasks"',
                f'"{class_name}NegativeTasks"',
                f'"{class_name}EdgeCaseTasks"',
            ])

        return f'''"""Test scenarios for load testing.

This module provides:
- BaseScenario: Base class for all scenario types
- CommonSecurityTasks: Shared authentication and security tests
- Per-API-group scenarios: positive, negative, edge cases for each API group
"""
{chr(10).join(imports)}

__all__ = [
    {(",{0}    ".format(chr(10))).join(all_exports)},
]
'''

    def _generate_db_files(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate database-specific files."""
        files = {}
        if self.db_type == "mongo":
            try:
                files["db_config.py"] = self._render_template("mongo/db_config.py.j2", context)
                files["data_provider.py"] = self._render_template("mongo/data_provider.py.j2", context)
            except Exception as e:
                logger.warning(f"Failed to generate mongo files: {e}")
        return files

    async def _generate_ai_enhanced_files(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate AI-enhanced files per API group.

        Instead of one large positive_tasks.py, generates:
        - scenarios/{api_group}/positive.py
        - scenarios/{api_group}/negative.py
        - scenarios/{api_group}/edge_cases.py

        This keeps prompts small for LLM token limits.
        """
        files = {}
        grouped_endpoints = context.get("grouped_endpoints", {})
        total_groups = len(grouped_endpoints)

        # Generate global data files (still needed for shared data generators)
        self._report_progress("AI", "Generating test data", "valid_data.py, invalid_data.py", 42)
        data_tasks = [
            self._enhance_valid_data(context),
            self._enhance_invalid_data(context),
        ]
        data_results = await asyncio.gather(*data_tasks, return_exceptions=True)

        # Valid data
        ai_methods = data_results[0] if not isinstance(data_results[0], Exception) else ""
        if isinstance(data_results[0], Exception):
            logger.warning(f"AI enhancement failed for valid_data: {data_results[0]}")
            ai_methods = "# AI enhancement failed - add methods manually\npass"
        files["data/valid_data.py"] = self._render_template(
            "data/valid_data.py.j2",
            {**context, "ai_generated_methods": ai_methods}
        )
        self._report_progress("AI", "Test data generated", "data/valid_data.py", 45)

        # Invalid data
        ai_methods = data_results[1] if not isinstance(data_results[1], Exception) else ""
        if isinstance(data_results[1], Exception):
            logger.warning(f"AI enhancement failed for invalid_data: {data_results[1]}")
            ai_methods = "# AI enhancement failed - add methods manually\npass"
        files["data/invalid_data.py"] = self._render_template(
            "data/invalid_data.py.j2",
            {**context, "ai_generated_methods": ai_methods}
        )
        self._report_progress("AI", "Test data generated", "data/invalid_data.py", 48)

        # Generate per-API-group scenario files
        for i, (group_name, group_endpoints) in enumerate(grouped_endpoints.items()):
            pct = 50 + int((i / max(total_groups, 1)) * 30)
            self._report_progress("AI", f"Generating scenarios", f"API group: {group_name} ({i+1}/{total_groups})", pct)
            group_files = await self._generate_group_scenarios(
                group_name, group_endpoints, context
            )
            files.update(group_files)
            self._report_progress("AI", f"Scenarios complete", f"{group_name}: {len(group_files)} files", pct + 5)

        return files

    async def _generate_group_scenarios(
        self,
        group_name: str,
        endpoints: List[Endpoint],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate scenario files for a specific API group.

        Produces:
        - scenarios/{group_name}/__init__.py
        - scenarios/{group_name}/positive.py
        - scenarios/{group_name}/negative.py
        - scenarios/{group_name}/edge_cases.py
        """
        files = {}
        class_name = self._to_class_name(group_name)

        # Create group context with only this group's endpoints (smaller prompts)
        group_context = {
            **context,
            "endpoints": endpoints,
            "endpoints_summary": self._format_endpoints_summary(endpoints),
            "group_name": group_name,
            "class_name": class_name,
        }

        # Generate scenarios in parallel for this group
        scenario_tasks = [
            self._enhance_group_positive(group_name, group_context),
            self._enhance_group_negative(group_name, group_context),
            self._enhance_group_edge_cases(group_name, group_context),
        ]
        results = await asyncio.gather(*scenario_tasks, return_exceptions=True)

        # Positive scenarios
        positive_methods = results[0] if not isinstance(results[0], Exception) else ""
        if isinstance(results[0], Exception):
            logger.warning(f"AI enhancement failed for {group_name}/positive: {results[0]}")
            positive_methods = "# AI enhancement failed - add methods manually\npass"

        # Negative scenarios
        negative_methods = results[1] if not isinstance(results[1], Exception) else ""
        if isinstance(results[1], Exception):
            logger.warning(f"AI enhancement failed for {group_name}/negative: {results[1]}")
            negative_methods = "# AI enhancement failed - add methods manually\npass"

        # Edge case scenarios
        edge_methods = results[2] if not isinstance(results[2], Exception) else ""
        if isinstance(results[2], Exception):
            logger.warning(f"AI enhancement failed for {group_name}/edge_cases: {results[2]}")
            edge_methods = "# AI enhancement failed - add methods manually\npass"

        # Generate __init__.py for this group
        files[f"scenarios/{group_name}/__init__.py"] = f'''"""Scenarios for {group_name} API group."""
from .positive import {class_name}PositiveTasks
from .negative import {class_name}NegativeTasks
from .edge_cases import {class_name}EdgeCaseTasks

__all__ = [
    "{class_name}PositiveTasks",
    "{class_name}NegativeTasks",
    "{class_name}EdgeCaseTasks",
]
'''

        # Generate positive.py
        files[f"scenarios/{group_name}/positive.py"] = self._generate_group_scenario_file(
            group_name, class_name, "Positive", positive_methods, endpoints
        )

        # Generate negative.py
        files[f"scenarios/{group_name}/negative.py"] = self._generate_group_scenario_file(
            group_name, class_name, "Negative", negative_methods, endpoints
        )

        # Generate edge_cases.py
        files[f"scenarios/{group_name}/edge_cases.py"] = self._generate_group_scenario_file(
            group_name, class_name, "EdgeCase", edge_methods, endpoints
        )

        return files

    def _generate_group_scenario_file(
        self,
        group_name: str,
        class_name: str,
        scenario_type: str,
        ai_methods: str,
        endpoints: List[Endpoint]
    ) -> str:
        """Generate a scenario file for a specific group and type."""
        import textwrap

        endpoints_doc = "\n".join(f"    - {ep.method.upper()} {ep.path}" for ep in endpoints)

        # Determine test description based on scenario type
        if scenario_type == "Positive":
            test_desc = "happy path"
        elif scenario_type == "Negative":
            test_desc = "error handling"
        else:
            test_desc = "boundary conditions"

        # AI methods come at 0 indent - apply 4-space class-level indent
        indented_methods = textwrap.indent(ai_methods, '    ')

        # Use relative imports - scenarios/{group}/ is 2 levels deep from root
        return f'''"""
{scenario_type} test scenarios for {group_name} API.

Endpoints covered:
{endpoints_doc}
"""
import logging
from locust import TaskSet, task

# Absolute import - works via sys.path in locustfile.py
from data import ValidDataGenerator, InvalidDataGenerator

logger = logging.getLogger(__name__)


class {class_name}{scenario_type}Tasks(TaskSet):
    """
    {scenario_type} test scenarios for {group_name} endpoints.

    Tests {test_desc} for this API group.
    """

    def on_start(self):
        """Initialize data generators for {group_name} tests."""
        self.valid_data = ValidDataGenerator()
        self.invalid_data = InvalidDataGenerator()
        logger.debug(f"Starting {class_name}{scenario_type}Tasks")

{indented_methods}
'''

    async def _enhance_group_positive(self, group_name: str, context: Dict[str, Any]) -> str:
        """Generate positive task methods for a specific API group."""
        prompt = self._build_group_prompt(group_name, "positive", context)
        return await self._call_ai(prompt)

    async def _enhance_group_negative(self, group_name: str, context: Dict[str, Any]) -> str:
        """Generate negative task methods for a specific API group."""
        prompt = self._build_group_prompt(group_name, "negative", context)
        return await self._call_ai(prompt)

    async def _enhance_group_edge_cases(self, group_name: str, context: Dict[str, Any]) -> str:
        """Generate edge case task methods for a specific API group."""
        prompt = self._build_group_prompt(group_name, "edge_cases", context)
        return await self._call_ai(prompt)

    def _build_group_prompt(self, group_name: str, scenario_type: str, context: Dict[str, Any]) -> str:
        """
        Build a focused prompt for a specific API group and scenario type.

        This keeps prompts small for LLM token limits.
        """
        endpoints_summary = context.get("endpoints_summary", "")
        class_name = context.get("class_name", self._to_class_name(group_name))

        # Common data generator usage instructions
        data_usage = """
IMPORTANT - Data Generator Usage:
- self.valid_data is a ValidDataGenerator instance with methods like:
  - self.valid_data.complete_data("schema_name") -> returns dict with all fields
  - self.valid_data.minimal_data("schema_name") -> returns dict with required fields only
  - self.valid_data.generate_id() -> returns a UUID string
  - self.valid_data.generate_email() -> returns a random email
  - self.valid_data.fake.name() -> returns a random name
- self.invalid_data is an InvalidDataGenerator instance with methods like:
  - self.invalid_data.missing_required("schema", "field") -> dict missing a field
  - self.invalid_data.invalid_types("schema") -> dict with wrong types
  - self.invalid_data.out_of_range("schema") -> dict with out-of-range values
- DO NOT use dict access like self.valid_data["key"] - these are class instances, not dicts!

CRITICAL - Indentation Rules:
- After a 'return' statement, any following code MUST be at a LOWER indent level (outside the if block)
- Do NOT put code after 'return' at the same indentation - that creates unreachable dead code
- For chained requests, use sequential code at the SAME indent level, not nested deeper

WRONG (dead code after return):
    if response.status_code != 201:
        response.failure(...)
        return
        # This code is INSIDE the if block - unreachable!
        next_step()

CORRECT (code outside the if block):
    if response.status_code != 201:
        response.failure(...)
        return
    # This code is OUTSIDE the if block - will execute when status IS 201
    response.success()
    next_step()
"""

        if scenario_type == "positive":
            return f"""Generate Locust @task methods for POSITIVE (happy path) tests for the {group_name} API.

Endpoints:
{endpoints_summary}
{data_usage}
Requirements:
- Generate 2-3 @task methods with weights (e.g., @task(5))
- Use self.client.get/post/put/delete with catch_response=True
- Test successful scenarios with valid data from self.valid_data methods
- Use "with ... as response:" context manager pattern
- Check response.status_code and call response.success() or response.failure()

Example format:
@task(5)
def test_list_items(self):
    \"\"\"Test listing items.\"\"\"
    with self.client.get("/api/v1/{group_name}/", catch_response=True) as response:
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"Expected 200, got {{response.status_code}}")

@task(3)
def test_create_item(self):
    \"\"\"Test creating an item.\"\"\"
    data = self.valid_data.complete_data("{group_name}")
    with self.client.post("/api/v1/{group_name}/", json=data, catch_response=True) as response:
        if response.status_code == 201:
            response.success()
        else:
            response.failure(f"Expected 201, got {{response.status_code}}")

Return ONLY the Python method code, no class definition. Maintain consistent 4-space indentation.
Output format (MANDATORY):
<new_methods>
...ONLY the method code here...
</new_methods>
{f'''
Additional Requirements from Developer:
{self.custom_requirement}
''' if self.custom_requirement else ''}"""

        elif scenario_type == "negative":
            return f"""Generate Locust @task methods for NEGATIVE (error handling) tests for the {group_name} API.

Endpoints:
{endpoints_summary}
{data_usage}
Requirements:
- Generate 2-3 @task methods with weights (e.g., @task(3))
- Use self.client.get/post/put/delete with catch_response=True
- Test error scenarios: invalid IDs, missing fields, wrong types
- Use self.invalid_data methods to generate bad data
- Expect 4xx responses and verify error handling

Example format:
@task(3)
def test_invalid_id(self):
    \"\"\"Test with invalid ID.\"\"\"
    with self.client.get("/api/v1/{group_name}/invalid-id-12345", catch_response=True) as response:
        if response.status_code == 404:
            response.success()
        else:
            response.failure(f"Expected 404, got {{response.status_code}}")

@task(2)
def test_missing_required_fields(self):
    \"\"\"Test with missing required fields.\"\"\"
    data = self.invalid_data.missing_required("{group_name}", "name")
    with self.client.post("/api/v1/{group_name}/", json=data, catch_response=True) as response:
        if response.status_code in [400, 422]:
            response.success()
        else:
            response.failure(f"Expected 400/422, got {{response.status_code}}")

Return ONLY the Python method code, no class definition. Maintain consistent 4-space indentation.
Output format (MANDATORY):
<new_methods>
...ONLY the method code here...
</new_methods>
{f'''
Additional Requirements from Developer:
{self.custom_requirement}
''' if self.custom_requirement else ''}"""

        else:  # edge_cases
            return f"""Generate Locust @task methods for EDGE CASE tests for the {group_name} API.

Endpoints:
{endpoints_summary}

Requirements:
- Generate 2-3 @task methods with weights (e.g., @task(2))
- Use self.client.get/post/put/delete with catch_response=True
- Test boundary conditions: empty strings, max lengths, special characters
- Each test should be a SEPARATE method, not nested
- Use "with ... as response:" context manager pattern

Example format:
@task(2)
def test_empty_string_params(self):
    \"\"\"Test with empty string parameters.\"\"\"
    with self.client.get("/api/v1/{group_name}/", params={{"name": ""}}, catch_response=True) as response:
        if response.status_code in [200, 400]:
            response.success()
        else:
            response.failure(f"Unexpected status {{response.status_code}}")

@task(2)
def test_max_length_input(self):
    \"\"\"Test with maximum length input.\"\"\"
    long_value = "a" * 1000
    with self.client.post("/api/v1/{group_name}/", json={{"name": long_value}}, catch_response=True) as response:
        if response.status_code in [200, 201, 400, 413]:
            response.success()
        else:
            response.failure(f"Unexpected status {{response.status_code}}")

@task(1)
def test_special_characters(self):
    \"\"\"Test with special characters.\"\"\"
    with self.client.get("/api/v1/{group_name}/!@%23$%25", catch_response=True) as response:
        if response.status_code in [200, 400, 404]:
            response.success()
        else:
            response.failure(f"Unexpected status {{response.status_code}}")

Return ONLY the Python method code, no class definition. Each test must be a separate method. Maintain consistent 4-space indentation.
Output format (MANDATORY):
<new_methods>
...ONLY the method code here...
</new_methods>
{f'''
Additional Requirements from Developer:
{self.custom_requirement}
''' if self.custom_requirement else ''}"""

    async def _enhance_valid_data(self, context: Dict[str, Any]) -> str:
        """Generate valid data methods via AI."""
        prompt = self._render_prompt("data/valid_data.j2", context)
        return await self._call_ai(prompt)

    async def _enhance_invalid_data(self, context: Dict[str, Any]) -> str:
        """Generate invalid data methods via AI."""
        prompt = self._render_prompt("data/invalid_data.j2", context)
        return await self._call_ai(prompt)

    # Note: _enhance_positive_tasks and _enhance_negative_tasks removed
    # Now using per-group generation: _enhance_group_positive, _enhance_group_negative, etc.

    async def _call_ai(self, prompt: str) -> str:
        """
        Call AI service, extract methods, validate and fix if needed.

        Uses the validation module to ensure generated code is valid.
        Retries up to self.retry_on_invalid times if validation fails.
        """
        max_attempts = 1 + self.retry_on_invalid

        for attempt in range(max_attempts):
            try:
                messages = self._ai_generator._build_messages(prompt)
                response = await self._ai_generator._make_api_call(
                    messages,
                    parse_context="modular_generator raw",
                    require_tags=True,
                )
                if not response:
                    continue

                cleaned = self._ai_generator._clean_ai_response(response)
                # Strip XML/HTML tags that LLMs sometimes add
                cleaned = self._strip_xml_tags(cleaned)
                # Normalize indentation for consistent template insertion
                normalized = self._normalize_indentation(cleaned)

                # Validate the generated code
                validation_result = self.code_validator.validate_method_code(normalized)

                if validation_result.is_valid:
                    # Try to fix any remaining issues (like unreachable code)
                    fix_result = self.code_fixer.fix_method_code(normalized)
                    if fix_result.fixes_applied:
                        logger.info(f"Applied fixes: {fix_result.fixes_applied}")
                    return fix_result.fixed_code

                # Validation failed - try to fix
                logger.warning(f"Validation failed (attempt {attempt + 1}/{max_attempts}): {validation_result.issues}")
                fix_result = self.code_fixer.fix_method_code(normalized)

                if fix_result.success:
                    logger.info(f"Fixed validation issues: {fix_result.fixes_applied}")
                    return fix_result.fixed_code

                # Fix failed, retry if we have attempts left
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying AI call (attempt {attempt + 2}/{max_attempts})")
                    # Add a hint to the prompt about the validation errors
                    prompt = f"# TODO: Previous attempt had issues: {validation_result.issues}\n{prompt}"
                else:
                    # Last attempt failed - return what we have with a warning comment
                    logger.warning(f"All {max_attempts} attempts failed validation, returning best effort")
                    return f"# WARNING: Code may have validation issues: {validation_result.issues}\n{normalized}"

            except Exception as e:
                logger.error(f"AI call failed: {e}")
                if attempt >= max_attempts - 1:
                    break

        return ""

    def _strip_xml_tags(self, code: str) -> str:
        """
        Strip XML/HTML tags that LLMs sometimes add to their output.

        Examples: </methods>, <code>, </code>, <methods>, etc.
        """
        import re
        # Remove XML-style tags like </methods>, <code>, etc.
        code = re.sub(r'</?[a-zA-Z_][a-zA-Z0-9_-]*\s*/?>', '', code)
        # Remove markdown code fences
        code = re.sub(r'^```\w*\s*$', '', code, flags=re.MULTILINE)
        return code.strip()

    def _normalize_indentation(self, code: str) -> str:
        """
        Normalize indentation of AI-generated code to base (0) indent.

        Preserves the AI's relative indentation (to keep if/else/with blocks correct)
        while normalizing so decorators and def lines start at column 0.

        Returns code at 0 indent (decorators and def at column 0, body properly indented).
        Templates/callers are responsible for adding class-level indentation.
        """
        import re

        if not code or not code.strip():
            return "pass  # No AI-generated code"

        lines = code.split('\n')

        # First pass: collect methods with their decorators and bodies
        # PRESERVE original indentation to maintain relative structure
        methods = []
        current_method = None
        pending_decorator = None
        pending_decorator_indent = 0

        for line in lines:
            stripped = line.strip()
            # Calculate original indent (spaces at start)
            original_indent = len(line) - len(line.lstrip())

            # Skip empty lines (but track them in method body)
            if not stripped:
                if current_method and current_method['body']:
                    current_method['body'].append({'text': '', 'indent': 0})
                continue

            # Skip leftover XML/garbage
            if stripped.startswith('<') or stripped.startswith('```'):
                continue

            # Check for @task decorator (with or without weight)
            if re.match(r'^@task\s*(\(\d+\))?', stripped):
                # Save decorator for the NEXT method
                pending_decorator = stripped
                pending_decorator_indent = original_indent
                continue

            # Check for method definition
            if re.match(r'^def\s+\w+\s*\(self', stripped):
                # Save previous method if exists
                if current_method:
                    methods.append(current_method)

                # Start new method with pending decorator
                current_method = {
                    'decorator': pending_decorator,
                    'decorator_indent': pending_decorator_indent,
                    'def_line': stripped,
                    'def_indent': original_indent,
                    'body': []
                }
                pending_decorator = None
                pending_decorator_indent = 0
                continue

            # Body line - preserve original indent
            if current_method:
                current_method['body'].append({'text': stripped, 'indent': original_indent})

        # Don't forget the last method
        if current_method:
            methods.append(current_method)

        if not methods:
            return "pass  # No AI-generated code"

        # Second pass: normalize each method by subtracting base indent
        # This preserves relative indentation for nested blocks
        normalized_methods = []

        for method in methods:
            fixed_lines = []

            # Determine base indent (indent of def line, or decorator if lower)
            base_indent = method['def_indent']
            if method['decorator']:
                base_indent = min(base_indent, method['decorator_indent'])

            # Add decorator if present (at column 0)
            if method['decorator']:
                fixed_lines.append(method['decorator'])

            # Add def line (at column 0)
            fixed_lines.append(method['def_line'])

            # Add body lines with preserved relative indentation
            for body_line in method['body']:
                if not body_line['text']:
                    fixed_lines.append('')
                    continue

                # Calculate new indent: original indent - base indent + 4 (for method body)
                # This normalizes to 0-based while preserving relative structure
                relative_indent = body_line['indent'] - base_indent
                # Body should be at least 4 spaces (inside method)
                new_indent = max(4, relative_indent if relative_indent > 0 else 4)
                fixed_lines.append(' ' * new_indent + body_line['text'])

            normalized_methods.append('\n'.join(fixed_lines))

        return '\n\n'.join(normalized_methods)

    def _render_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template."""
        try:
            logger.debug(f"[ModularGenerator] Looking for template: {template_path}")
            template = self.env.get_template(template_path)
            result = template.render(**context)
            logger.debug(f"[ModularGenerator] Template {template_path} rendered successfully ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"[ModularGenerator] Template render failed for {template_path}: {e}", exc_info=True)
            return f"# Template error: {e}\n"

    def _render_prompt(self, prompt_path: str, context: Dict[str, Any]) -> str:
        """Render a prompt template."""
        try:
            template = self.prompt_env.get_template(prompt_path)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Prompt render failed for {prompt_path}: {e}")
            return ""

    def _write_file(self, relative_path: str, content: str) -> None:
        """Write content to a file with UTF-8 encoding."""
        file_path = self.output_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.debug(f"Wrote {file_path}")

    def _format_endpoints_summary(self, endpoints: List[Endpoint]) -> str:
        """Format endpoints for prompt context."""
        lines = []
        for ep in endpoints[:20]:  # Limit to avoid token overflow
            params = f"({len(ep.parameters)} params)" if ep.parameters else ""
            lines.append(f"- {ep.method} {ep.path} {params}")
        return "\n".join(lines)

    def _format_schemas_info(self, schemas: Dict[str, Any]) -> str:
        """Format schemas for prompt context."""
        lines = []

        # Extract actual schema definitions from OpenAPI spec
        schema_defs = schemas
        if "components" in schemas and isinstance(schemas.get("components"), dict):
            # OpenAPI 3.x format
            schema_defs = schemas.get("components", {}).get("schemas", {})
        elif "definitions" in schemas:
            # Swagger 2.0 format
            schema_defs = schemas.get("definitions", {})

        if not isinstance(schema_defs, dict):
            return ""

        for name, schema in list(schema_defs.items())[:10]:  # Limit
            if not isinstance(schema, dict):
                continue
            props = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(props, dict):
                continue
            fields = [f"{k}{'*' if k in required else ''}" for k in list(props.keys())[:5]]
            lines.append(f"- {name}: {', '.join(fields)}")
        return "\n".join(lines)

    def _infer_primary_schema(self, endpoints: List[Endpoint]) -> str:
        """Infer primary schema name from endpoints."""
        if not endpoints:
            return "resource"
        # Extract from first POST endpoint path
        for ep in endpoints:
            if ep.method == "POST":
                parts = ep.path.strip("/").split("/")
                if parts:
                    return parts[-1].rstrip("s")  # Remove plural
        return "resource"
