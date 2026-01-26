"""
Scenario-based Workflow Generator

Generates separate workflow files for different test scenario types per endpoint:
- Positive scenarios (happy path + state-dependent)
- Negative scenarios (validation errors + edge cases + error handling)
- Security scenarios (injection, auth bypass)

Uses 3 LLM calls per endpoint for focused, non-truncated output.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from jinja2 import Environment, FileSystemLoader
from devdox_ai_locust.utils.http_fallback_presets import FallbackHttpResponseRegistry
from devdox_ai_locust.utils.code_validator import CodeValidator

if TYPE_CHECKING:
    from devdox_ai_locust.utils.debug_recorder import DebugRecorder
    from devdox_ai_locust.utils.generation_progress import GenerationProgress

logger = logging.getLogger(__name__)


class ScenarioGenerationError(Exception):
    """Raised when scenario generation fails"""
    pass


class CodeValidationError(ScenarioGenerationError):
    """Raised when generated code fails syntax validation"""
    def __init__(self, scenario_type: str, error: str, code: str, endpoint_info: str = ""):
        self.scenario_type = scenario_type
        self.error = error
        self.code = code
        self.endpoint_info = endpoint_info
        msg = f"Generated {scenario_type} code failed validation"
        if endpoint_info:
            msg += f" for [{endpoint_info}]"
        msg += f": {error}"
        super().__init__(msg)


class AIServiceError(ScenarioGenerationError):
    """Raised when AI service fails after all retries"""
    pass


class ScenarioType(Enum):
    """Types of test scenarios (all LLM-generated)"""
    POSITIVE = "positive"      # Happy path + state-dependent tests
    NEGATIVE = "negative"      # Validation errors + edge cases + error handling
    SECURITY = "security"      # Injection attacks + auth bypass


@dataclass
class RateLimitInfo:
    """Rate limit information from API response"""
    requests_per_second: int
    requests_per_minute: int
    remaining: int
    reset_seconds: float

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "RateLimitInfo":
        """Parse rate limit info from response headers"""
        rps = int(headers.get("x-ratelimit-limit", "1"))
        remaining = int(headers.get("x-ratelimit-remaining", "0"))
        reset = float(headers.get("x-ratelimit-reset", "1"))

        return cls(
            requests_per_second=rps,
            requests_per_minute=rps * 60,
            remaining=remaining,
            reset_seconds=reset,
        )

    @classmethod
    def default(cls) -> "RateLimitInfo":
        """Default rate limit (conservative estimate)"""
        return cls(
            requests_per_second=1,
            requests_per_minute=60,
            remaining=60,
            reset_seconds=1,
        )


@dataclass
class TimeEstimate:
    """Time estimate for generation"""
    total_calls: int
    rpm: int
    estimated_minutes: float
    estimated_seconds: float

    def __str__(self) -> str:
        if self.estimated_minutes < 1:
            return f"~{self.estimated_seconds:.0f} seconds"
        return f"~{self.estimated_minutes:.1f} minutes"


class ScenarioWorkflowGenerator:
    """
    Generates scenario-based workflow files using LLM.

    Uses 3 LLM calls per endpoint for focused, non-truncated output:
    - Call 1: Positive scenarios (happy path + state-dependent)
    - Call 2: Negative scenarios (validation + edge cases + error handling)
    - Call 3: Security scenarios (injection, auth bypass)
    """

    # Mapping of scenario types to output filenames
    SCENARIO_FILES = {
        ScenarioType.POSITIVE: "positive_workflow.py",
        ScenarioType.NEGATIVE: "negative_workflow.py",
        ScenarioType.SECURITY: "security_workflow.py",
    }

    # Prompt templates for LLM-based scenarios
    PROMPT_TEMPLATES = {
        ScenarioType.POSITIVE: "workflow_positive.j2",
        ScenarioType.NEGATIVE: "workflow_negative.j2",
        ScenarioType.SECURITY: "workflow_security.j2",
    }

    # Max concurrency limit
    MAX_CONCURRENCY = 50

    def __init__(
        self,
        prompt_dir: Path,
        ai_client: Any,
        ai_config: Any,
        max_concurrency: int = MAX_CONCURRENCY,
        debug_recorder: Optional["DebugRecorder"] = None,
        progress: Optional["GenerationProgress"] = None,
    ):
        """
        Initialize the scenario generator.

        Args:
            prompt_dir: Directory containing LLM prompt templates
            ai_client: Together AI client for LLM calls
            ai_config: AI configuration (model, timeout, etc.)
            max_concurrency: Maximum concurrent API calls
            debug_recorder: Optional debug recorder for capturing intermediate states
            progress: Optional progress display for verbose output
        """
        self.prompt_dir = prompt_dir
        self.ai_client = ai_client
        self.ai_config = ai_config
        self._rate_limit_info: Optional[RateLimitInfo] = None
        self._max_concurrency = max_concurrency
        self._current_concurrency = max_concurrency
        self._api_semaphore = asyncio.Semaphore(self._current_concurrency)
        self.debug_recorder = debug_recorder
        self.progress: Optional["GenerationProgress"] = progress
        self._fallback_registry = FallbackHttpResponseRegistry()
        self._code_validator = CodeValidator()

        # Setup Jinja environment for prompts
        self.prompt_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Extract allowed imports dynamically from prompt templates
        # This replaces the hardcoded ALLOWED_IMPORTS to avoid maintenance burden
        self._allowed_imports = self._extract_allowed_imports_from_templates()

    def _update_concurrency(self, rpm: int) -> None:
        """
        Update concurrency based on rate limit.

        Args:
            rpm: Requests per minute from rate limit headers
        """
        # Target: stay at ~80% of rate limit to avoid hitting it
        # Divide by 60 to get per-second, multiply by avg response time (~3s)
        optimal = min(int(rpm * 0.8 / 20), self._max_concurrency)  # ~3 req/s sustained
        new_concurrency = max(2, optimal)  # Never go below 2

        if new_concurrency != self._current_concurrency:
            logger.info(f"Adjusting concurrency: {self._current_concurrency} → {new_concurrency} (based on {rpm} RPM)")
            self._current_concurrency = new_concurrency
            self._api_semaphore = asyncio.Semaphore(new_concurrency)

    @property
    def num_scenarios(self) -> int:
        """Number of scenario types"""
        return len(self.SCENARIO_FILES)

    def estimate_time(self, num_endpoints: int) -> TimeEstimate:
        """
        Estimate generation time based on rate limits and concurrency.

        Args:
            num_endpoints: Number of endpoints to process

        Returns:
            TimeEstimate with calculated values
        """
        # 3 LLM calls per endpoint (positive + negative + security)
        total_calls = num_endpoints * self.num_scenarios

        rpm = self._rate_limit_info.requests_per_minute if self._rate_limit_info else 60

        # Factor in concurrency - with N concurrent workers, effective throughput is higher
        # But still bounded by rate limit (rpm)
        effective_rpm = min(rpm, self._current_concurrency * 20)  # ~3s per call avg
        estimated_minutes = total_calls / effective_rpm
        estimated_seconds = estimated_minutes * 60

        return TimeEstimate(
            total_calls=total_calls,
            rpm=effective_rpm,
            estimated_minutes=estimated_minutes,
            estimated_seconds=estimated_seconds,
        )

    def update_rate_limit(self, headers: Dict[str, str]) -> RateLimitInfo:
        """
        Update rate limit info from API response headers and adjust concurrency.

        Args:
            headers: Response headers from API call

        Returns:
            Updated RateLimitInfo
        """
        self._rate_limit_info = RateLimitInfo.from_headers(headers)
        self._update_concurrency(self._rate_limit_info.requests_per_minute)
        return self._rate_limit_info

    def get_rate_limit_info(self) -> RateLimitInfo:
        """Get current rate limit info or default"""
        return self._rate_limit_info or RateLimitInfo.default()

    @property
    def current_concurrency(self) -> int:
        """Current concurrency level"""
        return self._current_concurrency

    async def generate_all_endpoints(
        self,
        endpoints: List[Any],
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Dict[ScenarioType, str]]:
        """
        Generate workflows for ALL endpoints in parallel (bounded by concurrency).

        This is the main entry point for batch generation. It processes
        multiple endpoints concurrently, respecting rate limits.

        Args:
            endpoints: List of all endpoints to process
            base_workflow_content: Content of base_workflow.py
            test_data_content: Content of test_data.py
            auth_endpoints: Authentication endpoints (optional)
            progress_callback: Optional async callback(endpoint, results) for progress updates

        Returns:
            Dict mapping operation_id to scenario results

        Raises:
            CodeValidationError: If generated code fails syntax validation
            AIServiceError: If AI service fails after all retries
        """
        results: Dict[str, Dict[ScenarioType, str]] = {}

        # Create tasks for all endpoints
        async def process_endpoint(endpoint: Any) -> Tuple[str, Dict[ScenarioType, str]]:
            operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
            scenarios = await self.generate_endpoint_workflows(
                endpoint=endpoint,
                base_workflow_content=base_workflow_content,
                test_data_content=test_data_content,
                auth_endpoints=auth_endpoints,
                all_endpoints=endpoints,  # Pass all endpoints for CREATE endpoint lookup
            )
            if progress_callback:
                await progress_callback(endpoint, scenarios)
            return operation_id, scenarios

        # Process all endpoints concurrently (semaphore limits actual API calls)
        tasks = [process_endpoint(ep) for ep in endpoints]
        completed = await asyncio.gather(*tasks)

        for operation_id, scenarios in completed:
            results[operation_id] = scenarios

        return results

    async def generate_endpoint_workflows(
        self,
        endpoint: Any,
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        tag_name: str = "default",
        all_endpoints: Optional[List[Any]] = None,
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> Dict[ScenarioType, str]:
        """
        Generate all scenario workflow files for a single endpoint using LLM.

        Args:
            endpoint: Single endpoint to generate tests for
            base_workflow_content: Content of base_workflow.py
            test_data_content: Content of test_data.py
            auth_endpoints: Authentication endpoints (optional)
            tag_name: Tag/group name for this endpoint (used for debug logging)
            all_endpoints: All endpoints from OpenAPI spec (for finding related CREATE endpoints)
            custom_requirement: User-provided custom requirements for test generation
            db_type: Database type ("mongo" for MongoDB integration)

        Returns:
            Dict mapping ScenarioType to generated code content

        Raises:
            CodeValidationError: If generated code fails syntax validation
            AIServiceError: If AI service fails after all retries
        """
        # Generate all 3 scenarios in parallel using LLM
        scenario_types = list(ScenarioType)

        llm_tasks = [
            self._generate_llm_scenario(
                scenario_type,
                endpoint,
                base_workflow_content,
                test_data_content,
                auth_endpoints,
                tag_name,
                all_endpoints,
                custom_requirement,
                db_type,
            )
            for scenario_type in scenario_types
        ]

        # Use return_exceptions=True to preserve successful results even if some fail
        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)

        # Separate successes from failures
        endpoint_info = f"{endpoint.method} {endpoint.path}"
        results = {}
        errors = []
        for scenario_type, result in zip(scenario_types, llm_results):
            if isinstance(result, Exception):
                errors.append((scenario_type, result))
            elif result is not None:
                results[scenario_type] = result

        # If there are errors but also successes, return partial results
        if errors and results:
            for scenario_type, error in errors:
                logger.debug(
                    f"Scenario {scenario_type.value} failed for [{endpoint_info}], "
                    f"but other scenarios succeeded: {error}"
                )
            return results

        # If ALL scenarios failed, raise the first error
        if errors and not results:
            _, first_error = errors[0]
            raise first_error

        return results

    async def generate_tag_orchestrator(
        self,
        tag_name: str,
        tag_endpoints: List[Any],
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> str:
        """
        Generate an orchestrator workflow for a tag that sequences endpoints with data flow.

        The orchestrator includes:
        - CRUD lifecycle (create -> read -> update -> delete)
        - State-dependent tests (409 conflict, double delete, operation on deleted)
        - Auth tests (401 unauthorized, 403 forbidden, expired token)
        - Concurrent conflict tests
        - Resource limit tests (429)
        - Cleanup in on_stop()

        Args:
            tag_name: Name of the tag/resource group
            tag_endpoints: All endpoints in this tag
            base_workflow_content: Content of base_workflow.py
            test_data_content: Content of test_data.py
            auth_endpoints: Authentication endpoints (optional)
            custom_requirement: User-provided custom requirements
            db_type: Database type ("mongo" for MongoDB integration)

        Returns:
            Generated orchestrator Python code

        Raises:
            CodeValidationError: If generated code fails syntax validation
            AIServiceError: If AI service fails after all retries
        """
        # Load orchestrator template
        template = self.prompt_env.get_template("workflow_orchestrator.j2")

        # Format all endpoints in this tag with full details
        endpoints_list = self._format_endpoints_for_orchestrator(tag_endpoints)

        # Build class name from tag name
        class_name = self._tag_to_class_name(tag_name)

        # Render prompt
        prompt = template.render(
            tag_name=tag_name,
            endpoints_list=endpoints_list,
            auth_endpoints=self._format_endpoints_list(auth_endpoints) if auth_endpoints else "",
            base_workflow=base_workflow_content,
            test_data_content=test_data_content,
            class_name=class_name,
            custom_requirement=custom_requirement or "",
            db_type=db_type,
        )

        # Record orchestrator context and prompt
        if self.debug_recorder and self.debug_recorder.enabled:
            orchestrator_context = {
                "tag_name": tag_name,
                "endpoints_count": len(tag_endpoints),
                "class_name": class_name,
                "custom_requirement": custom_requirement or "",
                "db_type": db_type,
                "endpoints_list": endpoints_list,
            }
            await self.debug_recorder.record_orchestrator_context(
                tag=tag_name,
                context=orchestrator_context,
            )
            await self.debug_recorder.record_orchestrator_prompt(
                tag=tag_name,
                prompt=prompt,
            )

        # Call LLM with validation retry
        max_validation_retries = 2
        last_error = None
        last_code = None
        current_prompt = prompt

        for attempt in range(max_validation_retries):
            if attempt > 0 and last_error and last_code:
                current_prompt = self._render_fix_prompt(last_code, last_error)

            # Record LLM request
            if self.debug_recorder and self.debug_recorder.enabled:
                await self.debug_recorder.record_orchestrator_llm_request(
                    tag=tag_name,
                    request_data={
                        "model": self.ai_config.model,
                        "max_tokens": self.ai_config.max_tokens,
                        "temperature": self.ai_config.temperature,
                        "timeout": self.ai_config.timeout,
                        "attempt": attempt + 1,
                        "is_retry": attempt > 0,
                    },
                )

            content = await self._call_ai_service(current_prompt, f"orchestrator_{tag_name}")

            # Record LLM response
            if self.debug_recorder and self.debug_recorder.enabled:
                await self.debug_recorder.record_orchestrator_llm_response(
                    tag=tag_name,
                    response=content or "(empty response)",
                )

            if not content:
                raise AIServiceError(f"AI service returned empty response for orchestrator [{tag_name}]")

            # Detect HTML error page
            if content.strip().startswith('<') and '<html' in content.lower():
                raise AIServiceError(f"API returned HTML error page for orchestrator [{tag_name}]")

            # Extract and clean code
            extracted = self._extract_code(content)
            sanitized = self._sanitize_unicode(extracted)
            after_class_fix = self._fix_orchestrator_class_name(sanitized, class_name)
            after_bytes_fix = self._fix_bytes_literals(after_class_fix)
            after_regex_fix = self._fix_regex_strings(after_bytes_fix)
            content = after_regex_fix

            is_valid, error = self._validate_python_code(content)

            if is_valid:
                # Record final orchestrator code
                if self.debug_recorder and self.debug_recorder.enabled:
                    await self.debug_recorder.record_orchestrator_final(
                        tag=tag_name,
                        code=content,
                        summary={
                            "success": True,
                            "attempts": attempt + 1,
                            "used_fallback": False,
                            "code_length": len(content),
                        },
                    )
                if attempt > 0:
                    logger.info(
                        f"Retry SUCCEEDED for orchestrator [{tag_name}] "
                        f"on attempt {attempt + 1}/{max_validation_retries}"
                    )
                return content

            last_error = error
            last_code = content

            if attempt < max_validation_retries - 1:
                logger.debug(
                    f"Validation failed for orchestrator [{tag_name}], "
                    f"attempt {attempt + 1}/{max_validation_retries}: {error}. Retrying..."
                )
                await asyncio.sleep(1)

        raise CodeValidationError(
            "orchestrator",
            last_error,
            last_code,
            endpoint_info=f"tag: {tag_name}"
        )

    def _format_endpoints_for_orchestrator(self, endpoints: List[Any]) -> str:
        """Format all endpoints in a tag for the orchestrator prompt."""
        lines = []

        # Group by HTTP method for clarity
        by_method = {"POST": [], "GET": [], "PUT": [], "PATCH": [], "DELETE": []}
        for ep in endpoints:
            method = ep.method.upper()
            if method in by_method:
                by_method[method].append(ep)
            else:
                by_method.setdefault("OTHER", []).append(ep)

        for method, eps in by_method.items():
            if eps:
                lines.append(f"\n{method} endpoints:")
                for ep in eps:
                    operation_id = getattr(ep, "operation_id", "") or self._generate_operation_id(ep)
                    summary = getattr(ep, "summary", "") or "No summary"
                    lines.append(f"  - {ep.path}")
                    lines.append(f"    Operation ID: {operation_id}")
                    lines.append(f"    Summary: {summary}")

                    # Include request body schema for POST/PUT/PATCH
                    if method in ["POST", "PUT", "PATCH"] and hasattr(ep, "request_body") and ep.request_body:
                        rb = ep.request_body
                        schema = getattr(rb, "schema", {})
                        if schema and isinstance(schema, dict):
                            lines.append("    Request Body Schema:")
                            schema_lines = self._format_schema(schema, indent=3)
                            lines.extend(schema_lines)

                    # Include response schema for understanding ID field
                    if hasattr(ep, "responses") and ep.responses:
                        for response in ep.responses:
                            if str(response.status_code).startswith("2"):
                                resp_schema = response.schema if response.schema else None
                                if resp_schema:
                                    lines.append(f"    Response ({response.status_code}) Schema:")
                                    schema_lines = self._format_response_schema(resp_schema, indent=3)
                                    lines.extend(schema_lines)
                                break

        return "\n".join(lines)

    def _tag_to_class_name(self, tag_name: str) -> str:
        """Convert tag name to valid Python class name."""
        sanitized = self._sanitize_identifier(tag_name)
        words = sanitized.replace("_", " ").split()
        return "".join(word.capitalize() for word in words) or "Default"

    def _fix_orchestrator_class_name(self, code: str, expected_class_name: str) -> str:
        """Fix orchestrator class name to match expected naming convention."""
        import re

        expected_full_name = f"{expected_class_name}Orchestrator"

        # Find class definition that inherits from BaseWorkflow and SequentialTaskSet
        pattern = r'class\s+(\w+)\s*\([^)]*(?:BaseWorkflow|SequentialTaskSet)[^)]*\)\s*:'
        match = re.search(pattern, code)

        if match:
            actual_class_name = match.group(1)
            if actual_class_name != expected_full_name:
                logger.debug(f"Fixing orchestrator class name: {actual_class_name} -> {expected_full_name}")
                code = re.sub(
                    rf'\bclass\s+{re.escape(actual_class_name)}\s*\(',
                    f'class {expected_full_name}(',
                    code
                )
                code = re.sub(
                    rf'\b{re.escape(actual_class_name)}\b',
                    expected_full_name,
                    code
                )

        return code

    async def _generate_llm_scenario(
        self,
        scenario_type: ScenarioType,
        endpoint: Any,
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        tag_name: str = "default",
        all_endpoints: Optional[List[Any]] = None,
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> Optional[str]:
        """
        Generate a scenario using LLM for a single endpoint.

        Args:
            scenario_type: Type of scenario to generate
            endpoint: Single endpoint to generate tests for
            base_workflow_content: Base workflow code
            test_data_content: Test data code
            auth_endpoints: Auth endpoints
            tag_name: Tag/group name for debug logging
            all_endpoints: All endpoints from OpenAPI spec (for finding related CREATE endpoints)
            custom_requirement: User-provided custom requirements for test generation
            db_type: Database type ("mongo" for MongoDB integration)

        Returns:
            Generated Python code, or None if generation was skipped
        """
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        scenario_name = scenario_type.value
        endpoint_info = f"{endpoint.method} {endpoint.path}"

        if self.progress:
            self.progress.scenario_start(endpoint_info, scenario_name)

        template_name = self.PROMPT_TEMPLATES.get(scenario_type)
        if not template_name:
            raise ValueError(f"No prompt template for scenario type: {scenario_type}")

        # Let template loading errors propagate naturally
        template = self.prompt_env.get_template(template_name)

        # Format single endpoint with full details
        # For negative/security tests: exclude 2xx responses so the LLM doesn't copy success codes
        exclude_2xx = scenario_type in (ScenarioType.NEGATIVE, ScenarioType.SECURITY)
        endpoint_details = self._format_single_endpoint(endpoint, exclude_2xx=exclude_2xx)

        # Build class name from operation_id
        class_name = self._operation_to_class_name(endpoint)

        # Pre-compute exact status codes + descriptions for this scenario type
        # Handles spec vs fallback transparently - the template just gets the codes
        has_auth = bool(auth_endpoints)
        codes_with_desc = self._precompute_scenario_status_codes(endpoint, scenario_type, has_auth)
        expected_status_codes = [code for code, _ in codes_with_desc]
        expected_status_info = self._format_status_codes_for_prompt(codes_with_desc)

        # Skip generation if spec defines responses but no appropriate codes for this scenario type
        all_status_codes = self._extract_expected_status_codes(endpoint)

        # Skip positive generation if spec defines responses but no 2xx codes
        # (e.g., response-test endpoints that intentionally return 4xx/5xx)
        if scenario_type == ScenarioType.POSITIVE and all_status_codes and not expected_status_codes:
            skip_reason = f"spec defines no 2xx codes (defined: {all_status_codes})"
            logger.info(f"Skipping positive workflow for [{endpoint_info}] - {skip_reason}")
            if self.progress:
                self.progress.scenario_skipped(endpoint_info, scenario_name, skip_reason)
            return None

        # Skip negative generation if spec defines responses but no 4xx codes
        if scenario_type == ScenarioType.NEGATIVE and all_status_codes and not expected_status_codes:
            skip_reason = f"spec defines no 4xx codes (defined: {all_status_codes})"
            logger.info(f"Skipping negative workflow for [{endpoint_info}] - {skip_reason}")
            if self.progress:
                self.progress.scenario_skipped(endpoint_info, scenario_name, skip_reason)
            return None

        # Ensure expected_status_codes is never empty - use sensible defaults as last resort
        if not expected_status_codes:
            logger.warning(f"No expected status codes for {endpoint_info} {scenario_name} - using fallback")
            if scenario_type == ScenarioType.POSITIVE:
                expected_status_codes = [200]
            elif scenario_type == ScenarioType.NEGATIVE:
                expected_status_codes = [400, 422]
            elif scenario_type == ScenarioType.SECURITY:
                expected_status_codes = [200, 400, 422]

        # Pre-compute security injection points - skip if no valid targets
        injection_points = ""
        if scenario_type == ScenarioType.SECURITY:
            injection_points_result = self._precompute_injection_points(endpoint)
            if injection_points_result is None:
                skip_reason = "no valid injection points (no body string fields or query params)"
                logger.info(f"Skipping security workflow for [{endpoint_info}] - {skip_reason}")
                if self.progress:
                    self.progress.scenario_skipped(endpoint_info, scenario_name, skip_reason)
                return None
            injection_points = injection_points_result

        # Pre-compute negative test scenarios - skip if no valid tests possible
        negative_scenarios = ""
        if scenario_type == ScenarioType.NEGATIVE:
            negative_scenarios = self._precompute_negative_scenarios(endpoint)
            if not negative_scenarios:
                skip_reason = "no testable scenarios (no path params, body fields, or query params)"
                logger.info(f"Skipping negative workflow for [{endpoint_info}] - {skip_reason}")
                if self.progress:
                    self.progress.scenario_skipped(endpoint_info, scenario_name, skip_reason)
                return None

        # Pre-compute positive field details - skip if nothing to test
        positive_fields = ""
        if scenario_type == ScenarioType.POSITIVE:
            positive_fields = self._precompute_positive_fields(endpoint)

        # Find related CREATE endpoints for setup steps
        # Provide setup for: non-POST endpoints, OR POST endpoints with path params
        # (POST with path params needs parent resources created first)
        setup_endpoints_section = ""
        setup_count = 0
        has_path_params = any(
            seg.startswith("{") for seg in endpoint.path.split("/") if seg
        )
        if all_endpoints and (endpoint.method.upper() != "POST" or has_path_params):
            related_create_endpoints = self._find_related_create_endpoints(endpoint, all_endpoints)
            setup_count = len(related_create_endpoints)
            setup_endpoints_section = self._format_related_create_endpoints(related_create_endpoints)

        # Log pre-computation results
        if self.progress:
            detail_parts = [f"expected_status={expected_status_codes}"]
            if setup_count:
                detail_parts.append(f"{setup_count} setup endpoints")
            if positive_fields:
                detail_parts.append("fields pre-computed")
            if negative_scenarios:
                detail_parts.append("scenarios pre-computed")
            if injection_points:
                detail_parts.append("injection points found")
            self.progress.scenario_detail(endpoint_info, scenario_name, ", ".join(detail_parts))

        # Build context dict for template rendering
        template_context = {
            "endpoint": endpoint_details,
            "auth_endpoints": self._format_endpoints_list(auth_endpoints) if auth_endpoints else "",
            "base_workflow": base_workflow_content,
            "test_data_content": test_data_content,
            "class_name": class_name,
            "operation_id": operation_id,
            "method": endpoint.method,
            "path": endpoint.path,
            "endpoint_expected_status": expected_status_codes,
            "expected_status_info": expected_status_info,
            "injection_points": injection_points,
            "negative_scenarios": negative_scenarios,
            "positive_fields": positive_fields,
            "setup_endpoints": setup_endpoints_section,
            "custom_requirement": custom_requirement or "",
            "db_type": db_type,
        }

        # Validate required context variables exist (prevents LLM hallucination from empty context)
        self._validate_template_context(template_context, scenario_type)

        # Render prompt
        prompt = template.render(**template_context)

        # Get endpoint directory name for debug recording
        endpoint_dir_name = self.get_endpoint_dir_name(endpoint)

        # Record endpoint details and context
        if self.debug_recorder and self.debug_recorder.enabled:
            await self.debug_recorder.record_endpoint_details(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                endpoint=endpoint,
                formatted_details=endpoint_details,
            )
            # Record context without large content (base_workflow, test_data) to save space
            context_for_debug = {
                "endpoint_details": endpoint_details,
                "class_name": class_name,
                "operation_id": operation_id,
                "method": endpoint.method,
                "path": endpoint.path,
                "expected_status_codes": expected_status_codes,
                "setup_endpoints": setup_endpoints_section,
                "custom_requirement": custom_requirement or "",
                "db_type": db_type,
            }
            await self.debug_recorder.record_scenario_context(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                scenario_type=scenario_name,
                context=context_for_debug,
            )
            await self.debug_recorder.record_scenario_prompt(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                scenario_type=scenario_name,
                prompt=prompt,
            )

        # Call LLM with validation retry (error-aware on retry)
        max_validation_retries = 2
        last_error = None
        last_code = None
        last_is_semantic = False
        current_prompt = prompt  # Start with the original prompt
        all_errors: List[str] = []  # Track all errors for debug recording

        for attempt in range(max_validation_retries):
            # On retry, use error-aware fix prompt instead of original
            if attempt > 0 and last_error and last_code:
                if last_is_semantic:
                    current_prompt = self._render_semantic_fix_prompt(
                        last_code, last_error,
                        endpoint_expected_status=expected_status_codes,
                        endpoint_path=endpoint.path,
                        endpoint_method=endpoint.method.upper(),
                    )
                else:
                    current_prompt = self._render_fix_prompt(last_code, last_error)

            # Record LLM request
            if self.debug_recorder and self.debug_recorder.enabled:
                llm_request_data = {
                    "model": self.ai_config.model,
                    "max_tokens": self.ai_config.max_tokens,
                    "temperature": self.ai_config.temperature,
                    "timeout": self.ai_config.timeout,
                    "attempt": attempt + 1,
                    "is_retry": attempt > 0,
                }
                await self.debug_recorder.record_llm_request(
                    tag=tag_name,
                    endpoint_dir_name=endpoint_dir_name,
                    scenario_type=scenario_name,
                    request_data=llm_request_data,
                )

            if self.progress:
                attempt_label = f"attempt {attempt + 1}/{max_validation_retries}" if attempt > 0 else "calling LLM"
                self.progress.scenario_detail(endpoint_info, scenario_name, attempt_label)

            content = await self._call_ai_service(current_prompt, scenario_type.value)

            # Record raw LLM response
            if self.debug_recorder and self.debug_recorder.enabled:
                await self.debug_recorder.record_llm_response(
                    tag=tag_name,
                    endpoint_dir_name=endpoint_dir_name,
                    scenario_type=scenario_name,
                    response=content or "(empty response)",
                )

            if not content:
                raise AIServiceError(f"AI service returned empty response for {scenario_type.value} [{endpoint.method} {endpoint.path}]")

            # Detect if API returned HTML error page instead of code
            if content.strip().startswith('<') and '<html' in content.lower():
                raise AIServiceError(
                    f"API returned HTML error page instead of code for {scenario_type.value} "
                    f"[{endpoint.method} {endpoint.path}]. "
                    f"This may indicate an API error or rate limiting. First 200 chars: {content[:200]}"
                )

            # Extract code from response
            extracted = self._extract_code(content)

            # Record extracted code
            if self.debug_recorder and self.debug_recorder.enabled:
                await self.debug_recorder.record_extracted_code(
                    tag=tag_name,
                    endpoint_dir_name=endpoint_dir_name,
                    scenario_type=scenario_name,
                    code=extracted,
                )

            # Sanitize any non-ASCII Unicode characters the LLM may have injected
            sanitized = self._sanitize_unicode(extracted)

            # Fix class name to match expected naming convention
            # LLMs sometimes ignore the template and generate their own class names
            after_class_fix = self._fix_class_name(sanitized, class_name, scenario_type.value)

            # Fix bytes literals with unicode (b'tëst' → 'tëst'.encode('utf-8'))
            after_bytes_fix = self._fix_bytes_literals(after_class_fix)

            # Fix regex strings (convert to raw strings to avoid SyntaxWarnings)
            after_regex_fix = self._fix_regex_strings(after_bytes_fix)

            # Fix missing imports (LLM sometimes forgets to import get_task_weight, etc.)
            after_import_fix = self._fix_missing_imports(after_regex_fix)

            # Fix redundant .isoformat() calls on date methods that already return strings
            after_isoformat_fix = self._fix_isoformat_calls(after_import_fix)

            content = after_isoformat_fix

            # Record processed code (after all fixes)
            if self.debug_recorder and self.debug_recorder.enabled:
                await self.debug_recorder.record_processed_code(
                    tag=tag_name,
                    endpoint_dir_name=endpoint_dir_name,
                    scenario_type=scenario_name,
                    code=content,
                )

            is_valid, error = self._validate_python_code(content)

            # Record validation result
            if self.debug_recorder and self.debug_recorder.enabled:
                await self.debug_recorder.record_validation_result(
                    tag=tag_name,
                    endpoint_dir_name=endpoint_dir_name,
                    scenario_type=scenario_name,
                    is_valid=is_valid,
                    error=error if not is_valid else None,
                    checks=[{"check": "python_syntax", "passed": is_valid, "error": error}],
                )

            if not is_valid:
                if self.progress:
                    self.progress.scenario_detail(endpoint_info, scenario_name, f"syntax FAILED: {error[:120]}")

            if is_valid:
                # Syntax OK - now run semantic validation
                all_endpoint_paths = []
                if all_endpoints:
                    all_endpoint_paths = [
                        getattr(ep, "path", "") for ep in all_endpoints
                        if getattr(ep, "path", "")
                    ]

                # Extract request body schema for schema compliance checks
                request_body_schema = None
                if hasattr(endpoint, "request_body") and endpoint.request_body:
                    request_body_schema = getattr(endpoint.request_body, "schema", None)

                semantic_result = self._code_validator.validate(
                    code=content,
                    scenario_type=scenario_type.value,
                    endpoint_path=endpoint.path,
                    all_endpoint_paths=all_endpoint_paths,
                    request_body_schema=request_body_schema,
                )

                if semantic_result.is_valid:
                    # Both syntax and semantic validation passed
                    if self.debug_recorder and self.debug_recorder.enabled:
                        await self.debug_recorder.record_final_code(
                            tag=tag_name,
                            endpoint_dir_name=endpoint_dir_name,
                            scenario_type=scenario_name,
                            code=content,
                        )
                        await self.debug_recorder.record_scenario_summary(
                            tag=tag_name,
                            endpoint_dir_name=endpoint_dir_name,
                            scenario_type=scenario_name,
                            summary={
                                "success": True,
                                "attempts": attempt + 1,
                                "used_fallback": False,
                                "code_length": len(content),
                            },
                        )
                    if attempt > 0:
                        logger.info(
                            f"Retry SUCCEEDED for {scenario_type.value} "
                            f"[{endpoint.method} {endpoint.path}] on attempt {attempt + 1}/{max_validation_retries}"
                        )
                    if self.progress:
                        self.progress.scenario_done(endpoint_info, scenario_name)
                    return content
                else:
                    # Semantic validation failed - use semantic fix prompt on retry
                    error = semantic_result.error_message
                    is_semantic_error = True
                    logger.info(
                        f"Semantic validation failed for {scenario_type.value} "
                        f"[{endpoint.method} {endpoint.path}]: "
                        f"{len(semantic_result.violations)} violation(s)"
                    )
                    if self.progress:
                        violations_summary = "; ".join(
                            f"[{v.rule}] {v.message[:80]}" for v in semantic_result.violations[:3]
                        )
                        self.progress.scenario_detail(
                            endpoint_info, scenario_name, f"semantic FAILED: {violations_summary}"
                        )
            else:
                is_semantic_error = False

            # Save for error reporting and fix prompt on retry
            last_error = error
            last_code = content
            last_is_semantic = is_semantic_error if is_valid else False
            all_errors.append(f"Attempt {attempt + 1}: {error}")

            if attempt < max_validation_retries - 1:
                # Record retry attempt
                if self.debug_recorder and self.debug_recorder.enabled:
                    fix_prompt = (
                        self._render_semantic_fix_prompt(
                            content, error,
                            endpoint_expected_status=expected_status_codes,
                            endpoint_path=endpoint.path,
                            endpoint_method=endpoint.method.upper(),
                        )
                        if last_is_semantic
                        else self._render_fix_prompt(content, error)
                    )
                    await self.debug_recorder.record_retry_attempt(
                        tag=tag_name,
                        endpoint_dir_name=endpoint_dir_name,
                        scenario_type=scenario_name,
                        attempt=attempt + 1,
                        error=error,
                        bad_code=content,
                        fix_prompt=fix_prompt,
                        llm_response="(pending next attempt)",
                        extracted_code="(pending next attempt)",
                        validation_result={"valid": False, "error": error},
                    )
                error_type = "Semantic" if last_is_semantic else "Syntax"
                logger.debug(
                    f"{error_type} validation failed for {scenario_type.value} "
                    f"[{endpoint.method} {endpoint.path}], attempt {attempt + 1}/{max_validation_retries}: "
                    f"{error}. Retrying..."
                )
                if self.progress:
                    self.progress.scenario_retry(
                        endpoint_info, scenario_name, attempt + 1, max_validation_retries,
                        f"{error_type}: {error}"
                    )
                await asyncio.sleep(1)
            else:
                logger.debug(
                    f"Retry FAILED for {scenario_type.value} "
                    f"[{endpoint.method} {endpoint.path}] after {max_validation_retries} attempts. "
                    f"Final error: {error}"
                )

        # All retries exhausted
        raise CodeValidationError(
            scenario_type.value,
            last_error,
            last_code,
            endpoint_info=f"{endpoint.method} {endpoint.path}"
        )

    @staticmethod
    def _unwrap_nullable_schema(schema: dict) -> tuple:
        """Unwrap OpenAPI 3.1 anyOf nullable pattern and 3.0 nullable flag.

        Handles:
        - OpenAPI 3.1: anyOf: [{actual_type_schema}, {type: null}]
        - OpenAPI 3.0: {type: string, nullable: true}

        Returns:
            (unwrapped_schema, is_nullable) tuple.
            If nullable pattern detected, returns the real type schema and True.
            Otherwise returns the original schema unchanged and False.
        """
        if not isinstance(schema, dict):
            return schema, False

        # OpenAPI 3.0: nullable flag alongside type
        if schema.get("nullable") is True and schema.get("type"):
            return schema, True

        # OpenAPI 3.1: anyOf with exactly one null type variant
        any_of = schema.get("anyOf") or schema.get("oneOf")
        if any_of and isinstance(any_of, list):
            null_variants = [v for v in any_of if isinstance(v, dict) and v.get("type") == "null"]
            real_variants = [v for v in any_of if isinstance(v, dict) and v.get("type") != "null"]
            if len(null_variants) >= 1 and len(real_variants) == 1:
                return real_variants[0], True

        return schema, False

    def _extract_all_properties(self, schema: dict) -> Tuple[Dict[str, Any], List[str]]:
        """Extract all properties from a schema, handling allOf merging and discriminated unions.

        Handles:
        - Direct properties
        - allOf: merges properties from all items (items are already $ref-resolved by parser)
        - oneOf/anyOf discriminated unions: merges properties from all variants
          (union of all possible fields, so all can be recognized)

        Args:
            schema: Request body schema dict

        Returns:
            (properties_dict, required_list) tuple
        """
        if not isinstance(schema, dict):
            return {}, []

        properties = dict(schema.get("properties", {}))
        required_list = list(schema.get("required", []))

        # Handle allOf: merge properties from all items
        all_of = schema.get("allOf")
        if all_of and isinstance(all_of, list):
            for item in all_of:
                if not isinstance(item, dict):
                    continue
                # Item may itself have allOf (nested composition)
                item_props = item.get("properties", {})
                if item_props:
                    properties.update(item_props)
                    required_list.extend(item.get("required", []))
                # If item has its own allOf (e.g. resolved $ref that uses composition)
                nested_all_of = item.get("allOf")
                if nested_all_of and isinstance(nested_all_of, list):
                    for sub in nested_all_of:
                        if isinstance(sub, dict) and sub.get("properties"):
                            properties.update(sub["properties"])
                            required_list.extend(sub.get("required", []))

        # Handle discriminated unions: merge all variant properties
        one_of = schema.get("oneOf") or schema.get("anyOf")
        if one_of and isinstance(one_of, list) and not properties:
            for variant in one_of:
                if not isinstance(variant, dict):
                    continue
                v_props = variant.get("properties", {})
                if v_props:
                    properties.update(v_props)
                    required_list.extend(variant.get("required", []))
                # Variant might use allOf internally
                v_all_of = variant.get("allOf")
                if v_all_of and isinstance(v_all_of, list):
                    for sub in v_all_of:
                        if isinstance(sub, dict) and sub.get("properties"):
                            properties.update(sub["properties"])
                            required_list.extend(sub.get("required", []))

        return properties, list(set(required_list))

    @staticmethod
    def _escape_for_python_string(value: str) -> str:
        """Escape a string for safe embedding in a Python double-quoted string literal.

        Handles backslashes, quotes, and control characters.
        """
        if not isinstance(value, str):
            return str(value)
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

    @staticmethod
    def _escape_for_raw_string(value: str) -> str:
        """Escape a string for safe embedding in a Python raw string literal (r"...").

        For raw strings, backslashes are NOT escaped (they're literal in raw strings).
        Only quotes need escaping. Used for regex patterns since the LLM wraps them in r"...".
        """
        if not isinstance(value, str):
            return str(value)
        # In raw strings, only quotes need escaping (and raw strings can't end with odd backslashes)
        # For simplicity, we just escape quotes - regex patterns rarely have quotes
        return value.replace('"', '\\"')

    def _get_type_instruction(
        self,
        field_schema: dict,
        _object_ancestors: Optional[frozenset] = None,
        field_name: str = "",
    ) -> str:
        """Map a field schema to a Python code instruction for generating a valid value.

        Handles enums, patterns, formats, string/integer/number/boolean/array/object types,
        and nested structures. Used by both _precompute_positive_fields and
        _precompute_object_instruction to avoid duplicated logic.

        Args:
            field_schema: The (already unwrapped) field schema dict
            _object_ancestors: Optional frozenset of schema IDs for circular reference detection
                             (passed through to _precompute_object_instruction for nested objects)
            field_name: Optional field name for inferring the appropriate generator

        Returns:
            Python code string for generating a valid test value
        """
        if not isinstance(field_schema, dict):
            return "test_data_generator.generate_string(length=10)"

        field_type = field_schema.get("type", "string")
        field_format = field_schema.get("format", "")
        field_enum = field_schema.get("enum")
        field_pattern = field_schema.get("pattern")
        field_items = field_schema.get("items", {})
        field_name_lower = field_name.lower() if field_name else ""

        # Enum: always takes priority
        if field_enum:
            return f"random.choice({field_enum})"

        # Pattern: regex-based generation (use raw string since prompts tell LLM to use r"...")
        if field_pattern:
            escaped = self._escape_for_raw_string(field_pattern)
            return f'test_data_generator.generate_string(pattern=r"{escaped}")'

        # Format-specific generators - use dedicated methods that return strings directly
        # IMPORTANT: These methods return strings, NOT datetime objects!
        if field_format == "date":
            return "test_data_generator.random_date()"  # Returns "2024-03-15" string
        if field_format == "date-time":
            return "test_data_generator.random_datetime()"  # Returns "2024-03-15T14:30:00" string
        if field_format == "time":
            return "test_data_generator.random_time()"  # Returns "14:30:00" string
        if field_format == "email":
            return "test_data_generator.generate_email()"
        if field_format == "uuid":
            return "test_data_generator.random_uuid()"
        if field_format in ("uri", "url"):
            return "test_data_generator.random_uri()"
        if field_format == "ipv4":
            return "test_data_generator.random_ipv4()"
        if field_format == "ipv6":
            return "test_data_generator.random_ipv6()"
        if field_format == "hostname":
            return "test_data_generator.random_hostname()"
        if field_format == "byte":
            return "test_data_generator.random_base64()"

        # Type-based generators
        if field_type == "string":
            min_length = field_schema.get("minLength", 0)
            max_length = field_schema.get("maxLength", 50)

            # Infer generator from field name (generic patterns, not hardcoded)
            if field_name_lower:
                # Country codes: 2-3 char fields with "country" in name
                if "country" in field_name_lower and max_length <= 3:
                    return "test_data_generator.random_country_code()"
                # Currency codes: 3-char fields with "currency" in name
                if "currency" in field_name_lower and max_length <= 4:
                    return "test_data_generator.random_currency_code()"
                # Locale fields
                if "locale" in field_name_lower:
                    return "test_data_generator.random_locale()"
                # Color fields
                if "color" in field_name_lower or "colour" in field_name_lower:
                    return "test_data_generator.random_hex_color()"
                # Date/time fields by name (when format isn't specified)
                if any(kw in field_name_lower for kw in ["created_at", "updated_at", "modified_at", "timestamp"]):
                    return "test_data_generator.random_datetime()"
                if any(kw in field_name_lower for kw in ["birth_date", "start_date", "end_date", "date"]) and "time" not in field_name_lower:
                    return "test_data_generator.random_date()"
                # Email fields by name
                if "email" in field_name_lower:
                    return "test_data_generator.generate_email()"
                # URL fields by name
                if any(kw in field_name_lower for kw in ["url", "uri", "website", "link"]):
                    return "test_data_generator.random_uri()"
                # Phone fields by name
                if "phone" in field_name_lower:
                    return "test_data_generator.random_ipv4()"  # Faker phone can have issues, use simple format
                # IP address fields by name
                if "ipv4" in field_name_lower or "ip_address" in field_name_lower:
                    return "test_data_generator.random_ipv4()"
                if "ipv6" in field_name_lower:
                    return "test_data_generator.random_ipv6()"
                # Hostname fields by name
                if "hostname" in field_name_lower or "host" in field_name_lower:
                    return "test_data_generator.random_hostname()"

            # Default string generation with length constraints
            length = max_length if isinstance(max_length, int) and max_length <= 50 else 10
            if not isinstance(length, int):
                length = 10
            return f"test_data_generator.generate_string(length={length})"

        if field_type == "integer":
            exclusive_min = field_schema.get("exclusiveMinimum")
            exclusive_max = field_schema.get("exclusiveMaximum")
            min_val = exclusive_min if exclusive_min is not None else field_schema.get("minimum", 1)
            max_val = exclusive_max if exclusive_max is not None else field_schema.get("maximum", 1000)
            exclusive = exclusive_min is not None or exclusive_max is not None
            multiple_of = field_schema.get("multipleOf")
            parts = [f"min_val={min_val}", f"max_val={max_val}"]
            if exclusive:
                parts.append("exclusive=True")
            if multiple_of:
                parts.append(f"multiple_of={multiple_of}")
            return f"test_data_generator.generate_integer({', '.join(parts)})"

        if field_type == "number":
            exclusive_min = field_schema.get("exclusiveMinimum")
            exclusive_max = field_schema.get("exclusiveMaximum")
            min_val = exclusive_min if exclusive_min is not None else field_schema.get("minimum", 0.0)
            max_val = exclusive_max if exclusive_max is not None else field_schema.get("maximum", 1000.0)
            exclusive = exclusive_min is not None or exclusive_max is not None
            if exclusive:
                return f"test_data_generator.generate_float(min_val={min_val}, max_val={max_val}, exclusive=True)"
            return f"test_data_generator.generate_float(min_val={min_val}, max_val={max_val})"

        if field_type == "boolean":
            return "test_data_generator.generate_boolean()"

        if field_type == "object":
            sub_props = field_schema.get("properties", {})
            if sub_props:
                return self._precompute_object_instruction(field_schema, _object_ancestors)
            add_props = field_schema.get("additionalProperties")
            if add_props and isinstance(add_props, dict):
                val_type = add_props.get("type", "string")
                if val_type == "integer":
                    val_gen = "test_data_generator.generate_integer()"
                elif val_type == "number":
                    val_gen = "test_data_generator.generate_float()"
                elif val_type == "boolean":
                    val_gen = "test_data_generator.generate_boolean()"
                else:
                    val_gen = "test_data_generator.generate_string()"
                return '{f"key_{i}": ' + val_gen + ' for i in range(3)}'
            if add_props:
                return '{"key1": "value1", "key2": "value2"}'
            return "{}"

        if field_type == "array":
            items_type = field_items.get("type", "string") if isinstance(field_items, dict) else "string"
            items_enum = field_items.get("enum") if isinstance(field_items, dict) else None
            items_ref = field_items.get("$ref") if isinstance(field_items, dict) else None
            if items_enum:
                return f"[random.choice({items_enum}) for _ in range(3)]"
            if items_type == "object" or items_ref:
                items_props = field_items.get("properties", {}) if isinstance(field_items, dict) else {}
                if items_props:
                    obj_instr = self._precompute_object_instruction(field_items, _object_ancestors)
                    return f"[{obj_instr} for _ in range(2)]"
                return "[{}]"
            if items_type == "string":
                return "[test_data_generator.generate_string() for _ in range(3)]"
            if items_type == "integer":
                return "[test_data_generator.generate_integer() for _ in range(3)]"
            if items_type == "number":
                return "[test_data_generator.generate_float() for _ in range(3)]"
            if items_type == "boolean":
                return "[test_data_generator.generate_boolean() for _ in range(3)]"
            if items_type == "array":
                return "[[test_data_generator.generate_string() for _ in range(2)] for _ in range(3)]"
            return "[test_data_generator.generate_string() for _ in range(3)]"

        # Fallback for unknown types
        return "test_data_generator.generate_string(length=10)"

    def _format_single_endpoint(self, endpoint: Any, exclude_2xx: bool = False) -> str:
        """Format a single endpoint with full details for the prompt.

        Args:
            endpoint: Endpoint object to format
            exclude_2xx: If True, omit 2xx responses from the output (for negative/security tests)
        """
        lines = []

        # Basic info
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        summary = getattr(endpoint, "summary", "") or "No summary"
        description = getattr(endpoint, "description", "") or ""

        lines.append(f"Operation: {endpoint.method.upper()} {endpoint.path}")
        lines.append(f"Operation ID: {operation_id}")
        lines.append(f"Summary: {summary}")
        if description:
            lines.append(f"Description: {description}")

        # Parameters with full details
        has_cookie_params = False
        has_header_params = False
        if hasattr(endpoint, "parameters") and endpoint.parameters:
            lines.append("\nParameters:")
            for param in endpoint.parameters:
                param_name = getattr(param, "name", "unknown")
                param_in = getattr(param, "location", None)
                if param_in is None:
                    param_in = getattr(param, "in_", "query")
                # Handle ParameterType enum
                if hasattr(param_in, "value"):
                    param_in = param_in.value

                # Track cookie and header params for type coercion warning
                if param_in == "cookie":
                    has_cookie_params = True
                if param_in == "header":
                    has_header_params = True

                param_required = getattr(param, "required", False)
                param_type = getattr(param, "type", "string")
                param_format = getattr(param, "format", None)
                param_enum = getattr(param, "enum", None)
                param_desc = getattr(param, "description", None)
                param_pattern = getattr(param, "pattern", None)
                param_min_length = getattr(param, "min_length", None)
                param_max_length = getattr(param, "max_length", None)
                param_minimum = getattr(param, "minimum", None)
                param_maximum = getattr(param, "maximum", None)

                required_str = "(required)" if param_required else "(optional)"
                type_str = param_type
                if param_format:
                    type_str = f"{param_type} [{param_format}]"

                lines.append(f"  - {param_name} [{param_in}]: {type_str} {required_str}")
                if param_enum:
                    # Standardized format for enum values
                    lines.append(f"      allowed values: {param_enum}")
                if param_pattern:
                    lines.append(f"      pattern: {param_pattern}")
                if param_min_length is not None or param_max_length is not None:
                    length_constraints = []
                    if param_min_length is not None:
                        length_constraints.append(f"minLength={param_min_length}")
                    if param_max_length is not None:
                        length_constraints.append(f"maxLength={param_max_length}")
                    lines.append(f"      constraints: {', '.join(length_constraints)}")
                if param_minimum is not None or param_maximum is not None:
                    range_constraints = []
                    if param_minimum is not None:
                        range_constraints.append(f"min={param_minimum}")
                    if param_maximum is not None:
                        range_constraints.append(f"max={param_maximum}")
                    lines.append(f"      constraints: {', '.join(range_constraints)}")
                if param_desc:
                    lines.append(f"      description: {param_desc[:80]}")

            # Add type coercion warnings for cookies and headers
            if has_cookie_params:
                lines.append("")
                lines.append("  *** COOKIE VALUES MUST BE STRINGS ***")
                lines.append("  When passing cookies, ALL values must be strings, not integers or other types.")
                lines.append("  WRONG: cookies={'session_id': 123}")
                lines.append("  CORRECT: cookies={'session_id': '123'}")

            if has_header_params:
                lines.append("")
                lines.append("  *** HEADER VALUES MUST BE STRINGS ***")
                lines.append("  When passing headers, ALL values must be strings.")
                lines.append("  WRONG: headers={'X-Count': 10}")
                lines.append("  CORRECT: headers={'X-Count': '10'}")

        # Request body with FULL SCHEMA DETAILS
        if hasattr(endpoint, "request_body") and endpoint.request_body:
            rb = endpoint.request_body
            content_type = getattr(rb, "content_type", "application/json")
            schema = getattr(rb, "schema", {})
            rb_required = getattr(rb, "required", True)
            rb_desc = getattr(rb, "description", None)

            lines.append("\nRequest Body:")
            lines.append(f"  Content-Type: {content_type}")
            if rb_desc:
                lines.append(f"  Description: {rb_desc[:100]}")
            lines.append(f"  Required: {rb_required}")

            # Check for file upload - add explicit instructions
            if content_type in ["multipart/form-data", "application/octet-stream"]:
                lines.append("")
                lines.append("  *** FILE UPLOAD ENDPOINT ***")
                lines.append("  This endpoint requires multipart/form-data file upload.")
                lines.append("  DO NOT use json= parameter. Use files= with actual file data:")
                lines.append("  Example: files={'file': ('test.txt', b'file content', 'text/plain')}")
                lines.append("")

            # Format full schema with all details
            if schema and isinstance(schema, dict):
                schema_lines = self._format_schema(schema, indent=2)
                lines.extend(schema_lines)

        # Responses with FULL SCHEMA DETAILS
        if hasattr(endpoint, "responses") and endpoint.responses:
            responses = endpoint.responses
            lines.append("\nResponses:")
            if isinstance(responses, dict):
                for status_code, response in responses.items():
                    # Skip 2xx responses when exclude_2xx is set (for negative/security tests)
                    if exclude_2xx:
                        try:
                            if 200 <= int(status_code) < 300:
                                continue
                        except (ValueError, TypeError):
                            pass
                    desc = getattr(response, "description", "") if hasattr(response, "description") else str(response)
                    lines.append(f"  - {status_code}: {desc[:50] if len(str(desc)) > 50 else desc}")
                    # Add response schema if available
                    resp_schema = getattr(response, "schema", None) if hasattr(response, "schema") else None
                    if resp_schema and isinstance(resp_schema, dict):
                        lines.append(f"    Response Schema (use these EXACT field names when accessing response):")
                        schema_lines = self._format_response_schema(resp_schema, indent=3)
                        lines.extend(schema_lines)
            elif isinstance(responses, list):
                for response in responses:
                    status_code = getattr(response, "status_code", "???")
                    # Skip 2xx responses when exclude_2xx is set
                    if exclude_2xx:
                        try:
                            if 200 <= int(status_code) < 300:
                                continue
                        except (ValueError, TypeError):
                            pass
                    desc = getattr(response, "description", "")
                    lines.append(f"  - {status_code}: {desc[:50] if len(str(desc)) > 50 else desc}")
                    # Add response schema if available
                    resp_schema = getattr(response, "schema", None)
                    if resp_schema and isinstance(resp_schema, dict):
                        lines.append(f"    Response Schema (use these EXACT field names when accessing response):")
                        schema_lines = self._format_response_schema(resp_schema, indent=3)
                        lines.extend(schema_lines)

        return "\n".join(lines)

    def _format_schema(self, schema: dict, indent: int = 0) -> List[str]:
        """
        Format a JSON Schema with full details for the LLM prompt.

        Includes:
        - Property names and types
        - Required fields (marked with *)
        - Field descriptions
        - Constraints (minLength, maxLength, pattern, enum, min, max, etc.)
        - Nested objects and arrays
        - oneOf/anyOf union types with discriminator info
        """
        lines = []
        prefix = "  " * indent

        # Handle oneOf/anyOf union types (discriminated unions)
        one_of = schema.get("oneOf") or schema.get("anyOf")
        discriminator = schema.get("discriminator")

        if one_of and isinstance(one_of, list):
            # This is a union type - format it specially
            if discriminator:
                prop_name = discriminator.get("propertyName", "type")
                lines.append(f"{prefix}Schema: DISCRIMINATED UNION")
                lines.append(f"{prefix}  *** DISCRIMINATOR FIELD: {prop_name} (REQUIRED) ***")
                lines.append(f"{prefix}  You MUST include '{prop_name}' to specify which variant to use.")
                lines.append(f"{prefix}")

                # Get mapping if available
                mapping = discriminator.get("mapping", {})
                if mapping:
                    lines.append(f"{prefix}  Valid '{prop_name}' values and their schemas:")
                    for disc_value, ref in mapping.items():
                        lines.append(f"{prefix}    - {prop_name}=\"{disc_value}\":")
                        # Try to resolve the reference and show fields
                        variant_schema = self._resolve_ref_in_union(ref, one_of)
                        if variant_schema:
                            variant_props = variant_schema.get("properties", {})
                            variant_required = variant_schema.get("required", [])
                            for vp_name, vp_schema in variant_props.items():
                                if vp_name == prop_name:
                                    continue  # Skip discriminator field, already shown
                                vp_type = vp_schema.get("type", "any")
                                req_marker = " (REQUIRED)" if vp_name in variant_required else ""
                                lines.append(f"{prefix}        {vp_name}: {vp_type}{req_marker}")
                        lines.append(f"{prefix}")
            else:
                # oneOf/anyOf without discriminator
                lines.append(f"{prefix}Schema: UNION TYPE (oneOf/anyOf)")
                lines.append(f"{prefix}  Send ONE of the following object types:")
                for i, variant in enumerate(one_of, 1):
                    variant_schema = variant
                    if "$ref" in variant:
                        # Just note the reference, we can't resolve it fully here
                        lines.append(f"{prefix}  Option {i}: {variant['$ref']}")
                    else:
                        variant_props = variant.get("properties", {})
                        if variant_props:
                            lines.append(f"{prefix}  Option {i}:")
                            for vp_name, vp_schema in variant_props.items():
                                vp_type = vp_schema.get("type", "any")
                                lines.append(f"{prefix}    - {vp_name}: {vp_type}")
            return lines

        schema_type = schema.get("type", "object")
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        if properties:
            lines.append(f"{prefix}Schema:")
            if required_fields:
                lines.append(f"{prefix}  Required fields: {required_fields}")
            else:
                lines.append(f"{prefix}  Required fields: none")
            lines.append(f"{prefix}  Properties (use these EXACT field names in your code):")

            for prop_name, prop_schema in properties.items():
                is_required = prop_name in required_fields

                # Unwrap nullable pattern (OpenAPI 3.1 anyOf/3.0 nullable)
                unwrapped, is_nullable = self._unwrap_nullable_schema(prop_schema)
                nullable_marker = ", nullable" if is_nullable else ""
                req_marker = f" (REQUIRED{nullable_marker})" if is_required else f" (optional{nullable_marker})"

                prop_type = unwrapped.get("type", "any")
                prop_format = unwrapped.get("format")

                # Build type string with format
                type_str = prop_type
                if prop_format:
                    type_str = f"{prop_type} [{prop_format}]"

                lines.append(f"{prefix}    - {prop_name}: {type_str}{req_marker}")

                # Add description (check both original and unwrapped)
                desc_text = prop_schema.get("description") or unwrapped.get("description")
                if desc_text:
                    lines.append(f"{prefix}        description: {desc_text[:80]}")

                # Add constraints from unwrapped schema
                constraints = []
                if unwrapped.get("minLength") is not None:
                    constraints.append(f"minLength={unwrapped['minLength']}")
                if unwrapped.get("maxLength") is not None:
                    constraints.append(f"maxLength={unwrapped['maxLength']}")
                if unwrapped.get("pattern"):
                    pattern = unwrapped["pattern"][:40]
                    constraints.append(f"pattern='{pattern}'")
                if unwrapped.get("minimum") is not None:
                    constraints.append(f"min={unwrapped['minimum']}")
                if unwrapped.get("maximum") is not None:
                    constraints.append(f"max={unwrapped['maximum']}")
                if unwrapped.get("exclusiveMinimum") is not None:
                    constraints.append(f"exclusiveMin={unwrapped['exclusiveMinimum']}")
                if unwrapped.get("exclusiveMaximum") is not None:
                    constraints.append(f"exclusiveMax={unwrapped['exclusiveMaximum']}")
                if unwrapped.get("multipleOf") is not None:
                    constraints.append(f"multipleOf={unwrapped['multipleOf']}")

                if constraints:
                    lines.append(f"{prefix}        constraints: {', '.join(constraints)}")

                # Add enum values from unwrapped schema
                if unwrapped.get("enum"):
                    enum_vals = str(unwrapped["enum"])
                    lines.append(f"{prefix}        allowed values: {enum_vals}")

                # Add default value
                if unwrapped.get("default") is not None:
                    lines.append(f"{prefix}        default: {unwrapped['default']}")

                # Handle nested objects
                if prop_type == "object" and unwrapped.get("properties"):
                    lines.append(f"{prefix}        nested object properties:")
                    nested_lines = self._format_schema(unwrapped, indent + 3)
                    lines.extend(nested_lines)
                elif prop_type == "object" and unwrapped.get("additionalProperties"):
                    # Map/dict type
                    add_props = unwrapped["additionalProperties"]
                    if isinstance(add_props, dict):
                        val_type = add_props.get("type", "any")
                        lines.append(f"{prefix}        map type: string keys → {val_type} values")
                    else:
                        lines.append(f"{prefix}        map type: string keys → any values")

                # Handle arrays
                if prop_type == "array" and unwrapped.get("items"):
                    items = unwrapped["items"]
                    items_unwrapped, _ = self._unwrap_nullable_schema(items)
                    items_type = items_unwrapped.get("type", "any")
                    lines.append(f"{prefix}        array items type: {items_type}")
                    if items_unwrapped.get("properties"):
                        lines.append(f"{prefix}        array item properties:")
                        nested_lines = self._format_schema(items_unwrapped, indent + 3)
                        lines.extend(nested_lines)

        return lines

    def _resolve_ref_in_union(self, ref: str, one_of: List[dict]) -> Optional[dict]:
        """
        Try to resolve a $ref within a oneOf/anyOf array.

        When a discriminator mapping references a schema like "#/components/schemas/CreditCard",
        we look for that $ref in the oneOf array and return its inline schema if available.

        Args:
            ref: The $ref string (e.g., "#/components/schemas/CreditCard")
            one_of: The oneOf/anyOf array from the parent schema

        Returns:
            The resolved schema dict, or None if not found
        """
        for variant in one_of:
            if variant.get("$ref") == ref:
                # Found the reference, but it's not resolved here
                # We need to look for inline properties or allOf patterns
                if "properties" in variant:
                    return variant
                # Check if it's an allOf with the ref and additional properties
                if "allOf" in variant:
                    # Merge allOf schemas
                    merged = {"properties": {}, "required": []}
                    for sub in variant["allOf"]:
                        if "properties" in sub:
                            merged["properties"].update(sub["properties"])
                        if "required" in sub:
                            merged["required"].extend(sub["required"])
                    if merged["properties"]:
                        return merged

            # Sometimes the oneOf items have the schema inline, not as $ref
            # Check if this variant has a $ref that matches
            if "allOf" in variant:
                for sub in variant["allOf"]:
                    if sub.get("$ref") == ref:
                        # Found it in allOf - return the merged schema
                        merged = {"properties": {}, "required": []}
                        for all_sub in variant["allOf"]:
                            if "properties" in all_sub:
                                merged["properties"].update(all_sub["properties"])
                            if "required" in all_sub:
                                merged["required"].extend(all_sub["required"])
                        if merged["properties"]:
                            return merged

        # If the variant has inline properties, return it directly
        for variant in one_of:
            variant_props = variant.get("properties", {})
            if variant_props:
                # Check if this variant matches the ref name
                # Extract schema name from ref: "#/components/schemas/CreditCard" -> "CreditCard"
                ref_name = ref.split("/")[-1].lower() if ref else ""
                # Check if any property const value matches
                for prop_name, prop_schema in variant_props.items():
                    if prop_schema.get("const", "").lower().replace("_", "").replace("-", "") == ref_name.lower().replace("_", "").replace("-", ""):
                        return variant

        return None

    def _format_response_schema(self, schema: dict, indent: int = 0) -> List[str]:
        """
        Format a response JSON Schema for the LLM prompt.

        Simpler than _format_schema - focuses on field names and types
        so the LLM knows what fields to expect in API responses.
        """
        lines = []
        prefix = "  " * indent

        schema_type = schema.get("type", "object")
        properties = schema.get("properties", {})

        if properties:
            for prop_name, prop_schema in properties.items():
                # Unwrap nullable pattern
                unwrapped, is_nullable = self._unwrap_nullable_schema(prop_schema)
                prop_type = unwrapped.get("type", "any")
                prop_format = unwrapped.get("format")

                # Build type string with format
                type_str = prop_type
                if prop_format:
                    type_str = f"{prop_type} [{prop_format}]"
                if is_nullable:
                    type_str += " (nullable)"

                lines.append(f"{prefix}- {prop_name}: {type_str}")

                # Handle nested objects (show one level deep)
                if prop_type == "object" and unwrapped.get("properties"):
                    for nested_name, nested_schema in unwrapped["properties"].items():
                        n_unwrapped, _ = self._unwrap_nullable_schema(nested_schema)
                        nested_type = n_unwrapped.get("type", "any")
                        lines.append(f"{prefix}    - {nested_name}: {nested_type}")

                # Handle arrays
                if prop_type == "array" and unwrapped.get("items"):
                    items = unwrapped["items"]
                    items_unwrapped, _ = self._unwrap_nullable_schema(items)
                    items_type = items_unwrapped.get("type", "any")
                    lines.append(f"{prefix}    (array of {items_type})")
                    if items_unwrapped.get("properties"):
                        for item_name, item_schema in items_unwrapped["properties"].items():
                            i_unwrapped, _ = self._unwrap_nullable_schema(item_schema)
                            item_type = i_unwrapped.get("type", "any")
                            lines.append(f"{prefix}    - {item_name}: {item_type}")
        elif schema_type == "array":
            items = schema.get("items", {})
            items_unwrapped, _ = self._unwrap_nullable_schema(items)
            items_type = items_unwrapped.get("type", "any")
            lines.append(f"{prefix}(array of {items_type})")
            if items_unwrapped.get("properties"):
                for item_name, item_schema in items_unwrapped["properties"].items():
                    i_unwrapped, _ = self._unwrap_nullable_schema(item_schema)
                    item_type = i_unwrapped.get("type", "any")
                    lines.append(f"{prefix}- {item_name}: {item_type}")
        else:
            lines.append(f"{prefix}type: {schema_type}")

        return lines

    def _find_related_create_endpoints(
        self,
        target_endpoint: Any,
        all_endpoints: List[Any],
    ) -> List[Tuple[Any, float, str]]:
        """
        Find CREATE (POST) endpoints that are hierarchical parents of the target endpoint.

        Uses path prefix matching to find POST endpoints that create resources
        at each level of the target's path hierarchy. Only endpoints sharing
        the same path namespace are considered.

        For target: GET /api/v1/foo/users/{user_id}/posts/{post_id}/comments
        Parent paths are:
          - /api/v1/foo/users (creates user → user_id)
          - /api/v1/foo/users/{user_id}/posts (creates post → post_id)

        Args:
            target_endpoint: The endpoint being tested
            all_endpoints: All endpoints from the OpenAPI spec

        Returns:
            List of (endpoint, score, reason) tuples, sorted by relevance score (highest first).
            Returns empty list if the target is itself a POST endpoint.
        """
        # Skip if target is already a POST - it doesn't need setup endpoints
        if target_endpoint.method.upper() == "POST":
            return []

        results: List[Tuple[Any, float, str]] = []
        target_path = target_endpoint.path
        target_tags = set(getattr(target_endpoint, "tags", []) or [])

        # Build the hierarchy of parent paths from the target endpoint.
        # For /a/b/{id}/c/{id2}/d, parent paths are:
        #   /a/b (creates resource with id)
        #   /a/b/{id}/c (creates resource with id2)
        parent_paths = self._extract_parent_paths(target_path)

        # Extract the static path prefix (segments before first {param})
        target_prefix = self._get_static_prefix(target_path)

        for endpoint in all_endpoints:
            if endpoint.method.upper() != "POST":
                continue
            if endpoint.path == target_endpoint.path and endpoint.method == target_endpoint.method:
                continue

            score = 0.0
            reasons = []
            candidate_path = endpoint.path

            # Factor 1 (highest): Exact parent path match
            # The POST endpoint's path matches one of the target's hierarchy levels
            for i, parent in enumerate(parent_paths):
                if self._paths_match(candidate_path, parent):
                    # Higher score for deeper matches (more specific)
                    score += 100.0 + (i * 10.0)
                    reasons.append(f"parent path level {i + 1}: {parent}")
                    break

            # Factor 2: Shared path prefix (namespace matching)
            # Only consider endpoints in the same namespace
            if score == 0:
                candidate_prefix = self._get_static_prefix(candidate_path)
                common_prefix_len = self._common_prefix_length(target_prefix, candidate_prefix)
                # Require at least 3 segments in common (e.g., /api/v1/comprehensive)
                # to avoid matching /api/v1/users with /api/v1/comprehensive/nested/users
                if common_prefix_len >= 3:
                    score += 20.0 + (common_prefix_len * 5.0)
                    reasons.append(f"shared prefix ({common_prefix_len} segments)")

            # Factor 3: Tag matching (tiebreaker, lower weight than path matching)
            endpoint_tags = set(getattr(endpoint, "tags", []) or [])
            common_tags = target_tags & endpoint_tags
            if common_tags and score > 0:
                score += 10.0 * len(common_tags)
                reasons.append(f"same tag: {', '.join(common_tags)}")

            if score > 0:
                reason_str = "; ".join(reasons)
                results.append((endpoint, score, reason_str))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _extract_parent_paths(path: str) -> List[str]:
        """
        Extract parent hierarchy paths from a parameterized path.

        For /api/v1/users/{user_id}/posts/{post_id}/comments:
        Returns: ['/api/v1/users', '/api/v1/users/{user_id}/posts']

        Each parent path is where a POST would create the resource
        whose ID is used as a path parameter in the target.
        """
        segments = path.strip("/").split("/")
        parents = []
        for i, seg in enumerate(segments):
            if seg.startswith("{") and seg.endswith("}"):
                # Everything before this param is a parent path
                parent = "/" + "/".join(segments[:i])
                if parent != "/":
                    parents.append(parent)
        return parents

    @staticmethod
    def _get_static_prefix(path: str) -> List[str]:
        """Get path segments before the first path parameter."""
        segments = path.strip("/").split("/")
        prefix = []
        for seg in segments:
            if seg.startswith("{"):
                break
            prefix.append(seg.lower())
        return prefix

    @staticmethod
    def _common_prefix_length(prefix_a: List[str], prefix_b: List[str]) -> int:
        """Count how many leading segments match between two prefixes."""
        count = 0
        for a, b in zip(prefix_a, prefix_b):
            if a == b:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _paths_match(candidate: str, parent: str) -> bool:
        """
        Check if a candidate POST path matches a parent path pattern.

        Handles path parameter wildcards: /users/{id}/posts matches
        /users/{user_id}/posts because both have a param at the same position.
        """
        c_segments = candidate.strip("/").split("/")
        p_segments = parent.strip("/").split("/")

        if len(c_segments) != len(p_segments):
            return False

        for c, p in zip(c_segments, p_segments):
            c_is_param = c.startswith("{") and c.endswith("}")
            p_is_param = p.startswith("{") and p.endswith("}")
            if c_is_param or p_is_param:
                # Both are params or one is param - matches
                continue
            if c.lower() != p.lower():
                return False
        return True

    def _format_related_create_endpoints(
        self,
        related_endpoints: List[Tuple[Any, float, str]],
    ) -> str:
        """
        Format related CREATE endpoints for the LLM prompt.

        Handles edge cases:
        - Edge 1: Pass available endpoints with descriptions (even if not perfect match)
        - Edge 2: Return message saying no CREATE endpoints available
        - Edge 3: Rank and pass all with guidance message

        Args:
            related_endpoints: List of (endpoint, score, reason) from _find_related_create_endpoints

        Returns:
            Formatted string for the LLM prompt, or empty string for POST endpoints
        """
        if not related_endpoints:
            # Edge case 2: No CREATE endpoints found
            return """
=== SETUP ENDPOINTS (for creating test data) ===
No CREATE (POST) endpoints found that are related to this resource.
You may need to use test_data_generator or assume test data already exists.
Do NOT invent or call POST endpoints that are not documented here.
"""

        lines = []
        lines.append("=== SETUP ENDPOINTS (for creating test data) ===")
        lines.append("")
        lines.append("These POST endpoints can be used to create resources before testing.")
        lines.append("They are ranked by relevance to the endpoint you are testing.")
        lines.append("Use ONLY these endpoints for setup - do NOT invent endpoints that don't exist.")
        lines.append("**CRITICAL: Use the expected_status shown for EACH setup endpoint.**")
        lines.append("")

        # Edge case 3: Pass all ranked endpoints with guidance
        for i, (endpoint, score, reason) in enumerate(related_endpoints, 1):
            operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
            summary = getattr(endpoint, "summary", "") or "No summary"
            description = getattr(endpoint, "description", "") or ""

            # Extract expected status codes for THIS setup endpoint from OpenAPI spec
            setup_all_codes = self._extract_expected_status_codes(endpoint)
            setup_status_codes = self._filter_status_codes_for_scenario(
                setup_all_codes, ScenarioType.POSITIVE, method="POST"
            )
            # If no codes found, use sensible POST defaults
            if not setup_status_codes:
                setup_status_codes = [200, 201]

            lines.append(f"--- Rank #{i} (relevance: {score:.0f}) ---")
            lines.append(f"POST {endpoint.path}")
            lines.append(f"Operation ID: {operation_id}")
            lines.append(f"**expected_status={setup_status_codes}**  <-- USE THIS for this setup call")
            lines.append(f"Why relevant: {reason}")
            lines.append(f"Summary: {summary}")
            if description:
                lines.append(f"Description: {description[:150]}")

            # Include request body schema so LLM knows what fields to send
            if hasattr(endpoint, "request_body") and endpoint.request_body:
                rb = endpoint.request_body
                schema = getattr(rb, "schema", {})
                if schema and isinstance(schema, dict):
                    lines.append("Request Body Schema:")
                    schema_lines = self._format_schema(schema, indent=1)
                    lines.extend(schema_lines)

            lines.append("")

        # Add explicit example showing correct usage
        lines.append("=== SETUP CALL PATTERN ===")
        lines.append("```python")
        lines.append("# Use the expected_status from the setup endpoint above, NOT from the main endpoint")
        if related_endpoints:
            first_endpoint = related_endpoints[0][0]
            first_codes = self._extract_expected_status_codes(first_endpoint)
            first_filtered = self._filter_status_codes_for_scenario(first_codes, ScenarioType.POSITIVE, method="POST")
            if not first_filtered:
                first_filtered = [200, 201]
            lines.append(f'result = self.make_request("POST", "{first_endpoint.path}", expected_status={first_filtered}, json=data)')
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def _filter_status_codes_for_scenario(
        self, status_codes: List[int], scenario_type: ScenarioType,
        method: str = "GET", exclude_auth: bool = False,
    ) -> List[int]:
        """
        Filter status codes based on scenario type.

        Source of truth logic:
        - If OpenAPI spec defines responses: those are the ONLY codes used (no supplementation).
          Filter by scenario type. If filtering empties it, return empty.
        - If OpenAPI spec defines NO responses: use FallbackHttpResponseRegistry as fallback.

        Args:
            status_codes: All status codes from OpenAPI spec (empty if spec has no responses)
            scenario_type: The type of scenario being generated
            method: HTTP method (GET, POST, etc.) for fallback registry lookup
            exclude_auth: If True, exclude 401/403 from fallback codes

        Returns:
            Filtered list of status codes appropriate for the scenario type.
            Empty list means "no applicable codes" (e.g., positive for an endpoint with no 2xx).
        """
        if not status_codes:
            # No codes defined in spec - use FallbackHttpResponseRegistry
            return self._get_fallback_codes(method, scenario_type, exclude_auth)

        # Spec defines responses - those are the ONLY source of truth
        if scenario_type == ScenarioType.POSITIVE:
            # Positive tests: non-error responses (1xx, 2xx, 3xx) from spec
            # If spec has no non-error codes, return empty (caller should skip positive generation)
            return sorted([code for code in status_codes if code < 400])

        elif scenario_type == ScenarioType.NEGATIVE:
            # Negative tests: ONLY 4xx client error codes from spec
            # If spec has no 4xx, return empty (no negative test possible from spec)
            return sorted([code for code in status_codes if 400 <= code < 500])

        elif scenario_type == ScenarioType.SECURITY:
            # Security tests: all non-5xx codes from spec
            # 5xx is excluded because it indicates vulnerability (logged as failure)
            return sorted([code for code in status_codes if code < 500])

        return sorted(status_codes)

    def _get_fallback_codes(
        self, method: str, scenario_type: ScenarioType, exclude_auth: bool = False,
    ) -> List[int]:
        """
        Get fallback status codes from FallbackHttpResponseRegistry when
        the OpenAPI spec defines no responses for an endpoint.

        Args:
            method: HTTP method
            scenario_type: Scenario type to filter for
            exclude_auth: If True, exclude 401/403

        Returns:
            List of integer status codes appropriate for the scenario type
        """
        response_block = self._fallback_registry.get_responses(
            methods=method.upper(),
            exclude_auth=exclude_auth,
        )

        # Extract all codes from the response block for this method
        method_responses = response_block.as_dict().get(method.upper(), {})
        all_codes = []
        for code_str in method_responses.keys():
            try:
                all_codes.append(int(code_str))
            except (ValueError, TypeError):
                pass

        # Filter by scenario type
        if scenario_type == ScenarioType.POSITIVE:
            # Positive: non-error responses (1xx, 2xx, 3xx)
            return sorted([code for code in all_codes if code < 400])
        elif scenario_type == ScenarioType.NEGATIVE:
            return sorted([code for code in all_codes if 400 <= code < 500])
        elif scenario_type == ScenarioType.SECURITY:
            # Security: all non-5xx (5xx = vulnerability)
            return sorted([code for code in all_codes if code < 500])

        return sorted(all_codes)

    def _extract_expected_status_codes(self, endpoint: Any) -> List[int]:
        """
        Extract expected HTTP status codes from the OpenAPI spec responses.

        This allows per-endpoint validation instead of hardcoded method-based defaults.
        For example, a POST endpoint that defines both 201 (created) and 422 (validation error)
        as valid responses will return [201, 422].

        Args:
            endpoint: The endpoint object with responses attribute

        Returns:
            List of valid HTTP status codes defined in the OpenAPI spec.
            Returns empty list if no responses defined (will fall back to method-based defaults).
        """
        status_codes = []

        if not hasattr(endpoint, "responses") or not endpoint.responses:
            return status_codes

        responses = endpoint.responses

        if isinstance(responses, dict):
            # OpenAPI 3.x style: responses is a dict with status codes as keys
            for status_code_str in responses.keys():
                try:
                    # Handle string status codes like "200", "201", "default"
                    if status_code_str.lower() == "default":
                        continue  # Skip default response
                    status_code = int(status_code_str)
                    status_codes.append(status_code)
                except (ValueError, TypeError):
                    # Skip non-numeric status codes
                    pass
        elif isinstance(responses, list):
            # Some parsers return responses as a list with status_code attribute
            for response in responses:
                status_code = getattr(response, "status_code", None)
                if status_code is not None:
                    try:
                        status_codes.append(int(status_code))
                    except (ValueError, TypeError):
                        pass

        # Sort for consistency
        return sorted(status_codes)

    def _extract_status_codes_with_descriptions(self, endpoint: Any) -> List[Tuple[int, str]]:
        """
        Extract status codes with their descriptions from the OpenAPI spec responses.

        Returns:
            List of (code, description) tuples from the spec.
            Empty list if no responses defined.
        """
        result: List[Tuple[int, str]] = []

        if not hasattr(endpoint, "responses") or not endpoint.responses:
            return result

        responses = endpoint.responses

        if isinstance(responses, dict):
            for status_code_str, response in responses.items():
                try:
                    if status_code_str.lower() == "default":
                        continue
                    code = int(status_code_str)
                    desc = ""
                    if hasattr(response, "description"):
                        desc = getattr(response, "description", "") or ""
                    elif isinstance(response, dict):
                        desc = response.get("description", "")
                    result.append((code, desc))
                except (ValueError, TypeError):
                    pass
        elif isinstance(responses, list):
            for response in responses:
                status_code = getattr(response, "status_code", None)
                if status_code is not None:
                    try:
                        code = int(status_code)
                        desc = getattr(response, "description", "") or ""
                        result.append((code, desc))
                    except (ValueError, TypeError):
                        pass

        return sorted(result, key=lambda x: x[0])

    def _precompute_scenario_status_codes(
        self,
        endpoint: Any,
        scenario_type: ScenarioType,
        has_auth: bool,
    ) -> List[Tuple[int, str]]:
        """
        Pre-compute the exact status codes + descriptions for a scenario type.

        Logic:
        1. Try to get codes from spec responses (with descriptions)
        2. Filter by scenario type (2xx for positive, 4xx for negative, all <500 for security)
        3. If spec defines responses but filter is empty: return empty (caller handles skip)
        4. If spec defines NO responses at all: use FallbackHttpResponseRegistry
        5. Auth codes (401/403) included for negative/security when auth is enabled

        Returns:
            List of (code, description) tuples ready for the template.
        """
        method = endpoint.method.upper()

        # Step 1: Get codes + descriptions from spec
        spec_codes = self._extract_status_codes_with_descriptions(endpoint)

        # Step 2: Filter by scenario type
        if scenario_type == ScenarioType.POSITIVE:
            filtered = [(c, d) for c, d in spec_codes if 200 <= c < 300]
        elif scenario_type == ScenarioType.NEGATIVE:
            filtered = [(c, d) for c, d in spec_codes if 400 <= c < 500]
        elif scenario_type == ScenarioType.SECURITY:
            filtered = [(c, d) for c, d in spec_codes if c < 500]
        else:
            filtered = spec_codes

        # Step 3: If filter produced results, use them
        if filtered:
            return sorted(filtered, key=lambda x: x[0])

        # Step 4: Spec defines responses but none match this scenario type
        # For POSITIVE: spec is source of truth - no 2xx means no success case, skip
        # For NEGATIVE/SECURITY: endpoints may reject input even without documented 4xx codes
        if spec_codes and scenario_type == ScenarioType.POSITIVE:
            # Spec explicitly defines responses (e.g., only 431) but no 2xx codes.
            # Return empty to signal the caller should skip positive workflow.
            return []

        # Step 5: Spec defines NO responses at all - use fallback
        exclude_auth = not has_auth
        response_block = self._fallback_registry.get_responses(
            methods=method,
            exclude_auth=exclude_auth,
        )
        method_responses = response_block.as_dict().get(method, {})

        fallback_codes: List[Tuple[int, str]] = []
        for code_str, data in method_responses.items():
            try:
                code = int(code_str)
                desc = data.get("description", "") if isinstance(data, dict) else ""
                fallback_codes.append((code, desc))
            except (ValueError, TypeError):
                pass

        # Filter fallback by scenario type
        if scenario_type == ScenarioType.POSITIVE:
            result = sorted([(c, d) for c, d in fallback_codes if 200 <= c < 300], key=lambda x: x[0])
            return result if result else [(200, "OK")]
        elif scenario_type == ScenarioType.NEGATIVE:
            result = sorted([(c, d) for c, d in fallback_codes if 400 <= c < 500], key=lambda x: x[0])
            return result if result else [(400, "Bad Request"), (422, "Unprocessable Entity")]
        elif scenario_type == ScenarioType.SECURITY:
            result = sorted([(c, d) for c, d in fallback_codes if c < 500], key=lambda x: x[0])
            return result if result else [(400, "Bad Request"), (403, "Forbidden"), (422, "Unprocessable Entity")]

        return sorted(fallback_codes, key=lambda x: x[0])

    @staticmethod
    def _format_status_codes_for_prompt(codes_with_desc: List[Tuple[int, str]]) -> str:
        """Format pre-computed status codes with descriptions for the prompt."""
        if not codes_with_desc:
            return ""
        lines = []
        for code, desc in codes_with_desc:
            if desc:
                lines.append(f"- {code}: {desc}")
            else:
                lines.append(f"- {code}")
        return "\n".join(lines)

    def _precompute_injection_points(self, endpoint: Any) -> Optional[str]:
        """
        Pre-compute valid security injection points for an endpoint.

        Scans the endpoint's request body and query parameters for string fields
        that can receive injection payloads. If no valid injection points exist,
        returns None (caller should skip security generation).

        Returns:
            Formatted string listing injection points, or None if no valid targets.
        """
        body_fields: List[str] = []
        query_params: List[str] = []

        # Scan request body for string fields (handles allOf, oneOf, $ref)
        if hasattr(endpoint, "request_body") and endpoint.request_body:
            schema = getattr(endpoint.request_body, "schema", {})
            if schema and isinstance(schema, dict):
                properties, _ = self._extract_all_properties(schema)
                for field_name, field_schema in properties.items():
                    if isinstance(field_schema, dict):
                        unwrapped_fs, _ = self._unwrap_nullable_schema(field_schema)
                        field_type = unwrapped_fs.get("type", "")
                        if field_type == "string":
                            body_fields.append(field_name)

        # Scan parameters for string query params
        if hasattr(endpoint, "parameters") and endpoint.parameters:
            for param in endpoint.parameters:
                param_location = getattr(param, "location", None)
                if param_location is None:
                    param_location = getattr(param, "in_", "query")
                if hasattr(param_location, "value"):
                    param_location = param_location.value

                if param_location == "query":
                    param_type = getattr(param, "type", None) or "string"
                    if param_type == "string" or "string" in str(param_type):
                        query_params.append(getattr(param, "name", "unknown"))

        # No valid injection points - skip security generation
        if not body_fields and not query_params:
            return None

        # Format for prompt
        lines = []
        if body_fields:
            lines.append("Request body string fields (inject payloads here):")
            for f in body_fields:
                lines.append(f"  - {f}")
        if query_params:
            lines.append("Query parameters (inject payloads here):")
            for p in query_params:
                lines.append(f"  - {p}")

        return "\n".join(lines)

    def _precompute_negative_scenarios(self, endpoint: Any) -> str:
        """
        Pre-compute which negative test scenarios are valid for this endpoint.

        Based on the endpoint's schema, determines what can actually be tested:
        - Path params → non-existent ID test (only for resource endpoints, not parameter-test endpoints)
        - Required body fields → missing field test
        - Typed fields → wrong type test
        - Enum fields → invalid enum test
        - Pattern fields → invalid pattern test
        - Numeric constraints → boundary test

        Returns:
            Formatted string listing testable scenarios with details.
        """
        scenarios: List[str] = []

        # Detect parameter-test endpoints (these just echo inputs, no validation)
        endpoint_path = getattr(endpoint, "path", "")
        is_parameter_test_endpoint = "/parameters/" in endpoint_path

        # Check path parameters
        path_params: List[Tuple[str, str]] = []
        query_params: List[Tuple[str, str]] = []
        if hasattr(endpoint, "parameters") and endpoint.parameters:
            for param in endpoint.parameters:
                param_name = getattr(param, "name", "unknown")
                param_type = getattr(param, "type", None) or "string"
                param_location = getattr(param, "location", None)
                if param_location is None:
                    param_location = getattr(param, "in_", "query")
                if hasattr(param_location, "value"):
                    param_location = param_location.value

                if param_location == "path":
                    path_params.append((param_name, param_type))
                elif param_location == "query":
                    query_params.append((param_name, param_type))

        # Generate NON_EXISTENT_ID tests ONLY for resource endpoints (not parameter-test endpoints)
        # Parameter-test endpoints accept any value and echo it back - no resource lookup
        if path_params and not is_parameter_test_endpoint:
            for name, ptype in path_params:
                if ptype.lower() in ("integer", "number") or ptype.lower().startswith("array[int"):
                    scenarios.append(
                        f"NON_EXISTENT_ID: Test {{{name}}} with value 999999999 (integer path param)"
                    )
                else:
                    scenarios.append(
                        f"NON_EXISTENT_ID: Test {{{name}}} with value \"nonexistent-id-12345\" (string path param)"
                    )

        # Check request body fields
        required_fields: List[str] = []
        typed_fields: List[Tuple[str, str]] = []
        enum_fields: List[Tuple[str, List[Any]]] = []
        pattern_fields: List[Tuple[str, str]] = []
        numeric_fields: List[Tuple[str, Optional[float], Optional[float]]] = []

        if hasattr(endpoint, "request_body") and endpoint.request_body:
            schema = getattr(endpoint.request_body, "schema", {})
            if schema and isinstance(schema, dict):
                # Use shared helper for allOf/oneOf/$ref handling
                properties, required_list = self._extract_all_properties(schema)

                for field_name, field_schema in properties.items():
                    if not isinstance(field_schema, dict):
                        continue

                    # Unwrap nullable pattern
                    unwrapped_fs, _ = self._unwrap_nullable_schema(field_schema)
                    field_type = unwrapped_fs.get("type", "")

                    # Required fields
                    if field_name in required_list:
                        required_fields.append(field_name)

                    # Typed fields (for wrong-type tests)
                    if field_type in ("integer", "number", "boolean", "array"):
                        typed_fields.append((field_name, field_type))

                    # Enum fields
                    field_enum = unwrapped_fs.get("enum")
                    if field_enum:
                        enum_fields.append((field_name, field_enum))

                    # Pattern fields
                    field_pattern = unwrapped_fs.get("pattern")
                    if field_pattern:
                        pattern_fields.append((field_name, field_pattern))

                    # Numeric constraints
                    if field_type in ("integer", "number"):
                        minimum = unwrapped_fs.get("minimum")
                        maximum = unwrapped_fs.get("maximum")
                        exclusive_min = unwrapped_fs.get("exclusiveMinimum")
                        exclusive_max = unwrapped_fs.get("exclusiveMaximum")
                        if minimum is not None or maximum is not None or exclusive_min is not None or exclusive_max is not None:
                            effective_min = exclusive_min if exclusive_min is not None else minimum
                            effective_max = exclusive_max if exclusive_max is not None else maximum
                            numeric_fields.append((field_name, effective_min, effective_max))

        if required_fields:
            scenarios.append(
                f"MISSING_REQUIRED: Remove one of these required fields: {required_fields}"
            )

        if typed_fields:
            examples = []
            for name, ftype in typed_fields:
                if ftype == "integer" or ftype == "number":
                    examples.append(f'"{name}": "not_a_number" (expects {ftype})')
                elif ftype == "boolean":
                    examples.append(f'"{name}": "not_a_bool" (expects boolean)')
                elif ftype == "array":
                    examples.append(f'"{name}": "not_an_array" (expects array)')
            scenarios.append(f"WRONG_TYPE: Send wrong type: {examples}")

        if enum_fields:
            for name, values in enum_fields:
                scenarios.append(
                    f"INVALID_ENUM: Field \"{name}\" allows only {values}, send \"INVALID_VALUE_XYZ\""
                )

        if pattern_fields:
            for name, pattern in pattern_fields:
                scenarios.append(
                    f"INVALID_PATTERN: Field \"{name}\" must match pattern {pattern}, send \"!!!invalid!!!\""
                )

        if numeric_fields:
            for name, min_val, max_val in numeric_fields:
                if min_val is not None and isinstance(min_val, (int, float)):
                    scenarios.append(
                        f"BOUNDARY: Field \"{name}\" has min={min_val}, send {min_val - 1}"
                    )
                if max_val is not None and isinstance(max_val, (int, float)):
                    scenarios.append(
                        f"BOUNDARY: Field \"{name}\" has max={max_val}, send {max_val + 1}"
                    )

        # Fallback: test invalid query params ONLY if not a parameter-test endpoint
        # Parameter-test endpoints accept any value, so negative tests don't make sense
        if not scenarios and not is_parameter_test_endpoint:
            if query_params:
                for name, ptype in query_params:
                    if ptype.lower() in ("integer", "number") or ptype.lower().startswith("array[int"):
                        scenarios.append(f"INVALID_QUERY: Send \"{name}=not_a_number\" (expects integer)")
                    # Skip "very long string" tests - most APIs don't validate string length without explicit constraints

        if not scenarios:
            return ""

        lines = ["TESTABLE NEGATIVE SCENARIOS (implement ONLY these):"]
        for s in scenarios:
            lines.append(f"  - {s}")

        return "\n".join(lines)

    def _precompute_positive_fields(self, endpoint: Any) -> str:
        """
        Pre-compute field generation instructions for positive tests.

        Extracts field names, types, formats, enums, patterns, and constraints
        from the request body schema and formats them as explicit instructions
        so the LLM doesn't have to guess from the endpoint description.

        Returns:
            Formatted string with exact field generation instructions, or empty string.
        """
        lines: List[str] = []

        # Check if endpoint has request body
        if not hasattr(endpoint, "request_body") or not endpoint.request_body:
            return ""

        schema = getattr(endpoint.request_body, "schema", {})
        if not schema or not isinstance(schema, dict):
            return ""

        # Use shared helper for allOf merging and discriminated union extraction
        properties, required_list = self._extract_all_properties(schema)

        # Handle discriminated unions: provide per-variant instructions
        one_of = schema.get("oneOf") or schema.get("anyOf")
        discriminator = schema.get("discriminator", {})
        if one_of and isinstance(one_of, list) and not schema.get("properties"):
            disc_prop = discriminator.get("propertyName", "")
            lines.append("FIELD GENERATION INSTRUCTIONS (DISCRIMINATED UNION):")
            if disc_prop:
                lines.append(f"Discriminator field: \"{disc_prop}\"")
            lines.append("Pick ONE variant and include ALL its required fields:")
            lines.append("")
            for i, variant in enumerate(one_of[:4]):
                if not isinstance(variant, dict):
                    continue
                # Resolve variant allOf if present
                v_props = variant.get("properties", {})
                v_required = variant.get("required", [])
                v_all_of = variant.get("allOf")
                if v_all_of and isinstance(v_all_of, list):
                    for sub in v_all_of:
                        if isinstance(sub, dict) and sub.get("properties"):
                            v_props = {**v_props, **sub["properties"]}
                            v_required = v_required + sub.get("required", [])
                v_title = variant.get("title", f"Variant {i+1}")
                if v_props:
                    lines.append(f"  Variant \"{v_title}\" (required: {list(set(v_required))}):")
                    for vp_name, vp_schema in v_props.items():
                        if not isinstance(vp_schema, dict):
                            continue
                        unwrapped_vp, _ = self._unwrap_nullable_schema(vp_schema)
                        instruction = self._get_type_instruction(unwrapped_vp, field_name=vp_name)
                        lines.append(f"    \"{vp_name}\": {instruction}")
                    lines.append("")
            return "\n".join(lines)

        if not properties:
            return ""

        lines.append("FIELD GENERATION INSTRUCTIONS (use these EXACTLY):")
        lines.append(f"Required fields: {required_list if required_list else 'none'}")
        lines.append("")

        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue

            # Unwrap nullable pattern (OpenAPI 3.1 anyOf/3.0 nullable)
            unwrapped, _ = self._unwrap_nullable_schema(field_schema)

            field_type = unwrapped.get("type", "string")
            field_format = unwrapped.get("format", "")
            required_marker = " [REQUIRED]" if field_name in required_list else ""

            # Use shared type→instruction mapper (pass field_name for inference)
            instruction = self._get_type_instruction(unwrapped, field_name=field_name)

            lines.append(f"  \"{field_name}\": {instruction}  # type={field_type}{', format=' + field_format if field_format else ''}{required_marker}")

        return "\n".join(lines)

    def _precompute_object_instruction(self, schema: dict, _ancestors: Optional[frozenset] = None) -> str:
        """Generate a dict literal instruction for a nested object schema.

        Recurses into nested object properties. Uses identity-based ancestry
        tracking to detect circular schemas (stops recursion when the same
        schema object is encountered again in the chain).
        """
        properties = schema.get("properties", {})
        if not properties:
            return "{}"

        if _ancestors is None:
            _ancestors = frozenset()

        # Track this schema by identity to detect circular references
        schema_id = id(schema)
        if schema_id in _ancestors:
            return "{}"
        new_ancestors = _ancestors | {schema_id}

        parts = []
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue

            # Unwrap nullable pattern
            unwrapped, _ = self._unwrap_nullable_schema(prop_schema)

            # Delegate to shared type→instruction mapper with ancestry tracking and field name
            val = self._get_type_instruction(unwrapped, _object_ancestors=new_ancestors, field_name=prop_name)
            parts.append(f'"{prop_name}": {val}')

        return "{" + ", ".join(parts) + "}"

    def _format_endpoints_list(self, endpoints: List[Any]) -> str:
        """Format list of endpoints (for auth endpoints)"""
        if not endpoints:
            return ""

        lines = []
        for ep in endpoints:
            summary = getattr(ep, "summary", "") or "No summary"
            lines.append(f"- {ep.method.upper()} {ep.path} - {summary}")

        return "\n".join(lines)

    def _sanitize_identifier(self, name: str) -> str:
        """Sanitize string to be a valid Python identifier"""
        import re
        # Replace common separators with underscores
        name = name.replace("-", "_").replace(" ", "_").replace(".", "_").replace("/", "_")
        # Remove any remaining non-alphanumeric chars (except underscore)
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure doesn't start with a number
        if name and name[0].isdigit():
            name = f"n{name}"
        return name or "unnamed"

    def _operation_to_class_name(self, endpoint: Any) -> str:
        """Convert operation_id to valid Python class name"""
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        # Sanitize and convert to PascalCase
        sanitized = self._sanitize_identifier(operation_id)
        words = sanitized.replace("_", " ").split()
        return "".join(word.capitalize() for word in words) or "Unnamed"

    def _generate_operation_id(self, endpoint: Any) -> str:
        """Generate operation_id from method and path if not present"""
        path_parts = endpoint.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
        raw_id = f"{endpoint.method.lower()}_{path_parts}"
        return self._sanitize_identifier(raw_id)

    def get_endpoint_dir_name(self, endpoint: Any) -> str:
        """Get directory name for an endpoint"""
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        # Sanitize for filesystem
        return self._sanitize_identifier(operation_id).lower()

    async def _call_ai_service(self, prompt: str, scenario_type: str = "unknown") -> str:
        """Call AI service with retry logic. Raises AIServiceError after all retries fail."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer specializing in Locust load testing. "
                    "Generate clean, production-ready code for a SINGLE endpoint. "
                    "Keep the code focused and concise. "
                    "Return code in <code></code> tags. Do not truncate."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with self._api_semaphore:
                    response = await asyncio.wait_for(
                        self.ai_client.chat.completions.create(
                            model=self.ai_config.model,
                            messages=messages,
                            max_tokens=self.ai_config.max_tokens,
                            temperature=self.ai_config.temperature,
                        ),
                        timeout=self.ai_config.timeout,
                    )

                    # Update rate limit from headers if available
                    if hasattr(response, "headers"):
                        self.update_rate_limit(dict(response.headers))

                    if response.choices and response.choices[0].message:
                        return response.choices[0].message.content.strip()

            except asyncio.TimeoutError as e:
                last_error = e
                logger.debug(f"AI timeout on attempt {attempt + 1} for {scenario_type}")
            except Exception as e:
                last_error = e
                logger.debug(f"AI error on attempt {attempt + 1} for {scenario_type}: {e}")

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

        # All retries exhausted - raise exception
        raise AIServiceError(
            f"AI service failed after 3 attempts for {scenario_type}"
        ) from last_error

    def _extract_code(self, response: str) -> str:
        """Extract code from <code> tags with robust fallback handling.

        Handles chain-of-thought responses that have <analysis> before <code>.
        Only extracts the <code> section, ignoring analysis.
        """
        import re

        # First, strip any <analysis> sections (from chain-of-thought prompts)
        response = re.sub(r"<analysis>.*?</analysis>", "", response, flags=re.DOTALL | re.IGNORECASE)

        # Try case-insensitive match with closing tag (handle attributes like <code lang="python">)
        pattern = r"<code[^>]*>(.*?)</code>"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            code = max(matches, key=len).strip()
        else:
            # Fallback: handle <code> without closing tag
            code_start = re.search(r"<code[^>]*>", response, re.IGNORECASE)
            if code_start:
                code = response[code_start.end():]
            else:
                code = response

        # Aggressively strip ALL markdown code fence variations (anywhere in content)
        # Handles: ```python, ```code, ```py, ``` with any language identifier
        code = re.sub(r"```[\w]*\s*\n?", "", code)  # Opening fences with optional language
        code = re.sub(r"\n?```\s*$", "", code)  # Trailing fence at end
        code = re.sub(r"\n?```\s*\n", "\n", code)  # Fences in middle of content

        # Strip HTML code tags that may remain (with or without attributes)
        code = re.sub(r"</?code[^>]*>", "", code, flags=re.IGNORECASE)

        # Clean up garbage lines that aren't valid Python
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines starting with ! (not valid Python, often editor garbage)
            if stripped.startswith('!'):
                continue
            # Skip lines that look like file headers from editors
            if stripped.startswith('DO NOT EDIT') or stripped.startswith('generated by'):
                continue
            # Skip LLM explanatory notes that aren't valid Python
            # These often start with "Note:", "Since", "This", etc. and aren't comments
            if stripped.startswith('Note:') or stripped.startswith('Note that'):
                continue
            if stripped.startswith('Since ') and not stripped.startswith('Since('):
                continue
            if stripped.startswith('This endpoint') or stripped.startswith('This is'):
                continue
            if stripped.startswith('We ') and ('test' in stripped.lower() or 'endpoint' in stripped.lower()):
                continue
            # Skip chain-of-thought analysis remnants that leaked through
            if stripped.startswith('STEP ') and ':' in stripped:
                continue
            if stripped.startswith('Method:') or stripped.startswith('Path:'):
                continue
            if stripped.startswith('Required:') or stripped.startswith('Optional:'):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def _fix_class_name(self, code: str, expected_class_name: str, scenario_type: str) -> str:
        """
        Fix class name in generated code to match expected naming convention.

        LLMs sometimes generate their own class names instead of using the template.
        This post-processes the code to ensure the class name matches what __init__.py expects.
        """
        import re

        # Build the expected full class name (e.g., GetStringFormatsSecurityWorkflow)
        scenario_suffix = f"{scenario_type.capitalize()}Workflow"
        expected_full_name = f"{expected_class_name}{scenario_suffix}"

        # Find class definition that inherits from BaseWorkflow
        # Matches: class SomeName(BaseWorkflow): or class SomeName(TaskSet, BaseWorkflow):
        pattern = r'class\s+(\w+)\s*\([^)]*BaseWorkflow[^)]*\)\s*:'
        match = re.search(pattern, code)

        if match:
            actual_class_name = match.group(1)
            if actual_class_name != expected_full_name:
                logger.debug(f"Fixing class name: {actual_class_name} -> {expected_full_name}")
                # Replace the class name in definition only
                # Note: We only fix the class definition, not docstrings/comments
                # If LLM uses wrong name elsewhere, validation will fail and trigger retry
                code = re.sub(
                    rf'\bclass\s+{re.escape(actual_class_name)}\s*\(',
                    f'class {expected_full_name}(',
                    code
                )

        return code

    def _sanitize_unicode(self, code: str) -> str:
        """
        Remove non-ASCII characters that LLMs sometimes inject into code.

        LLMs occasionally output random Unicode characters (Chinese, Arabic, emoji)
        which corrupt variable names and cause ImportError/SyntaxError.
        This strips any non-ASCII characters from the generated code.

        Preserves ASCII printable characters (0x20-0x7E) and whitespace.
        """
        cleaned_lines = []
        for line in code.split('\n'):
            # Keep only ASCII characters (codes 0-127)
            cleaned = ''.join(c for c in line if ord(c) < 128)
            cleaned_lines.append(cleaned)
        return '\n'.join(cleaned_lines)

    def _fix_bytes_literals(self, code: str) -> str:
        """
        Fix bytes literals containing non-ASCII characters.

        LLMs sometimes generate b'tëst' which is invalid Python (bytes can only
        contain ASCII). This converts them to 'tëst'.encode('utf-8').

        Uses improved regex patterns that handle:
        - Escaped quotes: b'test\\'s data' or b"test\\"s data"
        - Multiple bytes literals on same line
        - Both single and double quoted strings
        """
        import re

        def fix_single_quoted(match):
            content = match.group(1)
            # Check if content has non-ASCII
            try:
                content.encode('ascii')
                return match.group(0)  # Valid ASCII, keep as-is
            except UnicodeEncodeError:
                # Has non-ASCII, convert to .encode() form
                return f"'{content}'.encode('utf-8')"

        def fix_double_quoted(match):
            content = match.group(1)
            # Check if content has non-ASCII
            try:
                content.encode('ascii')
                return match.group(0)  # Valid ASCII, keep as-is
            except UnicodeEncodeError:
                # Has non-ASCII, convert to .encode() form
                return f'"{content}".encode(\'utf-8\')'

        # Patterns that properly handle escaped quotes
        # (?:[^'\\]|\\.)* matches: non-quote-non-backslash OR backslash+anything
        single_quote_pattern = r"b'((?:[^'\\]|\\.)*)'"
        double_quote_pattern = r'b"((?:[^"\\]|\\.)*)"'

        # Apply fixes for both quote styles
        code = re.sub(single_quote_pattern, fix_single_quoted, code)
        code = re.sub(double_quote_pattern, fix_double_quoted, code)

        return code

    def _fix_regex_strings(self, code: str) -> str:
        """
        Convert strings with regex escape sequences to raw strings using tokenizer.

        LLMs sometimes generate "\\d" or "\\+" which triggers SyntaxWarnings
        in Python 3.12+. This converts them to raw strings: r"\\d", r"\\+".

        Uses Python's tokenizer for robust string detection that handles:
        - Escaped quotes within strings
        - Multi-line strings
        - Adjacent string literals
        - Nested quotes

        Converts: "\\d", "\\+", "\\s", etc. → r"\\d", r"\\+", r"\\s"
        """
        import tokenize
        import io
        import re

        # Problematic escape sequences that trigger SyntaxWarnings
        problematic_escapes = re.compile(
            r'\\[dDwWsS+*?^$.|()\\[\]{}]'
        )

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except tokenize.TokenizeError:
            # If tokenization fails, return unchanged (validation will catch issues)
            logger.debug("Tokenization failed in _fix_regex_strings, returning unchanged")
            return code

        # Find strings that need fixing and build replacement list
        # Each item: (start_row, start_col, end_row, end_col, old_string, new_string)
        replacements = []

        for tok in tokens:
            if tok.type != tokenize.STRING:
                continue

            string_val = tok.string

            # Skip if already a raw string (r"..." or r'...')
            if string_val.startswith(('r"', "r'", 'R"', "R'", 'br"', "br'", 'rb"', "rb'")):
                continue

            # Skip bytes literals (handled by _fix_bytes_literals)
            if string_val.startswith(('b"', "b'", 'B"', "B'")):
                continue

            # Skip f-strings (can't be made raw easily due to {} handling)
            if string_val.startswith(('f"', "f'", 'F"', "F'")):
                continue

            # Check if the string content has problematic escape sequences
            # We need to check the actual string value, not the repr
            if problematic_escapes.search(string_val):
                # Determine the quote style
                if string_val.startswith('"""') or string_val.startswith("'''"):
                    quote = string_val[:3]
                    content = string_val[3:-3]
                    new_string = f'r{quote}{content}{quote}'
                elif string_val.startswith('"'):
                    content = string_val[1:-1]
                    new_string = f'r"{content}"'
                elif string_val.startswith("'"):
                    content = string_val[1:-1]
                    new_string = f"r'{content}'"
                else:
                    # Unknown format, skip
                    continue

                replacements.append((
                    tok.start[0], tok.start[1],
                    tok.end[0], tok.end[1],
                    string_val, new_string
                ))

        # Apply replacements in reverse order to maintain positions
        if not replacements:
            return code

        lines = code.split('\n')

        # Sort by position (row, col) in reverse order
        replacements.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for start_row, start_col, end_row, end_col, old, new in replacements:
            # Tokenizer uses 1-indexed rows
            start_row -= 1
            end_row -= 1

            if start_row == end_row:
                # Single line replacement
                line = lines[start_row]
                lines[start_row] = line[:start_col] + new + line[end_col:]
            else:
                # Multi-line replacement (for triple-quoted strings)
                # Join all affected lines, make replacement, then split back
                first_part = lines[start_row][:start_col]
                last_part = lines[end_row][end_col:]
                lines[start_row] = first_part + new + last_part
                # Remove the lines that were part of the multi-line string
                del lines[start_row + 1:end_row + 1]

        return '\n'.join(lines)

    def _fix_missing_imports(self, code: str) -> str:
        """
        Add missing imports for commonly used functions that LLM forgets to import.

        The LLM sometimes uses functions like get_task_weight() but forgets
        to include the import statement. This post-processor adds them.
        """
        import re

        # Map of function/symbol usage patterns to their required import statements
        import_fixes = {
            r'\bget_task_weight\s*\(': 'from utils import get_task_weight',
        }

        lines = code.split('\n')

        for pattern, import_stmt in import_fixes.items():
            # Check if the symbol is used in the code
            if re.search(pattern, code):
                # Check if the import already exists
                if import_stmt not in code:
                    # Find the right place to insert (after existing imports)
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        # Track the last import line
                        if stripped.startswith('import ') or stripped.startswith('from '):
                            insert_idx = i + 1
                        # Stop at class definition
                        elif stripped.startswith('class '):
                            break

                    # Insert the import
                    if insert_idx > 0:
                        lines.insert(insert_idx, import_stmt)
                    else:
                        # No imports found, insert at beginning
                        lines.insert(0, import_stmt)

        return '\n'.join(lines)

    def _fix_isoformat_calls(self, code: str) -> str:
        """
        Remove redundant .isoformat() calls on test_data_generator date methods.

        LLM sometimes generates patterns like:
            test_data_generator.random_date().isoformat()

        But random_date(), random_datetime(), random_time() already return strings,
        so .isoformat() causes AttributeError. This removes the redundant call.
        """
        import re

        # Methods that return strings (not datetime objects)
        date_methods = [
            'random_date',
            'random_datetime',
            'random_time',
            'random_date_between',
            'random_future_date',
            'random_past_date',
        ]

        # Pattern: test_data_generator.random_date(...).isoformat()
        # Capture the method call and remove .isoformat()
        for method in date_methods:
            # Match method call with any arguments, followed by .isoformat()
            pattern = rf'(test_data_generator\.{method}\([^)]*\))\.isoformat\(\)'
            code = re.sub(pattern, r'\1', code)

        return code

    def _render_fix_prompt(self, failed_code: str, error_message: str) -> str:
        """
        Render the fix prompt template with the failed code and error.

        This is used for error-aware retries - instead of blindly retrying
        with the same prompt, we give the LLM context about what went wrong.

        Args:
            failed_code: The code that failed validation
            error_message: The validation error message

        Returns:
            Rendered fix prompt string
        """
        try:
            template = self.prompt_env.get_template("workflow_fix.j2")
            return template.render(
                failed_code=failed_code,
                error_message=error_message,
            )
        except Exception as e:
            # Fallback to simple prompt if template fails
            logger.warning(f"Failed to render fix template: {e}. Falling back to inline prompt.")
            return f"""Fix this Python syntax error:

Error: {error_message}

Code:
```python
{failed_code}
```

Output the complete corrected Python code:"""

    def _render_semantic_fix_prompt(
        self,
        failed_code: str,
        error_message: str,
        endpoint_expected_status: Optional[List[int]] = None,
        endpoint_path: str = "",
        endpoint_method: str = "",
    ) -> str:
        """
        Render the semantic fix prompt for code that passes syntax but fails semantic checks.

        Args:
            failed_code: The code that failed semantic validation
            error_message: The semantic violation details
            endpoint_expected_status: Pre-computed expected status codes for the endpoint
            endpoint_path: The endpoint path under test
            endpoint_method: The HTTP method under test

        Returns:
            Rendered semantic fix prompt string
        """
        try:
            template = self.prompt_env.get_template("workflow_semantic_fix.j2")
            return template.render(
                failed_code=failed_code,
                error_message=error_message,
                endpoint_expected_status=endpoint_expected_status,
                endpoint_path=endpoint_path,
                endpoint_method=endpoint_method,
            )
        except Exception as e:
            logger.warning(f"Failed to render semantic fix template: {e}. Falling back to inline prompt.")
            return f"""Fix these semantic issues in the generated code:

{error_message}

Code:
```python
{failed_code}
```

Fix ALL the violations and output the complete corrected Python code:"""

    # Base allowed imports (standard library + locust)
    # Project imports are extracted dynamically from prompt templates
    _BASE_ALLOWED_IMPORTS = {
        # Standard library
        "random", "logging", "datetime", "time", "json", "re", "uuid", "string",
        # Locust
        "locust",
    }

    def _extract_allowed_imports_from_templates(self) -> set:
        """
        Extract allowed imports dynamically from prompt templates.

        Parses the '=== ALLOWED IMPORTS ===' section from workflow templates
        to build the allowed imports set. This ensures the validation stays
        in sync with what the prompts actually allow.

        Returns:
            Set of allowed module names
        """
        import ast
        import re

        allowed = set(self._BASE_ALLOWED_IMPORTS)

        # Templates that define allowed imports
        template_files = [
            "workflow_positive.j2",
            "workflow_negative.j2",
            "workflow_security.j2",
        ]

        for template_file in template_files:
            try:
                template_path = self.prompt_dir / template_file
                if not template_path.exists():
                    continue

                content = template_path.read_text(encoding="utf-8")

                # Find the ALLOWED IMPORTS section
                # Pattern: === ALLOWED IMPORTS ... === followed by ```python ... ```
                match = re.search(
                    r'===\s*ALLOWED IMPORTS[^=]*===.*?```python\s*(.*?)```',
                    content,
                    re.DOTALL | re.IGNORECASE
                )

                if not match:
                    continue

                imports_code = match.group(1)

                # Parse the imports code block
                try:
                    tree = ast.parse(imports_code)
                except SyntaxError:
                    # Template might have Jinja syntax, try cleaning it
                    # Remove Jinja tags like {% if db_type == "mongo" %}
                    cleaned = re.sub(r'\{%.*?%\}', '', imports_code)
                    cleaned = re.sub(r'\{\{.*?\}\}', '', cleaned)
                    try:
                        tree = ast.parse(cleaned)
                    except SyntaxError:
                        continue

                # Extract module names from imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            # Add both the full path and the root module
                            allowed.add(alias.name)
                            allowed.add(alias.name.split('.')[0])

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            # Add both the full path and the root module
                            allowed.add(node.module)
                            allowed.add(node.module.split('.')[0])

            except Exception as e:
                logger.debug(f"Could not parse imports from {template_file}: {e}")
                continue

        logger.debug(f"Extracted allowed imports from templates: {allowed}")
        return allowed

    def _validate_template_context(self, context: Dict[str, Any], scenario_type: ScenarioType) -> None:
        """Validate that required template context variables are present.

        Logs warnings for missing/empty critical variables that could cause
        LLM hallucination. Does not raise exceptions - just logs issues.
        """
        # Required variables for all scenarios (must not be None or empty)
        required_vars = ["endpoint", "base_workflow", "test_data_content", "class_name", "method", "path"]

        for var in required_vars:
            value = context.get(var)
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.warning(f"Template context missing required variable '{var}' for {scenario_type.value} scenario")

        # Check endpoint_expected_status - should be a list, not None
        expected_status = context.get("endpoint_expected_status")
        if not expected_status or not isinstance(expected_status, list):
            logger.warning(f"Template context has invalid endpoint_expected_status: {expected_status}")

    def _validate_python_code(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax and check for suspicious imports"""
        # First check syntax
        try:
            compile(content, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - {e.text.strip() if e.text else ''}"

        # Check for potentially problematic imports
        warnings = self._check_imports(content)
        if warnings:
            # Log warnings but don't fail - LLM might have valid reason
            for warning in warnings:
                logger.warning(f"Suspicious import in generated code: {warning}")

        return True, ""

    def _check_imports(self, content: str) -> List[str]:
        """Check for potentially problematic imports in generated code.

        Returns list of warning messages for suspicious imports.
        Does not fail validation - just logs warnings.

        Uses dynamically extracted allowed imports from prompt templates,
        ensuring validation stays in sync with what prompts actually allow.
        """
        import ast

        warnings = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If AST parsing fails, syntax validation already caught it
            return []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name not in self._allowed_imports:
                        warnings.append(f"import {alias.name} - module '{module_name}' not in allowed list")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name not in self._allowed_imports:
                        warnings.append(f"from {node.module} import ... - module '{module_name}' not in allowed list")

        return warnings
