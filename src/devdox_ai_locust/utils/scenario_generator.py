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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from jinja2 import Environment, FileSystemLoader

from devdox_ai_locust.utils.constants import (
    CONTENT_TYPE_JSON,
    TYPE_ARRAY,
    TYPE_BOOLEAN,
    TYPE_INTEGER,
    TYPE_NUMBER,
    TYPE_OBJECT,
    TYPE_STRING,
)
from devdox_ai_locust.utils.http_fallback_presets import FallbackHttpResponseRegistry
from devdox_ai_locust.utils.code_validator import CodeValidator
from devdox_ai_locust.utils.code_processor import CodeProcessor
from devdox_ai_locust.utils.llm_client import (
    AIServiceError,
    RateLimitInfo,
    TimeEstimate,
)
from devdox_ai_locust.utils.schema_utils import (
    escape_for_raw_string,
    extract_all_properties,
    unwrap_nullable_schema,
)
from devdox_ai_locust.utils.type_instruction import (
    get_format_instruction,
    get_string_instruction,
    get_integer_instruction,
    get_number_instruction,
    get_object_instruction,
    get_array_instruction,
)

# Default summary for endpoints without a summary
_DEFAULT_SUMMARY = "No summary"

if TYPE_CHECKING:
    from devdox_ai_locust.utils.debug_recorder import DebugRecorder
    from devdox_ai_locust.utils.generation_progress import (
        GenerationProgress,
        EndpointAnalysis,
        OrchestratorAnalysis,
        SchemaAnalysis,
        SetupAnalysis,
    )
    from devdox_ai_locust.utils.open_ai_parser import Endpoint
    from devdox_ai_locust.config import AIEnhancementConfig
    from devdox_ai_locust.utils.llm_client import LLMClient
    from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedScenarioData:
    """Precomputed data for scenario generation."""

    injection_points: str = ""
    negative_scenarios: str = ""
    positive_fields: str = ""
    setup_endpoints_section: str = ""
    expected_status_codes: List[int] = field(default_factory=list)
    expected_status_info: str = ""


@dataclass
class TemplateContentData:
    """Template content data for scenario generation."""

    base_workflow_content: str = ""
    test_data_content: str = ""
    auth_endpoints: Optional[List[Any]] = None


class ScenarioGenerationError(Exception):
    """Raised when scenario generation fails"""

    pass


class CodeValidationError(ScenarioGenerationError):
    """Raised when generated code fails syntax validation"""

    def __init__(
        self, scenario_type: str, error: str, code: str, endpoint_info: str = ""
    ):
        self.scenario_type = scenario_type
        self.error = error
        self.code = code
        self.endpoint_info = endpoint_info
        msg = f"Generated {scenario_type} code failed validation"
        if endpoint_info:
            msg += f" for [{endpoint_info}]"
        msg += f": {error}"
        super().__init__(msg)


class ScenarioType(Enum):
    """Types of test scenarios (all LLM-generated)"""

    POSITIVE = "positive"  # Happy path + state-dependent tests
    NEGATIVE = "negative"  # Validation errors + edge cases + error handling
    SECURITY = "security"  # Injection attacks + auth bypass


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
        ai_client: "LLMClient",
        ai_config: "AIEnhancementConfig",
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
        self.replay_dir: Optional[Path] = None
        self._fallback_registry = FallbackHttpResponseRegistry()
        self._code_validator = CodeValidator()

        # Setup Jinja environment for prompts.
        # These templates generate Python source code for LLM prompts, not HTML.
        # XSS / auto-escaping (S5247) is not applicable here.
        self.prompt_env = Environment(  # NOSONAR
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Extract allowed imports dynamically from prompt templates
        # This replaces the hardcoded ALLOWED_IMPORTS to avoid maintenance burden
        self._allowed_imports = self._extract_allowed_imports_from_templates()

        # Initialize code processor for post-processing LLM output
        self._code_processor = CodeProcessor(self._allowed_imports)

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
            logger.info(
                f"Adjusting concurrency: {self._current_concurrency} → {new_concurrency} (based on {rpm} RPM)"
            )
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
        async def process_endpoint(
            endpoint: "Endpoint",
        ) -> Tuple[str, Dict[ScenarioType, str]]:
            operation_id = getattr(
                endpoint, "operation_id", ""
            ) or self._generate_operation_id(endpoint)
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
        endpoint: "Endpoint",
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        tag_name: str = "default",
        all_endpoints: Optional[List["Endpoint"]] = None,
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
        endpoint_info = f"{endpoint.method} {endpoint.path}"

        # Verbose mode: build and set endpoint analysis before generation
        if self.progress and self.progress.verbose:
            analysis = self._build_endpoint_analysis(endpoint, all_endpoints)
            self.progress.set_endpoint_analysis(endpoint_info, analysis)

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

        # Process results and separate successes from failures
        results, errors = self._process_llm_results(
            scenario_types, llm_results, endpoint_info
        )

        return self._handle_workflow_results(results, errors, endpoint_info)

    def _handle_workflow_results(
        self,
        results: Dict[ScenarioType, str],
        errors: List[Tuple[ScenarioType, Exception]],
        endpoint_info: str,
    ) -> Dict[ScenarioType, str]:
        """Handle mixed success/failure results from parallel scenario generation."""
        if errors:
            if not results:
                raise errors[0][1]
            for scenario_type, error in errors:
                logger.debug(
                    f"Scenario {scenario_type.value} failed for [{endpoint_info}], "
                    f"but other scenarios succeeded: {error}"
                )
        return results

    def _record_scenario_verbose_result(
        self,
        endpoint_info: str,
        scenario_type: ScenarioType,
        status: str,
        skip_reason: Optional[str] = None,
    ) -> None:
        """Record scenario result for verbose mode."""
        if not (self.progress and self.progress.verbose):
            return
        from devdox_ai_locust.utils.generation_progress import ScenarioResult

        result = ScenarioResult(
            scenario_type=scenario_type.value,
            status=status,
            skip_reason=skip_reason,
        )
        self.progress.record_scenario_result(endpoint_info, scenario_type.value, result)

    def _process_llm_results(
        self,
        scenario_types: List[ScenarioType],
        llm_results: List[Any],
        endpoint_info: str,
    ) -> Tuple[Dict[ScenarioType, str], List[Tuple[ScenarioType, Exception]]]:
        """Process LLM results, separating successes from failures."""
        results = {}
        errors = []
        for scenario_type, result in zip(scenario_types, llm_results):
            if isinstance(result, Exception):
                errors.append((scenario_type, result))
                self._record_scenario_verbose_result(
                    endpoint_info, scenario_type, "failed", str(result)[:100]
                )
            elif result is not None:
                results[scenario_type] = result
                self._record_scenario_verbose_result(
                    endpoint_info, scenario_type, "success"
                )
        return results, errors

    def _set_discriminator_info(self, schema_analysis: Any, schema: dict) -> None:
        """Set discriminator info on schema analysis if present."""
        one_of = schema.get("oneOf") or schema.get("anyOf")
        discriminator = schema.get("discriminator", {})
        if not (one_of and discriminator):
            return
        schema_analysis.schema_type = "discriminated_union"
        schema_analysis.discriminator = discriminator.get("propertyName", "")
        mapping = discriminator.get("mapping", {})
        schema_analysis.variants = list(mapping.keys()) if mapping else []

    def _count_property_constraints(
        self, schema_analysis: Any, properties: Dict[str, Any]
    ) -> None:
        """Count constraint types in schema properties."""
        for prop_schema in properties.values():
            unwrapped, _ = unwrap_nullable_schema(prop_schema)
            if unwrapped.get("pattern"):
                schema_analysis.patterns_found += 1
            if unwrapped.get("enum"):
                schema_analysis.enums_found += 1
            if unwrapped.get("format"):
                schema_analysis.formats_found += 1
            if unwrapped.get("type") == "array":
                if unwrapped.get("minItems") or unwrapped.get("maxItems"):
                    schema_analysis.arrays_with_constraints += 1

    def _analyze_schema_for_verbose(
        self, endpoint: "Endpoint", schema_analysis_cls: type
    ) -> Any:
        """Analyze request body schema for verbose mode."""
        schema_analysis = schema_analysis_cls()
        if not (hasattr(endpoint, "request_body") and endpoint.request_body):
            return schema_analysis

        schema = getattr(endpoint.request_body, "schema", {})
        if not (schema and isinstance(schema, dict)):
            return schema_analysis

        properties, required_list = extract_all_properties(schema)
        schema_analysis.total_fields = len(properties)
        schema_analysis.required_fields = len(required_list)

        self._set_discriminator_info(schema_analysis, schema)
        self._count_property_constraints(schema_analysis, properties)

        return schema_analysis

    def _parse_injection_result(
        self, injection_analysis: Any, injection_result: Optional[str]
    ) -> None:
        """Parse injection points from precomputed result."""
        if not injection_result:
            return
        for line in injection_result.split("\n"):
            if "HIGH_RISK" in line and ":" in line:
                field_name = line.split(":")[0].strip().lstrip("-").strip()
                injection_analysis.high_risk_fields.append(field_name)
            injection_analysis.total_injectable += 1

    def _determine_injection_locations(
        self, injection_analysis: Any, endpoint: "Endpoint"
    ) -> None:
        """Determine injection locations from endpoint structure."""
        if hasattr(endpoint, "request_body") and endpoint.request_body:
            injection_analysis.injection_locations.append("body")

        if not (hasattr(endpoint, "parameters") and endpoint.parameters):
            return

        for param in endpoint.parameters:
            loc = getattr(param, "location", None) or getattr(param, "in_", "query")
            if loc is not None and hasattr(loc, "value"):
                loc = loc.value
            if loc == "query" and "query" not in injection_analysis.injection_locations:
                injection_analysis.injection_locations.append("query")

    def _analyze_injection_for_verbose(
        self, endpoint: "Endpoint", injection_analysis_cls: type
    ) -> Any:
        """Analyze injection points for verbose mode."""
        injection_analysis = injection_analysis_cls()
        injection_result = self._precompute_injection_points(endpoint)

        self._parse_injection_result(injection_analysis, injection_result)
        self._determine_injection_locations(injection_analysis, endpoint)

        return injection_analysis

    def _parse_negative_scenario_types(self, negative_scenarios: str) -> List[str]:
        """Parse negative scenario types from precomputed string."""
        negative_types: list[str] = []
        if not negative_scenarios:
            return negative_types

        for line in negative_scenarios.split("\n"):
            if line.strip().startswith("-"):
                scenario_name = line.strip().lstrip("-").strip()
                if ":" in scenario_name:
                    scenario_name = scenario_name.split(":")[0].strip()
                if scenario_name:
                    negative_types.append(scenario_name)
        return negative_types

    def _build_endpoint_analysis(
        self,
        endpoint: "Endpoint",
        all_endpoints: Optional[List["Endpoint"]] = None,
    ) -> "EndpointAnalysis":
        """Build verbose analysis data for an endpoint."""
        from devdox_ai_locust.utils.generation_progress import (
            EndpointAnalysis,
            SchemaAnalysis,
            InjectionAnalysis,
        )

        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)

        responses_defined = self._extract_expected_status_codes(endpoint)
        content_type = self._get_endpoint_content_type(endpoint)
        schema_analysis = self._analyze_schema_for_verbose(endpoint, SchemaAnalysis)
        injection_analysis = self._analyze_injection_for_verbose(
            endpoint, InjectionAnalysis
        )
        setup_analysis = self._build_setup_analysis(endpoint, all_endpoints)
        positive_fields = self._count_positive_fields(endpoint)
        negative_types = self._get_negative_scenario_types(endpoint)
        warnings = self._build_analysis_warnings(responses_defined, schema_analysis)

        return EndpointAnalysis(
            method=endpoint.method.upper(),
            path=endpoint.path,
            operation_id=operation_id,
            responses_defined=responses_defined,
            source_of_truth="spec" if responses_defined else "fallback",
            content_type=content_type,
            schema=schema_analysis,
            strings_with_pattern=schema_analysis.patterns_found,
            numbers_with_bounds=0,
            fields_with_format=schema_analysis.formats_found,
            setup=setup_analysis,
            injection=injection_analysis,
            positive_fields_precomputed=positive_fields,
            negative_scenarios_precomputed=len(negative_types),
            negative_scenario_types=negative_types[:5],
            warnings=warnings,
        )

    def _get_endpoint_content_type(self, endpoint: "Endpoint") -> str:
        """Extract content type from endpoint request body."""
        if hasattr(endpoint, "request_body") and endpoint.request_body:
            ct = getattr(endpoint.request_body, "content_type", None)
            if ct:
                return str(ct)
        return CONTENT_TYPE_JSON

    def _build_setup_analysis(
        self,
        endpoint: "Endpoint",
        all_endpoints: Optional[List["Endpoint"]],
    ) -> "SetupAnalysis":
        """Build setup analysis for related create endpoints."""
        from devdox_ai_locust.utils.generation_progress import SetupAnalysis

        setup_analysis = SetupAnalysis()
        if not all_endpoints:
            return setup_analysis

        setup_results = self._find_related_create_endpoints(endpoint, all_endpoints)
        setup_analysis.setup_endpoints_found = len(setup_results)
        if setup_results:
            setup_analysis.needs_setup = True
            setup_analysis.setup_endpoints = [
                f"{ep[0].method} {ep[0].path}" for ep in setup_results[:3]
            ]
        return setup_analysis

    def _count_positive_fields(self, endpoint: "Endpoint") -> int:
        """Count the number of request body fields for positive scenarios."""
        if not (hasattr(endpoint, "request_body") and endpoint.request_body):
            return 0
        schema = getattr(endpoint.request_body, "schema", {})
        if not schema:
            return 0
        properties, _ = extract_all_properties(schema)
        return len(properties)

    def _get_negative_scenario_types(self, endpoint: "Endpoint") -> List[str]:
        """Get list of negative scenario type names for an endpoint."""
        negative_scenarios = self._precompute_negative_scenarios(endpoint)
        return self._parse_negative_scenario_types(negative_scenarios)

    def _build_analysis_warnings(
        self,
        responses_defined: List[int],
        schema_analysis: "SchemaAnalysis",
    ) -> List[str]:
        """Build warning messages for endpoint analysis."""
        warnings = []
        if not responses_defined:
            warnings.append("No responses defined in spec - using fallback codes")
        if schema_analysis.discriminator and not schema_analysis.variants:
            warnings.append("Discriminator without mapping - may generate invalid data")
        return warnings

    def _build_orchestrator_endpoint_info(
        self, endpoint: "Endpoint", endpoint_info_cls: Any
    ) -> Any:
        """Build endpoint info object for orchestrator analysis."""
        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)
        return endpoint_info_cls(
            method=endpoint.method.upper(),
            path=endpoint.path,
            operation_id=operation_id,
            has_positive=True,
            has_negative=True,
            has_security=True,
        )

    def _detect_crud_operations(
        self, tag_endpoints: List[Any]
    ) -> Tuple[bool, bool, bool, bool]:
        """Detect CRUD operations from endpoints."""
        has_create = any(ep.method.upper() == "POST" for ep in tag_endpoints)
        has_read = any(ep.method.upper() == "GET" for ep in tag_endpoints)
        has_update = any(ep.method.upper() in ("PUT", "PATCH") for ep in tag_endpoints)
        has_delete = any(ep.method.upper() == "DELETE" for ep in tag_endpoints)
        return has_create, has_read, has_update, has_delete

    def _detect_state_dependent_tests(
        self, has_create: bool, has_read: bool, has_update: bool, has_delete: bool
    ) -> List[str]:
        """Detect possible state-dependent tests based on CRUD operations."""
        tests = []
        if has_create and has_delete:
            tests.append("double_delete")
        if has_create and has_read:
            tests.append("read_after_delete")
        if has_create and has_update:
            tests.append("update_after_delete")
        if has_create:
            tests.append("409_conflict")
        return tests

    def _build_orchestrator_warnings(
        self,
        has_create: bool,
        crud_lifecycle_possible: bool,
        endpoint_count: int,
    ) -> List[str]:
        """Build warning messages for orchestrator analysis."""
        warnings = []
        if not has_create:
            warnings.append("No POST endpoint - create operations not possible")
        if not crud_lifecycle_possible:
            warnings.append(
                "Full CRUD lifecycle not possible - missing some operations"
            )
        if endpoint_count < 2:
            warnings.append("Only one endpoint - limited orchestration possibilities")
        return warnings

    def _build_orchestrator_analysis(
        self,
        tag_name: str,
        class_name: str,
        tag_endpoints: List[Any],
        auth_endpoints: Optional[List[Any]] = None,
    ) -> "OrchestratorAnalysis":
        """Build verbose analysis data for an orchestrator."""
        from devdox_ai_locust.utils.generation_progress import (
            OrchestratorAnalysis,
            OrchestratorEndpointInfo,
        )

        endpoints_info = [
            self._build_orchestrator_endpoint_info(ep, OrchestratorEndpointInfo)
            for ep in tag_endpoints
        ]

        has_create, has_read, has_update, has_delete = self._detect_crud_operations(
            tag_endpoints
        )
        crud_lifecycle_possible = has_create and has_read and has_delete
        state_dependent_tests = self._detect_state_dependent_tests(
            has_create, has_read, has_update, has_delete
        )

        auth_endpoints_count = len(auth_endpoints) if auth_endpoints else 0
        warnings = self._build_orchestrator_warnings(
            has_create, crud_lifecycle_possible, len(tag_endpoints)
        )

        return OrchestratorAnalysis(
            tag_name=tag_name,
            class_name=class_name,
            total_endpoints=len(tag_endpoints),
            valid_endpoints=len(tag_endpoints),
            endpoints=endpoints_info,
            has_create=has_create,
            has_read=has_read,
            has_update=has_update,
            has_delete=has_delete,
            crud_lifecycle_possible=crud_lifecycle_possible,
            auth_endpoints_found=auth_endpoints_count,
            auth_tests_possible=auth_endpoints_count > 0,
            state_dependent_tests=state_dependent_tests,
            concurrent_tests_possible=has_create or has_update,
            resource_limit_tests=has_create or has_update,
            warnings=warnings,
        )

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

        # Verbose mode: build and set orchestrator analysis
        if self.progress and self.progress.verbose:
            analysis = self._build_orchestrator_analysis(
                tag_name=tag_name,
                class_name=class_name,
                tag_endpoints=tag_endpoints,
                auth_endpoints=auth_endpoints,
            )
            self.progress.set_orchestrator_analysis(tag_name, analysis)

        # Render prompt
        prompt = template.render(
            tag_name=tag_name,
            endpoints_list=endpoints_list,
            auth_endpoints=(
                self._format_endpoints_list(auth_endpoints) if auth_endpoints else ""
            ),
            base_workflow=base_workflow_content,
            test_data_content=test_data_content,
            class_name=class_name,
            custom_requirement=custom_requirement or "",
            db_type=db_type,
        )

        # Record orchestrator context and prompt
        await self._record_orchestrator_context(
            tag_name,
            tag_endpoints,
            class_name,
            custom_requirement,
            db_type,
            endpoints_list,
            prompt,
        )

        return await self._orchestrator_retry_loop(tag_name, class_name, prompt)

    async def _orchestrator_retry_loop(
        self,
        tag_name: str,
        class_name: str,
        prompt: str,
    ) -> str:
        """Execute LLM call with validation retry for orchestrator generation."""
        max_retries = 1 if self.replay_dir else 2
        last_error = None
        last_code = None
        current_prompt = prompt

        for attempt in range(max_retries):
            if attempt > 0 and last_error and last_code:
                current_prompt = self._render_fix_prompt(last_code, last_error)

            content = await self._orchestrator_single_attempt(
                tag_name,
                class_name,
                current_prompt,
                attempt,
            )
            is_valid, error = self._code_processor.validate_python_code(content)

            if is_valid:
                return await self._finalize_orchestrator_success(
                    tag_name,
                    content,
                    attempt,
                    max_retries,
                )

            last_error = error
            last_code = content
            self._log_orchestrator_retry(tag_name, attempt, max_retries, error)

        raise CodeValidationError(
            "orchestrator",
            last_error or "",
            last_code or "",
            endpoint_info=f"tag: {tag_name}",
        )

    async def _orchestrator_single_attempt(
        self,
        tag_name: str,
        class_name: str,
        prompt: str,
        attempt: int,
    ) -> str:
        """Execute a single orchestrator LLM call and process the response."""
        if self.replay_dir:
            response_path = (
                self.replay_dir / tag_name / "orchestrator" / "llm_response.txt"
            )
            if not response_path.exists():
                raise FileNotFoundError(f"Replay fixture not found: {response_path}")
            content = response_path.read_text(encoding="utf-8")
            logger.info("Replay: loaded orchestrator for %s", tag_name)
        else:
            await self._record_orchestrator_llm_call(tag_name, attempt)
            content = await self._call_ai_service(prompt, f"orchestrator_{tag_name}")

        if self.debug_recorder and self.debug_recorder.enabled:
            await self.debug_recorder.record_orchestrator_llm_response(
                tag=tag_name, response=content or "(empty response)"
            )

        self._validate_orchestrator_response(content, tag_name)
        return self._apply_orchestrator_fixes(content, class_name)

    async def _finalize_orchestrator_success(
        self,
        tag_name: str,
        content: str,
        attempt: int,
        max_retries: int,
    ) -> str:
        """Record success and log retry info for orchestrator generation."""
        await self._record_orchestrator_success(tag_name, content, attempt)
        if attempt > 0:
            logger.info(
                f"Retry SUCCEEDED for orchestrator [{tag_name}] "
                f"on attempt {attempt + 1}/{max_retries}"
            )
        return content

    def _log_orchestrator_retry(
        self,
        tag_name: str,
        attempt: int,
        max_retries: int,
        error: Optional[str],
    ) -> None:
        """Log orchestrator validation failure and sleep before retry."""
        if attempt < max_retries - 1:
            logger.debug(
                f"Validation failed for orchestrator [{tag_name}], "
                f"attempt {attempt + 1}/{max_retries}: {error}. Retrying..."
            )

    def _format_endpoints_for_orchestrator(self, endpoints: List[Any]) -> str:
        """Format all endpoints in a tag for the orchestrator prompt."""
        by_method = self._group_endpoints_by_method(endpoints)
        lines = []
        for method, eps in by_method.items():
            if eps:
                lines.append(f"\n{method} endpoints:")
                for ep in eps:
                    lines.extend(self._format_orchestrator_endpoint(ep, method))
        return "\n".join(lines)

    def _group_endpoints_by_method(self, endpoints: List[Any]) -> Dict[str, List[Any]]:
        """Group endpoints by HTTP method."""
        by_method: Dict[str, List[Any]] = {
            "POST": [],
            "GET": [],
            "PUT": [],
            "PATCH": [],
            "DELETE": [],
        }
        for ep in endpoints:
            method = ep.method.upper()
            by_method.setdefault(method, []).append(ep)
        return by_method

    def _format_orchestrator_endpoint(self, ep: "Endpoint", method: str) -> List[str]:
        """Format a single endpoint for orchestrator prompt."""
        operation_id = getattr(ep, "operation_id", "") or self._generate_operation_id(
            ep
        )
        summary = getattr(ep, "summary", "") or _DEFAULT_SUMMARY
        lines = [
            f"  - {ep.path}",
            f"    Operation ID: {operation_id}",
            f"    Summary: {summary}",
        ]
        lines.extend(self._format_orchestrator_request_body(ep, method))
        lines.extend(self._format_orchestrator_response_schema(ep))
        return lines

    def _format_orchestrator_request_body(
        self, ep: "Endpoint", method: str
    ) -> List[str]:
        """Format request body schema for orchestrator prompt (POST/PUT/PATCH only)."""
        if method not in ("POST", "PUT", "PATCH"):
            return []
        if not (hasattr(ep, "request_body") and ep.request_body):
            return []
        schema = getattr(ep.request_body, "schema", {})
        if not (schema and isinstance(schema, dict)):
            return []
        lines = ["    Request Body Schema:"]
        lines.extend(self._format_schema(schema, indent=3))
        return lines

    def _format_orchestrator_response_schema(self, ep: "Endpoint") -> List[str]:
        """Format the first 2xx response schema for orchestrator prompt."""
        if not (hasattr(ep, "responses") and ep.responses):
            return []
        for response in ep.responses:
            if str(response.status_code).startswith("2") and response.schema:
                lines = [f"    Response ({response.status_code}) Schema:"]
                lines.extend(self._format_response_schema(response.schema, indent=3))
                return lines
        return []

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
        pattern = (
            r"class\s+(\w+)\s*\([^)]*(?:BaseWorkflow|SequentialTaskSet)[^)]*\)\s*:"
        )
        match = re.search(pattern, code)

        if match:
            actual_class_name = match.group(1)
            if actual_class_name != expected_full_name:
                logger.debug(
                    f"Fixing orchestrator class name: {actual_class_name} -> {expected_full_name}"
                )
                code = re.sub(
                    rf"\bclass\s+{re.escape(actual_class_name)}\s*\(",
                    f"class {expected_full_name}(",
                    code,
                )
                code = re.sub(
                    rf"\b{re.escape(actual_class_name)}\b", expected_full_name, code
                )

        return code

    async def _record_orchestrator_context(
        self,
        tag_name: str,
        tag_endpoints: List[Any],
        class_name: str,
        custom_requirement: Optional[str],
        db_type: str,
        endpoints_list: str,
        prompt: str,
    ) -> None:
        """Record orchestrator context and prompt for debugging."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return
        context = {
            "tag_name": tag_name,
            "endpoints_count": len(tag_endpoints),
            "class_name": class_name,
            "custom_requirement": custom_requirement or "",
            "db_type": db_type,
            "endpoints_list": endpoints_list,
        }
        await self.debug_recorder.record_orchestrator_context(
            tag=tag_name, context=context
        )
        await self.debug_recorder.record_orchestrator_prompt(
            tag=tag_name, prompt=prompt
        )

    async def _record_orchestrator_llm_call(self, tag_name: str, attempt: int) -> None:
        """Record orchestrator LLM request."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return
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

    def _validate_orchestrator_response(
        self, content: Optional[str], tag_name: str
    ) -> None:
        """Validate orchestrator LLM response, raise if invalid."""
        if not content:
            raise AIServiceError(
                f"AI service returned empty response for orchestrator [{tag_name}]"
            )
        if content.strip().startswith("<") and "<html" in content.lower():
            raise AIServiceError(
                f"API returned HTML error page for orchestrator [{tag_name}]"
            )

    def _apply_orchestrator_fixes(self, content: str, class_name: str) -> str:
        """Apply all code fixes to orchestrator code."""
        extracted = self._code_processor.extract_code(content)
        sanitized = self._code_processor.sanitize_unicode(extracted)
        after_class = self._fix_orchestrator_class_name(sanitized, class_name)
        after_bytes = self._code_processor.fix_bytes_literals(after_class)
        return self._code_processor.fix_regex_strings(after_bytes)

    async def _record_orchestrator_success(
        self, tag_name: str, content: str, attempt: int
    ) -> None:
        """Record successful orchestrator generation."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return
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

    def _should_skip_scenario(
        self,
        scenario_type: ScenarioType,
        expected_status_codes: List[int],
        all_status_codes: List[int],
    ) -> Optional[str]:
        """Check if scenario should be skipped. Returns skip reason or None."""
        if (
            scenario_type == ScenarioType.POSITIVE
            and all_status_codes
            and not expected_status_codes
        ):
            return f"spec defines no 2xx codes (defined: {all_status_codes})"
        if (
            scenario_type == ScenarioType.NEGATIVE
            and all_status_codes
            and not expected_status_codes
        ):
            return f"spec defines no 4xx codes (defined: {all_status_codes})"
        return None

    def _get_fallback_status_codes(self, scenario_type: ScenarioType) -> List[int]:
        """Get fallback status codes when none are found."""
        if scenario_type == ScenarioType.POSITIVE:
            return [200]
        elif scenario_type == ScenarioType.NEGATIVE:
            return [400, 422]
        elif scenario_type == ScenarioType.SECURITY:
            return [200, 400, 422]
        return []  # type: ignore[unreachable]

    def _precompute_scenario_specific_data(
        self,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
    ) -> Tuple[str, str, str, Optional[str]]:
        """Pre-compute scenario-specific data. Returns (injection_points, negative_scenarios, positive_fields, skip_reason)."""
        injection_points = ""
        negative_scenarios = ""
        positive_fields = ""
        skip_reason = None

        if scenario_type == ScenarioType.SECURITY:
            result = self._precompute_injection_points(endpoint)
            if result is None:
                skip_reason = (
                    "no valid injection points (no body string fields or query params)"
                )
            else:
                injection_points = result

        elif scenario_type == ScenarioType.NEGATIVE:
            negative_scenarios = self._precompute_negative_scenarios(endpoint)
            if not negative_scenarios:
                skip_reason = "no testable scenarios (no path params, body fields, or query params)"

        elif scenario_type == ScenarioType.POSITIVE:
            positive_fields = self._precompute_positive_fields(endpoint)

        return injection_points, negative_scenarios, positive_fields, skip_reason

    def _find_setup_endpoints(
        self, endpoint: "Endpoint", all_endpoints: Optional[List["Endpoint"]]
    ) -> Tuple[str, int]:
        """Find related CREATE endpoints for setup. Returns (section_text, count)."""
        if not all_endpoints:
            return "", 0

        has_path_params = any(
            seg.startswith("{") for seg in endpoint.path.split("/") if seg
        )
        if endpoint.method.upper() == "POST" and not has_path_params:
            return "", 0

        related = self._find_related_create_endpoints(endpoint, all_endpoints)
        return self._format_related_create_endpoints(related), len(related)

    def _log_precomputation_results(
        self,
        endpoint_info: str,
        scenario_name: str,
        expected_status_codes: List[int],
        setup_count: int,
        positive_fields: str,
        negative_scenarios: str,
        injection_points: str,
    ) -> None:
        """Log pre-computation results to progress tracker."""
        if not self.progress:
            return
        parts = [f"expected_status={expected_status_codes}"]
        if setup_count:
            parts.append(f"{setup_count} setup endpoints")
        if positive_fields:
            parts.append("fields pre-computed")
        if negative_scenarios:
            parts.append("scenarios pre-computed")
        if injection_points:
            parts.append("injection points found")
        self.progress.scenario_detail(endpoint_info, scenario_name, ", ".join(parts))

    def _build_scenario_template_context(
        self,
        endpoint_details: str,
        content_data: TemplateContentData,
        class_name: str,
        operation_id: str,
        endpoint: "Endpoint",
        precomputed: PrecomputedScenarioData,
        custom_requirement: Optional[str],
        db_type: str,
    ) -> Dict[str, Any]:
        """Build template context dict for scenario generation."""
        auth_formatted = ""
        if content_data.auth_endpoints:
            auth_formatted = self._format_endpoints_list(content_data.auth_endpoints)

        return {
            "endpoint": endpoint_details,
            "auth_endpoints": auth_formatted,
            "base_workflow": content_data.base_workflow_content,
            "test_data_content": content_data.test_data_content,
            "class_name": class_name,
            "operation_id": operation_id,
            "method": endpoint.method.upper(),
            "path": endpoint.path,
            "endpoint_expected_status": precomputed.expected_status_codes,
            "expected_status_info": precomputed.expected_status_info,
            "injection_points": precomputed.injection_points,
            "negative_scenarios": precomputed.negative_scenarios,
            "positive_fields": precomputed.positive_fields,
            "setup_endpoints": precomputed.setup_endpoints_section,
            "custom_requirement": custom_requirement or "",
            "db_type": db_type,
        }

    def _render_scenario_template(
        self,
        template: "Template",
        context: Dict[str, Any],
        scenario_type: ScenarioType,
    ) -> str:
        """Render a scenario template with the given context."""
        try:
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template render failed for {scenario_type.value}: {e}")
            raise

    async def _record_pre_generation_debug(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        endpoint: "Endpoint",
        endpoint_details: str,
        scenario_name: str,
        class_name: str,
        operation_id: str,
        expected_status_codes: List[int],
        setup_endpoints_section: str,
        custom_requirement: Optional[str],
        db_type: str,
        prompt: str,
    ) -> None:
        """Record debug info before LLM generation."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return

        context = self._build_debug_context(
            endpoint,
            endpoint_details,
            class_name,
            operation_id,
            expected_status_codes,
            setup_endpoints_section,
            custom_requirement,
            db_type,
        )
        await self.debug_recorder.record_scenario_context(
            tag=tag_name,
            endpoint_dir_name=endpoint_dir_name,
            scenario_type=scenario_name,
            context=context,
        )
        await self.debug_recorder.record_scenario_prompt(
            tag=tag_name,
            endpoint_dir_name=endpoint_dir_name,
            scenario_type=scenario_name,
            prompt=prompt,
        )

    def _build_debug_context(
        self,
        endpoint: "Endpoint",
        endpoint_details: str,
        class_name: str,
        operation_id: str,
        expected_status_codes: List[int],
        setup_endpoints_section: str,
        custom_requirement: Optional[str],
        db_type: str,
    ) -> Dict[str, Any]:
        """Build context dict for debug recording."""
        return {
            "endpoint_path": endpoint.path,
            "endpoint_method": endpoint.method.upper(),
            "endpoint_details_length": len(endpoint_details),
            "class_name": class_name,
            "operation_id": operation_id,
            "expected_status_codes": expected_status_codes,
            "has_setup_endpoints": bool(setup_endpoints_section),
            "custom_requirement": custom_requirement or "",
            "db_type": db_type,
        }

    def _apply_code_fixes(
        self, raw_content: str, class_name: str, scenario_type: str
    ) -> str:
        """Apply all code fixes to extracted code."""
        sanitized = self._code_processor.sanitize_unicode(raw_content)
        after_class = self._code_processor.fix_class_name(
            sanitized, class_name, scenario_type
        )
        after_bytes = self._code_processor.fix_bytes_literals(after_class)
        after_regex = self._code_processor.fix_regex_strings(after_bytes)
        after_import = self._code_processor.fix_missing_imports(after_regex)
        return self._code_processor.fix_isoformat_calls(after_import)

    def _validate_llm_response(
        self, content: Optional[str], scenario_type: ScenarioType, endpoint: "Endpoint"
    ) -> None:
        """Validate that LLM response is usable, raise if not."""
        if not content:
            raise AIServiceError(
                f"AI service returned empty response for {scenario_type.value} "
                f"[{endpoint.method} {endpoint.path}]"
            )
        if content.strip().startswith("<") and "<html" in content.lower():
            raise AIServiceError(
                f"API returned HTML error page instead of code for {scenario_type.value} "
                f"[{endpoint.method} {endpoint.path}]. "
                f"This may indicate an API error or rate limiting. "
                f"First 200 chars: {content[:200]}"
            )

    def _prepare_retry_prompt(
        self,
        last_code: str,
        last_error: str,
        last_is_semantic: bool,
        expected_status_codes: List[int],
        endpoint_path: str,
        endpoint_method: str,
    ) -> str:
        """Prepare the prompt for a retry attempt."""
        if last_is_semantic:
            return self._render_semantic_fix_prompt(
                last_code,
                last_error,
                endpoint_expected_status=expected_status_codes,
                endpoint_path=endpoint_path,
                endpoint_method=endpoint_method,
            )
        return self._render_fix_prompt(last_code, last_error)

    def _get_semantic_validation_context(
        self, endpoint: "Endpoint", all_endpoints: Optional[List["Endpoint"]]
    ) -> Tuple[List[str], Any]:
        """Get context needed for semantic validation."""
        all_paths = []
        if all_endpoints:
            all_paths = [
                getattr(ep, "path", "")
                for ep in all_endpoints
                if getattr(ep, "path", "")
            ]

        schema = None
        if hasattr(endpoint, "request_body") and endpoint.request_body:
            schema = getattr(endpoint.request_body, "schema", None)

        return all_paths, schema

    async def _record_llm_request(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        attempt: int,
    ) -> None:
        """Record LLM request details for debugging."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return
        data = {
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
            request_data=data,
        )

    async def _record_validation_success(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        content: str,
        attempt: int,
    ) -> None:
        """Record successful validation to debug recorder."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return
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

    async def _record_retry_debug(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        attempt: int,
        error: str,
        content: str,
        last_is_semantic: bool,
        expected_status_codes: List[int],
        endpoint_path: str,
        endpoint_method: str,
    ) -> None:
        """Record retry attempt details for debugging."""
        if not self.debug_recorder or not self.debug_recorder.enabled:
            return
        fix_prompt = self._prepare_retry_prompt(
            content,
            error,
            last_is_semantic,
            expected_status_codes,
            endpoint_path,
            endpoint_method,
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

    def _log_semantic_failure(
        self,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        violations: List[Any],
        endpoint_info: str,
        scenario_name: str,
    ) -> None:
        """Log semantic validation failure."""
        logger.info(
            f"Semantic validation failed for {scenario_type.value} "
            f"[{endpoint.method} {endpoint.path}]: {len(violations)} violation(s)"
        )
        if self.progress:
            summary = "; ".join(f"[{v.rule}] {v.message[:80]}" for v in violations[:3])
            self.progress.scenario_detail(
                endpoint_info, scenario_name, f"semantic FAILED: {summary}"
            )

    async def _record_code_extraction(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        extracted: str,
    ) -> None:
        """Record extracted code to debug recorder."""
        if self.debug_recorder and self.debug_recorder.enabled:
            await self.debug_recorder.record_extracted_code(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                scenario_type=scenario_name,
                code=extracted,
            )

    async def _record_processed_code(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        content: str,
    ) -> None:
        """Record processed code (after fixes) to debug recorder."""
        if self.debug_recorder and self.debug_recorder.enabled:
            await self.debug_recorder.record_processed_code(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                scenario_type=scenario_name,
                code=content,
            )

    async def _record_syntax_validation(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        is_valid: bool,
        error: Optional[str],
    ) -> None:
        """Record syntax validation result to debug recorder."""
        if self.debug_recorder and self.debug_recorder.enabled:
            await self.debug_recorder.record_validation_result(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                scenario_type=scenario_name,
                is_valid=is_valid,
                error=error if not is_valid else None,
                checks=[{"check": "python_syntax", "passed": is_valid, "error": error}],
            )

    def _prepare_scenario_precomputation(
        self,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        auth_endpoints: Optional[List[Any]],
        endpoint_info: str,
        scenario_name: str,
    ) -> Optional[Tuple[List[int], str, List[int]]]:
        """Pre-compute status codes. Returns None if scenario should be skipped."""
        has_auth = bool(auth_endpoints)
        codes_with_desc = self._precompute_scenario_status_codes(
            endpoint, scenario_type, has_auth
        )
        expected_codes = [code for code, _ in codes_with_desc]
        status_info = self._format_status_codes_for_prompt(codes_with_desc)
        all_codes = self._extract_expected_status_codes(endpoint)

        skip = self._should_skip_scenario(scenario_type, expected_codes, all_codes)
        if skip:
            self._skip_scenario(endpoint_info, scenario_name, skip)
            return None

        if not expected_codes:
            logger.warning(
                f"No expected status codes for {endpoint_info} "
                f"{scenario_name} - using fallback"
            )
            expected_codes = self._get_fallback_status_codes(scenario_type)

        return expected_codes, status_info, all_codes

    def _skip_scenario(
        self, endpoint_info: str, scenario_name: str, reason: str
    ) -> None:
        """Log and record a scenario skip."""
        logger.info(
            f"Skipping {scenario_name} workflow for [{endpoint_info}] - {reason}"
        )
        if self.progress:
            self.progress.scenario_skipped(endpoint_info, scenario_name, reason)

    async def _build_and_record_prompt(
        self,
        template: "Template",
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        endpoint_details: str,
        content_data: TemplateContentData,
        class_name: str,
        operation_id: str,
        precomputed: PrecomputedScenarioData,
        custom_requirement: Optional[str],
        db_type: str,
        tag_name: str,
        scenario_name: str,
    ) -> str:
        """Build template context, render prompt, and record debug info."""
        context = self._build_scenario_template_context(
            endpoint_details=endpoint_details,
            content_data=content_data,
            class_name=class_name,
            operation_id=operation_id,
            endpoint=endpoint,
            precomputed=precomputed,
            custom_requirement=custom_requirement,
            db_type=db_type,
        )
        self._validate_template_context(context, scenario_type)
        prompt = self._render_scenario_template(template, context, scenario_type)

        endpoint_dir_name = self.get_endpoint_dir_name(endpoint)
        await self._record_pre_generation_debug(
            tag_name=tag_name,
            endpoint_dir_name=endpoint_dir_name,
            endpoint=endpoint,
            endpoint_details=endpoint_details,
            scenario_name=scenario_name,
            class_name=class_name,
            operation_id=operation_id,
            expected_status_codes=precomputed.expected_status_codes,
            setup_endpoints_section=precomputed.setup_endpoints_section,
            custom_requirement=custom_requirement,
            db_type=db_type,
            prompt=prompt,
        )
        return prompt

    async def _record_llm_response_debug(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        content: Optional[str],
    ) -> None:
        """Record raw LLM response for debugging."""
        if self.debug_recorder and self.debug_recorder.enabled:
            await self.debug_recorder.record_llm_response(
                tag=tag_name,
                endpoint_dir_name=endpoint_dir_name,
                scenario_type=scenario_name,
                response=content or "(empty response)",
            )

    async def _extract_and_validate_code(
        self,
        content: str,
        class_name: str,
        scenario_type: ScenarioType,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
    ) -> Tuple[str, bool, Optional[str]]:
        """Extract code, apply fixes, and validate syntax."""
        extracted = self._code_processor.extract_code(content)
        await self._record_code_extraction(
            tag_name, endpoint_dir_name, scenario_name, extracted
        )
        fixed = self._apply_code_fixes(extracted, class_name, scenario_type.value)
        await self._record_processed_code(
            tag_name, endpoint_dir_name, scenario_name, fixed
        )
        is_valid, error = self._code_processor.validate_python_code(fixed)
        await self._record_syntax_validation(
            tag_name, endpoint_dir_name, scenario_name, is_valid, error
        )
        return fixed, is_valid, error

    async def _handle_retry_or_fail(
        self,
        attempt: int,
        max_retries: int,
        error: str,
        content: str,
        last_is_semantic: bool,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        endpoint_info: str,
        scenario_name: str,
        tag_name: str,
        endpoint_dir_name: str,
        expected_status_codes: List[int],
    ) -> None:
        """Handle retry logging or final failure logging."""
        if attempt < max_retries - 1:
            await self._record_retry_debug(
                tag_name,
                endpoint_dir_name,
                scenario_name,
                attempt,
                error,
                content,
                last_is_semantic,
                expected_status_codes,
                endpoint.path,
                endpoint.method.upper(),
            )
            error_type = "Semantic" if last_is_semantic else "Syntax"
            logger.debug(
                f"{error_type} validation failed for {scenario_type.value} "
                f"[{endpoint.method} {endpoint.path}], "
                f"attempt {attempt + 1}/{max_retries}: {error}. Retrying..."
            )
            if self.progress:
                self.progress.scenario_retry(
                    endpoint_info,
                    scenario_name,
                    attempt + 1,
                    max_retries,
                    f"{error_type}: {error}",
                )
            await asyncio.sleep(1)
        else:
            logger.debug(
                f"Retry FAILED for {scenario_type.value} "
                f"[{endpoint.method} {endpoint.path}] after "
                f"{max_retries} attempts. Final error: {error}"
            )

    async def _generate_llm_scenario(
        self,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        tag_name: str = "default",
        all_endpoints: Optional[List["Endpoint"]] = None,
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> Optional[str]:
        """Generate a scenario using LLM for a single endpoint."""
        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)
        scenario_name = scenario_type.value
        endpoint_info = f"{endpoint.method} {endpoint.path}"

        if self.progress:
            self.progress.scenario_start(endpoint_info, scenario_name)

        template_name = self.PROMPT_TEMPLATES.get(scenario_type)
        if not template_name:
            raise ValueError(f"No prompt template for scenario type: {scenario_type}")

        template = self.prompt_env.get_template(template_name)
        exclude_2xx = scenario_type in (ScenarioType.NEGATIVE, ScenarioType.SECURITY)
        endpoint_details = self._format_single_endpoint(
            endpoint, exclude_2xx=exclude_2xx
        )
        class_name = self._operation_to_class_name(endpoint)

        # Pre-compute status codes (returns None if skip)
        precomp = self._prepare_scenario_precomputation(
            scenario_type, endpoint, auth_endpoints, endpoint_info, scenario_name
        )
        if precomp is None:
            return None
        expected_status_codes, expected_status_info, _ = precomp

        # Pre-compute scenario-specific data
        injection_points, negative_scenarios, positive_fields, skip_reason = (
            self._precompute_scenario_specific_data(scenario_type, endpoint)
        )
        if skip_reason:
            self._skip_scenario(endpoint_info, scenario_name, skip_reason)
            return None

        # Find setup endpoints and log
        setup_section, setup_count = self._find_setup_endpoints(endpoint, all_endpoints)
        self._log_precomputation_results(
            endpoint_info,
            scenario_name,
            expected_status_codes,
            setup_count,
            positive_fields,
            negative_scenarios,
            injection_points,
        )

        # Build prompt and record debug
        content_data = TemplateContentData(
            base_workflow_content=base_workflow_content,
            test_data_content=test_data_content,
            auth_endpoints=auth_endpoints,
        )
        precomputed = PrecomputedScenarioData(
            injection_points=injection_points,
            negative_scenarios=negative_scenarios,
            positive_fields=positive_fields,
            setup_endpoints_section=setup_section,
            expected_status_codes=expected_status_codes,
            expected_status_info=expected_status_info,
        )
        prompt = await self._build_and_record_prompt(
            template,
            scenario_type,
            endpoint,
            endpoint_details,
            content_data,
            class_name,
            operation_id,
            precomputed,
            custom_requirement,
            db_type,
            tag_name,
            scenario_name,
        )

        # Run LLM with retry loop
        return await self._run_llm_retry_loop(
            scenario_type,
            endpoint,
            endpoint_info,
            scenario_name,
            class_name,
            tag_name,
            all_endpoints,
            expected_status_codes,
            prompt,
        )

    async def _run_llm_retry_loop(
        self,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        endpoint_info: str,
        scenario_name: str,
        class_name: str,
        tag_name: str,
        all_endpoints: Optional[List["Endpoint"]],
        expected_status_codes: List[int],
        prompt: str,
    ) -> str:
        """Execute the LLM call with validation retry loop."""
        max_retries = 1 if self.replay_dir else 2
        last_error = None
        last_code = None
        last_is_semantic = False
        current_prompt = prompt
        endpoint_dir_name = self.get_endpoint_dir_name(endpoint)

        for attempt in range(max_retries):
            if attempt > 0 and last_error and last_code:
                current_prompt = self._prepare_retry_prompt(
                    last_code,
                    last_error,
                    last_is_semantic,
                    expected_status_codes,
                    endpoint.path,
                    endpoint.method.upper(),
                )

            raw = await self._fetch_llm_response(
                tag_name,
                endpoint_dir_name,
                scenario_name,
                scenario_type,
                endpoint_info,
                current_prompt,
                attempt,
                max_retries,
            )
            self._validate_llm_response(raw, scenario_type, endpoint)

            content, is_valid, error = await self._extract_and_validate_code(
                raw,
                class_name,
                scenario_type,
                tag_name,
                endpoint_dir_name,
                scenario_name,
            )

            if not is_valid:
                self._report_syntax_failure(endpoint_info, scenario_name, error)
                is_semantic_error = False
            else:
                result = await self._check_and_finalize_scenario(
                    content,
                    scenario_type,
                    endpoint,
                    all_endpoints,
                    tag_name,
                    endpoint_dir_name,
                    scenario_name,
                    endpoint_info,
                    attempt,
                    max_retries,
                )
                if isinstance(result, str):
                    return result
                error, is_semantic_error = result

            last_error = error
            last_code = content
            last_is_semantic = is_semantic_error if is_valid else False

            await self._handle_retry_or_fail(
                attempt,
                max_retries,
                error or "",
                content,
                last_is_semantic,
                scenario_type,
                endpoint,
                endpoint_info,
                scenario_name,
                tag_name,
                endpoint_dir_name,
                expected_status_codes,
            )

        raise CodeValidationError(
            scenario_type.value,
            last_error or "",
            last_code or "",
            endpoint_info=f"{endpoint.method} {endpoint.path}",
        )

    def _load_replay_response(
        self, tag_name: str, endpoint_dir_name: str, scenario_name: str
    ) -> str:
        """Load a recorded LLM response from the replay directory."""
        assert self.replay_dir is not None
        response_path = (
            self.replay_dir
            / tag_name
            / endpoint_dir_name
            / scenario_name
            / "llm_response.txt"
        )
        if not response_path.exists():
            raise FileNotFoundError(f"Replay fixture not found: {response_path}")
        return response_path.read_text(encoding="utf-8")

    async def _fetch_llm_response(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        scenario_type: ScenarioType,
        endpoint_info: str,
        prompt: str,
        attempt: int,
        max_retries: int,
    ) -> str:
        """Record, call LLM, and validate the raw response."""
        if self.replay_dir:
            raw = self._load_replay_response(tag_name, endpoint_dir_name, scenario_name)
            logger.info(
                "Replay: loaded %s/%s/%s", tag_name, endpoint_dir_name, scenario_name
            )
            return raw

        await self._record_llm_request(
            tag_name, endpoint_dir_name, scenario_name, attempt
        )
        if self.progress:
            label = (
                f"attempt {attempt + 1}/{max_retries}" if attempt > 0 else "calling LLM"
            )
            self.progress.scenario_detail(endpoint_info, scenario_name, label)

        raw = await self._call_ai_service(prompt, scenario_type.value)
        await self._record_llm_response_debug(
            tag_name, endpoint_dir_name, scenario_name, raw
        )
        return raw

    def _report_syntax_failure(
        self,
        endpoint_info: str,
        scenario_name: str,
        error: Optional[str],
    ) -> None:
        """Report syntax validation failure via progress."""
        if self.progress:
            self.progress.scenario_detail(
                endpoint_info,
                scenario_name,
                f"syntax FAILED: {error[:120] if error else 'unknown'}",
            )

    async def _check_and_finalize_scenario(
        self,
        content: str,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        all_endpoints: Optional[List["Endpoint"]],
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        endpoint_info: str,
        attempt: int,
        max_retries: int,
    ) -> "str | Tuple[Optional[str], bool]":
        """Run semantic check; return code string on success or (error, is_semantic) on failure."""
        result = self._run_semantic_check(
            content, scenario_type, endpoint, all_endpoints
        )
        if result is None:
            return await self._finalize_scenario_success(
                tag_name,
                endpoint_dir_name,
                scenario_name,
                content,
                endpoint_info,
                scenario_type,
                endpoint,
                attempt,
                max_retries,
            )
        self._log_semantic_failure(
            scenario_type,
            endpoint,
            result.violations,
            endpoint_info,
            scenario_name,
        )
        return (result.error_message, True)

    async def _finalize_scenario_success(
        self,
        tag_name: str,
        endpoint_dir_name: str,
        scenario_name: str,
        content: str,
        endpoint_info: str,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        attempt: int,
        max_retries: int,
    ) -> str:
        """Record success, log retry info, and return validated content."""
        await self._record_validation_success(
            tag_name,
            endpoint_dir_name,
            scenario_name,
            content,
            attempt,
        )
        if attempt > 0:
            logger.info(
                f"Retry SUCCEEDED for {scenario_type.value} "
                f"[{endpoint.method} {endpoint.path}] on attempt "
                f"{attempt + 1}/{max_retries}"
            )
        if self.progress:
            self.progress.scenario_done(endpoint_info, scenario_name)
        return content

    def _run_semantic_check(
        self,
        content: str,
        scenario_type: ScenarioType,
        endpoint: "Endpoint",
        all_endpoints: Optional[List["Endpoint"]],
    ) -> Optional[Any]:
        """Run semantic validation. Returns None if valid, result if failed."""
        all_paths, request_schema = self._get_semantic_validation_context(
            endpoint, all_endpoints
        )
        result = self._code_validator.validate(
            code=content,
            scenario_type=scenario_type.value,
            endpoint_path=endpoint.path,
            all_endpoint_paths=all_paths,
            request_body_schema=request_schema,
        )
        return None if result.is_valid else result

    def _get_type_instruction(
        self,
        field_schema: dict,
        _object_ancestors: Optional[frozenset] = None,
        field_name: str = "",
    ) -> str:
        """Map a field schema to a Python code instruction for generating a valid value.

        Delegates to type-specific helper functions in type_instruction module.

        Args:
            field_schema: The (already unwrapped) field schema dict
            _object_ancestors: Frozenset for circular reference detection
            field_name: Optional field name for inferring the appropriate generator

        Returns:
            Python code string for generating a valid test value
        """
        if not isinstance(field_schema, dict):
            return "test_data_generator.generate_string(length=10)"  # type: ignore[unreachable]

        field_type = field_schema.get("type", TYPE_STRING)
        field_format = field_schema.get("format", "")
        field_enum = field_schema.get("enum")
        field_pattern = field_schema.get("pattern")
        field_name_lower = field_name.lower() if field_name else ""

        # Enum takes priority
        if field_enum:
            return f"random.choice({field_enum})"

        # IPv4 early detection (before pattern check)
        if field_name_lower and field_type == TYPE_STRING:
            if "ipv4" in field_name_lower or field_name_lower == "ip_address":
                return "test_data_generator.random_ipv4()"

        # Pattern-based generation
        if field_pattern:
            escaped = escape_for_raw_string(field_pattern)
            return f'test_data_generator.generate_string(pattern=r"{escaped}")'

        # Format-specific generators
        format_instr = get_format_instruction(field_format)
        if format_instr:
            return format_instr

        # Type-specific generators
        if field_type == TYPE_STRING:
            return get_string_instruction(field_schema, field_name)

        if field_type == TYPE_INTEGER:
            return get_integer_instruction(field_schema, field_name)

        if field_type == TYPE_NUMBER:
            return get_number_instruction(field_schema)

        if field_type == TYPE_BOOLEAN:
            return "test_data_generator.generate_boolean()"

        if field_type == TYPE_OBJECT:
            return get_object_instruction(
                field_schema,
                self._precompute_object_instruction,
                _object_ancestors,
            )

        if field_type == TYPE_ARRAY:
            return get_array_instruction(
                field_schema,
                self._precompute_object_instruction,
                _object_ancestors,
                field_name,
            )

        # Fallback for unknown types
        return "test_data_generator.generate_string(length=10)"

    def _format_endpoint_parameters(
        self, endpoint: "Endpoint"
    ) -> Tuple[List[str], bool, bool]:
        """Format endpoint parameters section.

        Returns:
            Tuple of (lines, has_cookie_params, has_header_params)
        """
        lines: list[str] = []
        has_cookie_params = False
        has_header_params = False

        if not (hasattr(endpoint, "parameters") and endpoint.parameters):
            return lines, has_cookie_params, has_header_params

        lines.append("\nParameters:")
        for param in endpoint.parameters:
            param_lines, is_cookie, is_header = self._format_single_parameter(param)
            lines.extend(param_lines)
            has_cookie_params = has_cookie_params or is_cookie
            has_header_params = has_header_params or is_header

        # Add type coercion warnings
        if has_cookie_params:
            lines.extend(self._get_cookie_warning())
        if has_header_params:
            lines.extend(self._get_header_warning())

        return lines, has_cookie_params, has_header_params

    def _format_single_parameter(self, param: Any) -> Tuple[List[str], bool, bool]:
        """Format a single parameter. Returns (lines, is_cookie, is_header)."""
        param_name = getattr(param, "name", "unknown")
        param_in = self._get_param_location(param)
        param_required = getattr(param, "required", False)
        param_type = getattr(param, "type", "string")
        param_format = getattr(param, "format", None)

        required_str = "(required)" if param_required else "(optional)"
        type_str = f"{param_type} [{param_format}]" if param_format else param_type

        lines = [f"  - {param_name} [{param_in}]: {type_str} {required_str}"]
        lines.extend(self._format_param_constraints(param))

        return lines, param_in == "cookie", param_in == "header"

    @staticmethod
    def _format_param_constraints(param: Any) -> List[str]:
        """Format constraint lines for a parameter."""
        lines = []
        if getattr(param, "enum", None):
            lines.append(f"      allowed values: {param.enum}")
        if getattr(param, "pattern", None):
            lines.append(f"      pattern: {param.pattern}")

        length_parts = []
        if getattr(param, "min_length", None) is not None:
            length_parts.append(f"minLength={param.min_length}")
        if getattr(param, "max_length", None) is not None:
            length_parts.append(f"maxLength={param.max_length}")
        if length_parts:
            lines.append(f"      constraints: {', '.join(length_parts)}")

        range_parts = []
        if getattr(param, "minimum", None) is not None:
            range_parts.append(f"min={param.minimum}")
        if getattr(param, "maximum", None) is not None:
            range_parts.append(f"max={param.maximum}")
        if range_parts:
            lines.append(f"      constraints: {', '.join(range_parts)}")

        if getattr(param, "description", None):
            lines.append(f"      description: {param.description[:80]}")
        return lines

    def _get_cookie_warning(self) -> List[str]:
        """Return cookie type coercion warning lines."""
        return [
            "",
            "  *** COOKIE VALUES MUST BE STRINGS ***",
            "  When passing cookies, ALL values must be strings, not integers.",
            "  WRONG: cookies={'session_id': 123}",
            "  CORRECT: cookies={'session_id': '123'}",
        ]

    def _get_header_warning(self) -> List[str]:
        """Return header type coercion warning lines."""
        return [
            "",
            "  *** HEADER VALUES MUST BE STRINGS ***",
            "  When passing headers, ALL values must be strings.",
            "  WRONG: headers={'X-Count': 10}",
            "  CORRECT: headers={'X-Count': '10'}",
        ]

    def _format_endpoint_request_body(self, endpoint: "Endpoint") -> List[str]:
        """Format endpoint request body section."""
        lines: list[str] = []
        if not (hasattr(endpoint, "request_body") and endpoint.request_body):
            return lines

        rb = endpoint.request_body
        content_type = getattr(rb, "content_type", CONTENT_TYPE_JSON)
        schema = getattr(rb, "schema", {})

        lines.append("\nRequest Body:")
        lines.append(f"  Content-Type: {content_type}")
        rb_desc = getattr(rb, "description", None)
        if rb_desc:
            lines.append(f"  Description: {rb_desc[:100]}")
        lines.append(f"  Required: {getattr(rb, 'required', True)}")

        # File upload warning
        if content_type in ["multipart/form-data", "application/octet-stream"]:
            lines.extend(self._get_file_upload_warning())

        # Format schema
        if schema and isinstance(schema, dict):
            lines.extend(self._format_schema(schema, indent=2))

        return lines

    def _get_file_upload_warning(self) -> List[str]:
        """Return file upload warning lines."""
        return [
            "",
            "  *** FILE UPLOAD ENDPOINT ***",
            "  This endpoint requires multipart/form-data file upload.",
            "  DO NOT use json= parameter. Use files= with actual file data:",
            "  Example: files={'file': ('test.txt', b'file content', 'text/plain')}",
            "",
        ]

    def _format_endpoint_responses(
        self, endpoint: "Endpoint", exclude_2xx: bool = False
    ) -> List[str]:
        """Format endpoint responses section."""
        lines: list[str] = []
        if not (hasattr(endpoint, "responses") and endpoint.responses):
            return lines

        responses = endpoint.responses
        lines.append("\nResponses:")

        if isinstance(responses, dict):  # type: ignore[unreachable]
            for status_code, response in responses.items():  # type: ignore[unreachable]
                if exclude_2xx and self._is_2xx_status(status_code):
                    continue
                lines.extend(self._format_single_response(status_code, response))
        elif isinstance(responses, list):
            for response in responses:
                status_code = getattr(response, "status_code", "???")
                if exclude_2xx and self._is_2xx_status(status_code):
                    continue
                lines.extend(self._format_single_response(status_code, response))

        return lines

    def _is_2xx_status(self, status_code: Any) -> bool:
        """Check if status code is in 2xx range."""
        try:
            return 200 <= int(status_code) < 300
        except (ValueError, TypeError):
            return False

    def _format_single_response(self, status_code: Any, response: Any) -> List[str]:
        """Format a single response entry."""
        lines = []
        desc = (
            getattr(response, "description", "")
            if hasattr(response, "description")
            else str(response)
        )
        desc_str = desc[:50] if len(str(desc)) > 50 else desc
        lines.append(f"  - {status_code}: {desc_str}")

        resp_schema = (
            getattr(response, "schema", None) if hasattr(response, "schema") else None
        )
        if resp_schema and isinstance(resp_schema, dict):
            lines.append("    Response Schema (use these EXACT field names):")
            lines.extend(self._format_response_schema(resp_schema, indent=3))

        return lines

    def _format_single_endpoint(
        self, endpoint: "Endpoint", exclude_2xx: bool = False
    ) -> str:
        """Format a single endpoint with full details for the prompt.

        Args:
            endpoint: Endpoint object to format
            exclude_2xx: If True, omit 2xx responses (for negative/security tests)
        """
        lines = []

        # Basic info
        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)
        summary = getattr(endpoint, "summary", "") or _DEFAULT_SUMMARY
        description = getattr(endpoint, "description", "") or ""

        lines.append(f"Operation: {endpoint.method.upper()} {endpoint.path}")
        lines.append(f"Operation ID: {operation_id}")
        lines.append(f"Summary: {summary}")
        if description:
            lines.append(f"Description: {description}")

        # Parameters, request body, and responses
        param_lines, _, _ = self._format_endpoint_parameters(endpoint)
        lines.extend(param_lines)
        lines.extend(self._format_endpoint_request_body(endpoint))
        lines.extend(self._format_endpoint_responses(endpoint, exclude_2xx))

        return "\n".join(lines)

    def _format_property_constraints(self, schema: dict) -> List[str]:
        """Extract constraint strings from a schema."""
        constraints = []
        constraint_keys = [
            ("minLength", "minLength"),
            ("maxLength", "maxLength"),
            ("minimum", "min"),
            ("maximum", "max"),
            ("exclusiveMinimum", "exclusiveMin"),
            ("exclusiveMaximum", "exclusiveMax"),
            ("multipleOf", "multipleOf"),
        ]
        for key, label in constraint_keys:
            if schema.get(key) is not None:
                constraints.append(f"{label}={schema[key]}")
        if schema.get("pattern"):
            pattern = schema["pattern"][:40]
            constraints.append(f"pattern='{pattern}'")
        return constraints

    def _format_discriminated_union(
        self, one_of: List[dict], discriminator: dict, prefix: str
    ) -> List[str]:
        """Format a discriminated union schema."""
        lines = []
        prop_name = discriminator.get("propertyName", "type")
        lines.append(f"{prefix}Schema: DISCRIMINATED UNION")
        lines.append(f"{prefix}  *** DISCRIMINATOR FIELD: {prop_name} (REQUIRED) ***")
        lines.append(
            f"{prefix}  You MUST include '{prop_name}' to specify which variant."
        )
        lines.append(f"{prefix}")

        mapping = discriminator.get("mapping", {})
        if mapping:
            lines.append(f"{prefix}  Valid '{prop_name}' values and their schemas:")
            for disc_value, ref in mapping.items():
                lines.append(f'{prefix}    - {prop_name}="{disc_value}":')
                variant_schema = self._resolve_ref_in_union(ref, one_of)
                if variant_schema:
                    lines.extend(
                        self._format_variant_properties(
                            variant_schema, prop_name, prefix + "        "
                        )
                    )
                lines.append(f"{prefix}")
        return lines

    def _format_variant_properties(
        self, variant_schema: dict, discriminator_prop: str, prefix: str
    ) -> List[str]:
        """Format properties of a union variant."""
        lines = []
        variant_props = variant_schema.get("properties", {})
        variant_required = variant_schema.get("required", [])
        for vp_name, vp_schema in variant_props.items():
            if vp_name == discriminator_prop:
                continue
            unwrapped_vp, _ = unwrap_nullable_schema(vp_schema)
            vp_type = unwrapped_vp.get("type", "any")
            req_marker = " (REQUIRED)" if vp_name in variant_required else ""
            generator = self._get_type_instruction(unwrapped_vp, field_name=vp_name)
            lines.append(f"{prefix}{vp_name}: {vp_type}{req_marker}")
            lines.append(f"{prefix}    USE: {generator}")
        return lines

    def _format_union_without_discriminator(
        self, one_of: List[dict], prefix: str
    ) -> List[str]:
        """Format a union type without discriminator."""
        lines = []
        lines.append(f"{prefix}Schema: UNION TYPE (oneOf/anyOf)")
        lines.append(f"{prefix}  Send ONE of the following object types:")
        for i, variant in enumerate(one_of, 1):
            if "$ref" in variant:
                lines.append(f"{prefix}  Option {i}: {variant['$ref']}")
            else:
                variant_props = variant.get("properties", {})
                if variant_props:
                    lines.append(f"{prefix}  Option {i}:")
                    for vp_name, vp_schema in variant_props.items():
                        unwrapped_vp, _ = unwrap_nullable_schema(vp_schema)
                        vp_type = unwrapped_vp.get("type", "any")
                        generator = self._get_type_instruction(
                            unwrapped_vp, field_name=vp_name
                        )
                        lines.append(f"{prefix}    - {vp_name}: {vp_type}")
                        lines.append(f"{prefix}        USE: {generator}")
        return lines

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
            if discriminator:
                return self._format_discriminated_union(one_of, discriminator, prefix)
            else:
                return self._format_union_without_discriminator(one_of, prefix)

        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        if properties:
            lines.extend(self._format_schema_header(prefix, required_fields))
            for prop_name, prop_schema in properties.items():
                lines.extend(
                    self._format_schema_property(
                        prop_name, prop_schema, required_fields, prefix, indent
                    )
                )

        return lines

    def _format_schema_header(
        self, prefix: str, required_fields: List[str]
    ) -> List[str]:
        """Format the schema header with required fields info."""
        lines = [f"{prefix}Schema:"]
        if required_fields:
            lines.append(f"{prefix}  Required fields: {required_fields}")
        else:
            lines.append(f"{prefix}  Required fields: none")
        lines.append(
            f"{prefix}  Properties (use these EXACT field names in your code):"
        )
        return lines

    def _format_schema_property(
        self,
        prop_name: str,
        prop_schema: dict,
        required_fields: List[str],
        prefix: str,
        indent: int,
    ) -> List[str]:
        """Format a single property in the schema."""
        lines = []
        is_required = prop_name in required_fields
        unwrapped, is_nullable = unwrap_nullable_schema(prop_schema)

        lines.extend(
            self._format_property_type_line(
                prop_name, unwrapped, is_required, is_nullable, prefix
            )
        )
        lines.extend(self._format_property_metadata(prop_schema, unwrapped, prefix))
        lines.extend(self._format_nested_type(unwrapped, prefix, indent))

        return lines

    def _format_property_type_line(
        self,
        prop_name: str,
        unwrapped: dict,
        is_required: bool,
        is_nullable: bool,
        prefix: str,
    ) -> List[str]:
        """Format the property type line with markers."""
        nullable_marker = ", nullable" if is_nullable else ""
        req_marker = (
            f" (REQUIRED{nullable_marker})"
            if is_required
            else f" (optional{nullable_marker})"
        )

        prop_type = unwrapped.get("type", "any")
        prop_format = unwrapped.get("format")
        type_str = f"{prop_type} [{prop_format}]" if prop_format else prop_type

        generator = self._get_type_instruction(unwrapped, field_name=prop_name)
        return [
            f"{prefix}    - {prop_name}: {type_str}{req_marker}",
            f"{prefix}        USE: {generator}",
        ]

    def _format_property_metadata(
        self,
        prop_schema: dict,
        unwrapped: dict,
        prefix: str,
    ) -> List[str]:
        """Format property metadata (description, constraints, enum, default)."""
        lines = []
        desc_text = prop_schema.get("description") or unwrapped.get("description")
        if desc_text:
            lines.append(f"{prefix}        description: {desc_text[:80]}")

        constraints = self._format_property_constraints(unwrapped)
        if constraints:
            lines.append(f"{prefix}        constraints: {', '.join(constraints)}")

        if unwrapped.get("enum"):
            lines.append(f"{prefix}        allowed values: {unwrapped['enum']}")

        if unwrapped.get("default") is not None:
            lines.append(f"{prefix}        default: {unwrapped['default']}")

        return lines

    def _format_nested_type(
        self, unwrapped: dict, prefix: str, indent: int
    ) -> List[str]:
        """Format nested object or array types."""
        prop_type = unwrapped.get("type", "any")

        if prop_type == "object":
            return self._format_nested_object(unwrapped, prefix, indent)
        elif prop_type == "array" and unwrapped.get("items"):
            return self._format_array_items(unwrapped, prefix, indent)
        return []

    def _format_nested_object(
        self, unwrapped: dict, prefix: str, indent: int
    ) -> List[str]:
        """Format nested object properties or map types."""
        if unwrapped.get("properties"):
            return [f"{prefix}        nested object properties:"] + self._format_schema(
                unwrapped, indent + 3
            )

        add_props = unwrapped.get("additionalProperties")
        if add_props:
            val_type = (
                add_props.get("type", "any") if isinstance(add_props, dict) else "any"
            )
            return [f"{prefix}        map type: string keys → {val_type} values"]
        return []

    def _format_array_items(
        self, unwrapped: dict, prefix: str, indent: int
    ) -> List[str]:
        """Format array item types."""
        lines = []
        items = unwrapped["items"]
        items_unwrapped, _ = unwrap_nullable_schema(items)
        items_one_of = items_unwrapped.get("oneOf") or items_unwrapped.get("anyOf")

        if items_one_of and isinstance(items_one_of, list):
            lines.extend(self._format_union_array_items(items_one_of, prefix, indent))
        else:
            items_type = items_unwrapped.get("type", "any")
            lines.append(f"{prefix}        array items type: {items_type}")

        lines.extend(self._format_array_constraints(unwrapped, prefix))

        if items_unwrapped.get("properties"):
            lines.append(f"{prefix}        array item properties:")
            lines.extend(self._format_schema(items_unwrapped, indent + 3))

        return lines

    def _format_union_array_items(
        self, items_one_of: List[dict], prefix: str, indent: int
    ) -> List[str]:
        """Format array items that are a union type."""
        lines = []
        variant_names = []
        first_variant_with_props = None

        for variant in items_one_of:
            if not isinstance(variant, dict):
                continue  # type: ignore[unreachable]
            name = self._get_variant_name(variant)
            if name:
                variant_names.append(name)
            if first_variant_with_props is None:
                variant_props, _ = extract_all_properties(variant)
                if variant_props:
                    first_variant_with_props = variant

        if variant_names:
            lines.append(
                f"{prefix}        array items type: oneOf ({' | '.join(variant_names)})"
            )
        else:
            lines.append(
                f"{prefix}        array items type: oneOf (union of {len(items_one_of)} variants)"
            )

        if first_variant_with_props:
            lines.append(f"{prefix}        first variant schema (use this structure):")
            lines.extend(self._format_schema(first_variant_with_props, indent + 3))

        return lines

    def _get_variant_name(self, variant: dict) -> Optional[str]:
        """Get a name for a union variant."""
        if "$ref" in variant:
            return str(variant["$ref"].split("/")[-1])
        if "properties" in variant:
            for p_schema in variant.get("properties", {}).values():
                if isinstance(p_schema, dict) and p_schema.get("const"):
                    return str(p_schema["const"])
        return None

    def _format_array_constraints(self, unwrapped: dict, prefix: str) -> List[str]:
        """Format array minItems/maxItems constraints."""
        min_items = unwrapped.get("minItems")
        max_items = unwrapped.get("maxItems")
        if min_items is None and max_items is None:
            return []

        constraints = []
        if min_items is not None:
            constraints.append(f"minItems={min_items}")
        if max_items is not None:
            constraints.append(f"maxItems={max_items}")
        return [f"{prefix}        array constraints: {', '.join(constraints)}"]

    def _resolve_ref_in_union(self, ref: str, one_of: List[dict]) -> Optional[dict]:
        """Try to resolve a $ref within a oneOf/anyOf array."""
        # Direct $ref match or allOf containing the $ref
        for variant in one_of:
            result = self._try_resolve_variant_by_ref(variant, ref)
            if result is not None:
                return result

        # Fallback: match by const value in inline properties
        return self._try_resolve_variant_by_const(one_of, ref)

    def _try_merge_all_of_with_properties(self, variant: dict) -> Optional[dict]:
        """Try to merge allOf and return if it has properties."""
        if "allOf" not in variant:
            return None
        merged = self._merge_all_of(variant["allOf"])
        if merged and merged.get("properties"):
            return merged
        return None

    def _try_resolve_variant_by_ref(
        self,
        variant: dict,
        ref: str,
    ) -> Optional[dict]:
        """Try to resolve a variant by direct $ref match or allOf containing the $ref."""
        # Direct $ref match
        if variant.get("$ref") == ref:
            if "properties" in variant:
                return variant
            return self._try_merge_all_of_with_properties(variant)

        # Check allOf for $ref match
        if "allOf" not in variant:
            return None
        for sub in variant["allOf"]:
            if sub.get("$ref") == ref:
                return self._try_merge_all_of_with_properties(variant)
        return None

    @staticmethod
    def _merge_all_of(all_of: List[dict]) -> dict:
        """Merge allOf schemas into a single schema with properties and required."""
        merged: dict = {"properties": {}, "required": []}
        for sub in all_of:
            if "properties" in sub:
                merged["properties"].update(sub["properties"])
            if "required" in sub:
                merged["required"].extend(sub["required"])
        return merged

    @staticmethod
    def _try_resolve_variant_by_const(
        one_of: List[dict],
        ref: str,
    ) -> Optional[dict]:
        """Try to match a variant by const value matching the ref name."""
        ref_name = (
            ref.split("/")[-1].lower().replace("_", "").replace("-", "") if ref else ""
        )
        for variant in one_of:
            variant_props = variant.get("properties", {})
            if not variant_props:
                continue
            for _, prop_schema in variant_props.items():
                const_val = (
                    str(prop_schema.get("const", ""))
                    .lower()
                    .replace("_", "")
                    .replace("-", "")
                )
                if const_val == ref_name:
                    return variant
        return None

    def _build_response_type_string(self, unwrapped: dict, is_nullable: bool) -> str:
        """Build type string for response schema property."""
        prop_type = unwrapped.get("type", "any")
        prop_format = unwrapped.get("format")
        type_str = f"{prop_type} [{prop_format}]" if prop_format else prop_type
        if is_nullable:
            type_str += " (nullable)"
        return type_str

    def _format_response_nested_object(self, unwrapped: dict, prefix: str) -> List[str]:
        """Format nested object properties in response schema."""
        lines = []
        for nested_name, nested_schema in unwrapped.get("properties", {}).items():
            n_unwrapped, _ = unwrap_nullable_schema(nested_schema)
            nested_type = n_unwrapped.get("type", "any")
            lines.append(f"{prefix}    - {nested_name}: {nested_type}")
        return lines

    def _format_response_array_type(
        self, items_unwrapped: dict, prefix: str
    ) -> List[str]:
        """Format array type description in response schema."""
        items_one_of = items_unwrapped.get("oneOf") or items_unwrapped.get("anyOf")
        if items_one_of and isinstance(items_one_of, list):
            return [f"{prefix}    (array of oneOf variants)"]
        items_type = items_unwrapped.get("type", "any")
        return [f"{prefix}    (array of {items_type})"]

    def _format_response_array_properties(
        self, items_unwrapped: dict, prefix: str
    ) -> List[str]:
        """Format array item properties in response schema."""
        lines = []
        for item_name, item_schema in items_unwrapped.get("properties", {}).items():
            i_unwrapped, _ = unwrap_nullable_schema(item_schema)
            item_type = i_unwrapped.get("type", "any")
            lines.append(f"{prefix}    - {item_name}: {item_type}")
        return lines

    def _format_response_property(
        self, prop_name: str, prop_schema: dict, prefix: str
    ) -> List[str]:
        """Format a single property in response schema."""
        lines = []
        unwrapped, is_nullable = unwrap_nullable_schema(prop_schema)
        prop_type = unwrapped.get("type", "any")
        type_str = self._build_response_type_string(unwrapped, is_nullable)

        lines.append(f"{prefix}- {prop_name}: {type_str}")

        if prop_type == "object" and unwrapped.get("properties"):
            lines.extend(self._format_response_nested_object(unwrapped, prefix))

        if prop_type == "array" and unwrapped.get("items"):
            items_unwrapped, _ = unwrap_nullable_schema(unwrapped["items"])
            lines.extend(self._format_response_array_type(items_unwrapped, prefix))
            if items_unwrapped.get("properties"):
                lines.extend(
                    self._format_response_array_properties(items_unwrapped, prefix)
                )

        return lines

    def _format_response_schema(self, schema: dict, indent: int = 0) -> List[str]:
        """Format a response JSON Schema for the LLM prompt."""
        prefix = "  " * indent
        schema_type = schema.get("type", "object")
        properties = schema.get("properties", {})

        if properties:
            lines = []
            for prop_name, prop_schema in properties.items():
                lines.extend(
                    self._format_response_property(prop_name, prop_schema, prefix)
                )
            return lines

        if schema_type == "array":
            items_unwrapped, _ = unwrap_nullable_schema(schema.get("items", {}))
            lines = self._format_response_array_type(items_unwrapped, prefix)
            if items_unwrapped.get("properties"):
                lines.extend(
                    self._format_response_array_properties(items_unwrapped, prefix)
                )
            return lines

        return [f"{prefix}type: {schema_type}"]

    def _score_parent_path_match(
        self, candidate_path: str, parent_paths: List[str]
    ) -> Tuple[float, Optional[str]]:
        """Score endpoint based on parent path match."""
        for i, parent in enumerate(parent_paths):
            if self._paths_match(candidate_path, parent):
                score = 100.0 + (i * 10.0)
                reason = f"parent path level {i + 1}: {parent}"
                return score, reason
        return 0.0, None

    def _score_prefix_match(
        self, candidate_path: str, target_prefix: List[str]
    ) -> Tuple[float, Optional[str]]:
        """Score endpoint based on shared prefix."""
        candidate_prefix = self._get_static_prefix(candidate_path)
        common_len = self._common_prefix_length(target_prefix, candidate_prefix)
        if common_len >= 3:
            score = 20.0 + (common_len * 5.0)
            return score, f"shared prefix ({common_len} segments)"
        return 0.0, None

    def _score_tag_match(
        self, endpoint: "Endpoint", target_tags: set, current_score: float
    ) -> Tuple[float, Optional[str]]:
        """Score endpoint based on tag matching."""
        if current_score == 0:
            return 0.0, None
        endpoint_tags = set(getattr(endpoint, "tags", []) or [])
        common_tags = target_tags & endpoint_tags
        if common_tags:
            score = 10.0 * len(common_tags)
            return score, f"same tag: {', '.join(common_tags)}"
        return 0.0, None

    def _calculate_endpoint_relevance(
        self,
        endpoint: "Endpoint",
        parent_paths: List[str],
        target_prefix: List[str],
        target_tags: set,
    ) -> Optional[Tuple[Any, float, str]]:
        """Calculate relevance score for a candidate endpoint."""
        score = 0.0
        reasons = []

        parent_score, parent_reason = self._score_parent_path_match(
            endpoint.path, parent_paths
        )
        if parent_score > 0:
            score += parent_score
            reasons.append(parent_reason)
        else:
            prefix_score, prefix_reason = self._score_prefix_match(
                endpoint.path, target_prefix
            )
            if prefix_score > 0:
                score += prefix_score
                reasons.append(prefix_reason)

        tag_score, tag_reason = self._score_tag_match(endpoint, target_tags, score)
        if tag_score > 0:
            score += tag_score
            reasons.append(tag_reason)

        if score > 0:
            return (endpoint, score, "; ".join(r for r in reasons if r))
        return None

    def _find_related_create_endpoints(
        self,
        target_endpoint: "Endpoint",
        all_endpoints: List[Any],
    ) -> List[Tuple[Any, float, str]]:
        """Find CREATE (POST) endpoints that are parents of the target endpoint."""
        if target_endpoint.method.upper() == "POST":
            return []

        parent_paths = self._extract_parent_paths(target_endpoint.path)
        target_prefix = self._get_static_prefix(target_endpoint.path)
        target_tags = set(getattr(target_endpoint, "tags", []) or [])

        results = []
        for endpoint in all_endpoints:
            if endpoint.method.upper() != "POST":
                continue
            if endpoint.path == target_endpoint.path:
                continue

            result = self._calculate_endpoint_relevance(
                endpoint, parent_paths, target_prefix, target_tags
            )
            if result:
                results.append(result)

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

    def _get_empty_setup_endpoints_message(self) -> str:
        """Return message when no CREATE endpoints are found."""
        return """
=== SETUP ENDPOINTS (for creating test data) ===
No CREATE (POST) endpoints found that are related to this resource.
You may need to use test_data_generator or assume test data already exists.
Do NOT invent or call POST endpoints that are not documented here.
"""

    def _get_setup_endpoints_header(self) -> List[str]:
        """Return header lines for setup endpoints section."""
        return [
            "=== SETUP ENDPOINTS (for creating test data) ===",
            "",
            "These POST endpoints can be used to create resources before testing.",
            "They are ranked by relevance to the endpoint you are testing.",
            "Use ONLY these endpoints for setup - do NOT invent endpoints that don't exist.",
            "**CRITICAL: Use the expected_status shown for EACH setup endpoint.**",
            "",
        ]

    def _get_setup_endpoint_status_codes(self, endpoint: "Endpoint") -> List[int]:
        """Get status codes for a setup endpoint, with defaults."""
        all_codes = self._extract_expected_status_codes(endpoint)
        filtered = self._filter_status_codes_for_scenario(
            all_codes, ScenarioType.POSITIVE, method="POST"
        )
        return filtered if filtered else [200, 201]

    def _format_single_setup_endpoint(
        self, endpoint: "Endpoint", rank: int, score: float, reason: str
    ) -> List[str]:
        """Format a single setup endpoint for the prompt."""
        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)
        summary = getattr(endpoint, "summary", "") or _DEFAULT_SUMMARY
        description = getattr(endpoint, "description", "") or ""
        status_codes = self._get_setup_endpoint_status_codes(endpoint)

        lines = [
            f"--- Rank #{rank} (relevance: {score:.0f}) ---",
            f"POST {endpoint.path}",
            f"Operation ID: {operation_id}",
            f"**expected_status={status_codes}**  <-- USE THIS for this setup call",
            f"Why relevant: {reason}",
            f"Summary: {summary}",
        ]
        if description:
            lines.append(f"Description: {description[:150]}")

        lines.extend(self._format_setup_endpoint_schema(endpoint))
        lines.append("")
        return lines

    def _format_setup_endpoint_schema(self, endpoint: "Endpoint") -> List[str]:
        """Format request body schema for a setup endpoint."""
        if not hasattr(endpoint, "request_body") or not endpoint.request_body:
            return []
        schema = getattr(endpoint.request_body, "schema", {})
        if not schema or not isinstance(schema, dict):
            return []
        return ["Request Body Schema:"] + self._format_schema(schema, indent=1)

    def _format_setup_call_pattern(self, first_endpoint: "Endpoint") -> List[str]:
        """Format the setup call pattern example."""
        status_codes = self._get_setup_endpoint_status_codes(first_endpoint)
        return [
            "=== SETUP CALL PATTERN ===",
            "```python",
            "# Use the expected_status from the setup endpoint above, NOT from the main endpoint",
            f'result = self.make_request("POST", "{first_endpoint.path}", expected_status={status_codes}, json=data)',
            "```",
            "",
        ]

    def _format_related_create_endpoints(
        self,
        related_endpoints: List[Tuple[Any, float, str]],
    ) -> str:
        """Format related CREATE endpoints for the LLM prompt."""
        if not related_endpoints:
            return self._get_empty_setup_endpoints_message()

        lines = self._get_setup_endpoints_header()

        for i, (endpoint, score, reason) in enumerate(related_endpoints, 1):
            lines.extend(self._format_single_setup_endpoint(endpoint, i, score, reason))

        lines.extend(self._format_setup_call_pattern(related_endpoints[0][0]))
        return "\n".join(lines)

    def _filter_status_codes_for_scenario(
        self,
        status_codes: List[int],
        scenario_type: ScenarioType,
        method: str = "GET",
        exclude_auth: bool = False,
    ) -> List[int]:
        """Filter status codes based on scenario type.

        Uses spec codes if available, otherwise falls back to FallbackHttpResponseRegistry.
        """
        if not status_codes:
            return self._get_fallback_codes(method, scenario_type, exclude_auth)
        return self._filter_int_codes_by_scenario(status_codes, scenario_type)

    @staticmethod
    def _filter_int_codes_by_scenario(
        codes: List[int],
        scenario_type: ScenarioType,
    ) -> List[int]:
        """Filter integer status codes by scenario type."""
        if scenario_type == ScenarioType.POSITIVE:
            return sorted(c for c in codes if c < 400)
        if scenario_type == ScenarioType.NEGATIVE:
            return sorted(c for c in codes if 400 <= c < 500)
        if scenario_type == ScenarioType.SECURITY:
            return sorted(c for c in codes if c < 500)
        return sorted(codes)  # type: ignore[unreachable]

    def _get_fallback_codes(
        self,
        method: str,
        scenario_type: ScenarioType,
        exclude_auth: bool = False,
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

        return sorted(all_codes)  # type: ignore[unreachable]

    def _extract_expected_status_codes(self, endpoint: "Endpoint") -> List[int]:
        """Extract expected HTTP status codes from the OpenAPI spec responses."""
        if not hasattr(endpoint, "responses") or not endpoint.responses:
            return []

        responses = endpoint.responses
        if isinstance(responses, dict):  # type: ignore[unreachable]
            return sorted(self._parse_status_codes_from_dict(responses))  # type: ignore[unreachable]
        if isinstance(responses, list):
            return sorted(self._parse_status_codes_from_list(responses))
        return []  # type: ignore[unreachable]

    @staticmethod
    def _parse_status_codes_from_dict(responses: dict) -> List[int]:
        """Parse integer status codes from a dict-style responses object."""
        codes = []
        for key in responses.keys():
            try:
                if str(key).lower() == "default":
                    continue
                codes.append(int(key))
            except (ValueError, TypeError):
                pass
        return codes

    @staticmethod
    def _parse_status_codes_from_list(responses: list) -> List[int]:
        """Parse integer status codes from a list-style responses object."""
        codes = []
        for response in responses:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                try:
                    codes.append(int(status_code))
                except (ValueError, TypeError):
                    pass
        return codes

    def _extract_status_codes_with_descriptions(
        self,
        endpoint: "Endpoint",
    ) -> List[Tuple[int, str]]:
        """Extract status codes with descriptions from OpenAPI spec responses."""
        if not hasattr(endpoint, "responses") or not endpoint.responses:
            return []

        responses = endpoint.responses
        if isinstance(responses, dict):  # type: ignore[unreachable]
            result = self._parse_codes_with_desc_from_dict(responses)  # type: ignore[unreachable]
        elif isinstance(responses, list):
            result = self._parse_codes_with_desc_from_list(responses)
        else:
            return []  # type: ignore[unreachable]
        return sorted(result, key=lambda x: x[0])

    @staticmethod
    def _parse_codes_with_desc_from_dict(
        responses: dict,
    ) -> List[Tuple[int, str]]:
        """Parse (code, description) tuples from dict-style responses."""
        result = []
        for status_code_str, response in responses.items():
            try:
                if str(status_code_str).lower() == "default":
                    continue
                code = int(status_code_str)
                if hasattr(response, "description"):
                    desc = getattr(response, "description", "") or ""
                elif isinstance(response, dict):
                    desc = response.get("description", "")
                else:
                    desc = ""
                result.append((code, desc))
            except (ValueError, TypeError):
                pass
        return result

    @staticmethod
    def _parse_codes_with_desc_from_list(
        responses: list,
    ) -> List[Tuple[int, str]]:
        """Parse (code, description) tuples from list-style responses."""
        result = []
        for response in responses:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                try:
                    code = int(status_code)
                    desc = getattr(response, "description", "") or ""
                    result.append((code, desc))
                except (ValueError, TypeError):
                    pass
        return result

    def _filter_codes_by_scenario_type(
        self,
        codes: List[Tuple[int, str]],
        scenario_type: ScenarioType,
    ) -> List[Tuple[int, str]]:
        """Filter status codes by scenario type."""
        if scenario_type == ScenarioType.POSITIVE:
            return [(c, d) for c, d in codes if 200 <= c < 300]
        elif scenario_type == ScenarioType.NEGATIVE:
            return [(c, d) for c, d in codes if 400 <= c < 500]
        elif scenario_type == ScenarioType.SECURITY:
            return [(c, d) for c, d in codes if c < 500]
        return codes  # type: ignore[unreachable]

    def _get_fallback_codes_with_descriptions(
        self, method: str, has_auth: bool
    ) -> List[Tuple[int, str]]:
        """Get fallback status codes with descriptions from registry."""
        response_block = self._fallback_registry.get_responses(
            methods=method, exclude_auth=not has_auth
        )
        method_responses = response_block.as_dict().get(method, {})

        codes: List[Tuple[int, str]] = []
        for code_str, data in method_responses.items():
            try:
                code = int(code_str)
                desc = data.get("description", "") if isinstance(data, dict) else ""
                codes.append((code, desc))
            except (ValueError, TypeError):
                pass
        return codes

    def _get_default_codes_for_scenario(
        self, scenario_type: ScenarioType
    ) -> List[Tuple[int, str]]:
        """Get default status codes when no codes found for scenario type."""
        if scenario_type == ScenarioType.POSITIVE:
            return [(200, "OK")]
        elif scenario_type == ScenarioType.NEGATIVE:
            return [(400, "Bad Request"), (422, "Unprocessable Entity")]
        elif scenario_type == ScenarioType.SECURITY:
            return [
                (400, "Bad Request"),
                (403, "Forbidden"),
                (422, "Unprocessable Entity"),
            ]
        return []  # type: ignore[unreachable]

    def _precompute_scenario_status_codes(
        self,
        endpoint: "Endpoint",
        scenario_type: ScenarioType,
        has_auth: bool,
    ) -> List[Tuple[int, str]]:
        """Pre-compute status codes + descriptions for a scenario type."""
        spec_codes = self._extract_status_codes_with_descriptions(endpoint)
        filtered = self._filter_codes_by_scenario_type(spec_codes, scenario_type)

        if filtered:
            return sorted(filtered, key=lambda x: x[0])

        # No 2xx in spec means skip positive workflow
        if spec_codes and scenario_type == ScenarioType.POSITIVE:
            return []

        # Use fallback registry when spec has no responses
        fallback_codes = self._get_fallback_codes_with_descriptions(
            endpoint.method.upper(), has_auth
        )
        filtered_fallback = self._filter_codes_by_scenario_type(
            fallback_codes, scenario_type
        )

        if filtered_fallback:
            return sorted(filtered_fallback, key=lambda x: x[0])
        return self._get_default_codes_for_scenario(scenario_type)

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

    def _precompute_injection_points(self, endpoint: "Endpoint") -> Optional[str]:
        """Pre-compute valid security injection points for an endpoint.

        Returns:
            Formatted string listing injection points, or None if no valid targets.
        """
        body_fields = self._scan_body_string_fields(endpoint)
        query_params = self._scan_query_string_params(endpoint)

        if not body_fields and not query_params:
            return None

        return self._format_injection_points(body_fields, query_params)

    def _scan_body_string_fields(self, endpoint: "Endpoint") -> List[str]:
        """Scan request body for string fields suitable for injection."""
        if not (hasattr(endpoint, "request_body") and endpoint.request_body):
            return []
        schema = getattr(endpoint.request_body, "schema", {})
        if not (schema and isinstance(schema, dict)):
            return []

        fields = []
        properties, _ = extract_all_properties(schema)
        for field_name, field_schema in properties.items():
            if isinstance(field_schema, dict):
                unwrapped_fs, _ = unwrap_nullable_schema(field_schema)
                if unwrapped_fs.get("type", "") == "string":
                    fields.append(field_name)
        return fields

    def _scan_query_string_params(self, endpoint: "Endpoint") -> List[str]:
        """Scan endpoint parameters for string query params."""
        if not (hasattr(endpoint, "parameters") and endpoint.parameters):
            return []

        params = []
        for param in endpoint.parameters:
            location = self._get_param_location(param)
            if location == "query":
                param_type = getattr(param, "type", None) or "string"
                if param_type == "string" or "string" in str(param_type):
                    params.append(getattr(param, "name", "unknown"))
        return params

    @staticmethod
    def _get_param_location(param: Any) -> str:
        """Extract the location string from a parameter object."""
        location = getattr(param, "location", None)
        if location is None:
            location = getattr(param, "in_", "query")
        if location is not None and hasattr(location, "value"):
            location = location.value
        return str(location) if location is not None else "query"

    @staticmethod
    def _format_injection_points(
        body_fields: List[str],
        query_params: List[str],
    ) -> str:
        """Format injection points into a prompt-ready string."""
        lines = []
        if body_fields:
            lines.append("Request body string fields (inject payloads here):")
            lines.extend(f"  - {f}" for f in body_fields)
        if query_params:
            lines.append("Query parameters (inject payloads here):")
            lines.extend(f"  - {p}" for p in query_params)
        return "\n".join(lines)

    def _extract_endpoint_params(
        self, endpoint: "Endpoint"
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Extract path and query parameters from endpoint.

        Returns:
            Tuple of (path_params, query_params) as lists of (name, type) tuples
        """
        path_params: List[Tuple[str, str]] = []
        query_params: List[Tuple[str, str]] = []

        if not (hasattr(endpoint, "parameters") and endpoint.parameters):
            return path_params, query_params

        for param in endpoint.parameters:
            param_name = getattr(param, "name", "unknown")
            param_type = getattr(param, "type", None) or "string"
            param_location: Any = getattr(param, "location", None)
            if param_location is None:
                param_location = getattr(param, "in_", "query")
            if param_location is not None and hasattr(param_location, "value"):
                param_location = param_location.value

            if param_location == "path":
                path_params.append((param_name, param_type))
            elif param_location == "query":
                query_params.append((param_name, param_type))

        return path_params, query_params

    def _generate_nonexistent_id_scenarios(
        self, path_params: List[Tuple[str, str]]
    ) -> List[str]:
        """Generate NON_EXISTENT_ID test scenarios for path params."""
        scenarios = []
        for name, ptype in path_params:
            if ptype.lower() in ("integer", "number") or ptype.lower().startswith(
                "array[int"
            ):
                scenarios.append(
                    f"NON_EXISTENT_ID: Test {{{name}}} with value 999999999 (integer path param)"
                )
            else:
                scenarios.append(
                    f'NON_EXISTENT_ID: Test {{{name}}} with value "nonexistent-id-12345" (string path param)'
                )
        return scenarios

    def _extract_body_field_categories(self, endpoint: "Endpoint") -> Tuple[
        List[str],
        List[Tuple[str, str]],
        List[Tuple[str, List[Any]]],
        List[Tuple[str, str]],
        List[Tuple[str, Optional[float], Optional[float]]],
    ]:
        """Categorize request body fields for negative testing.

        Returns:
            Tuple of (required, typed, enum, pattern, numeric) field lists
        """
        empty_result: Tuple[
            List[str],
            List[Tuple[str, str]],
            List[Tuple[str, List[Any]]],
            List[Tuple[str, str]],
            List[Tuple[str, Optional[float], Optional[float]]],
        ] = ([], [], [], [], [])

        properties, required_list = self._get_body_properties(endpoint)
        if properties is None:
            return empty_result

        required_fields: List[str] = []
        typed_fields: List[Tuple[str, str]] = []
        enum_fields: List[Tuple[str, List[Any]]] = []
        pattern_fields: List[Tuple[str, str]] = []
        numeric_fields: List[Tuple[str, Optional[float], Optional[float]]] = []

        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            self._categorize_field(
                field_name,
                field_schema,
                required_list,
                required_fields,
                typed_fields,
                enum_fields,
                pattern_fields,
                numeric_fields,
            )

        return (
            required_fields,
            typed_fields,
            enum_fields,
            pattern_fields,
            numeric_fields,
        )

    def _get_body_properties(
        self,
        endpoint: "Endpoint",
    ) -> Tuple[Optional[Dict], List[str]]:
        """Extract properties and required list from endpoint body schema."""
        if not (hasattr(endpoint, "request_body") and endpoint.request_body):
            return None, []
        schema = getattr(endpoint.request_body, "schema", {})
        if not (schema and isinstance(schema, dict)):
            return None, []
        properties, required_list = extract_all_properties(schema)
        return properties, required_list

    def _categorize_field(
        self,
        field_name: str,
        field_schema: dict,
        required_list: List[str],
        required_fields: List[str],
        typed_fields: List[Tuple[str, str]],
        enum_fields: List[Tuple[str, List[Any]]],
        pattern_fields: List[Tuple[str, str]],
        numeric_fields: List[Tuple[str, Optional[float], Optional[float]]],
    ) -> None:
        """Categorize a single field into the appropriate category lists."""
        unwrapped_fs, _ = unwrap_nullable_schema(field_schema)
        field_type = unwrapped_fs.get("type", "")

        if field_name in required_list:
            required_fields.append(field_name)
        if field_type in ("integer", "number", "boolean", "array"):
            typed_fields.append((field_name, field_type))
        if unwrapped_fs.get("enum"):
            enum_fields.append((field_name, unwrapped_fs["enum"]))
        if unwrapped_fs.get("pattern"):
            pattern_fields.append((field_name, unwrapped_fs["pattern"]))
        if field_type in ("integer", "number"):
            self._add_numeric_field_if_constrained(
                field_name, unwrapped_fs, numeric_fields
            )

    def _add_numeric_field_if_constrained(
        self,
        field_name: str,
        schema: dict,
        numeric_fields: List[Tuple[str, Optional[float], Optional[float]]],
    ) -> None:
        """Add numeric field to list if it has constraints."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_min = schema.get("exclusiveMinimum")
        exclusive_max = schema.get("exclusiveMaximum")

        if any(v is not None for v in [minimum, maximum, exclusive_min, exclusive_max]):
            effective_min = exclusive_min if exclusive_min is not None else minimum
            effective_max = exclusive_max if exclusive_max is not None else maximum
            numeric_fields.append((field_name, effective_min, effective_max))

    def _generate_field_type_scenarios(
        self,
        required_fields: List[str],
        typed_fields: List[Tuple[str, str]],
        enum_fields: List[Tuple[str, List[Any]]],
        pattern_fields: List[Tuple[str, str]],
        numeric_fields: List[Tuple[str, Optional[float], Optional[float]]],
    ) -> List[str]:
        """Generate negative test scenarios based on field categories."""
        scenarios: List[str] = []
        if required_fields:
            scenarios.append(
                f"MISSING_REQUIRED: Remove one of these required fields: {required_fields}"
            )
        if typed_fields:
            scenarios.append(self._build_wrong_type_scenario(typed_fields))
        scenarios.extend(self._build_enum_scenarios(enum_fields))
        scenarios.extend(self._build_pattern_scenarios(pattern_fields))
        scenarios.extend(self._build_boundary_scenarios(numeric_fields))
        return scenarios

    @staticmethod
    def _build_wrong_type_scenario(typed_fields: List[Tuple[str, str]]) -> str:
        """Build a WRONG_TYPE scenario string from typed fields."""
        type_mismatch = {
            "integer": "not_a_number",
            "number": "not_a_number",
            "boolean": "not_a_bool",
            "array": "not_an_array",
        }
        examples = [
            f'"{name}": "{type_mismatch.get(ftype, "wrong")}" (expects {ftype})'
            for name, ftype in typed_fields
        ]
        return f"WRONG_TYPE: Send wrong type: {examples}"

    @staticmethod
    def _build_enum_scenarios(enum_fields: List[Tuple[str, List[Any]]]) -> List[str]:
        """Build INVALID_ENUM scenarios."""
        return [
            f'INVALID_ENUM: Field "{name}" allows only {values}, send "INVALID_VALUE_XYZ"'
            for name, values in enum_fields
        ]

    @staticmethod
    def _build_pattern_scenarios(pattern_fields: List[Tuple[str, str]]) -> List[str]:
        """Build INVALID_PATTERN scenarios."""
        return [
            f'INVALID_PATTERN: Field "{name}" must match pattern {pattern}, send "!!!invalid!!!"'
            for name, pattern in pattern_fields
        ]

    @staticmethod
    def _build_boundary_scenarios(
        numeric_fields: List[Tuple[str, Optional[float], Optional[float]]],
    ) -> List[str]:
        """Build BOUNDARY scenarios for numeric fields with min/max constraints."""
        scenarios = []
        for name, min_val, max_val in numeric_fields:
            if min_val is not None and isinstance(min_val, (int, float)):
                scenarios.append(
                    f'BOUNDARY: Field "{name}" has min={min_val}, send {min_val - 1}'
                )
            if max_val is not None and isinstance(max_val, (int, float)):
                scenarios.append(
                    f'BOUNDARY: Field "{name}" has max={max_val}, send {max_val + 1}'
                )
        return scenarios

    def _generate_fallback_query_scenarios(
        self, query_params: List[Tuple[str, str]]
    ) -> List[str]:
        """Generate fallback scenarios for query params when no body scenarios exist."""
        scenarios = []
        for name, ptype in query_params:
            if ptype.lower() in ("integer", "number") or ptype.lower().startswith(
                "array[int"
            ):
                scenarios.append(
                    f'INVALID_QUERY: Send "{name}=not_a_number" (expects integer)'
                )
        return scenarios

    def _precompute_negative_scenarios(self, endpoint: "Endpoint") -> str:
        """
        Pre-compute which negative test scenarios are valid for this endpoint.

        Determines testable scenarios based on endpoint schema.

        Returns:
            Formatted string listing testable scenarios with details.
        """
        scenarios: List[str] = []

        # Detect parameter-test endpoints (these just echo inputs, no validation)
        endpoint_path = getattr(endpoint, "path", "")
        is_parameter_test_endpoint = "/parameters/" in endpoint_path

        # Extract parameters
        path_params, query_params = self._extract_endpoint_params(endpoint)

        # Non-existent ID tests (only for resource endpoints)
        if path_params and not is_parameter_test_endpoint:
            scenarios.extend(self._generate_nonexistent_id_scenarios(path_params))

        # Extract body field categories and generate scenarios
        (
            required_fields,
            typed_fields,
            enum_fields,
            pattern_fields,
            numeric_fields,
        ) = self._extract_body_field_categories(endpoint)

        scenarios.extend(
            self._generate_field_type_scenarios(
                required_fields,
                typed_fields,
                enum_fields,
                pattern_fields,
                numeric_fields,
            )
        )

        # Fallback: query param tests (only if no body scenarios)
        if not scenarios and not is_parameter_test_endpoint and query_params:
            scenarios.extend(self._generate_fallback_query_scenarios(query_params))

        if not scenarios:
            return ""

        lines = ["TESTABLE NEGATIVE SCENARIOS (implement ONLY these):"]
        lines.extend(f"  - {s}" for s in scenarios)
        return "\n".join(lines)

    def _resolve_variant_properties(
        self, variant: dict
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Resolve properties from a variant, including allOf."""
        v_props = dict(variant.get("properties", {}))
        v_required = list(variant.get("required", []))
        v_all_of = variant.get("allOf")
        if v_all_of and isinstance(v_all_of, list):
            for sub in v_all_of:
                if isinstance(sub, dict) and sub.get("properties"):
                    v_props.update(sub["properties"])
                    v_required.extend(sub.get("required", []))
        return v_props, v_required

    def _format_variant_fields(
        self, v_props: Dict[str, Any], v_required: List[str], v_title: str
    ) -> List[str]:
        """Format field instructions for a single variant."""
        lines = [f'  Variant "{v_title}" (required: {list(set(v_required))}):']
        for vp_name, vp_schema in v_props.items():
            if not isinstance(vp_schema, dict):
                continue
            unwrapped_vp, _ = unwrap_nullable_schema(vp_schema)
            instruction = self._get_type_instruction(unwrapped_vp, field_name=vp_name)
            lines.append(f'    "{vp_name}": {instruction}')
        lines.append("")
        return lines

    def _format_union_field_instructions(
        self, one_of: List[dict], discriminator: dict
    ) -> str:
        """Format field instructions for discriminated union schemas."""
        lines = ["FIELD GENERATION INSTRUCTIONS (DISCRIMINATED UNION):"]
        disc_prop = discriminator.get("propertyName", "")
        if disc_prop:
            lines.append(f'Discriminator field: "{disc_prop}"')
        lines.append("Pick ONE variant and include ALL its required fields:")
        lines.append("")

        for i, variant in enumerate(one_of[:4]):
            if not isinstance(variant, dict):
                continue  # type: ignore[unreachable]
            v_props, v_required = self._resolve_variant_properties(variant)
            v_title = variant.get("title", f"Variant {i+1}")
            if v_props:
                lines.extend(self._format_variant_fields(v_props, v_required, v_title))

        return "\n".join(lines)

    def _format_field_instruction_line(
        self, field_name: str, field_schema: dict, required_list: List[str]
    ) -> Optional[str]:
        """Format a single field instruction line."""
        if not isinstance(field_schema, dict):
            return None  # type: ignore[unreachable]
        unwrapped, _ = unwrap_nullable_schema(field_schema)
        field_type = unwrapped.get("type", "string")
        field_format = unwrapped.get("format", "")
        required_marker = " [REQUIRED]" if field_name in required_list else ""
        instruction = self._get_type_instruction(unwrapped, field_name=field_name)
        format_part = f", format={field_format}" if field_format else ""
        return f'  "{field_name}": {instruction}  # type={field_type}{format_part}{required_marker}'

    def _precompute_positive_fields(self, endpoint: "Endpoint") -> str:
        """Pre-compute field generation instructions for positive tests."""
        if not hasattr(endpoint, "request_body") or not endpoint.request_body:
            return ""

        schema = getattr(endpoint.request_body, "schema", {})
        if not schema or not isinstance(schema, dict):
            return ""

        properties, required_list = extract_all_properties(schema)
        one_of = schema.get("oneOf") or schema.get("anyOf")
        discriminator = schema.get("discriminator", {})

        if one_of and isinstance(one_of, list) and not schema.get("properties"):
            return self._format_union_field_instructions(one_of, discriminator)

        if not properties:
            return ""

        lines = [
            "FIELD GENERATION INSTRUCTIONS (use these EXACTLY):",
            f"Required fields: {required_list if required_list else 'none'}",
            "",
        ]

        for field_name, field_schema in properties.items():
            line = self._format_field_instruction_line(
                field_name, field_schema, required_list
            )
            if line:
                lines.append(line)

        return "\n".join(lines)

    def _precompute_object_instruction(
        self, schema: dict, _ancestors: Optional[frozenset] = None
    ) -> str:
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
            unwrapped, _ = unwrap_nullable_schema(prop_schema)

            # Delegate to shared type→instruction mapper with ancestry tracking and field name
            val = self._get_type_instruction(
                unwrapped, _object_ancestors=new_ancestors, field_name=prop_name
            )
            parts.append(f'"{prop_name}": {val}')

        return "{" + ", ".join(parts) + "}"

    def _format_endpoints_list(self, endpoints: List[Any]) -> str:
        """Format list of endpoints (for auth endpoints)"""
        if not endpoints:
            return ""

        lines = []
        for ep in endpoints:
            summary = getattr(ep, "summary", "") or _DEFAULT_SUMMARY
            lines.append(f"- {ep.method.upper()} {ep.path} - {summary}")

        return "\n".join(lines)

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """Sanitize string to be a valid Python identifier."""
        from devdox_ai_locust.utils.constants import sanitize_identifier

        return sanitize_identifier(name)

    def _operation_to_class_name(self, endpoint: "Endpoint") -> str:
        """Convert operation_id to valid Python class name"""
        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)
        # Sanitize and convert to PascalCase
        sanitized = self._sanitize_identifier(operation_id)
        words = sanitized.replace("_", " ").split()
        return "".join(word.capitalize() for word in words) or "Unnamed"

    def _generate_operation_id(self, endpoint: "Endpoint") -> str:
        """Generate operation_id from method and path if not present"""
        path_parts = (
            endpoint.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
        )
        raw_id = f"{endpoint.method.lower()}_{path_parts}"
        return self._sanitize_identifier(raw_id)

    def get_endpoint_dir_name(self, endpoint: "Endpoint") -> str:
        """Get directory name for an endpoint"""
        operation_id = getattr(
            endpoint, "operation_id", ""
        ) or self._generate_operation_id(endpoint)
        # Sanitize for filesystem
        return self._sanitize_identifier(operation_id).lower()

    async def _call_ai_service(
        self, prompt: str, scenario_type: str = "unknown"
    ) -> str:
        """Call AI service with retry logic. Raises AIServiceError after all retries fail."""
        messages = self._build_ai_messages(prompt)

        last_error: Optional[Exception] = None
        for attempt in range(3):
            result = await self._try_ai_call(messages, scenario_type, attempt)
            if isinstance(result, str):
                return result
            last_error = result

            if attempt < 2:
                await asyncio.sleep(2**attempt)

        raise AIServiceError(
            f"AI service failed after 3 attempts for {scenario_type}"
        ) from last_error

    @staticmethod
    def _build_ai_messages(prompt: str) -> List[Dict[str, str]]:
        """Build the message payload for the AI service."""
        return [
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

    async def _try_ai_call(
        self,
        messages: List[Dict[str, str]],
        scenario_type: str,
        attempt: int,
    ) -> "str | Exception":
        """Attempt a single AI call. Returns content string on success, Exception on failure."""
        try:
            async with self._api_semaphore:
                response = await asyncio.wait_for(
                    self.ai_client.chat.completions.create(  # type: ignore[attr-defined]
                        model=self.ai_config.model,
                        messages=messages,
                        max_tokens=self.ai_config.max_tokens,
                        temperature=self.ai_config.temperature,
                    ),
                    timeout=self.ai_config.timeout,
                )

                if hasattr(response, "headers"):
                    self.update_rate_limit(dict(response.headers))

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    return str(content).strip() if content else ""

            return Exception("Empty AI response")
        except asyncio.TimeoutError as e:
            logger.debug(f"AI timeout on attempt {attempt + 1} for {scenario_type}")
            return e
        except Exception as e:
            logger.debug(f"AI error on attempt {attempt + 1} for {scenario_type}: {e}")
            return e

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
            logger.warning(
                f"Failed to render fix template: {e}. Falling back to inline prompt."
            )
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
            logger.warning(
                f"Failed to render semantic fix template: {e}. Falling back to inline prompt."
            )
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
        "random",
        "logging",
        "datetime",
        "time",
        "json",
        "re",
        "uuid",
        "string",
        # Locust
        "locust",
    }

    def _extract_allowed_imports_from_templates(self) -> set:
        """Extract allowed imports dynamically from prompt templates."""
        allowed = set(self._BASE_ALLOWED_IMPORTS)

        template_files = [
            "workflow_positive.j2",
            "workflow_negative.j2",
            "workflow_security.j2",
        ]

        for template_file in template_files:
            self._extract_imports_from_template(template_file, allowed)

        logger.debug(f"Extracted allowed imports from templates: {allowed}")
        return allowed

    def _extract_imports_from_template(
        self,
        template_file: str,
        allowed: set,
    ) -> None:
        """Extract imports from a single template file into the allowed set."""
        try:
            template_path = self.prompt_dir / template_file
            if not template_path.exists():
                return

            content = template_path.read_text(encoding="utf-8")
            from devdox_ai_locust.utils.constants import ALLOWED_IMPORTS_RE

            match = ALLOWED_IMPORTS_RE.search(content)
            if not match:
                return

            tree = self._parse_imports_code(match.group(1))
            if tree:
                self._collect_import_names(tree, allowed)

        except Exception as e:
            logger.debug(f"Could not parse imports from {template_file}: {e}")

    def _parse_imports_code(self, imports_code: str) -> Any:
        """Parse Python import code, cleaning Jinja syntax if needed."""
        import ast

        try:
            return ast.parse(imports_code)
        except SyntaxError:
            from devdox_ai_locust.utils.constants import JINJA_BLOCK_RE, JINJA_VAR_RE

            cleaned = JINJA_BLOCK_RE.sub("", imports_code)
            cleaned = JINJA_VAR_RE.sub("", cleaned)
            try:
                return ast.parse(cleaned)
            except SyntaxError:
                return None

    @staticmethod
    def _collect_import_names(tree: Any, allowed: set) -> None:
        """Collect module names from an AST into the allowed set."""
        import ast

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    allowed.add(alias.name)
                    allowed.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                allowed.add(node.module)
                allowed.add(node.module.split(".")[0])

    def _validate_template_context(
        self, context: Dict[str, Any], scenario_type: ScenarioType
    ) -> None:
        """Validate that required template context variables are present.

        Logs warnings for missing/empty critical variables that could cause
        LLM hallucination. Does not raise exceptions - just logs issues.
        """
        # Required variables for all scenarios (must not be None or empty)
        required_vars = [
            "endpoint",
            "base_workflow",
            "test_data_content",
            "class_name",
            "method",
            "path",
        ]

        for var in required_vars:
            value = context.get(var)
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.warning(
                    f"Template context missing required variable '{var}' for {scenario_type.value} scenario"
                )

        # Check endpoint_expected_status - should be a list, not None
        expected_status = context.get("endpoint_expected_status")
        if not expected_status or not isinstance(expected_status, list):
            logger.warning(
                f"Template context has invalid endpoint_expected_status: {expected_status}"
            )
