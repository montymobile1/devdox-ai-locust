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
from typing import Dict, List, Any, Optional, Tuple
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


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

    # Default and max concurrency limits
    DEFAULT_CONCURRENCY = 10
    MAX_CONCURRENCY = 50

    def __init__(
        self,
        prompt_dir: Path,
        ai_client: Any,
        ai_config: Any,
        max_concurrency: int = MAX_CONCURRENCY,
    ):
        """
        Initialize the scenario generator.

        Args:
            prompt_dir: Directory containing LLM prompt templates
            ai_client: Together AI client for LLM calls
            ai_config: AI configuration (model, timeout, etc.)
            max_concurrency: Maximum concurrent API calls
        """
        self.prompt_dir = prompt_dir
        self.ai_client = ai_client
        self.ai_config = ai_config
        self._rate_limit_info: Optional[RateLimitInfo] = None
        self._max_concurrency = max_concurrency
        self._current_concurrency = self.DEFAULT_CONCURRENCY
        self._api_semaphore = asyncio.Semaphore(self._current_concurrency)

        # Setup Jinja environment for prompts
        self.prompt_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _update_concurrency(self, rpm: int) -> None:
        """
        Update concurrency based on rate limit.

        Args:
            rpm: Requests per minute from rate limit headers
        """
        # Target: stay at ~80% of rate limit to avoid hitting it
        # Divide by 60 to get per-second, multiply by avg response time (~3s)
        optimal = min(int(rpm * 0.8 / 20), self._max_concurrency)  # ~3 req/s sustained
        new_concurrency = max(self.DEFAULT_CONCURRENCY, optimal)

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
            )
            for scenario_type in scenario_types
        ]

        # Let exceptions propagate naturally - no return_exceptions=True
        llm_results = await asyncio.gather(*llm_tasks)

        return dict(zip(scenario_types, llm_results))

    async def _generate_llm_scenario(
        self,
        scenario_type: ScenarioType,
        endpoint: Any,
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
        tag_name: str = "default",
        all_endpoints: Optional[List[Any]] = None,
    ) -> str:
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

        Returns:
            Generated Python code
        """
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        scenario_name = scenario_type.value


        template_name = self.PROMPT_TEMPLATES.get(scenario_type)
        if not template_name:
            raise ValueError(f"No prompt template for scenario type: {scenario_type}")

        # Let template loading errors propagate naturally
        template = self.prompt_env.get_template(template_name)

        # Format single endpoint with full details
        endpoint_details = self._format_single_endpoint(endpoint)

        # Build class name from operation_id
        class_name = self._operation_to_class_name(endpoint)

        # Extract expected status codes from OpenAPI spec responses
        expected_status_codes = self._extract_expected_status_codes(endpoint)

        # Find related CREATE endpoints for setup steps (only for non-POST endpoints)
        setup_endpoints_section = ""
        if all_endpoints and endpoint.method.upper() != "POST":
            related_create_endpoints = self._find_related_create_endpoints(endpoint, all_endpoints)
            setup_endpoints_section = self._format_related_create_endpoints(related_create_endpoints)

        # Render prompt
        prompt = template.render(
            endpoint=endpoint_details,
            auth_endpoints=self._format_endpoints_list(auth_endpoints) if auth_endpoints else "",
            base_workflow=base_workflow_content,
            test_data_content=test_data_content,
            class_name=class_name,
            operation_id=operation_id,
            method=endpoint.method,
            path=endpoint.path,
            endpoint_expected_status=expected_status_codes,
            setup_endpoints=setup_endpoints_section,
        )

        # Call LLM with validation retry (error-aware on retry)
        max_validation_retries = 2
        last_error = None
        last_code = None
        current_prompt = prompt  # Start with the original prompt

        for attempt in range(max_validation_retries):
            # On retry, use error-aware fix prompt instead of original
            if attempt > 0 and last_error and last_code:
                current_prompt = self._render_fix_prompt(last_code, last_error)

            content = await self._call_ai_service(current_prompt, scenario_type.value)

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

            # Sanitize any non-ASCII Unicode characters the LLM may have injected
            sanitized = self._sanitize_unicode(extracted)

            # Fix class name to match expected naming convention
            # LLMs sometimes ignore the template and generate their own class names
            after_class_fix = self._fix_class_name(sanitized, class_name, scenario_type.value)

            # Fix bytes literals with unicode (b'tëst' → 'tëst'.encode('utf-8'))
            after_bytes_fix = self._fix_bytes_literals(after_class_fix)

            # Fix regex strings (convert to raw strings to avoid SyntaxWarnings)
            after_regex_fix = self._fix_regex_strings(after_bytes_fix)

            content = after_regex_fix

            is_valid, error = self._validate_python_code(content)

            if is_valid:
                if attempt > 0:
                    console.print(
                        f"[green]✓ Retry SUCCEEDED[/green] for {scenario_type.value} "
                        f"[{endpoint.method} {endpoint.path}] on attempt {attempt + 1}/{max_validation_retries}"
                    )
                return content

            # Save for error reporting and fix prompt on retry
            last_error = error
            last_code = content

            if attempt < max_validation_retries - 1:
                console.print(
                    f"[yellow]⚠ Validation failed[/yellow] for {scenario_type.value} "
                    f"[{endpoint.method} {endpoint.path}], attempt {attempt + 1}/{max_validation_retries}: "
                    f"{error}. Retrying with fix prompt..."
                )
                await asyncio.sleep(1)
            else:
                console.print(
                    f"[red]✗ Retry FAILED[/red] for {scenario_type.value} "
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

    def _format_single_endpoint(self, endpoint: Any) -> str:
        """Format a single endpoint with full details for the prompt"""
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

                required_str = "(required)" if param_required else "(optional)"
                type_str = param_type
                if param_format:
                    type_str = f"{param_type} [{param_format}]"

                lines.append(f"  - {param_name} [{param_in}]: {type_str} {required_str}")
                if param_enum:
                    lines.append(f"      enum: {param_enum}")
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
                req_marker = " (REQUIRED)" if is_required else " (optional)"
                prop_type = prop_schema.get("type", "any")
                prop_format = prop_schema.get("format")

                # Build type string with format
                type_str = prop_type
                if prop_format:
                    type_str = f"{prop_type} [{prop_format}]"

                lines.append(f"{prefix}    - {prop_name}: {type_str}{req_marker}")

                # Add description
                if prop_schema.get("description"):
                    desc = prop_schema["description"][:80]
                    lines.append(f"{prefix}        description: {desc}")

                # Add constraints
                constraints = []
                if prop_schema.get("minLength") is not None:
                    constraints.append(f"minLength={prop_schema['minLength']}")
                if prop_schema.get("maxLength") is not None:
                    constraints.append(f"maxLength={prop_schema['maxLength']}")
                if prop_schema.get("pattern"):
                    pattern = prop_schema["pattern"][:40]
                    constraints.append(f"pattern='{pattern}'")
                if prop_schema.get("minimum") is not None:
                    constraints.append(f"min={prop_schema['minimum']}")
                if prop_schema.get("maximum") is not None:
                    constraints.append(f"max={prop_schema['maximum']}")
                if prop_schema.get("exclusiveMinimum") is not None:
                    constraints.append(f"exclusiveMin={prop_schema['exclusiveMinimum']}")
                if prop_schema.get("exclusiveMaximum") is not None:
                    constraints.append(f"exclusiveMax={prop_schema['exclusiveMaximum']}")
                if prop_schema.get("multipleOf") is not None:
                    constraints.append(f"multipleOf={prop_schema['multipleOf']}")

                if constraints:
                    lines.append(f"{prefix}        constraints: {', '.join(constraints)}")

                # Add enum values
                if prop_schema.get("enum"):
                    enum_vals = str(prop_schema["enum"])[:60]
                    lines.append(f"{prefix}        allowed values: {enum_vals}")

                # Add default value
                if prop_schema.get("default") is not None:
                    lines.append(f"{prefix}        default: {prop_schema['default']}")

                # Handle nested objects
                if prop_type == "object" and prop_schema.get("properties"):
                    lines.append(f"{prefix}        nested object properties:")
                    nested_lines = self._format_schema(prop_schema, indent + 3)
                    lines.extend(nested_lines)

                # Handle arrays
                if prop_type == "array" and prop_schema.get("items"):
                    items = prop_schema["items"]
                    items_type = items.get("type", "any")
                    lines.append(f"{prefix}        array items type: {items_type}")
                    if items.get("properties"):
                        lines.append(f"{prefix}        array item properties:")
                        nested_lines = self._format_schema(items, indent + 3)
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
                    if prop_schema.get("const", "").lower() == ref_name.replace("_", ""):
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
                prop_type = prop_schema.get("type", "any")
                prop_format = prop_schema.get("format")

                # Build type string with format
                type_str = prop_type
                if prop_format:
                    type_str = f"{prop_type} [{prop_format}]"

                lines.append(f"{prefix}- {prop_name}: {type_str}")

                # Handle nested objects (show one level deep)
                if prop_type == "object" and prop_schema.get("properties"):
                    for nested_name, nested_schema in prop_schema["properties"].items():
                        nested_type = nested_schema.get("type", "any")
                        lines.append(f"{prefix}    - {nested_name}: {nested_type}")

                # Handle arrays
                if prop_type == "array" and prop_schema.get("items"):
                    items = prop_schema["items"]
                    items_type = items.get("type", "any")
                    lines.append(f"{prefix}    (array of {items_type})")
                    if items.get("properties"):
                        for item_name, item_schema in items["properties"].items():
                            item_type = item_schema.get("type", "any")
                            lines.append(f"{prefix}    - {item_name}: {item_type}")
        elif schema_type == "array":
            items = schema.get("items", {})
            items_type = items.get("type", "any")
            lines.append(f"{prefix}(array of {items_type})")
            if items.get("properties"):
                for item_name, item_schema in items["properties"].items():
                    item_type = item_schema.get("type", "any")
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
        Find CREATE (POST) endpoints related to the target endpoint using fuzzy matching.

        Uses tag similarity and path segment matching to find relevant POST endpoints
        that could be used as setup steps (e.g., creating items before testing GET/PUT/DELETE).

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
        target_path = target_endpoint.path.lower()
        target_tags = set(getattr(target_endpoint, "tags", []) or [])

        # Extract resource name from path (e.g., "/users/{id}" -> "users")
        target_path_segments = [s for s in target_path.strip("/").split("/") if not s.startswith("{")]

        for endpoint in all_endpoints:
            # Only consider POST endpoints
            if endpoint.method.upper() != "POST":
                continue

            # Don't return the target itself
            if endpoint.path == target_endpoint.path and endpoint.method == target_endpoint.method:
                continue

            score = 0.0
            reasons = []

            # Factor 1: Tag matching (high weight)
            endpoint_tags = set(getattr(endpoint, "tags", []) or [])
            common_tags = target_tags & endpoint_tags
            if common_tags:
                score += 50.0 * len(common_tags)  # 50 points per matching tag
                reasons.append(f"shares tag(s): {', '.join(common_tags)}")

            # Factor 2: Path segment matching
            endpoint_path = endpoint.path.lower()
            endpoint_segments = [s for s in endpoint_path.strip("/").split("/") if not s.startswith("{")]

            # Check for exact base path match (e.g., "/users" matches "/users/{id}")
            if endpoint_segments and target_path_segments:
                # First segment match is most important (resource type)
                if endpoint_segments[0] == target_path_segments[0]:
                    score += 30.0
                    reasons.append(f"same resource type: {endpoint_segments[0]}")

                # Additional matching segments add smaller bonus
                common_segments = set(endpoint_segments) & set(target_path_segments)
                if len(common_segments) > 1:
                    score += 5.0 * (len(common_segments) - 1)
                    reasons.append(f"path segments: {', '.join(common_segments)}")

            # Factor 3: Path contains target resource name
            # e.g., target is GET /orders/{id}/items, look for POST /items or POST /orders/{id}/items
            if target_path_segments:
                primary_resource = target_path_segments[-1] if target_path_segments else ""
                if primary_resource and primary_resource in endpoint_path:
                    score += 20.0
                    if f"same endpoint resource: {primary_resource}" not in reasons:
                        reasons.append(f"creates resource: {primary_resource}")

            # Factor 4: Boost for simpler paths (collection endpoints like POST /users)
            # These are more likely to be general create endpoints
            if len(endpoint_segments) == 1 and score > 0:
                score += 10.0
                reasons.append("collection endpoint")

            # Only include if there's some relevance
            if score > 0:
                reason_str = "; ".join(reasons)
                results.append((endpoint, score, reason_str))

        # Sort by score (highest first) and return
        results.sort(key=lambda x: x[1], reverse=True)
        return results

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
        lines.append("")

        # Edge case 3: Pass all ranked endpoints with guidance
        for i, (endpoint, score, reason) in enumerate(related_endpoints, 1):
            operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
            summary = getattr(endpoint, "summary", "") or "No summary"
            description = getattr(endpoint, "description", "") or ""

            lines.append(f"--- Rank #{i} (relevance: {score:.0f}) ---")
            lines.append(f"POST {endpoint.path}")
            lines.append(f"Operation ID: {operation_id}")
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

        return "\n".join(lines)

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
                        content = response.choices[0].message.content.strip()
                        return self._extract_code(content)

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"AI timeout on attempt {attempt + 1} for {scenario_type}")
            except Exception as e:
                last_error = e
                logger.warning(f"AI error on attempt {attempt + 1} for {scenario_type}: {e}")

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
                # Replace the class name in definition and any self-references
                code = re.sub(
                    rf'\bclass\s+{re.escape(actual_class_name)}\s*\(',
                    f'class {expected_full_name}(',
                    code
                )
                # Also replace docstrings or comments mentioning the old name
                code = re.sub(
                    rf'\b{re.escape(actual_class_name)}\b',
                    expected_full_name,
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
        """
        import re

        def fix_match(match):
            quote_char = match.group(1)  # ' or "
            content = match.group(2)
            # Check if content has non-ASCII
            try:
                content.encode('ascii')
                return match.group(0)  # Valid ASCII, keep as-is
            except UnicodeEncodeError:
                # Has non-ASCII, convert to .encode() form
                return f"{quote_char}{content}{quote_char}.encode('utf-8')"

        # Match b'...' or b"..." - non-greedy to handle multiple on same line
        pattern = r"b(['\"])([^'\"]*?)\1"
        return re.sub(pattern, fix_match, code)

    def _fix_regex_strings(self, code: str) -> str:
        """
        Convert strings with regex escape sequences to raw strings.

        LLMs sometimes generate "\\d" or "\\+" which triggers SyntaxWarnings
        in Python 3.12+. This converts them to raw strings: r"\\d", r"\\+".

        Converts: "\\d", "\\+", "\\s", etc. → r"\\d", r"\\+", r"\\s"
        """
        import re

        # Problematic escape sequences that trigger SyntaxWarnings
        problematic_escapes = [
            '\\d', '\\D', '\\w', '\\W', '\\s', '\\S',
            '\\+', '\\*', '\\?', '\\^', '\\$', '\\|',
            '\\(', '\\)', '\\[', '\\]', '\\{', '\\}',
            '\\.'  # Escaped dot in regex patterns like \.
        ]

        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            # Skip comments
            if line.strip().startswith('#'):
                fixed_lines.append(line)
                continue

            # Check if line contains problematic escape sequences in strings
            # and doesn't already use raw strings
            has_problematic = any(escape in line for escape in problematic_escapes)
            already_raw = 'r"' in line or "r'" in line

            if has_problematic and not already_raw:
                # Find and fix string literals containing regex escapes
                # Pattern matches quoted strings not preceded by 'r'
                line = re.sub(
                    r'(?<!r)(["\'])([^"\']*(?:\\[dDwWsS+*?^$.|\\()\[\]{}])[^"\']*)\1',
                    lambda m: f'r{m.group(1)}{m.group(2)}{m.group(1)}',
                    line
                )

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

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

    def _validate_python_code(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax and return error details if invalid"""
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - {e.text.strip() if e.text else ''}"
