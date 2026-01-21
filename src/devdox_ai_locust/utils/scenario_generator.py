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
    ) -> Dict[ScenarioType, str]:
        """
        Generate all scenario workflow files for a single endpoint using LLM.

        Args:
            endpoint: Single endpoint to generate tests for
            base_workflow_content: Content of base_workflow.py
            test_data_content: Content of test_data.py
            auth_endpoints: Authentication endpoints (optional)

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
    ) -> str:
        """
        Generate a scenario using LLM for a single endpoint.

        Args:
            scenario_type: Type of scenario to generate
            endpoint: Single endpoint to generate tests for
            base_workflow_content: Base workflow code
            test_data_content: Test data code
            auth_endpoints: Auth endpoints

        Returns:
            Generated Python code
        """
        template_name = self.PROMPT_TEMPLATES.get(scenario_type)
        if not template_name:
            raise ValueError(f"No prompt template for scenario type: {scenario_type}")

        # Let template loading errors propagate naturally
        template = self.prompt_env.get_template(template_name)

        # Format single endpoint with full details
        endpoint_details = self._format_single_endpoint(endpoint)

        # Build class name from operation_id
        class_name = self._operation_to_class_name(endpoint)

        # Render prompt
        prompt = template.render(
            endpoint=endpoint_details,
            auth_endpoints=self._format_endpoints_list(auth_endpoints) if auth_endpoints else "",
            base_workflow=base_workflow_content,
            test_data_content=test_data_content,
            class_name=class_name,
            operation_id=getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint),
            method=endpoint.method,
            path=endpoint.path,
        )

        # Call LLM with validation retry
        max_validation_retries = 2
        last_error = None
        last_code = None

        for attempt in range(max_validation_retries):
            content = await self._call_ai_service(prompt, scenario_type.value)

            if not content:
                raise AIServiceError(f"AI service returned empty response for {scenario_type.value} [{endpoint.method} {endpoint.path}]")

            # Detect if API returned HTML error page instead of code
            if content.strip().startswith('<') and '<html' in content.lower():
                raise AIServiceError(
                    f"API returned HTML error page instead of code for {scenario_type.value} "
                    f"[{endpoint.method} {endpoint.path}]. "
                    f"This may indicate an API error or rate limiting. First 200 chars: {content[:200]}"
                )

            # Fix class name to match expected naming convention
            # LLMs sometimes ignore the template and generate their own class names
            content = self._fix_class_name(content, class_name, scenario_type.value)

            # Fix bytes literals with unicode (b'tëst' → 'tëst'.encode('utf-8'))
            content = self._fix_bytes_literals(content)

            is_valid, error = self._validate_python_code(content)
            if is_valid:
                return content

            # Save for error reporting
            last_error = error
            last_code = content

            if attempt < max_validation_retries - 1:
                logger.warning(
                    f"Validation failed for {scenario_type.value} [{endpoint.method} {endpoint.path}], "
                    f"attempt {attempt + 1}/{max_validation_retries}: {error}"
                )
                await asyncio.sleep(1)

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
        """
        lines = []
        prefix = "  " * indent

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

    def _validate_python_code(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax and return error details if invalid"""
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - {e.text.strip() if e.text else ''}"
