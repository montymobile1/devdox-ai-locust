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
    def __init__(self, scenario_type: str, error: str, code: str):
        self.scenario_type = scenario_type
        self.error = error
        self.code = code
        super().__init__(f"Generated {scenario_type} code failed validation: {error}")


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
        Estimate generation time based on rate limits.

        Args:
            num_endpoints: Number of endpoints to process

        Returns:
            TimeEstimate with calculated values
        """
        # 5 LLM calls per endpoint (positive + negative + edge + state + security)
        total_calls = num_endpoints * self.num_scenarios

        rpm = self._rate_limit_info.requests_per_minute if self._rate_limit_info else 60
        estimated_minutes = total_calls / rpm
        estimated_seconds = estimated_minutes * 60

        return TimeEstimate(
            total_calls=total_calls,
            rpm=rpm,
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
                raise AIServiceError(f"AI service returned empty response for {scenario_type.value}")

            # Fix class name to match expected naming convention
            # LLMs sometimes ignore the template and generate their own class names
            content = self._fix_class_name(content, class_name, scenario_type.value)

            is_valid, error = self._validate_python_code(content)
            if is_valid:
                return content

            # Save for error reporting
            last_error = error
            last_code = content

            if attempt < max_validation_retries - 1:
                logger.warning(f"Validation failed for {scenario_type.value}, retrying: {error}")
                await asyncio.sleep(1)

        # All retries exhausted
        raise CodeValidationError(scenario_type.value, last_error, last_code)

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

        # Parameters
        if hasattr(endpoint, "parameters") and endpoint.parameters:
            lines.append("\nParameters:")
            for param in endpoint.parameters:
                param_name = getattr(param, "name", "unknown")
                param_in = getattr(param, "in_", "query")
                param_required = getattr(param, "required", False)
                param_type = "string"
                if hasattr(param, "schema") and param.schema:
                    param_type = getattr(param.schema, "type", "string")
                required_str = "(required)" if param_required else "(optional)"
                lines.append(f"  - {param_name} [{param_in}]: {param_type} {required_str}")

        # Request body
        if hasattr(endpoint, "request_body") and endpoint.request_body:
            lines.append("\nRequest Body:")
            if hasattr(endpoint.request_body, "content"):
                content = endpoint.request_body.content
                # Handle both dict and other formats
                if isinstance(content, dict):
                    for content_type, media_type in content.items():
                        lines.append(f"  Content-Type: {content_type}")
                        if hasattr(media_type, "schema") and media_type.schema:
                            schema = media_type.schema
                            if hasattr(schema, "properties") and schema.properties:
                                props = schema.properties
                                if isinstance(props, dict):
                                    lines.append("  Properties:")
                                    for prop_name, prop_schema in props.items():
                                        prop_type = getattr(prop_schema, "type", "any")
                                        lines.append(f"    - {prop_name}: {prop_type}")
                else:
                    lines.append(f"  Content: {type(content).__name__}")

        # Responses
        if hasattr(endpoint, "responses") and endpoint.responses:
            responses = endpoint.responses
            if isinstance(responses, dict):
                lines.append("\nResponses:")
                for status_code, response in responses.items():
                    desc = getattr(response, "description", "") if hasattr(response, "description") else str(response)
                    lines.append(f"  - {status_code}: {desc[:50] if len(str(desc)) > 50 else desc}")

        return "\n".join(lines)

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
        """Extract code from <code> tags with robust fallback handling"""
        import re

        # Try case-insensitive match with closing tag
        pattern = r"<code>(.*?)</code>"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            code = max(matches, key=len).strip()
        else:
            # Fallback: handle <code> without closing tag
            code_start = re.search(r"<code>", response, re.IGNORECASE)
            if code_start:
                code = response[code_start.end():]
            else:
                code = response

            # Clean up markdown code blocks
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]

            # Final cleanup - strip any remaining tags
            code = re.sub(r"^</?code>", "", code, flags=re.IGNORECASE)
            code = re.sub(r"</?code>$", "", code, flags=re.IGNORECASE)

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

    def _validate_python_code(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax and return error details if invalid"""
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - {e.text.strip() if e.text else ''}"
