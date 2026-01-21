"""
Scenario-based Workflow Generator

Generates separate workflow files for different test scenario types per endpoint:
- Positive scenarios (happy path)
- Negative scenarios (validation errors + error handling)
- Edge scenarios (boundary conditions)
- State scenarios (state-dependent tests)
- Security scenarios (injection, auth bypass)

Uses 5 LLM calls per endpoint for focused, non-truncated output.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of test scenarios (all LLM-generated)"""
    POSITIVE = "positive"      # Happy path tests
    NEGATIVE = "negative"      # Validation errors + error handling
    EDGE = "edge"              # Edge cases + boundary conditions
    STATE = "state"            # State-dependent tests
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

    Uses 5 LLM calls per endpoint for focused, non-truncated output:
    - Call 1: Positive scenarios (happy path)
    - Call 2: Negative scenarios (validation + error handling)
    - Call 3: Edge scenarios (boundary conditions)
    - Call 4: State scenarios (state-dependent)
    - Call 5: Security scenarios (injection, auth bypass)
    """

    # Mapping of scenario types to output filenames
    SCENARIO_FILES = {
        ScenarioType.POSITIVE: "positive_workflow.py",
        ScenarioType.NEGATIVE: "negative_workflow.py",
        ScenarioType.EDGE: "edge_workflow.py",
        ScenarioType.STATE: "state_workflow.py",
        ScenarioType.SECURITY: "security_workflow.py",
    }

    # Prompt templates for LLM-based scenarios
    PROMPT_TEMPLATES = {
        ScenarioType.POSITIVE: "workflow_positive.j2",
        ScenarioType.NEGATIVE: "workflow_negative.j2",
        ScenarioType.EDGE: "workflow_edge.j2",
        ScenarioType.STATE: "workflow_state.j2",
        ScenarioType.SECURITY: "workflow_security.j2",
    }

    def __init__(
        self,
        prompt_dir: Path,
        ai_client: Any,
        ai_config: Any,
    ):
        """
        Initialize the scenario generator.

        Args:
            prompt_dir: Directory containing LLM prompt templates
            ai_client: Together AI client for LLM calls
            ai_config: AI configuration (model, timeout, etc.)
        """
        self.prompt_dir = prompt_dir
        self.ai_client = ai_client
        self.ai_config = ai_config
        self._rate_limit_info: Optional[RateLimitInfo] = None
        self._api_semaphore = asyncio.Semaphore(5)

        # Setup Jinja environment for prompts
        self.prompt_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

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
        Update rate limit info from API response headers.

        Args:
            headers: Response headers from API call

        Returns:
            Updated RateLimitInfo
        """
        self._rate_limit_info = RateLimitInfo.from_headers(headers)
        return self._rate_limit_info

    def get_rate_limit_info(self) -> RateLimitInfo:
        """Get current rate limit info or default"""
        return self._rate_limit_info or RateLimitInfo.default()

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
        """
        results = {}

        # Generate all 5 scenarios in parallel using LLM
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

        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)

        for scenario_type, result in zip(scenario_types, llm_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate {scenario_type.value} scenario: {result}")
                results[scenario_type] = ""
            else:
                results[scenario_type] = result

        return results

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

        try:
            template = self.prompt_env.get_template(template_name)
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return ""

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

        # Call LLM
        content = await self._call_ai_service(prompt)

        # Validate Python syntax
        if content:
            is_valid, error = self._validate_python_code(content)
            if is_valid:
                return content
            logger.warning(
                f"Generated {scenario_type.value} code failed validation: {error}"
            )
            # Log first few lines to help debug
            lines = content.split('\n')[:10]
            logger.debug(f"First 10 lines of failed code:\n" + '\n'.join(lines))
        else:
            logger.warning(f"Generated {scenario_type.value} code was empty")

        return ""

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
                for content_type, media_type in endpoint.request_body.content.items():
                    lines.append(f"  Content-Type: {content_type}")
                    if hasattr(media_type, "schema") and media_type.schema:
                        schema = media_type.schema
                        if hasattr(schema, "properties") and schema.properties:
                            lines.append("  Properties:")
                            for prop_name, prop_schema in schema.properties.items():
                                prop_type = getattr(prop_schema, "type", "any")
                                lines.append(f"    - {prop_name}: {prop_type}")

        # Responses
        if hasattr(endpoint, "responses") and endpoint.responses:
            lines.append("\nResponses:")
            for status_code, response in endpoint.responses.items():
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

    def _operation_to_class_name(self, endpoint: Any) -> str:
        """Convert operation_id to valid Python class name"""
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        # Remove special characters and convert to PascalCase
        words = operation_id.replace("-", " ").replace("_", " ").split()
        return "".join(word.capitalize() for word in words)

    def _generate_operation_id(self, endpoint: Any) -> str:
        """Generate operation_id from method and path if not present"""
        path_parts = endpoint.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
        return f"{endpoint.method.lower()}_{path_parts}"

    def get_endpoint_dir_name(self, endpoint: Any) -> str:
        """Get directory name for an endpoint"""
        operation_id = getattr(endpoint, "operation_id", "") or self._generate_operation_id(endpoint)
        # Sanitize for filesystem
        return operation_id.lower().replace(" ", "_").replace("-", "_")

    async def _call_ai_service(self, prompt: str) -> str:
        """Call AI service with retry logic"""
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

            except asyncio.TimeoutError:
                logger.warning(f"AI timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"AI error on attempt {attempt + 1}: {e}")

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

        return ""

    def _extract_code(self, response: str) -> str:
        """Extract code from <code> tags with robust fallback handling"""
        import re

        # Try case-insensitive match with closing tag
        pattern = r"<code>(.*?)</code>"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return max(matches, key=len).strip()

        # Fallback: handle <code> without closing tag
        code_start = re.search(r"<code>", response, re.IGNORECASE)
        if code_start:
            response = response[code_start.end():]

        # Clean up markdown code blocks
        if response.startswith("```python"):
            response = response[9:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        # Final cleanup - strip any remaining tags
        response = re.sub(r"^</?code>", "", response, flags=re.IGNORECASE)
        response = re.sub(r"</?code>$", "", response, flags=re.IGNORECASE)

        return response.strip()

    def _validate_python_code(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax and return error details if invalid"""
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - {e.text.strip() if e.text else ''}"
