"""
Scenario-based Workflow Generator

Generates separate workflow files for different test scenario types:
- Positive + State scenarios (LLM-generated)
- Negative + Edge + Error scenarios (LLM-generated)
- Security scenarios (LLM-generated)

Uses 3 LLM calls per API tag to generate focused, high-quality test code.
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
    POSITIVE = "positive"      # Happy path + state-dependent
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

    Uses 3 LLM calls per tag:
    - Call 1: Positive + State scenarios
    - Call 2: Negative + Edge + Error scenarios
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

    def estimate_time(self, num_tags: int) -> TimeEstimate:
        """
        Estimate generation time based on rate limits.

        Args:
            num_tags: Number of API tags to process

        Returns:
            TimeEstimate with calculated values
        """
        # 3 LLM calls per tag (positive + negative + security)
        total_calls = num_tags * 3

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

    async def generate_scenario_workflows(
        self,
        tag_name: str,
        endpoints: List[Any],
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
    ) -> Dict[ScenarioType, str]:
        """
        Generate all scenario workflow files for a tag using LLM.

        Args:
            tag_name: Name of the API tag
            endpoints: List of endpoints for this tag
            base_workflow_content: Content of base_workflow.py
            test_data_content: Content of test_data.py
            auth_endpoints: Authentication endpoints (optional)

        Returns:
            Dict mapping ScenarioType to generated code content
        """
        results = {}

        # Generate all scenarios in parallel using LLM
        llm_tasks = [
            self._generate_llm_scenario(
                ScenarioType.POSITIVE,
                tag_name,
                endpoints,
                base_workflow_content,
                test_data_content,
                auth_endpoints,
            ),
            self._generate_llm_scenario(
                ScenarioType.NEGATIVE,
                tag_name,
                endpoints,
                base_workflow_content,
                test_data_content,
                auth_endpoints,
            ),
            self._generate_llm_scenario(
                ScenarioType.SECURITY,
                tag_name,
                endpoints,
                base_workflow_content,
                test_data_content,
                auth_endpoints,
            ),
        ]

        scenario_types = [ScenarioType.POSITIVE, ScenarioType.NEGATIVE, ScenarioType.SECURITY]
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
        tag_name: str,
        endpoints: List[Any],
        base_workflow_content: str,
        test_data_content: str,
        auth_endpoints: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate a scenario using LLM.

        Args:
            scenario_type: Type of scenario to generate
            tag_name: Name of the API tag
            endpoints: List of endpoints
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

        # Format endpoints for prompt
        endpoints_str = self._format_endpoints(endpoints)

        # Build class name from tag
        class_name = self._tag_to_class_name(tag_name)

        # Render prompt
        prompt = template.render(
            grouped_endpoints=endpoints_str,
            auth_endpoints=self._format_endpoints(auth_endpoints) if auth_endpoints else "",
            base_workflow=base_workflow_content,
            test_data_content=test_data_content,
            class_name=class_name,
            tag_name=tag_name,
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

    def _format_endpoints(self, endpoints: List[Any]) -> str:
        """Format endpoints for prompt"""
        if not endpoints:
            return ""

        lines = []
        for ep in endpoints:
            params = f"({len(ep.parameters)} params)" if hasattr(ep, "parameters") and ep.parameters else ""
            body = "(with body)" if hasattr(ep, "request_body") and ep.request_body else ""
            summary = getattr(ep, "summary", "") or "No summary"
            lines.append(f"- {ep.method} {ep.path} {params} {body} - {summary}")

        return "\n".join(lines)

    def _tag_to_class_name(self, tag_name: str) -> str:
        """Convert tag name to valid Python class name"""
        # Remove special characters and convert to PascalCase
        words = tag_name.replace("-", " ").replace("_", " ").split()
        return "".join(word.capitalize() for word in words)

    async def _call_ai_service(self, prompt: str) -> str:
        """Call AI service with retry logic"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer specializing in Locust load testing. "
                    "Generate clean, production-ready code. "
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
        """Extract code from <code> tags"""
        import re
        pattern = r"<code>(.*?)</code>"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return max(matches, key=len).strip()

        # Fallback: clean up response
        if response.startswith("```python"):
            response = response[9:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        return response.strip()

    def _validate_python_code(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax and return error details if invalid"""
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - {e.text.strip() if e.text else ''}"
