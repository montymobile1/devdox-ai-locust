"""
Hybrid Locust Test Generator

Combines reliable template-based generation with LLM enhancement for creativity
and domain-specific optimizations.
"""

import re
import traceback
import sys
import os
import asyncio
import logging
import subprocess
import importlib.util as importlib_util
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass
import uuid
import shutil
import warnings
import signal
from contextlib import contextmanager


from devdox_ai_locust.utils.open_ai_parser import Endpoint
from devdox_ai_locust.utils.file_creation import FileCreationConfig, SafeFileCreator
from devdox_ai_locust.locust_generator import LocustTestGenerator, TestDataConfig
from together import AsyncTogether

logger = logging.getLogger(__name__)


test_data_file_path = "test_data.py"
data_provider_path = "data_provider.py"


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Module execution timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class ErrorClassification:
    """Classification of an error for retry logic"""

    is_retryable: bool
    backoff_seconds: float
    error_type: str


@dataclass
class AIEnhancementConfig:
    """Configuration for AI enhancement"""

    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 60
    enhance_workflows: bool = True
    enhance_test_data: bool = True
    enhance_validation: bool = True
    create_domain_flows: bool = True
    update_main_locust: bool = True
    check_flows:bool = False


@dataclass
class EnhancementResult:
    """Result of AI enhancement"""

    success: bool
    enhanced_files: Dict[str, str]
    enhanced_directory_files: List[Dict[str, Any]]
    enhancements_applied: List[str]
    errors: List[str]
    processing_time: float


def trace_error_paths(created_files:list, error_output: str) -> None:
    """
    Extract and analyze file paths from Python error output
    """

    lines = error_output.strip().split('\n')
    file_paths = []

    # Comprehensive patterns to match different traceback formats
    patterns = [
        # Standard traceback: File "/path/to/file.py", line 123, in function_name
        r'File "([^"]+\.py)"',
        # Direct file path mentions (most relevant for project files)
        r'File "?([^"\s]+\.py)"?',
        # Module loading errors showing full paths
        r'^\s*File "([^"]+\.py)"',
    ]

    # Extract file paths using patterns
    for line in lines:
        line = line.strip()

        # Skip system/library lines that aren't relevant to user code
        # Keep site-packages in case user wants to see those too, but prioritize project files
        if any(skip in line for skip in ['<frozen', '__pycache__']):
            continue

        for pattern in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                # Clean up the path
                file_path = match.strip().strip('"\'')

                # Only keep Python files
                if file_path.endswith('.py') and len(file_path) > 3:
                    file_paths.append(file_path)

    # Remove duplicates while preserving order
    seen = set()
    unique_file_paths = []
    for path in file_paths:
        if path not in seen:
            seen.add(path)
            unique_file_paths.append(path)

    file_paths = unique_file_paths


    files_to_check = []
    for i, path in enumerate(file_paths, 1):

        # Check if file exists
        exists = os.path.exists(path)
        path_obj = Path(path)

        if exists:
            abs_path = os.path.abspath(path)
            files_to_check.extend(
                [f.get("full_path") for f in created_files if str(f.get("full_path") )== str(abs_path)]
            )

            new_path = os.path.join(os.getcwd(), path)

            # Get parent directory
            parent = os.path.dirname(abs_path)


    return files_to_check

class EnhancementProcessor:
    """Handles individual enhancement operations"""

    def __init__(
        self,
        ai_config: Optional[AIEnhancementConfig],
        locust_generator: "HybridLocustGenerator",
    ) -> None:
        self.ai_config = ai_config
        self.locust_generator = locust_generator

    async def process_main_locust_enhancement(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process main locustfile enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config and self.ai_config.update_main_locust:
            enhanced_content = await self.locust_generator._enhance_locustfile(
                base_files.get("locustfile.py", ""), endpoints, api_info
            )
            if enhanced_content:
                enhanced_files["locustfile.py"] = enhanced_content
                enhancements.append("main_locust_update")
        return enhanced_files, enhancements

    async def process_domain_flows_enhancement(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: Optional[str] = "",
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process domain flows enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config and self.ai_config.create_domain_flows:
            domain_flows = await self.locust_generator._generate_domain_flows(
                endpoints, api_info, custom_requirement=custom_requirement
            )
            if domain_flows:
                enhanced_files["custom_flows.py"] = domain_flows
                enhancements.append("domain_flows")

        return enhanced_files, enhancements

    async def process_workflow_enhancements(
        self,
        base_files: Dict[str, str],
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process workflow enhancements"""
        enhanced_directory_files: List[Dict[str, Any]] = []
        enhancements: List[str] = []

        if self.ai_config and not self.ai_config.enhance_workflows:
            return enhanced_directory_files, enhancements

        base_workflow_files = self.locust_generator.get_files_by_key(
            directory_files, "base_workflow.py"
        )
        base_workflow_content = ""
        if base_workflow_files:
            first_workflow = base_workflow_files[0]
            # Get the content from the dictionary - adjust key name as needed
            base_workflow_content = first_workflow.get("base_workflow.py", "")
        for workflow_item in directory_files:
            enhanced_workflow_item = await self._enhance_single_workflow(
                workflow_item,
                base_files,
                base_workflow_content,
                grouped_endpoints,
                db_type,
            )
            if enhanced_workflow_item:
                enhanced_directory_files.append(enhanced_workflow_item["files"])
                enhancements.extend(enhanced_workflow_item["enhancements"])

        return enhanced_directory_files, enhancements

    async def _enhance_single_workflow(
        self,
        workflow_item: Dict[str, Any],
        base_files: Dict[str, str],
        base_workflow_files: str,
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str = "",
    ) -> Dict[str, Any] | None:
        """Enhance a single workflow file"""
        for key, value in workflow_item.items():
            workflow_key = key.replace("_workflow.py", "")
            endpoints_for_workflow = grouped_endpoints.get(workflow_key, [])
            auth_endpoints = grouped_endpoints.get("Authentication", [])
            workflow_endpoints_dict = {workflow_key: endpoints_for_workflow}
            enhanced_workflow = await self.locust_generator._enhance_workflows(
                base_content=value,
                test_data_content=base_files.get(test_data_file_path, ""),
                base_workflow=base_workflow_files,
                grouped_enpoints=workflow_endpoints_dict,
                auth_endpoints=auth_endpoints,
                db_type=db_type,
            )
            if enhanced_workflow:
                return {
                    "files": {key: enhanced_workflow},
                    "enhancements": [f"enhanced_workflows_{key}"],
                }

        return None

    async def process_test_data_enhancement(
        self, base_files: Dict[str, str], endpoints: List[Endpoint], db_type: str = ""
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process test data enhancement"""
        enhanced_files = {}
        enhancements = []
        if self.ai_config and self.ai_config.enhance_test_data:
            enhanced_test_data = await self.locust_generator.enhance_test_data_file(
                base_files.get(test_data_file_path, ""),
                endpoints,
                db_type,
                base_files.get(data_provider_path, ""),
                base_files.get("db_config.py", ""),
                data_provider_path,
            )
            if enhanced_test_data:
                enhanced_files[test_data_file_path] = enhanced_test_data
                enhancements.append("smart_test_data")
        return enhanced_files, enhancements

    async def process_validation_enhancement(
        self, base_files: Dict[str, str], endpoints: List[Endpoint]
    ) -> Tuple[Dict[str, str], List[str]]:
        """Process validation enhancement"""
        enhanced_files = {}
        enhancements = []
        if self.ai_config and self.ai_config.enhance_validation:
            enhanced_validation = await self.locust_generator._enhance_validation(
                base_files.get("utils.py", ""), endpoints
            )
            if enhanced_validation:
                enhanced_files["utils.py"] = enhanced_validation
                enhancements.append("advanced_validation")
        return enhanced_files, enhancements


class HybridLocustGenerator:
    """
    Hybrid generator that combines template-based reliability with AI creativity
    """

    def __init__(
        self,
        ai_client: AsyncTogether,
        ai_config: Optional[AIEnhancementConfig] = None,
        test_config: Optional[TestDataConfig] = None,
        prompt_dir: str = "prompt",
    ):
        self.ai_client = ai_client
        self.ai_config = ai_config or AIEnhancementConfig()
        self.template_generator = LocustTestGenerator(test_config)
        self.prompt_dir = self._find_project_root() / prompt_dir
        self._api_semaphore = asyncio.Semaphore(5)
        self._setup_jinja_env()
        self.MAX_RETRIES = 3
        self.RATE_LIMIT_BACKOFF = 10
        self.NON_RETRYABLE_CODES = ["401", "403", "unauthorized", "forbidden"]
        self.RATE_LIMIT_INDICATORS = ["429", "rate limit"]

    def _find_project_root(self) -> Path:
        """Find the project root by looking for setup.py, pyproject.toml, or .git"""
        current_path = Path(__file__).parent

        return current_path

    def _setup_jinja_env(self) -> None:
        """Setup Jinja2 environment with custom filters"""
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def _classify_error(self, error: Exception, attempt: int) -> ErrorClassification:
        """
        Classify an error to determine retry behavior.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            ErrorClassification with retry decision and backoff time
        """
        error_str = str(error).lower()

        # Non-retryable errors (auth/permission)
        if any(code in error_str for code in self.NON_RETRYABLE_CODES):
            logger.error(f"Authentication error, not retrying: {error}")
            return ErrorClassification(
                is_retryable=False, backoff_seconds=0, error_type="auth"
            )

        # Rate limit errors (retryable with longer backoff)
        if any(indicator in error_str for indicator in self.RATE_LIMIT_INDICATORS):
            logger.warning(f"Rate limit hit on attempt {attempt + 1}")
            return ErrorClassification(
                is_retryable=True,
                backoff_seconds=self.RATE_LIMIT_BACKOFF,
                error_type="rate_limit",
            )

        # Other retryable errors (exponential backoff)
        logger.warning(
            f"Retryable error on attempt {attempt + 1}: {type(error).__name__}"
        )
        return ErrorClassification(
            is_retryable=True,
            backoff_seconds=2**attempt,  # Exponential: 1s, 2s, 4s
            error_type="retryable",
        )

    async def generate_from_endpoints(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: Optional[str] = None,
        target_host: Optional[str] = None,
        include_auth: bool = True,
        db_type: str = "",
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Generate Locust tests using hybrid approach

        1. Generate reliable base structure with templates
        2. Enhance with AI for domain-specific improvements
        3. Validate and merge results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Generate reliable base structure
            logger.info("🔧 Generating base test structure with templates...")
            base_files, directory_files, grouped_enpoints = (
                self.template_generator.generate_from_endpoints(
                    endpoints,
                    api_info,
                    include_auth=include_auth,
                    target_host=target_host,
                    db_type=db_type,
                )
            )

            base_files = self.template_generator.fix_indent(base_files)
            # Step 2: Enhance with AI if available
            if self.ai_client and self._should_enhance(endpoints, api_info):
                logger.info("🤖 Enhancing tests with AI...")
                enhancement_result = await self._enhance_with_ai(
                    base_files,
                    endpoints,
                    api_info,
                    directory_files,
                    grouped_enpoints,
                    custom_requirement,
                    db_type,
                )
                if enhancement_result.success:
                    logger.info(
                        f"✅ AI enhancements applied: {', '.join(enhancement_result.enhancements_applied)}"
                    )
                    return (
                        enhancement_result.enhanced_files,
                        enhancement_result.enhanced_directory_files,
                    )
                else:
                    logger.warning(
                        f"⚠️ AI enhancement failed, using template base: {', '.join(enhancement_result.errors)}"
                    )
            else:
                logger.info("📋 Using template-based generation only")

            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"⏱️ Generation completed in {processing_time:.2f}s")

            return base_files, directory_files

        except Exception as e:
            logger.error(f"Hybrid generation failed line 267 : {e}")

            return {}, []

    def _should_enhance(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> bool:
        """Determine if AI enhancement is worthwhile"""
        # Enhance if we have enough endpoints or complex schemas
        complex_endpoints = [
            ep
            for ep in endpoints
            if ep.request_body or len(ep.parameters) > 3 or len(ep.responses) > 2
        ]

        return (
            len(endpoints) >= 3
            or len(complex_endpoints)  # Enough endpoints for meaningful enhancement
            >= 1
            or self._detect_domain_patterns(  # Has complex endpoints
                endpoints, api_info
            )  # Has recognizable domain patterns
        )

    def _detect_domain_patterns(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> bool:
        """Detect if API belongs to known domains that benefit from custom flows"""
        domain_keywords = {
            "ecommerce": ["product", "cart", "order", "payment", "checkout"],
            "user_management": ["user", "auth", "login", "register", "profile"],
            "content_management": ["post", "article", "comment", "media", "upload"],
            "financial": ["transaction", "account", "balance", "transfer"],
            "social": ["friend", "follow", "message", "notification", "feed"],
        }

        api_text = f"{api_info.get('title', '')} {api_info.get('description', '')}"
        endpoint_paths = " ".join([ep.path for ep in endpoints])
        combined_text = f"{api_text} {endpoint_paths}".lower()

        for domain, keywords in domain_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return True

        return False

    async def _enhance_locustfile(
        self, base_content: str, endpoints: List[Any], api_info: Dict[str, Any]
    ) -> Optional[str]:
        # Configuration

        try:
            template = self.jinja_env.get_template("locust.j2")

            # Prepare context for template
            context = {
                "base_content": base_content,
                "endpoints_for_prompt": self._format_endpoints_for_prompt(
                    endpoints[:5]
                ),
                "api_info": api_info,
            }
            # Render enhanced content
            prompt = template.render(**context)
            enhanced_content = await self._call_ai_service(prompt)
            return enhanced_content
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return base_content

    async def _enhance_with_ai(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> EnhancementResult:
        """Enhance base files with AI - Refactored for reduced cognitive complexity"""
        start_time = asyncio.get_event_loop().time()

        try:
            enhancement_result = await self._process_all_enhancements(
                base_files,
                endpoints,
                api_info,
                directory_files,
                grouped_endpoints,
                custom_requirement,
                db_type,
            )

            processing_time = asyncio.get_event_loop().time() - start_time
            enhancement_result.processing_time = processing_time

            return enhancement_result

        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time

            return EnhancementResult(
                success=False,
                enhanced_files=base_files,
                enhancements_applied=[],
                enhanced_directory_files=[],
                errors=[str(e)],
                processing_time=processing_time,
            )

    async def _process_all_enhancements(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        custom_requirement: Optional[str] = None,
        db_type: str = "",
    ) -> EnhancementResult:
        """Process all enhancements using the enhancement processor"""
        processor = EnhancementProcessor(self.ai_config, self)

        enhanced_files = base_files.copy()
        enhanced_directory_files = []
        enhancements_applied: List[str] = []
        errors = []
        # Process each enhancement type
        enhancement_tasks = [
            processor.process_main_locust_enhancement(base_files, endpoints, api_info),
            processor.process_domain_flows_enhancement(
                endpoints, api_info, custom_requirement
            ),
            processor.process_test_data_enhancement(base_files, endpoints, db_type),
            processor.process_validation_enhancement(base_files, endpoints),
        ]

        # Execute file-based enhancements concurrently
        file_enhancement_results = await asyncio.gather(
            *enhancement_tasks, return_exceptions=True
        )

        # Process results from file-based enhancements
        for result in file_enhancement_results:
            if isinstance(result, BaseException):
                errors.append(str(result))
                continue

            files, enhancements = result
            enhanced_files.update(files)
            enhancements_applied.extend(enhancements)

        # Process workflow enhancements separately (more complex logic)
        try:
            (
                workflow_files,
                workflow_enhancements,
            ) = await processor.process_workflow_enhancements(
                base_files, directory_files, grouped_endpoints, db_type
            )
            enhanced_directory_files.extend(workflow_files)
            enhancements_applied.extend(workflow_enhancements)
        except Exception as e:
            errors.append(f"Workflow enhancement error: {str(e)}")

        return EnhancementResult(
            success=len(errors) == 0,
            enhanced_files=enhanced_files,
            enhanced_directory_files=enhanced_directory_files,
            enhancements_applied=enhancements_applied,
            errors=errors,
            processing_time=0,  # Will be set by caller
        )

    async def _generate_domain_flows(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: Optional[str] = "",
    ) -> Optional[str]:
        """Generate domain-specific user flows"""

        # Analyze endpoints to determine domain
        domain_analysis = self._analyze_api_domain(endpoints, api_info)
        try:
            template = self.jinja_env.get_template("domain.j2")
            # Render enhanced content
            prompt = template.render(
                domain_analysis=domain_analysis,
                custom_requirement=custom_requirement,
                endpoints=self._format_endpoints_for_prompt(endpoints),
            )

            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content:
                return enhanced_content
        except Exception as e:
            logger.warning(f"Domain flows generation failed: {e}")

        return ""

    def get_files_by_key(
        self, directory_files: List[Dict[str, Any]], target_key: str
    ) -> List[Dict[str, Any]]:
        """Return directory items that contain the specified key"""
        return [items for items in directory_files if target_key in items]

    async def _enhance_workflows(
        self,
        base_content: str,
        test_data_content: str,
        base_workflow: str,
        grouped_enpoints: Dict[str, List[Endpoint]],
        auth_endpoints: List[Endpoint],
        db_type: str = "",
    ) -> Optional[str]:
        try:
            template = self.jinja_env.get_template("workflow.j2")

            # Render enhanced content
            prompt = template.render(
                grouped_enpoints=grouped_enpoints,
                test_data_content=test_data_content,
                base_workflow=base_workflow,
                auth_endpoints=auth_endpoints,
                base_content=base_content,
                db_type=db_type,
            )
            enhanced_content = await self._call_ai_service(prompt)
            return enhanced_content
        except Exception as e:
            logger.warning(f"Workflow enhancement failed: {e}")

        return ""

    async def enhance_test_data_file(
        self,
        base_content: str,
        endpoints: List[Endpoint],
        db_type: str = "",
        data_provider: str = "",
        db_config: str = "",
        data_provider_path: str = "",
    ) -> Optional[str]:
        """Enhance test data generation with domain knowledge"""

        # Extract schema information
        schemas_info = self._extract_schema_patterns(endpoints)

        try:
            template = self.jinja_env.get_template("test_data.j2")

            # Prepare context for template
            context = {
                "base_content": base_content,
                "schemas_info": schemas_info,
                "endpoints": endpoints,
                "db_type": db_type,
                "data_provider_content": data_provider,
                "db_config": db_config,
                "data_provider_path": data_provider_path,
            }

            # Render enhanced content
            prompt = template.render(**context)
            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content and self._validate_python_code(enhanced_content):
                return enhanced_content
        except Exception as e:
            logger.warning(f"Test data enhancement failed: {e}")

        return ""

    async def _enhance_validation(
        self, base_content: str, endpoints: List[Endpoint]
    ) -> Optional[str]:
        """Enhance response validation with endpoint-specific checks"""

        validation_patterns = self._extract_validation_patterns(endpoints)
        try:
            template = self.jinja_env.get_template("validation.j2")

            # Render enhanced content
            prompt = template.render(
                base_content=base_content, validation_patterns=validation_patterns
            )
            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content:
                return enhanced_content
        except Exception as e:
            logger.warning(f"Validation enhancement failed: {e}")

        return ""

    def _build_messages(self, prompt: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "You are an expert Python developer specializing in Locust load testing. Generate clean, production-ready code with proper error handling. "
                "Always return your code wrapped in <code></code> tags with no explanations outside the tags and DO NOT TRUNCATE THE CODE. "
                "Format: <code>your_python_code_here</code>",
            },
            {"role": "user", "content": prompt},
        ]

    async def _make_api_call(self, messages: list[dict],multiple_file:bool= False) -> Optional[str]:
        """Make API call - ONE job"""
        async with self._api_semaphore:
            api_call = self.ai_client.chat.completions.create(
                model=self.ai_config.model,
                messages=messages,
                max_tokens=self.ai_config.max_tokens,
                temperature=self.ai_config.temperature,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
            )

            # Wait for the API call with timeout
            response = await asyncio.wait_for(
                api_call,
                timeout=self.ai_config.timeout,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content.strip()
                if not multiple_file:
                    # Clean up the response
                    content = self._clean_ai_response(
                        self.extract_code_from_response(content)
                    )
                return content

        return None

    async def _call_ai_service_for_locust_fixes(self, prompt: str,file_contents:list) -> Optional[str]:
        """Call AI service specifically for Locust file validation and correction tasks."""
        file_updates = {}
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in Locust performance testing and Python development. "
                    "Analyze the given Locust files and validation errors. "
                    "Return corrected code for each file that needs changes. "
                    "Follow the response format exactly as instructed by the user. "
                    "DO NOT use <code> tags or any custom formatting — only return markdown blocks starting with ### UPDATED_FILE."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._api_semaphore:
                    content = await self._make_api_call(messages, multiple_file=True)

                    if content:
                        return content

            except asyncio.TimeoutError:
                logger.warning(f"AI service timeout on attempt {attempt + 1}")

            except Exception as e:
                classification = self._classify_error(e, attempt)
                if not classification.is_retryable:
                    return ""
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(classification.backoff_seconds)
                    continue

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)

        return content

    async def _call_ai_service(self, prompt: str) -> Optional[str]:
        """Call AI service with retry logic and validation"""
        messages = self._build_messages(prompt)

        for attempt in range(self.MAX_RETRIES):  # Retry logic
            try:
                async with self._api_semaphore:
                    content = await self._make_api_call(messages)
                    if content:
                        return content

            except asyncio.TimeoutError:
                logger.warning(f"AI service timeout on attempt {attempt + 1}")

            except Exception as e:
                classification = self._classify_error(e, attempt)  # Helper 3

                if not classification.is_retryable:
                    return ""

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(classification.backoff_seconds)

                    continue

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)

        return ""

    def extract_code_from_response(self, response_text: str) -> str:
        # Extract content between <code> tags
        pattern = r"<code>(.*?)</code>"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if not matches:
            logger.warning("No <code> tags found, using full response")
            return response_text.strip()

        content = max(matches, key=len).strip()

        # Content too short - use full response
        if not content or len(content) <= 10:
            logger.warning(
                f"Code in tags too short ({len(content)} chars), using full response"
            )
            return response_text.strip()

        logger.debug(f"Extracted {len(content)} chars from <code> tags")
        return str(content)

    def _clean_ai_response(self, content: str) -> str:
        """Clean and validate AI response"""
        # Remove markdown code blocks if present
        if content.startswith("```python") and content.endswith("```"):
            content = content[9:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()

        # Remove any explanatory text before/after code
        lines = content.split("\n")
        start_idx = 0
        end_idx = len(lines)

        # Find actual Python code start
        for i, line in enumerate(lines):
            if line.strip().startswith(
                ("import ", "from ", "class ", "def ", '"""', "'''")
            ):
                start_idx = i
                break

        # Find actual Python code end (remove trailing explanations)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if (
                line
                and not line.startswith("#")
                and not line.lower().startswith(("note:", "this", "the "))
            ):
                end_idx = i + 1
                break

        return "\n".join(lines[start_idx:end_idx])

    def _analyze_api_domain(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> str:
        """Analyze API to determine domain and patterns"""
        analysis = []

        # API info analysis
        analysis.append(f"API Title: {api_info.get('title', 'Unknown')}")
        analysis.append(f"Description: {api_info.get('description', 'No description')}")

        # Endpoint analysis
        methods = [ep.method for ep in endpoints]
        paths = [ep.path for ep in endpoints]

        analysis.append(f"Total Endpoints: {len(endpoints)}")
        analysis.append(f"HTTP Methods: {', '.join(set(methods))}")
        analysis.append(f"Common Path Patterns: {self._extract_path_patterns(paths)}")

        # Resource analysis
        resources = self._extract_resources_from_paths(paths)
        analysis.append(f"Main Resources: {', '.join(resources[:5])}")

        return "\n".join(analysis)

    def _format_endpoints_for_prompt(self, endpoints: List[Endpoint]) -> str:
        """Format endpoints for AI prompt"""
        formatted = []
        for ep in endpoints:
            params = f"({len(ep.parameters)} params)" if ep.parameters else ""
            body = "(with body)" if ep.request_body else ""
            formatted.append(
                f"- {ep.method} {ep.path} {params} {body} - {ep.summary or 'No summary'}"
            )

        return "\n".join(formatted)

    def _extract_schema_patterns(self, endpoints: List[Endpoint]) -> str:
        """Extract common schema patterns from endpoints"""
        patterns = []

        for ep in endpoints:
            if ep.request_body and ep.request_body.schema:
                schema = ep.request_body.schema
                if schema.get("properties"):
                    fields = list(schema["properties"].keys())
                    patterns.append(f"{ep.path} ({ep.method}): {', '.join(fields[:5])}")

        return "\n".join(patterns[:10])  # Limit for token efficiency

    def _extract_validation_patterns(self, endpoints: List[Endpoint]) -> str:
        """Extract validation patterns needed for endpoints"""
        patterns = []

        for ep in endpoints:
            for response in ep.responses:
                if response.status_code.startswith("2"):  # Success responses
                    pattern = f"{ep.method} {ep.path} -> {response.status_code}"
                    if response.schema:
                        pattern += " (schema validation needed)"
                    patterns.append(pattern)

        return "\n".join(patterns[:10])

    def _analyze_performance_patterns(self, endpoints: List[Endpoint]) -> str:
        """Analyze endpoints for performance testing patterns"""
        analysis = []

        # Categorize endpoints by performance characteristics
        read_heavy = [ep for ep in endpoints if ep.method == "GET"]
        write_heavy = [ep for ep in endpoints if ep.method in ["POST", "PUT", "PATCH"]]
        bulk_candidates = [
            ep
            for ep in endpoints
            if "bulk" in ep.path.lower() or "batch" in ep.path.lower()
        ]

        analysis.append(
            f"Read-heavy endpoints: {len(read_heavy)} (good for load testing)"
        )
        analysis.append(
            f"Write-heavy endpoints: {len(write_heavy)} (good for stress testing)"
        )
        analysis.append(
            f"Bulk operation endpoints: {len(bulk_candidates)} (good for volume testing)"
        )

        # Identify endpoints that might be resource intensive
        complex_endpoints = [
            ep
            for ep in endpoints
            if ep.request_body
            and ep.request_body.schema
            and len(ep.request_body.schema.get("properties", {})) > 5
        ]
        analysis.append(
            f"Complex endpoints: {len(complex_endpoints)} (monitor for performance)"
        )

        return "\n".join(analysis)

    def _extract_path_patterns(self, paths: List[str]) -> str:
        """Extract common patterns from API paths"""
        patterns = set()
        for path in paths:
            # Extract patterns like /api/v1/{resource}
            parts = path.split("/")
            if len(parts) > 2:
                pattern = "/".join(parts[:3])
                if "{" in pattern:
                    pattern = (
                        pattern.replace("{id}", "{id}")
                        .replace("{", "{")
                        .replace("}", "}")
                    )
                patterns.add(pattern)

        return ", ".join(list(patterns)[:5])

    def _extract_resources_from_paths(self, paths: List[str]) -> List[str]:
        """Extract resource names from API paths"""
        resources = set()
        for path in paths:
            parts = [p for p in path.split("/") if p and not p.startswith("{")]
            for part in parts:
                if len(part) > 2 and part.isalpha():  # Likely a resource name
                    resources.add(part)

        return sorted(resources)

    async def _create_test_files_safely(
        self,
        test_files: Dict[str, str],
        output_path: Path,
        max_file_size: int = 1024 * 1024,
    ) -> List[dict]:
        """Create test files safely with reduced complexity"""

        if not test_files:
            return []

        # Setup
        config = FileCreationConfig()
        config.MAX_FILE_SIZE = max_file_size
        creator = SafeFileCreator(config)
        temp_dir = output_path / f"temp_{uuid.uuid4().hex[:8]}"

        try:
            return await self._process_file_creation(
                creator, test_files, output_path, temp_dir
            )
        finally:
            await self._cleanup_temp_directory(temp_dir)

    def run_locust_file(self,created_files:list, file_path: str) -> None:
        """
        Validate a Locust file for syntax and import errors WITHOUT running it.
        Returns validation results with compatibility for existing code.
        """
        validation_result = {
            "file_path": file_path,
            "files_affected":[],
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "user_classes_found": [],
            "tasks_found": 0,
            "stdout": "",
            "stderr": "",
            "returncode": 1,
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result["stderr"] = f"File not found: {file_path}"
                return validation_result
            # Check syntax
            with open(file_path, "r") as f:
                file_content = f.read()

            try:
                compile(file_content, file_path, "exec")
            except SyntaxError as e:
                validation_result["stderr"] = (
                    f"SyntaxError on line {e.lineno}: {str(e)}"
                )
                return validation_result

            # Import analysis
            import_result = self._safe_import_analysis(created_files,file_path)

            validation_result.update(import_result)

            # Set result fields
            if validation_result["errors"]:
                validation_result["stderr"] = "\n".join(
                    [
                        err if isinstance(err, str) else err.get("message", str(err))
                        for err in validation_result["errors"]
                    ]
                )
            else:
                validation_result["is_valid"] = True
                validation_result["stdout"] = (
                    f"✅ Validation successful: {len(validation_result['user_classes_found'])} user classes, {validation_result['tasks_found']} tasks"
                )
                validation_result["stderr"] = ""
                validation_result["returncode"] = 0

        except Exception as e:

            validation_result["stderr"] = f"Validation error: {str(e)}"

        return validation_result

    def _normalize_file_path(self, file_path: str) -> str:
        """
        Normalize file path to handle common issues like duplication.

        Args:
            file_path: Original file path (may have issues)

        Returns:
            Normalized file path
        """

        # Step 1: Handle if path already exists as-is
        if os.path.exists(file_path):
            result = os.path.abspath(file_path)
            return result

        # Step 2: Handle relative paths
        if not os.path.isabs(file_path):
            # Check if it's relative to current working directory
            potential_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(potential_path):
                result = potential_path
                return result

            # Step 3: Check for path duplication (like tests/tests/file.py)
            parts = file_path.split(os.sep)
            if len(parts) >= 2:
                # Look for consecutive duplicate parts
                for i in range(len(parts) - 1):
                    if parts[i] == parts[i + 1] and parts[i]:  # Skip empty parts
                        # Remove the duplicate
                        fixed_parts = parts[:i] + parts[i+1:]
                        fixed_path = os.sep.join(fixed_parts)
                        potential_path = os.path.join(os.getcwd(), fixed_path)


                        if os.path.exists(potential_path):
                            result = potential_path

                            return result

            # Step 4: Try removing common problematic prefixes
            # Sometimes paths get malformed like "project/project/tests/file.py"
            cwd_name = os.path.basename(os.getcwd())
            if file_path.startswith(f"{cwd_name}/"):
                potential_path = os.path.join(os.getcwd(), file_path[len(cwd_name)+1:])
                if os.path.exists(potential_path):
                    result = potential_path
                    return result

        # Step 5: If nothing worked, return the absolute path (even if it doesn't exist)
        result = os.path.abspath(file_path)

        return result

    def run_test_locust_file(self, created_files:list, file_path:str):
        files_affected = []
        try:

            result = subprocess.run(
                ["locust", "-f", file_path],  # command as list
                capture_output=True,  # capture stdout & stderr
                text=True,  # decode bytes to string,
                timeout=30,
            )
            if result.returncode != 0:
                # Trace the error
                files_affected = trace_error_paths(created_files,result.stderr)

            return result, files_affected
        except subprocess.TimeoutExpired:
            return None, files_affected
        except FileNotFoundError:

            return None, files_affected
        except Exception as e:
            return None, files_affected

    def _safe_import_analysis(self, created_files:list, file_path: str) -> Dict[str, Any]:
        """
        Safely analyze the Locust file by importing it in a controlled way.
        Fixed version with proper path handling.
        """

        result = {
            "errors": [],
            "warnings": [],
            "user_classes_found": [],
            "tasks_found": 0,
            "affected_files":[]
        }

        # ========================================
        # 🔧 FIXED PATH HANDLING
        # ========================================

        # Step 1: Normalize and resolve the path
        normalized_path = self._normalize_file_path(file_path)

        # Step 2: Validate the path exists
        if not os.path.exists(normalized_path):
            result["errors"].append(
                {
                    "type": "FileNotFound",
                    "message": f"File not found: {normalized_path} (original: {file_path})",
                    "suggestion": "Check the file path and ensure the file exists",
                }

            )
            result['affected_files'].append(file_path)
            return result

        # Step 3: Get directory and module info from normalized path
        dir_path = os.path.dirname(os.path.abspath(normalized_path))
        module_name = os.path.basename(normalized_path).replace(".py", "")

        # Save current state to restore later
        original_path = sys.path.copy()
        original_cwd = os.getcwd()
        original_modules = set(sys.modules.keys())

        try:
            # Add directory to Python path so imports work
            if dir_path not in sys.path:
                sys.path.insert(0, dir_path)

            # Change to file directory for relative imports
            os.chdir(dir_path)

            # Clean any existing module from cache for fresh import
            modules_to_remove = [
                name
                for name in sys.modules.keys()
                if name == module_name or name.startswith(f"{module_name}.")
            ]
            for mod_name in modules_to_remove:
                del sys.modules[mod_name]

            # Create module spec and load it
            spec = importlib_util.spec_from_file_location(module_name, normalized_path)

            if spec is None:
                result["errors"].append(
                    {
                        "type": "ImportError",
                        "message": f"Could not create module spec for {normalized_path}",
                        "suggestion": "Check that the file is a valid Python module",
                    }
                )
                result["affected_files"].append(file_path)
                return result

            if spec.loader is None:
                result["affected_files"].append(normalized_path)
                result["errors"].append(
                    {
                        "type": "ImportError",
                        "message": f"No loader available for {normalized_path}",
                        "suggestion": "Check file permissions and format",
                    }
                )
                return result

            res, affected_files  = self.run_test_locust_file(created_files, normalized_path)
            result["errors"].append(
                {
                    "type": "",
                    "message": f" {res.stderr}",
                    "missing_module": "",
                    "suggestion": res.stderr
                }
            )
            result['affected_files'].extend(affected_files)

        except ImportError as e:
            print(f"❌ ImportError: {str(e)}")
            error_msg = str(e)
            missing_module = None

            if "No module named" in error_msg:
                start = error_msg.find("'") + 1
                end = error_msg.find("'", start)
                if start > 0 and end > start:
                    missing_module = error_msg[start:end]

            result["errors"].append(
                {
                    "type": "ImportError",
                    "message": f"Missing dependency: {error_msg}",
                    "missing_module": missing_module,
                    "suggestion": (
                        f"Install missing module with: pip install {missing_module}"
                        if missing_module
                        else "Check that all imported modules are installed"
                    ),
                }
            )
            result['affected_files'].append(file_path)

        except SyntaxError as e:
            print(f"❌ SyntaxError: {str(e)}")
            result["errors"].append(
                {
                    "type": "SyntaxError",
                    "message": str(e),
                    "line": e.lineno,
                    "text": e.text,
                    "suggestion": "Fix the syntax error before proceeding",
                }
            )
            result["affected_files"].append(file_path)

        except NameError as e:
            print(f"❌ NameError: {str(e)}")
            result["errors"].append(
                {
                    "type": "NameError",
                    "message": str(e),
                    "suggestion": "Check that all variables and functions are properly defined or imported",
                }
            )
            result["affected_files"].append(file_path)

        except AttributeError as e:
            print(f"❌ AttributeError: {str(e)}")
            result["errors"].append(
                {
                    "type": "AttributeError",
                    "message": str(e),
                    "suggestion": "Check that all attributes and methods exist on the objects you're using",
                }
            )
            result["affected_files"].append(file_path)

        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            result["errors"].append(
                {
                    "type": "ExecutionError",
                    "message": f"Error during module execution: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "suggestion": "Check the full traceback for details about the execution error",
                }
            )
            result["affected_files"].append(file_path)

        finally:
            # Restore original state
            sys.path[:] = original_path
            os.chdir(original_cwd)

            # Clean up any modules we added
            current_modules = set(sys.modules.keys())
            added_modules = current_modules - original_modules
            for mod_name in added_modules:
                if mod_name.startswith(module_name):
                    try:
                        del sys.modules[mod_name]

                    except KeyError:
                        pass

        return result


    def _parse_ai_file_updates(self, ai_response: str, original_file_contents: dict) -> dict:
        """
        Parse AI response to extract individual file updates.

        Args:
            enhanced_response_files: The AI's response containing multiple file updates
            original_file_contents: Dictionary of original file contents for comparison

        Returns:
            Dictionary with file paths as keys and updated content as values
        """
        file_updates = {}

        # Pattern to match file headers: ### UPDATED_FILE: /path/to/file.py
        file_pattern = r'### UPDATED_FILE:\s*([^\n]+)\s*\n```python\s*\n(.*?)\n```'
        matches = re.findall(file_pattern, ai_response, re.DOTALL)

        if matches:

                for file_path, content in matches:
                    file_path = file_path.strip()
                    content = content.strip()

                    # Validate that this file was in our original set
                    if file_path in original_file_contents:
                        # Check if content actually changed
                        original_content = original_file_contents[file_path].strip()
                        if content != original_content:
                            file_updates[file_path] = content


        else:
                # Fallback: Try to parse as single file update (backward compatibility)

                # Clean up response for single file
                cleaned_content = ai_response.strip()

                # Remove code block markers
                if cleaned_content.startswith("```python"):
                    cleaned_content = cleaned_content[9:]
                elif cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content[3:]

                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]

                cleaned_content = cleaned_content.strip()

                # If we only have one file, assume the response is for that file
                if len(original_file_contents) == 1:
                    single_file = list(original_file_contents.keys())[0]
                    original_content = list(original_file_contents.values())[0].strip()

                    if cleaned_content != original_content:
                        file_updates[single_file] = cleaned_content

        return file_updates

    async def update_files_with_AI(self,created_files: list, locust_file: str, errors: list, affected_files: list):
        """Update the Locust file with AI suggestions."""
        file_content=""
        file_contents={}
        if len(affected_files)>0:
            for affected_file in affected_files:
                if os.path.exists(affected_file):
                    with open(affected_file, 'r') as f:
                        content = f.read()
                        file_contents[affected_file] = content

                        file_content += f"the content of  ```python {affected_file}``` is {content}"

        else:
            with open(locust_file, "r") as f:
                content = f.read()
                file_contents[locust_file] = content
                file_content = f"the content of  ```python {locust_file}``` is {content}"

        file_updates = {}
        prompt = f"""
        You are an expert in Locust performance testing framework and Python development.
        You are given a Locust test files and validation errors that occurred when running them.
        Analyze the errors and return corrected file content for ALL files that need changes

       **IMPORTANT INSTRUCTIONS:**
        1. If the error is about "No Locust User classes found", ensure at least one top-level class inherits from HttpUser
        2. Fix import errors (missing imports, incorrect imports, etc.)
        3. Fix syntax errors and type annotation issues
        4. Only modify files that actually need changes
        5. Return each file's corrected content separately


        **Files provided:**
        {file_content}
        
        **Errors encountered:**
        {errors}

        **Response format:**
        For each file that needs changes, respond with:
        
        ### UPDATED_FILE: /path/to/file.py
        ```python
        [corrected file content here]
        ```
        
        ### UPDATED_FILE: /path/to/another_file.py  
        ```python
        [corrected file content here]
        ```
        
        Only include files that actually need modifications. If a file doesn't need changes, don't include it in the response.

        ---
        """
        enhanced_response_files = await self._call_ai_service_for_locust_fixes(prompt,file_contents)

        if enhanced_response_files :
            # Parse the AI response to extract individual file updates
            file_updates = self._parse_ai_file_updates(enhanced_response_files, file_contents)

        return file_updates

    async def retry_until_run(
        self,
        created_files: list,
        locust_file: str,
        file_enhance_path:str,
        output_path:Path,
        attempt: int = 1,
        max_retries: int = 5,
    ) -> bool:
        """Recursively retry validating the Locust file until no errors occur."""

        result = self.run_locust_file(created_files,locust_file)

        if attempt >= max_retries:
            print(f"❌ Max retries ({max_retries}) reached for {locust_file}.")
            return False

        if len(result["errors"]) > 0:
            try:
                # Get AI-generated file updates

                file_updates = await self.update_files_with_AI(
                    created_files,file_enhance_path, result["errors"], result["affected_files"]
                )
                if file_updates:
                    # Apply updates safely
                    update_results = await self._create_test_files_safely(
                        file_updates,
                        output_path,
                        max_file_size=1024 * 1024,
                    )

                    # Add updated files to created_files
                    for result in update_results:
                        file_path = result.get("file_path")
                        if file_path and file_path not in created_files:
                            created_files.append(file_path)

                    return await self.retry_until_run(
                            created_files,
                            locust_file,
                        file_enhance_path,
                            output_path,
                            attempt + 1,
                            max_retries,
                        )

                else:
                    return False

            except Exception as e:
                print(f"❌ Error in retry with AI fixes: {str(e)}")
                return False
        return False

    async def _delete_existing_files(
            self,
            file_updates: Dict[str, str],
            output_path: Path
    ) -> None:
        """Delete existing files that will be updated"""

        deleted_count = 0

        for file_path in file_updates.keys():
            try:
                # Resolve the full file path
                if not os.path.isabs(file_path):
                    full_path = output_path / file_path
                else:
                    full_path = Path(file_path)

                # Delete if it exists
                if full_path.exists():
                    full_path.unlink()
                    deleted_count += 1


            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {str(e)}")
                continue


    async def _process_file_creation(
        self,
        creator: SafeFileCreator,
        test_files: Dict[str, str],
        output_path: Path,
        temp_dir: Path,
    ) -> List[dict]:
        """Process the file creation workflow"""

        # Ensure directories exist
        output_path.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        await self._delete_existing_files(test_files, output_path)

        # Prepare files in temp directory
        prepared_files = await self._prepare_files_in_temp(
            creator, test_files, temp_dir
        )

        if not prepared_files:
            return []

        # Move files atomically to final location
        return await creator.move_files_atomically(prepared_files, output_path)

    async def _prepare_files_in_temp(
        self, creator: SafeFileCreator, test_files: Dict[str, str], temp_dir: Path
    ) -> List[dict]:
        """Prepare all files in temporary directory"""

        prepared_files = []

        for filename, content in test_files.items():
            file_result = await self._prepare_single_file(
                creator, filename, content, temp_dir
            )
            if file_result:
                prepared_files.append(file_result)

        return prepared_files

    async def _prepare_single_file(
        self, creator: SafeFileCreator, filename: str, content: str, temp_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Prepare a single file, return None if failed"""

        try:
            # Validate file
            is_valid, clean_filename, processed_content = creator.validate_file(
                filename, content
            )
            if not is_valid:
                return None

            # Create temp file
            file_info = await creator.create_temp_file(
                clean_filename, processed_content, temp_dir
            )
            logger.info(f"Prepared: {clean_filename} ({len(processed_content)} chars)")
            return file_info

        except Exception as e:
            logger.error(f"Failed to prepare file {filename}: {e}")
            return None

    async def _cleanup_temp_directory(self, temp_dir: Path) -> None:
        """Clean up temporary directory"""
        if temp_dir.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

    def _validate_python_code(self, content: str) -> bool:
        """Validate Python code syntax"""
        try:
            compile(content, "<string>", "exec")
            return True
        except SyntaxError:
            return False
