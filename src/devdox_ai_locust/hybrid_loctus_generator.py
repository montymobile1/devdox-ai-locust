"""
Hybrid Locust Test Generator

Combines reliable template-based generation with LLM enhancement for creativity
and domain-specific optimizations.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass
import uuid
import shutil


from devdox_ai_locust.utils.open_ai_parser import Endpoint
from devdox_ai_locust.utils.file_creation import FileCreationConfig, SafeFileCreator
from devdox_ai_locust.locust_generator import LocustTestGenerator, TestDataConfig
from devdox_ai_locust.utils.patch_generator import PatchGenerator
from together import AsyncTogether

logger = logging.getLogger(__name__)

# Hardcoded batch size for endpoint processing to prevent LLM output truncation
ENDPOINT_BATCH_SIZE = 5


test_data_file_path = "test_data.py"
data_provider_path = "data_provider.py"
base_workflow_path = "base_workflow.py"
workflow_jinja_path = "workflow.j2"


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


@dataclass
class EnhancementResult:
    """Result of AI enhancement"""

    success: bool
    enhanced_files: Dict[str, str]
    enhanced_directory_files: List[Dict[str, Any]]
    enhancements_applied: List[str]
    errors: List[str]
    processing_time: float


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

    def _get_base_workflow_content(self, directory_files: List[Dict[str, Any]]) -> str:
        base_files = self.locust_generator.get_files_by_key(
            directory_files, base_workflow_path
        )
        if not base_files:
            return ""
        workflow_dict = base_files[0]
        return workflow_dict.get(base_workflow_path) or ""

    async def _process_workflow_item(
        self,
        file_dict: Dict[str, Any],
        base_files: Dict[str, str],
        base_workflow_content: str,
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str,
        template: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        return await self._enhance_single_workflow(
            file_dict,
            base_files,
            base_workflow_content,
            grouped_endpoints,
            db_type,
            template_path=template or workflow_jinja_path,
        )

    async def process_workflow_enhancements(
        self,
        base_files: Dict[str, str],
        directory_files: List[Dict[str, Any]],
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str = "",
        include_auth: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process workflow enhancements with minimal duplication."""

        # Early exit if enhancements disabled
        if not (self.ai_config and self.ai_config.enhance_workflows):
            return [], []

        enhanced_directory_files: List[Dict[str, Any]] = []
        enhancements: List[str] = []

        # Extract base workflow content
        base_workflow_content = self._get_base_workflow_content(directory_files)

        # Handle special case: enhancing base workflow if auth included
        if include_auth and base_workflow_content:
            result = await self._process_workflow_item(
                file_dict={base_workflow_path: base_workflow_content},
                base_files=base_files,
                base_workflow_content=base_workflow_content,
                grouped_endpoints=grouped_endpoints,
                db_type=db_type,
                template="base_workflow.j2",
            )
            if result:
                base_workflow_content = result["files"].get(base_workflow_path, "")
                enhancements.extend(result["enhancements"])

        # Enhance all other workflow files
        for workflow_item in directory_files:
            if workflow_item.get("file_name") == base_workflow_path:
                continue

            result = await self._process_workflow_item(
                file_dict=workflow_item,
                base_files=base_files,
                base_workflow_content=base_workflow_content,
                grouped_endpoints=grouped_endpoints,
                db_type=db_type,
                template="",
            )
            if result:
                enhanced_directory_files.append(result["files"])
                enhancements.extend(result["enhancements"])

        return enhanced_directory_files, enhancements

    async def _enhance_single_workflow(
        self,
        workflow_item: Dict[str, Any],
        base_files: Dict[str, str],
        base_workflow_files: str,
        grouped_endpoints: Dict[str, List[Endpoint]],
        db_type: str = "",
        template_path: str =workflow_jinja_path,
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
                template_path=template_path,
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
        output_dir: Optional[Path] = None,
        diagnostics: bool = False,
    ):
        """
        Initialize the hybrid generator.

        Args:
            ai_client: Together AI client for LLM calls.
            ai_config: Configuration for AI enhancement.
            test_config: Configuration for test data generation.
            prompt_dir: Directory containing Jinja2 prompt templates.
            output_dir: Output directory for generated tests (used for diagnostics).
            diagnostics: If True, saves pre/post LLM patches and prompts for debugging.
        """
        self.ai_client = ai_client
        self.ai_config = ai_config or AIEnhancementConfig()
        self.template_generator = LocustTestGenerator(test_config)
        self.prompt_dir = self._find_project_root() / prompt_dir
        self._api_semaphore = asyncio.Semaphore(5)
        self._setup_jinja_env()
        self.MAX_RETRIES = 3
        self.RATE_LIMIT_BACKOFF = 10
        self.NON_RETRYABLE_CODES = [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "authentication",
            "unauthorized",
            "invalid token",
        ]
        self.RATE_LIMIT_INDICATORS = ["429", "rate limit"]

        # Diagnostics support
        self.diagnostics = diagnostics
        self.output_dir = output_dir
        self.patch_generator: Optional[PatchGenerator] = None
        if diagnostics and output_dir:
            self.patch_generator = PatchGenerator(output_dir)
            logger.info("Diagnostics mode enabled - will save patches and prompts")

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
        Generate Locust tests using hybrid approach.

        1. Generate reliable base structure with templates
        2. Enhance with AI for domain-specific improvements
        3. Validate and merge results

        If diagnostics mode is enabled, saves pre-LLM and post-LLM patches
        for debugging purposes.
        """
        start_time = asyncio.get_event_loop().time()

        # Start diagnostics session if enabled
        if self.patch_generator:
            self.patch_generator.start_session()
            logger.info("Diagnostics session started")

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

            # Save pre-LLM patch if diagnostics enabled
            if self.patch_generator:
                self.patch_generator.save_pre_llm_patch(base_files, directory_files)
                logger.info("Saved pre-LLM patch")

            # Step 2: Enhance with AI if available
            if self.ai_client and self._should_enhance(endpoints, api_info):
                logger.info("🤖 Enhancing tests with AI...")
                enhancement_result = await self._enhance_with_ai(
                    base_files,
                    endpoints,
                    api_info,
                    include_auth,
                    directory_files,
                    grouped_enpoints,
                    custom_requirement,
                    db_type,
                )
                if enhancement_result.success:
                    logger.info(
                        f"✅ AI enhancements applied: {', '.join(enhancement_result.enhancements_applied)}"
                    )

                    # Save post-LLM patch if diagnostics enabled
                    if self.patch_generator:
                        self.patch_generator.save_post_llm_patch(
                            pre_files=base_files,
                            post_files=enhancement_result.enhanced_files,
                            pre_directory_files=directory_files,
                            post_directory_files=enhancement_result.enhanced_directory_files,
                        )
                        self.patch_generator.save_prompts_log()
                        logger.info("Saved post-LLM patch and prompts log")

                    return (
                        enhancement_result.enhanced_files,
                        enhancement_result.enhanced_directory_files,
                    )
                else:
                    logger.warning(
                        f"⚠️ AI enhancement failed, using template base: {', '.join(enhancement_result.errors)}"
                    )
                    # Still save prompts log even on failure
                    if self.patch_generator:
                        self.patch_generator.save_prompts_log()
            else:
                logger.info("📋 Using template-based generation only")

            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"⏱️ Generation completed in {processing_time:.2f}s")

            return base_files, directory_files

        except Exception as e:
            logger.error(f"Hybrid generation failed line 267 : {e}")
            # Save prompts log even on error
            if self.patch_generator:
                self.patch_generator.save_prompts_log()

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
        """
        Enhance the main locustfile with AI-generated improvements.

        Args:
            base_content: The template-generated locustfile content.
            endpoints: List of API endpoints to consider for enhancement.
            api_info: Dictionary containing API metadata (title, description, etc.).

        Returns:
            Enhanced locustfile content if successful and valid, otherwise
            falls back to base_content.
        """
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
            enhanced_content = await self._call_ai_service(
                prompt, file_context="locustfile.py"
            )

            # Validate the generated Python code
            if enhanced_content and self._validate_python_code(enhanced_content):
                return enhanced_content

            logger.warning(
                "Enhanced locustfile failed syntax validation, using base content"
            )
            return base_content
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return base_content

    async def _enhance_with_ai(
        self,
        base_files: Dict[str, str],
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        include_auth: bool,
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
                include_auth,
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
        include_auth: bool,
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
                base_files, directory_files, grouped_endpoints, db_type, include_auth
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
        """
        Generate domain-specific user flows using AI.

        Analyzes the API domain and generates custom Locust user flow classes
        tailored to the specific business domain (e.g., e-commerce, user management).

        Args:
            endpoints: List of API endpoints to analyze.
            api_info: Dictionary containing API metadata.
            custom_requirement: Optional custom requirements from the user.

        Returns:
            Generated domain flow Python code if successful and valid,
            empty string otherwise.
        """
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

            enhanced_content = await self._call_ai_service(
                prompt, file_context="domain_flows.py"
            )

            # Validate the generated Python code
            if enhanced_content and self._validate_python_code(enhanced_content):
                return enhanced_content

            logger.warning(
                "Generated domain flows failed syntax validation, skipping"
            )
        except Exception as e:
            logger.warning(f"Domain flows generation failed: {e}")

        return ""

    def get_files_by_key(
        self, directory_files: List[Dict[str, Any]], target_key: str
    ) -> List[Dict[str, Any]]:
        """Return directory items that contain the specified key"""
        return [items for items in directory_files if target_key in items]

    def _batch_endpoints(
        self, endpoints: List[Endpoint]
    ) -> List[List[Endpoint]]:
        """
        Split endpoints into batches to prevent LLM output truncation.

        Large endpoint lists can cause the LLM to generate responses that exceed
        the max_tokens limit, resulting in truncated (and syntactically invalid)
        Python code. This method splits endpoints into manageable batches.

        Args:
            endpoints: List of endpoints to batch.

        Returns:
            List of endpoint batches, each containing at most ENDPOINT_BATCH_SIZE
            endpoints.
        """
        if not endpoints:
            return []
        return [
            endpoints[i : i + ENDPOINT_BATCH_SIZE]
            for i in range(0, len(endpoints), ENDPOINT_BATCH_SIZE)
        ]

    async def _enhance_workflows(
        self,
        base_content: str,
        test_data_content: str,
        base_workflow: str,
        grouped_enpoints: Dict[str, List[Endpoint]],
        auth_endpoints: List[Endpoint],
        db_type: str = "",
        template_path: str = workflow_jinja_path,
    ) -> Optional[str]:
        """
        Enhance workflow files with AI-generated improvements.

        If the endpoint list is large, it will be processed in batches to prevent
        LLM output truncation. Each batch generates a separate workflow class
        (e.g., UsersWorkflowPart1, UsersWorkflowPart2).

        Args:
            base_content: The template-generated workflow content.
            test_data_content: Content of the test_data.py file.
            base_workflow: Content of the base_workflow.py file.
            grouped_enpoints: Dictionary mapping workflow names to endpoint lists.
            auth_endpoints: List of authentication-related endpoints.
            db_type: Database type for data provider integration.
            template_path: Path to the Jinja2 template.

        Returns:
            Enhanced workflow content if successful and valid, empty string otherwise.
        """
        try:
            template = self.jinja_env.get_template(template_path)

            # Get the workflow key and endpoints
            workflow_key = list(grouped_enpoints.keys())[0] if grouped_enpoints else ""
            endpoints = grouped_enpoints.get(workflow_key, [])

            # Check if batching is needed
            if len(endpoints) <= ENDPOINT_BATCH_SIZE:
                # No batching needed - process normally
                return await self._enhance_workflow_single(
                    template=template,
                    base_content=base_content,
                    test_data_content=test_data_content,
                    base_workflow=base_workflow,
                    grouped_enpoints=grouped_enpoints,
                    auth_endpoints=auth_endpoints,
                    db_type=db_type,
                )

            # Batching needed - process in batches and generate separate classes
            return await self._enhance_workflow_batched(
                template=template,
                base_content=base_content,
                test_data_content=test_data_content,
                base_workflow=base_workflow,
                workflow_key=workflow_key,
                endpoints=endpoints,
                auth_endpoints=auth_endpoints,
                db_type=db_type,
            )

        except Exception as e:
            logger.warning(f"Workflow enhancement failed: {e}")

        return ""

    async def _enhance_workflow_single(
        self,
        template,
        base_content: str,
        test_data_content: str,
        base_workflow: str,
        grouped_enpoints: Dict[str, List[Endpoint]],
        auth_endpoints: List[Endpoint],
        db_type: str,
    ) -> Optional[str]:
        """
        Enhance a workflow with a single LLM call (no batching).

        Args:
            template: Jinja2 template object.
            base_content: The template-generated workflow content.
            test_data_content: Content of the test_data.py file.
            base_workflow: Content of the base_workflow.py file.
            grouped_enpoints: Dictionary mapping workflow names to endpoint lists.
            auth_endpoints: List of authentication-related endpoints.
            db_type: Database type for data provider integration.

        Returns:
            Enhanced workflow content if valid, empty string otherwise.
        """
        # Get workflow key for logging
        workflow_key = list(grouped_enpoints.keys())[0] if grouped_enpoints else "unknown"

        prompt = template.render(
            grouped_enpoints=grouped_enpoints,
            test_data_content=test_data_content,
            base_workflow=base_workflow,
            auth_endpoints=auth_endpoints,
            base_content=base_content,
            db_type=db_type,
        )

        enhanced_content = await self._call_ai_service(
            prompt, file_context=f"workflows/{workflow_key}_workflow.py"
        )

        # Validate the generated Python code
        if enhanced_content and self._validate_python_code(enhanced_content):
            return enhanced_content

        logger.warning(
            "Enhanced workflow failed syntax validation, falling back to template"
        )
        return ""

    async def _enhance_workflow_batched(
        self,
        template,
        base_content: str,
        test_data_content: str,
        base_workflow: str,
        workflow_key: str,
        endpoints: List[Endpoint],
        auth_endpoints: List[Endpoint],
        db_type: str,
    ) -> Optional[str]:
        """
        Enhance a workflow by processing endpoints in batches.

        Generates separate workflow classes for each batch (e.g., UsersWorkflowPart1,
        UsersWorkflowPart2) to prevent LLM output truncation.

        Args:
            template: Jinja2 template object.
            base_content: The template-generated workflow content.
            test_data_content: Content of the test_data.py file.
            base_workflow: Content of the base_workflow.py file.
            workflow_key: The name/key of the workflow being enhanced.
            endpoints: Full list of endpoints to process.
            auth_endpoints: List of authentication-related endpoints.
            db_type: Database type for data provider integration.

        Returns:
            Combined workflow content with multiple classes, or empty string on failure.
        """
        batches = self._batch_endpoints(endpoints)
        all_parts = []
        batch_size = ENDPOINT_BATCH_SIZE

        for batch_idx, batch in enumerate(batches, start=1):
            logger.info(
                f"Processing batch {batch_idx}/{len(batches)} for {workflow_key} "
                f"({len(batch)} endpoints)"
            )

            batch_grouped = {workflow_key: batch}

            prompt = template.render(
                grouped_enpoints=batch_grouped,
                test_data_content=test_data_content,
                base_workflow=base_workflow,
                auth_endpoints=auth_endpoints if batch_idx == 1 else [],
                base_content=base_content if batch_idx == 1 else "",
                db_type=db_type,
            )

            enhanced_content = await self._call_ai_service(
                prompt, file_context=f"workflows/{workflow_key}_workflow_batch{batch_idx}.py"
            )

            # Validate with retry using smaller batches
            if not enhanced_content or not self._validate_python_code(enhanced_content):
                logger.warning(
                    f"Batch {batch_idx} failed validation, retrying with smaller batch"
                )
                enhanced_content = await self._retry_with_smaller_batch(
                    template=template,
                    batch=batch,
                    workflow_key=workflow_key,
                    test_data_content=test_data_content,
                    base_workflow=base_workflow,
                    auth_endpoints=auth_endpoints if batch_idx == 1 else [],
                    base_content=base_content if batch_idx == 1 else "",
                    db_type=db_type,
                    current_batch_size=batch_size,
                )

            if enhanced_content:
                # Rename class to include part number if multiple batches
                if len(batches) > 1:
                    enhanced_content = self._rename_workflow_class(
                        enhanced_content, workflow_key, batch_idx
                    )
                all_parts.append(enhanced_content)
            else:
                logger.warning(
                    f"Batch {batch_idx} failed completely, skipping"
                )

        if not all_parts:
            logger.error("All batches failed, returning empty")
            return ""

        # Combine all parts into a single file
        return self._combine_workflow_parts(all_parts)

    async def _retry_with_smaller_batch(
        self,
        template,
        batch: List[Endpoint],
        workflow_key: str,
        test_data_content: str,
        base_workflow: str,
        auth_endpoints: List[Endpoint],
        base_content: str,
        db_type: str,
        current_batch_size: int,
    ) -> Optional[str]:
        """
        Retry workflow enhancement with progressively smaller batches.

        If validation fails, this method halves the batch size and retries until
        either success or the batch size reaches 1. If all retries fail, returns
        empty string (caller should fall back to template).

        Args:
            template: Jinja2 template object.
            batch: Current batch of endpoints.
            workflow_key: The name/key of the workflow.
            test_data_content: Content of the test_data.py file.
            base_workflow: Content of the base_workflow.py file.
            auth_endpoints: List of authentication endpoints.
            base_content: Base workflow content.
            db_type: Database type.
            current_batch_size: Current batch size.

        Returns:
            Valid enhanced content or empty string.
        """
        batch_size = current_batch_size // 2
        results = []

        while batch_size >= 1:
            logger.info(f"Retrying with batch size: {batch_size}")
            sub_batches = [
                batch[i : i + batch_size]
                for i in range(0, len(batch), batch_size)
            ]

            all_valid = True
            batch_results = []

            for sub_batch in sub_batches:
                sub_grouped = {workflow_key: sub_batch}
                prompt = template.render(
                    grouped_enpoints=sub_grouped,
                    test_data_content=test_data_content,
                    base_workflow=base_workflow,
                    auth_endpoints=auth_endpoints,
                    base_content=base_content,
                    db_type=db_type,
                )

                content = await self._call_ai_service(
                    prompt, file_context=f"workflows/{workflow_key}_workflow_retry.py"
                )
                if content and self._validate_python_code(content):
                    batch_results.append(content)
                else:
                    all_valid = False
                    break

            if all_valid and batch_results:
                return self._combine_workflow_parts(batch_results)

            batch_size = batch_size // 2

        logger.error("All retry attempts failed")
        return ""

    def _rename_workflow_class(
        self, content: str, workflow_key: str, part_num: int
    ) -> str:
        """
        Rename workflow class to include part number.

        Transforms class names like 'UsersWorkflow' to 'UsersWorkflowPart1' to
        allow multiple classes in the same file without naming conflicts.

        Args:
            content: The Python code content.
            workflow_key: The workflow name/key.
            part_num: The part number (1-indexed).

        Returns:
            Content with renamed class.
        """
        # Common class name patterns to look for
        class_patterns = [
            f"class {workflow_key.title().replace('_', '')}Workflow",
            f"class {workflow_key.title()}Workflow",
            f"class {workflow_key.replace('_', ' ').title().replace(' ', '')}Workflow",
        ]

        for pattern in class_patterns:
            if pattern in content:
                new_name = f"{pattern}Part{part_num}"
                content = content.replace(pattern, new_name, 1)
                break

        return content

    def _combine_workflow_parts(self, parts: List[str]) -> str:
        """
        Combine multiple workflow parts into a single file.

        Deduplicates imports and combines class definitions.

        Args:
            parts: List of workflow code parts to combine.

        Returns:
            Combined Python code as a single string.
        """
        if not parts:
            return ""

        if len(parts) == 1:
            return parts[0]

        # Extract imports from all parts
        all_imports = set()
        class_definitions = []

        for part in parts:
            lines = part.split('\n')
            imports = []
            code_lines = []
            in_imports = True

            for line in lines:
                stripped = line.strip()
                if in_imports and (
                    stripped.startswith('import ')
                    or stripped.startswith('from ')
                    or stripped == ''
                    or stripped.startswith('#')
                ):
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        all_imports.add(line)
                    elif stripped.startswith('#') and not code_lines:
                        imports.append(line)
                else:
                    in_imports = False
                    code_lines.append(line)

            if code_lines:
                class_definitions.append('\n'.join(code_lines))

        # Build combined content
        sorted_imports = sorted(all_imports)
        combined = '\n'.join(sorted_imports)
        combined += '\n\n'
        combined += '\n\n'.join(class_definitions)

        return combined

    async def enhance_test_data_file(
        self,
        base_content: str,
        endpoints: List[Endpoint],
        db_type: str = "",
        data_provider: str = "",
        db_config: str = "",
        data_provider_path: str = "",
    ) -> Optional[str]:
        """
        Enhance test data generation with domain knowledge.

        Uses AI to improve the test data file with more realistic and
        domain-appropriate data generators based on the API schemas.

        Args:
            base_content: The template-generated test_data.py content.
            endpoints: List of API endpoints to analyze for data patterns.
            db_type: Database type (e.g., 'mongo') for data provider integration.
            data_provider: Content of the data_provider.py file.
            db_config: Content of the db_config.py file.
            data_provider_path: Path to the data provider module.

        Returns:
            Enhanced test data content if successful and valid, empty string otherwise.
        """
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
            enhanced_content = await self._call_ai_service(
                prompt, file_context="test_data.py"
            )
            if enhanced_content and self._validate_python_code(enhanced_content):
                return enhanced_content
        except Exception as e:
            logger.warning(f"Test data enhancement failed: {e}")

        return ""

    async def _enhance_validation(
        self, base_content: str, endpoints: List[Endpoint]
    ) -> Optional[str]:
        """
        Enhance response validation utilities with endpoint-specific checks.

        Generates improved validation functions based on the API response schemas
        defined in the endpoints.

        Args:
            base_content: The template-generated utils.py content.
            endpoints: List of API endpoints to analyze for validation patterns.

        Returns:
            Enhanced validation content if successful and valid, empty string otherwise.
        """
        validation_patterns = self._extract_validation_patterns(endpoints)
        try:
            template = self.jinja_env.get_template("validation.j2")

            # Render enhanced content
            prompt = template.render(
                base_content=base_content, validation_patterns=validation_patterns
            )
            enhanced_content = await self._call_ai_service(
                prompt, file_context="utils.py"
            )

            # Validate the generated Python code
            if enhanced_content and self._validate_python_code(enhanced_content):
                return enhanced_content

            logger.warning(
                "Enhanced validation failed syntax validation, skipping"
            )
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

    async def _make_api_call(self, messages: list[dict]) -> Optional[str]:
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
                # Clean up the response
                content = self._clean_ai_response(
                    self.extract_code_from_response(content)
                )
                return content

        return None

    async def _call_ai_service(
        self, prompt: str, file_context: str = "unknown"
    ) -> Optional[str]:
        """
        Call AI service with retry logic and validation.

        Args:
            prompt: The prompt to send to the LLM.
            file_context: Context about which file is being enhanced (for diagnostics).

        Returns:
            The LLM response content, or empty string on failure.
        """
        # Log prompt for diagnostics if enabled
        if self.patch_generator:
            self.patch_generator.log_prompt(file_context, prompt)

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
        """
        Validate Python code syntax using the compile() built-in.

        This is used to catch truncated or malformed code generated by the LLM
        before writing it to disk. Truncated code (due to max_tokens limits)
        typically results in SyntaxError.

        Args:
            content: Python source code to validate.

        Returns:
            True if the code compiles successfully, False if there's a SyntaxError.
        """
        try:
            compile(content, "<string>", "exec")
            return True
        except SyntaxError:
            return False
