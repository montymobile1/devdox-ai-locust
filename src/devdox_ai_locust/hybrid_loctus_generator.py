"""
Hybrid Locust Test Generator

Combines reliable template-based generation with LLM enhancement for creativity
and domain-specific optimizations.
"""

import os
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from dataclasses import dataclass
import uuid
import shutil


from devdox_ai_locust.utils.open_ai_parser import (
    Endpoint
)
from devdox_ai_locust.locust_generator import LocustTestGenerator, TestDataConfig

logger = logging.getLogger(__name__)


@dataclass
class AIEnhancementConfig:
    """Configuration for AI enhancement"""

    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 60
    enhance_workflows: bool = True
    enhance_test_data: bool = True
    enhance_validation: bool = False
    create_domain_flows: bool = False
    update_main_locust: bool = True


@dataclass
class EnhancementResult:
    """Result of AI enhancement"""

    success: bool
    enhanced_files: Dict[str, str]
    enhanced_directory_files  :List[Dict[str, Any]]
    enhancements_applied: List[str]
    errors: List[str]
    processing_time: float


class EnhancementProcessor:
    """Handles individual enhancement operations"""

    def __init__(self, ai_config, locust_generator):
        self.ai_config = ai_config
        self.locust_generator = locust_generator

    async def process_main_locust_enhancement(self, base_files: Dict[str, str],
                                              endpoints: List[Endpoint], api_info: Dict[str, Any]) -> Tuple[
        Dict[str, str], List[str]]:
        """Process main locustfile enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config.update_main_locust:
            enhanced_content = await self.locust_generator._enhance_locustfile(
                base_files.get("locustfile.py", ""), endpoints, api_info
            )
            if enhanced_content:
                enhanced_files['locustfile.py'] = enhanced_content
                enhancements.append("main_locust_update")

        return enhanced_files, enhancements

    async def process_domain_flows_enhancement(self, base_files: Dict[str, str],
                                               endpoints: List[Endpoint], api_info: Dict[str, Any]) -> Tuple[
        Dict[str, str], List[str]]:
        """Process domain flows enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config.create_domain_flows:
            domain_flows = await self.locust_generator._generate_domain_flows(endpoints, api_info)
            if domain_flows:
                enhanced_files["custom_flows.py"] = domain_flows
                enhancements.append("domain_flows")

        return enhanced_files, enhancements

    async def process_workflow_enhancements(self, base_files: Dict[str, str],
                                            directory_files: List[Dict[str, Any]],
                                            grouped_endpoints: Dict[str, List[Endpoint]]) -> Tuple[
        List[Dict[str, Any]], List[str]]:
        """Process workflow enhancements"""
        enhanced_directory_files = []
        enhancements = []

        if not self.ai_config.enhance_workflows:
            return enhanced_directory_files, enhancements

        base_workflow_files = self.locust_generator.get_files_by_key(directory_files, 'base_workflow.py')

        for workflow_item in directory_files:
            enhanced_workflow_item = await self._enhance_single_workflow(
                workflow_item, base_files, base_workflow_files, grouped_endpoints
            )
            if enhanced_workflow_item:
                enhanced_directory_files.append(enhanced_workflow_item['files'])
                enhancements.extend(enhanced_workflow_item['enhancements'])

        return enhanced_directory_files, enhancements

    async def _enhance_single_workflow(self, workflow_item: Dict[str, Any],
                                       base_files: Dict[str, str], base_workflow_files: str,
                                       grouped_endpoints: Dict[str, List[Endpoint]]) -> Dict[str, Any]:
        """Enhance a single workflow file"""
        for key, value in workflow_item.items():
            workflow_key = key.replace("_workflow.py", "")
            endpoints_for_workflow = grouped_endpoints.get(workflow_key, [])
            auth_endpoints = grouped_endpoints.get('Authentication', [])

            enhanced_workflow = await self.locust_generator._enhance_workflows(
                base_content=value,
                test_data_content=base_files.get("test_data.py", ""),
                base_workflow=base_workflow_files,
                grouped_enpoints=endpoints_for_workflow,
                auth_endpoints=auth_endpoints
            )

            if enhanced_workflow:
                return {
                    'files': {key: enhanced_workflow},
                    'enhancements': [f"enhanced_workflows_{key}"]
                }

        return None

    async def process_test_data_enhancement(self, base_files: Dict[str, str],
                                            endpoints: List[Endpoint], api_info: Dict[str, Any]) -> Tuple[
        Dict[str, str], List[str]]:
        """Process test data enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config.enhance_test_data:
            enhanced_test_data = await self.locust_generator.enhance_test_data_file(
                base_files.get("test_data.py", ""), endpoints, api_info
            )
            if enhanced_test_data:
                enhanced_files["test_data.py"] = enhanced_test_data
                enhancements.append("smart_test_data")

        return enhanced_files, enhancements

    async def process_validation_enhancement(self, base_files: Dict[str, str],
                                             endpoints: List[Endpoint], api_info: Dict[str, Any]) -> Tuple[
        Dict[str, str], List[str]]:
        """Process validation enhancement"""
        enhanced_files = {}
        enhancements = []

        if self.ai_config.enhance_validation:
            enhanced_validation = await self.locust_generator._enhance_validation(
                base_files.get("utils.py", ""), endpoints, api_info
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
        ai_client=None,
        ai_config: AIEnhancementConfig = None,
        test_config: TestDataConfig = None,
        prompt_dir:str = "prompt"
    ):
        self.ai_client = ai_client
        self.ai_config = ai_config or AIEnhancementConfig()
        self.template_generator = LocustTestGenerator(test_config)
        self.enhancement_cache = {}
        self.prompt_dir =  self._find_project_root() /prompt_dir
        self._setup_jinja_env()


    def _find_project_root(self) -> Path:
        """Find the project root by looking for setup.py, pyproject.toml, or .git"""
        current_path = Path(__file__).parent

        return current_path

    def _setup_jinja_env(self):
        """Setup Jinja2 environment with custom filters"""
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False
        )


    async def generate_from_endpoints(
        self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        output_dir: str = "locust_tests",
    ) -> Dict[str, str]:
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
            base_files, directory_files, grouped_enpoints = self.template_generator.generate_from_endpoints(
                endpoints, api_info, output_dir
            )

            #directory_files = self.template_generator.fix_indent(directory_files)
            base_files = self.template_generator.fix_indent(base_files)

            # Step 2: Enhance with AI if available
            if self.ai_client and self._should_enhance(endpoints, api_info):

                logger.info("🤖 Enhancing tests with AI...")
                enhancement_result = await self._enhance_with_ai(
                    base_files, endpoints, api_info, directory_files, grouped_enpoints
                )

                if enhancement_result.success:
                    logger.info(
                        f"✅ AI enhancements applied: {', '.join(enhancement_result.enhancements_applied)}"
                    )
                    return enhancement_result.enhanced_files, enhancement_result.enhanced_directory_files
                else:
                    logger.warning(
                        f"⚠️ AI enhancement failed, using template base: {', '.join(enhancement_result.errors)}"
                    )
            else:
                logger.info("📋 Using template-based generation only")

            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"⏱️ Generation completed in {processing_time:.2f}s")

            return base_files,directory_files

        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}")
            # Fallback to template-only
            return self.template_generator.generate_from_endpoints(
                endpoints, api_info, output_dir
            ),[]

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
                template = self.jinja_env.get_template('locust.j2')

                # Prepare context for template
                context = {
                    'base_content': base_content,
                    'endpoints_for_prompt': self._format_endpoints_for_prompt(endpoints[:5]),
                    'api_info': api_info

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
            grouped_endpoints: Dict[str, List[Endpoint]]
    ) -> EnhancementResult:
        """Enhance base files with AI - Refactored for reduced cognitive complexity"""
        start_time = asyncio.get_event_loop().time()

        try:
            enhancement_result = await self._process_all_enhancements(
                base_files, endpoints, api_info, directory_files, grouped_endpoints
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
            grouped_endpoints: Dict[str, List[Endpoint]]
    ) -> EnhancementResult:
        """Process all enhancements using the enhancement processor"""
        processor = EnhancementProcessor(self.ai_config, self)

        enhanced_files = base_files.copy()
        enhanced_directory_files = []
        enhancements_applied = []
        errors = []

        # Process each enhancement type
        enhancement_tasks = [
            processor.process_main_locust_enhancement(base_files, endpoints, api_info),
            processor.process_domain_flows_enhancement(base_files, endpoints, api_info),
            processor.process_test_data_enhancement(base_files, endpoints, api_info),
            processor.process_validation_enhancement(base_files, endpoints, api_info),
        ]

        # Execute file-based enhancements concurrently
        file_enhancement_results = await asyncio.gather(*enhancement_tasks, return_exceptions=True)

        # Process results from file-based enhancements
        for result in file_enhancement_results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue

            files, enhancements = result
            enhanced_files.update(files)
            enhancements_applied.extend(enhancements)

        # Process workflow enhancements separately (more complex logic)
        try:
            workflow_files, workflow_enhancements = await processor.process_workflow_enhancements(
                base_files, directory_files, grouped_endpoints
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
            processing_time=0  # Will be set by caller
        )

    async def _generate_domain_flows(
        self, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> Optional[str]:
        """Generate domain-specific user flows"""

        # Analyze endpoints to determine domain
        domain_analysis = self._analyze_api_domain(endpoints, api_info)


        try:
            template = self.jinja_env.get_template('workflow.j2')

            # Render enhanced content
            prompt = template.render(domain_analysis=domain_analysis,
                                     endpoints=self._format_endpoints_for_prompt(endpoints)
                                     )

            enhanced_content = await self._call_ai_service(prompt)

            if enhanced_content :
                return enhanced_content
        except Exception as e:
            logger.warning(f"Domain flows generation failed: {e}")

        return ""

    def get_files_by_key(self,directory_files, target_key):
        """Return directory items that contain the specified key"""
        return [items for items in directory_files if target_key in items]


    async def _enhance_workflows(
        self, base_content: str, test_data_content: str,base_workflow:str, grouped_enpoints: Dict[str, List[Endpoint]],auth_endpoints: List[Endpoint]
    ) -> Optional[str]:


        try:
            template = self.jinja_env.get_template('workflow.j2')


            # Render enhanced content
            prompt = template.render(grouped_enpoints=grouped_enpoints,
                                     test_data_content=test_data_content,
                                      base_workflow=base_workflow,
                                      auth_endpoints=auth_endpoints,

                                     base_content=base_content
                                     )
            enhanced_content = await self._call_ai_service(prompt)
            return enhanced_content
        except Exception as e:
            logger.warning(f"Workflow enhancement failed: {e}")

        return ""

    async def enhance_test_data_file(
        self, base_content: str, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> Optional[str]:
        """Enhance test data generation with domain knowledge"""

        # Extract schema information
        schemas_info = self._extract_schema_patterns(endpoints)


        try:
            template = self.jinja_env.get_template('test_data.j2')

            # Prepare context for template
            context = {
                'base_content': base_content,
                'schemas_info': schemas_info,
                'endpoints': endpoints

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
        self, base_content: str, endpoints: List[Endpoint], api_info: Dict[str, Any]
    ) -> Optional[str]:
        """Enhance response validation with endpoint-specific checks"""

        validation_patterns = self._extract_validation_patterns(endpoints)
        try:
            template = self.jinja_env.get_template('validation.j2')

            # Render enhanced content
            prompt = template.render(base_content=base_content,
                                     validation_patterns=validation_patterns

                                     )
            enhanced_content = await self._call_ai_service(prompt)
            if enhanced_content :
                return enhanced_content
        except Exception as e:
            logger.warning(f"Validation enhancement failed: {e}")

        return ""



    async def _call_ai_service(self, prompt: str) -> Optional[str]:
        """Call AI service with retry logic and validation"""


        messages = [
            {
                "role": "system",
                "content": "You are an expert Python developer specializing in Locust load testing. Generate clean, production-ready code with proper error handling. "
                           "Always return your code wrapped in <code></code> tags with no explanations outside the tags and DO NOT TRUNCATE THE CODE. "
                           "Format: <code>your_python_code_here</code>",
            },
            {"role": "user", "content": prompt},
        ]



        for attempt in range(3):  # Retry logic
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.ai_client.chat.completions.create,
                        model=self.ai_config.model,
                        messages=messages,
                        max_tokens=self.ai_config.max_tokens,
                        temperature=self.ai_config.temperature,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.1,
                    ),
                    timeout=self.ai_config.timeout,
                )

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content.strip()


                    # Clean up the response
                    content = self._clean_ai_response(self.extract_code_from_response(content))

                    if content:
                        return content

            except asyncio.TimeoutError:
                logger.warning(f"AI service timeout on attempt {attempt + 1}")

            except Exception as e:
                logger.warning(f"AI service error on attempt {attempt + 1}: {e}")


            if attempt < 2:  # Wait before retry
                await asyncio.sleep(2**attempt)

        return ""

    def extract_code_from_response(self,response_text):
        # Extract content between <code> tags

        code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()
            # Additional validation - ensure we got actual content
            if content and len(content) > 0:
                return content


        return response_text.strip()


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
                        pattern += f" (schema validation needed)"
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

        return sorted(list(resources))

    async def _create_test_files_safely(
        self,
        test_files: Dict[str, str],
        output_path: Path,
        max_file_size: int = 1024 * 1024,  # 1MB limit
    ) -> list:
        """
        Create test files safely with comprehensive security and error handling
        """
        created_files = []
        temp_dir = output_path / f"temp_{uuid.uuid4().hex[:8]}"

        # Security validation
        allowed_extensions = {
            ".py",
            ".md",
            ".txt",
            ".sh",
            ".yml",
            ".yaml",
            ".json",
            ".example",
        }

        try:
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

            for filename, content in test_files.items():
                try:
                    # Security checks
                    clean_filename = self._sanitize_filename(filename)

                    file_extension = Path(clean_filename).suffix.lower()


                    if file_extension not in allowed_extensions:
                        logger.warning(
                            f"⚠️ Skipping file with disallowed extension: {filename}"
                        )
                        continue

                    if len(content.encode("utf-8")) > max_file_size:
                        logger.warning(f"⚠️ File too large, truncating: {filename}")
                        content = content[: max_file_size // 2]  # Safe truncation



                    # Create file in temp directory first (atomic operation)
                    temp_file_path = temp_dir / clean_filename
                    await asyncio.to_thread(
                        temp_file_path.write_text, content, encoding="utf-8"
                    )

                    # Set appropriate permissions
                    if clean_filename.endswith(".sh"):
                        temp_file_path.chmod(0o755)  # Executable
                    else:
                        temp_file_path.chmod(0o644)  # Read/write

                    created_files.append(
                        {
                            "filename": clean_filename,
                            "temp_path": temp_file_path,
                            "final_path": output_path / clean_filename,
                            "size": len(content.encode("utf-8")),
                            "type": file_extension.lstrip("."),
                        }
                    )

                    logger.info(f"📄 Prepared: {clean_filename} ({len(content)} chars)")

                except Exception as e:
                    logger.error(f"❌ Failed to prepare file {filename}: {e}")
                    continue

            # Atomic move to final location (all or nothing)
            if created_files:
                for file_info in created_files:
                    try:
                        await asyncio.to_thread(
                            shutil.move,
                            str(file_info["temp_path"]),
                            str(file_info["final_path"]),
                        )
                        file_info["path"] = file_info["final_path"]
                        logger.info(f"✅ Created: {file_info['filename']}")
                    except Exception as e:
                        logger.error(
                            f"❌ Failed to move file {file_info['filename']}: {e}"
                        )
                        # Remove from created_files if move failed
                        created_files = [f for f in created_files if f != file_info]

            return created_files

        except Exception as e:
            logger.error(f"❌ File creation process failed: {e}")
            return []

        finally:
            # Always clean up temp directory
            if temp_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent security issues"""
        # Remove directory components
        clean_name = os.path.basename(filename).lower()

        # Remove dangerous characters
        clean_name = re.sub(r'[<>:"/\\|?*]', "", clean_name)
        # Replace spaces with underscores
        clean_name = clean_name.replace("- ", "_")

        # Ensure reasonable length
        if len(clean_name) > 255:
            name_part, ext = os.path.splitext(clean_name)
            clean_name = name_part[:250] + ext

        # Prevent hidden files and ensure not empty
        safe_dotfiles = {".env.example", ".gitignore", ".env.template"}
        if not clean_name or (
            clean_name.startswith(".") and clean_name not in safe_dotfiles
        ):
            clean_name = f"generated_{uuid.uuid4().hex[:8]}.py"

        return clean_name

    def _validate_python_code(self, content: str) -> bool:
        """Validate Python code syntax"""
        try:
            compile(content, "<string>", "exec")
            return True
        except SyntaxError:
            return False
