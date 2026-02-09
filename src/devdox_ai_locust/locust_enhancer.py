"""
Locust Test Enhancer

Core orchestrator that enhances existing Locust test files with new AI-generated
test scenarios. Ties together the file analyzer, AI client, Jinja2 prompt template,
and code merger to produce enriched load-test files based on user requirements.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import black
import jinja2

from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest
from devdox_ai_locust.utils.ai_client import AIEnhancementConfig, TogetherAIClient
from devdox_ai_locust.utils.code_merger import LocustCodeMerger, MergeResult
from devdox_ai_locust.utils.locust_file_analyzer import (
    LocustFileAnalysis,
    LocustFileAnalyzer,
)
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser
from devdox_ai_locust.utils.swagger_utils import get_api_schema

logger = logging.getLogger(__name__)


@dataclass
class EnhanceResult:
    """Result of enhancing a Locust test file with new AI-generated scenarios."""

    success: bool
    enhanced_source: str
    original_source: str
    added_imports: List[str] = field(default_factory=list)
    added_tasks: List[str] = field(default_factory=list)
    added_classes: List[str] = field(default_factory=list)
    added_helpers: List[str] = field(default_factory=list)
    replaced_tasks: List[str] = field(default_factory=list)
    replaced_helpers: List[str] = field(default_factory=list)
    replaced_classes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class LocustTestEnhancer:
    """Enhances existing Locust test files with new AI-generated test scenarios.

    Orchestrates the full enhancement pipeline:

    1. **Analyze** the existing Locust file to understand its structure.
    2. Optionally **fetch an API schema** for additional context.
    3. **Render** a Jinja2 prompt with all context for the AI.
    4. **Call the AI** to generate new test scenarios.
    5. **Merge** the generated code into the existing file.
    6. **Format** the result with Black.

    Usage::

        enhancer = LocustTestEnhancer(together_api_key="...")
        result = await enhancer.enhance_file(
            "locustfile.py",
            "Add scenarios for user profile CRUD operations",
        )
        if result.success:
            print(result.enhanced_source)
    """

    SYSTEM_PROMPT = (
        "You are an expert Python developer specializing in Locust load testing "
        "frameworks. You generate high-quality, production-ready test scenarios."
    )

    _SECTION_KEYS = [
        "new_imports", "new_tasks", "new_classes", "new_helpers",
        "replace_tasks", "replace_helpers", "replace_classes",
    ]

    def __init__(
        self,
        together_api_key: str,
        ai_config: Optional[AIEnhancementConfig] = None,
        verbose: bool = False,
    ) -> None:
        self._api_key = together_api_key
        self._ai_config = ai_config or AIEnhancementConfig()
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)
        self._analyzer = LocustFileAnalyzer()
        self._template_env = self._setup_jinja_env()

    # ------------------------------------------------------------------
    # Jinja2 setup
    # ------------------------------------------------------------------

    def _setup_jinja_env(self) -> jinja2.Environment:
        """Set up a Jinja2 environment for the prompt templates.

        Templates live in the ``prompt/`` directory next to this module,
        following the same pattern used by ``HybridLocustGenerator``.
        """
        prompt_dir = Path(__file__).parent / "prompt"
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enhance_file(
        self,
        file_path: str,
        custom_requirement: str,
        swagger_url: Optional[str] = None,
    ) -> EnhanceResult:
        """Enhance an existing Locust test file with new test scenarios.

        This is the primary entry point. It reads, analyses, enhances,
        merges, and formats the file in a single call.

        Args:
            file_path: Path to the existing Locust test file.
            custom_requirement: Natural-language description of the
                scenarios to generate (the primary input for the AI).
            swagger_url: Optional URL to an OpenAPI/Swagger spec that
                provides additional API context for the AI.

        Returns:
            An ``EnhanceResult`` containing the enhanced source code and
            a summary of changes.
        """
        try:
            # 1. Analyze the existing file
            analysis = self._analyze_file(file_path)
            self._logger.info(
                "Analyzed %s: found %d classes, %d task methods",
                file_path,
                len(analysis.user_classes),
                sum(len(c.task_methods) for c in analysis.user_classes),
            )

            return await self._run_enhancement_pipeline(
                analysis, custom_requirement, swagger_url
            )

        except (FileNotFoundError, ValueError) as e:
            self._logger.error("Failed to analyze file: %s", e)
            original = ""
            try:
                original = await asyncio.to_thread(
                    Path(file_path).read_text, encoding="utf-8"
                )
            except IOError:
                pass
            return EnhanceResult(
                success=False,
                enhanced_source=original,
                original_source=original,
                error=str(e),
            )
        except Exception as e:
            self._logger.error("Enhancement failed: %s", e)
            original = analysis.raw_source if "analysis" in dir() else ""
            return EnhanceResult(
                success=False,
                enhanced_source=original,
                original_source=original,
                error=str(e),
            )

    async def enhance_source(
        self,
        source: str,
        custom_requirement: str,
        swagger_url: Optional[str] = None,
    ) -> EnhanceResult:
        """Enhance Locust test source code directly (without reading from file).

        Behaves identically to :meth:`enhance_file` but accepts an
        in-memory source string instead of a file path.

        Args:
            source: Existing Locust test source code.
            custom_requirement: Natural-language requirement for the AI.
            swagger_url: Optional OpenAPI/Swagger spec URL.

        Returns:
            An ``EnhanceResult`` with the enhanced source and summary.
        """
        try:
            analysis = self._analyze_source(source)
            self._logger.info(
                "Analyzed source: found %d classes, %d task methods",
                len(analysis.user_classes),
                sum(len(c.task_methods) for c in analysis.user_classes),
            )

            return await self._run_enhancement_pipeline(
                analysis, custom_requirement, swagger_url
            )

        except ValueError as e:
            self._logger.error("Failed to analyze source: %s", e)
            return EnhanceResult(
                success=False,
                enhanced_source=source,
                original_source=source,
                error=str(e),
            )
        except Exception as e:
            self._logger.error("Enhancement failed: %s", e)
            original = analysis.raw_source if "analysis" in dir() else source
            return EnhanceResult(
                success=False,
                enhanced_source=original,
                original_source=original,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _run_enhancement_pipeline(
        self,
        analysis: LocustFileAnalysis,
        custom_requirement: str,
        swagger_url: Optional[str] = None,
    ) -> EnhanceResult:
        """Execute the core enhancement pipeline shared by both public methods.

        Args:
            analysis: Pre-computed analysis of the existing Locust file.
            custom_requirement: The user's natural-language requirement.
            swagger_url: Optional OpenAPI spec URL for extra context.

        Returns:
            An ``EnhanceResult`` with the enhanced source and summary.
        """
        if self._verbose:
            self._log_verbose_analysis(analysis)

        # 1. Optionally fetch API schema
        api_schema_summary = await self._fetch_api_schema_context(swagger_url)

        # 2. Render the AI prompt
        prompt = self._render_prompt(
            analysis, custom_requirement, api_schema_summary
        )

        if self._verbose:
            self._logger.info(
                "[prompt] rendered prompt: %d chars (~%d tokens estimate)",
                len(prompt),
                len(prompt) // 4,
            )
            self._logger.debug("Rendered prompt:\n%s", prompt)

        # 3. Call AI to generate new scenarios
        new_sections = await self._generate_new_scenarios(prompt)

        if self._verbose:
            self._log_verbose_ai_sections(new_sections)

        # Check if AI returned anything useful
        if not any(new_sections.get(k, "").strip() for k in self._SECTION_KEYS):
            self._logger.warning(
                "[ai-response] all sections empty — nothing to merge"
            )
            return EnhanceResult(
                success=False,
                enhanced_source=analysis.raw_source,
                original_source=analysis.raw_source,
                warnings=["AI did not generate any new test scenarios."],
                error="No new scenarios generated by AI",
            )

        if self._verbose:
            self._log_verbose_merge_preview(new_sections)

        # 4. Merge new scenarios into existing file
        merge_result = self._merge_scenarios(analysis, new_sections)

        if self._verbose:
            self._log_verbose_merge_result(merge_result)

        # 5. Format with Black
        formatted_source = self._format_with_black(merge_result.merged_source)

        if self._verbose:
            self._log_verbose_format_result(analysis, formatted_source)

        return EnhanceResult(
            success=True,
            enhanced_source=formatted_source,
            original_source=analysis.raw_source,
            added_imports=merge_result.added_imports,
            added_tasks=merge_result.added_tasks,
            added_classes=merge_result.added_classes,
            added_helpers=merge_result.added_helpers,
            replaced_tasks=merge_result.replaced_tasks,
            replaced_helpers=merge_result.replaced_helpers,
            replaced_classes=merge_result.replaced_classes,
            warnings=merge_result.warnings,
            error=None,
        )

    # ------------------------------------------------------------------
    # Pipeline verbose-logging helpers
    # ------------------------------------------------------------------

    def _log_verbose_analysis(self, analysis: LocustFileAnalysis) -> None:
        """Log a detailed breakdown of the file analysis."""
        total_tasks = sum(len(c.task_methods) for c in analysis.user_classes)
        total_other = sum(len(c.other_methods) for c in analysis.user_classes)
        self._logger.info(
            "[analysis] %d class(es), %d @task method(s), "
            "%d other method(s), %d import(s), %d module-level function(s)",
            len(analysis.user_classes),
            total_tasks,
            total_other,
            len(analysis.imports),
            len(analysis.module_level_functions),
        )
        for cls in analysis.user_classes:
            parent_str = ", ".join(cls.parent_classes) or "none"
            task_names = [t.name for t in cls.task_methods]
            self._logger.info(
                "[analysis]   class %s (parents: %s) — tasks: %s",
                cls.name,
                parent_str,
                ", ".join(task_names) if task_names else "(none)",
            )
        self._logger.info(
            "[analysis] auth detected: %s, sequential tasks: %s",
            self._has_auth(analysis),
            self._has_sequential_tasks(analysis),
        )
        self._logger.info(
            "[analysis] source size: %d chars, %d lines",
            len(analysis.raw_source),
            analysis.raw_source.count("\n") + 1,
        )

    async def _fetch_api_schema_context(
        self, swagger_url: Optional[str]
    ) -> str:
        """Fetch an API schema summary if a URL is provided.

        Returns an empty string when no URL is given or the fetch fails.
        """
        if not swagger_url:
            if self._verbose:
                self._logger.info(
                    "[schema] no swagger URL provided — skipping"
                )
            return ""

        try:
            summary = await self._fetch_api_schema_summary(swagger_url)
            self._logger.info("Fetched API schema from %s", swagger_url)
            if self._verbose and summary:
                ep_count = summary.count("\n- ")
                self._logger.info(
                    "[schema] summary: %d chars, ~%d endpoint(s) listed",
                    len(summary),
                    ep_count,
                )
            return summary
        except Exception as e:
            self._logger.warning("Could not fetch API schema: %s", e)
            return ""

    def _log_verbose_ai_sections(self, new_sections: dict) -> None:
        """Log the size of each AI response section."""
        self._logger.info("[ai-response] section sizes:")
        for key in self._SECTION_KEYS:
            content = new_sections.get(key, "")
            char_count = len(content.strip()) if content else 0
            label = "EMPTY" if char_count == 0 else f"{char_count} chars"
            self._logger.info("  %-18s %s", key, label)

    def _log_verbose_merge_preview(self, new_sections: dict) -> None:
        """Log what the upcoming merge will do."""
        preview_parts = []
        for key in self._SECTION_KEYS:
            content = new_sections.get(key, "")
            if content and content.strip():
                action = "replace" if key.startswith("replace_") else "add"
                kind = key.split("_", 1)[1]
                preview_parts.append(f"{action} {kind}")
        self._logger.info(
            "[merge] about to: %s",
            ", ".join(preview_parts) if preview_parts else "(nothing)",
        )

    def _log_verbose_merge_result(self, merge_result: MergeResult) -> None:
        """Log the merge result summary and any warnings."""
        self._logger.info(
            "[merge] result: +%d imports, +%d tasks, +%d classes, "
            "+%d helpers, ~%d tasks replaced, ~%d helpers replaced, "
            "~%d classes replaced, %d warning(s)",
            len(merge_result.added_imports),
            len(merge_result.added_tasks),
            len(merge_result.added_classes),
            len(merge_result.added_helpers),
            len(merge_result.replaced_tasks),
            len(merge_result.replaced_helpers),
            len(merge_result.replaced_classes),
            len(merge_result.warnings),
        )
        for w in merge_result.warnings:
            self._logger.warning("[merge] %s", w)

    def _log_verbose_format_result(
        self, analysis: LocustFileAnalysis, formatted_source: str
    ) -> None:
        """Log the line-count difference after Black formatting."""
        orig_lines = analysis.raw_source.count("\n") + 1
        new_lines = formatted_source.count("\n") + 1
        diff = new_lines - orig_lines
        sign = "+" if diff >= 0 else ""
        self._logger.info(
            "[format] %d -> %d lines (%s%d), Black formatting applied",
            orig_lines,
            new_lines,
            sign,
            diff,
        )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _analyze_file(self, file_path: str) -> LocustFileAnalysis:
        """Read and analyze an existing Locust test file.

        Args:
            file_path: Filesystem path to the test file.

        Returns:
            A ``LocustFileAnalysis`` describing the file structure.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid Python syntax.
        """
        return self._analyzer.analyze(file_path)

    def _analyze_source(self, source: str) -> LocustFileAnalysis:
        """Analyze Locust test source code directly.

        Args:
            source: Python source code as a string.

        Returns:
            A ``LocustFileAnalysis`` describing the code structure.

        Raises:
            ValueError: If the source contains invalid Python syntax.
        """
        return self._analyzer.analyze_source(source)

    # ------------------------------------------------------------------
    # API schema fetching
    # ------------------------------------------------------------------

    async def _fetch_api_schema_summary(self, swagger_url: str) -> str:
        """Fetch and parse an OpenAPI spec, returning a concise summary.

        Uses :func:`swagger_utils.get_api_schema` and
        :class:`OpenAPIParser` to fetch, parse, and summarise the spec.
        The summary is kept brief so it serves as useful context for the
        AI without consuming too many tokens.

        Args:
            swagger_url: URL to the OpenAPI/Swagger specification.

        Returns:
            A concise multi-line string summarising the API.
        """
        request = SwaggerProcessingRequest(swagger_url=swagger_url)
        raw_schema = await get_api_schema(request)
        if not raw_schema:
            return ""

        parser = OpenAPIParser()
        parser.parse_schema(raw_schema)

        api_info = parser.get_schema_info()
        endpoints = parser.parse_endpoints()

        lines: List[str] = [
            f"API: {api_info.get('title', 'Unknown')} v{api_info.get('version', '?')}",
            f"Base URL: {api_info.get('base_url', 'http://localhost')}",
            "Endpoints:",
        ]
        for ep in endpoints:
            summary_part = f" - {ep.summary}" if ep.summary else ""
            lines.append(f"- {ep.method} {ep.path}{summary_part}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        analysis: LocustFileAnalysis,
        custom_requirement: str,
        api_schema_summary: str = "",
    ) -> str:
        """Render the ``enhance.j2`` template with all context variables.

        Args:
            analysis: The structural analysis of the existing file.
            custom_requirement: The user's requirement string.
            api_schema_summary: Optional API schema summary.

        Returns:
            The fully rendered prompt string ready for the AI.
        """
        template = self._template_env.get_template("enhance.j2")

        existing_task_names: List[str] = [
            task.name
            for cls in analysis.user_classes
            for task in cls.task_methods
        ]
        existing_class_names: List[str] = [
            cls.name for cls in analysis.user_classes
        ]
        existing_imports: List[str] = [
            imp.statement for imp in analysis.imports
        ]

        user_class_name = (
            analysis.user_classes[0].name
            if analysis.user_classes
            else "LocustUser"
        )

        # Extract source code for existing methods so the AI can see
        # what it may be upgrading.
        existing_task_source = self._extract_method_sources(analysis)
        existing_helper_source = self._extract_helper_sources(analysis)
        existing_class_source = self._extract_class_sources(analysis)

        return template.render(
            custom_requirement=custom_requirement,
            existing_source_code=analysis.raw_source,
            existing_task_names=existing_task_names,
            existing_class_names=existing_class_names,
            existing_imports=existing_imports,
            user_class_name=user_class_name,
            api_schema_summary=api_schema_summary,
            has_auth=self._has_auth(analysis),
            has_sequential_tasks=self._has_sequential_tasks(analysis),
            existing_task_source=existing_task_source,
            existing_helper_source=existing_helper_source,
            existing_class_source=existing_class_source,
        )

    # ------------------------------------------------------------------
    # AI generation
    # ------------------------------------------------------------------

    MAX_REPAIR_ATTEMPTS = 2

    async def _generate_new_scenarios(self, prompt: str) -> dict:
        """Call the AI and parse the structured response.

        Creates a :class:`TogetherAIClient` context, sends the rendered
        prompt, and uses :meth:`extract_tagged_sections` to parse out
        the ``<new_imports>``, ``<new_tasks>``, ``<new_classes>``, and
        ``<new_helpers>`` sections.

        If the generated code has syntax errors, it will be sent back to
        the AI for repair (up to MAX_REPAIR_ATTEMPTS times).

        Args:
            prompt: The fully rendered user prompt.

        Returns:
            A dict with keys ``"new_imports"``, ``"new_tasks"``,
            ``"new_classes"``, ``"new_helpers"`` -- each mapped to a
            string (possibly empty).
        """
        import time

        if self._verbose:
            self._logger.info("[ai] calling Together AI (%s)...", self._ai_config.model)

        t0 = time.monotonic()
        async with TogetherAIClient(self._api_key, self._ai_config) as client:
            # Use raw=True to preserve XML-tagged sections in response
            response = await client.call(self.SYSTEM_PROMPT, prompt, raw=True)
            elapsed = time.monotonic() - t0

            if self._verbose:
                self._logger.info(
                    "[ai] response received in %.1fs (%d chars)",
                    elapsed,
                    len(response) if response else 0,
                )
                self._logger.debug("Raw AI response:\n%s", response)

            if not response:
                self._logger.warning("AI returned an empty response")
                return {}

            sections = TogetherAIClient.extract_tagged_sections(response)
            result = {
                "new_imports": sections.get("new_imports", ""),
                "new_tasks": sections.get("new_tasks", ""),
                "new_classes": sections.get("new_classes", ""),
                "new_helpers": sections.get("new_helpers", ""),
                "replace_tasks": sections.get("replace_tasks", ""),
                "replace_helpers": sections.get("replace_helpers", ""),
                "replace_classes": sections.get("replace_classes", ""),
            }

            # Validate and repair if needed
            for attempt in range(self.MAX_REPAIR_ATTEMPTS):
                errors = self._validate_sections(result)
                if not errors:
                    break

                if self._verbose:
                    self._logger.info(
                        "[ai] syntax errors detected, requesting repair (attempt %d/%d)",
                        attempt + 1,
                        self.MAX_REPAIR_ATTEMPTS,
                    )
                    for section, error in errors.items():
                        self._logger.info("[ai]   %s: %s", section, error)

                # Send back to AI for repair
                result = await self._repair_sections(client, result, errors)

        return result

    def _validate_sections(self, sections: dict) -> dict:
        """Validate Python syntax in code sections.

        Args:
            sections: Dict with code sections (new_tasks, new_classes, etc.)

        Returns:
            Dict mapping section name to error message for sections with errors.
            Empty dict if all sections are valid.
        """
        import ast
        import textwrap

        errors = {}

        # Sections that contain code needing validation
        code_sections = [
            "new_tasks",
            "new_classes",
            "new_helpers",
            "replace_tasks",
            "replace_helpers",
            "replace_classes",
        ]

        for section_name in code_sections:
            code = textwrap.dedent(sections.get(section_name, "")).strip()
            if not code:
                continue

            # For tasks/methods, wrap in a dummy class to make it valid module-level code
            if "tasks" in section_name:
                # Wrap methods in a class for validation
                test_code = "class _ValidationWrapper:\n"
                for line in code.splitlines():
                    test_code += f"    {line}\n"
            else:
                test_code = code

            try:
                ast.parse(test_code)
            except SyntaxError as e:
                errors[section_name] = f"line {e.lineno}: {e.msg}"

        return errors

    async def _repair_sections(
        self,
        client: TogetherAIClient,
        sections: dict,
        errors: dict,
    ) -> dict:
        """Send broken code sections back to the AI for repair.

        Args:
            client: The TogetherAIClient instance.
            sections: Dict with code sections.
            errors: Dict mapping section names to error messages.

        Returns:
            Updated sections dict with repaired code.
        """
        # Build repair prompt
        repair_prompt = self._build_repair_prompt(sections, errors)

        if self._verbose:
            self._logger.debug("Repair prompt:\n%s", repair_prompt)

        response = await client.call(self.SYSTEM_PROMPT, repair_prompt, raw=True)

        if not response:
            self._logger.warning("AI returned empty response for repair request")
            return sections

        # Extract repaired sections
        repaired = TogetherAIClient.extract_tagged_sections(response)

        # Merge repaired sections back (only update sections that had errors)
        result = sections.copy()
        for section_name in errors:
            if section_name in repaired and repaired[section_name].strip():
                result[section_name] = repaired[section_name]
                if self._verbose:
                    self._logger.info(
                        "[ai] repaired section '%s' (%d chars)",
                        section_name,
                        len(repaired[section_name]),
                    )

        return result

    def _build_repair_prompt(self, sections: dict, errors: dict) -> str:
        """Build a prompt asking the AI to fix syntax errors.

        Args:
            sections: Dict with code sections.
            errors: Dict mapping section names to error messages.

        Returns:
            Repair prompt string.
        """
        lines = [
            "The following code sections have Python syntax errors that need to be fixed.",
            "Please return ONLY the corrected sections using the same XML tags.",
            "Common issues to fix:",
            "- Incomplete try/except blocks (every 'try:' needs an 'except:' or 'finally:')",
            "- Missing colons after function/class definitions",
            "- Incorrect indentation",
            "- Unclosed parentheses, brackets, or strings",
            "",
            "ERRORS FOUND:",
        ]

        for section_name, error_msg in errors.items():
            lines.append(f"- <{section_name}>: {error_msg}")

        lines.append("")
        lines.append("CODE TO FIX:")
        lines.append("")

        for section_name, error_msg in errors.items():
            code = sections.get(section_name, "")
            lines.append(f"<{section_name}>")
            lines.append(code)
            lines.append(f"</{section_name}>")
            lines.append("")

        lines.append("Return the fixed code in the same XML tags. Ensure all try blocks have except/finally clauses.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Code merging
    # ------------------------------------------------------------------

    def _merge_scenarios(
        self,
        analysis: LocustFileAnalysis,
        new_sections: dict,
    ) -> MergeResult:
        """Merge AI-generated sections into the existing Locust file.

        Delegates to :class:`LocustCodeMerger` which handles insertion
        of imports, helpers, task methods, and new classes in the
        correct locations.

        Args:
            analysis: The structural analysis of the existing file.
            new_sections: Dict with ``new_imports``, ``new_tasks``,
                ``new_classes``, and ``new_helpers`` strings.

        Returns:
            A ``MergeResult`` with the merged source and change summary.
        """
        merger = LocustCodeMerger(analysis)
        return merger.merge(
            new_imports=new_sections.get("new_imports", ""),
            new_tasks=new_sections.get("new_tasks", ""),
            new_classes=new_sections.get("new_classes", ""),
            new_helpers=new_sections.get("new_helpers", ""),
            replace_tasks=new_sections.get("replace_tasks", ""),
            replace_helpers=new_sections.get("replace_helpers", ""),
            replace_classes=new_sections.get("replace_classes", ""),
        )

    # ------------------------------------------------------------------
    # Black formatting
    # ------------------------------------------------------------------

    def _format_with_black(self, source: str) -> str:
        """Format source code with Black.

        Uses the same approach as
        :meth:`LocustTestGenerator.fix_indent`. If Black raises an
        error the unformatted source is returned and a warning is logged.

        Args:
            source: Python source code to format.

        Returns:
            The Black-formatted source, or the original source on failure.
        """
        try:
            return black.format_str(source, mode=black.Mode(line_length=88))
        except black.InvalidInput:
            self._logger.warning(
                "Black could not parse merged code; returning unformatted source"
            )
            return source
        except Exception as e:
            self._logger.warning("Black formatting failed: %s", e)
            return source

    # ------------------------------------------------------------------
    # Analysis inspection helpers
    # ------------------------------------------------------------------

    def _has_auth(self, analysis: LocustFileAnalysis) -> bool:
        """Check whether any class has authentication-related methods.

        Scans both task methods and non-task methods for names that
        indicate login, authentication, or token-handling logic.

        Args:
            analysis: The file analysis to inspect.

        Returns:
            ``True`` if auth-related methods are detected.
        """
        auth_indicators = {
            "login",
            "auth",
            "authenticate",
            "on_start",
            "get_token",
            "signin",
        }
        for cls in analysis.user_classes:
            all_method_names = [m.name for m in cls.task_methods] + list(
                cls.other_methods
            )
            for name in all_method_names:
                if name.lower() in auth_indicators:
                    return True
                # Also check partial matches (e.g. "do_login", "refresh_token")
                if any(indicator in name.lower() for indicator in auth_indicators):
                    return True
        return False

    def _has_sequential_tasks(self, analysis: LocustFileAnalysis) -> bool:
        """Check whether any class uses SequentialTaskSet or TaskSet.

        Args:
            analysis: The file analysis to inspect.

        Returns:
            ``True`` if a sequential/task-set parent is found.
        """
        sequential_parents = {"SequentialTaskSet", "TaskSet"}
        for cls in analysis.user_classes:
            if any(parent in sequential_parents for parent in cls.parent_classes):
                return True
        return False

    # ------------------------------------------------------------------
    # Source extraction helpers (for upgrade context)
    # ------------------------------------------------------------------

    def _extract_method_sources(
        self, analysis: LocustFileAnalysis
    ) -> dict:
        """Extract source code for each @task method from the raw source.

        Returns a dict mapping method name to its source code string.
        """
        sources: dict = {}
        lines = analysis.raw_source.splitlines()
        for cls in analysis.user_classes:
            for task in cls.task_methods:
                start = task.line_number - 1  # 0-based
                # Find the end by looking at the next item or class end
                end = cls.end_line_number
                sources[task.name] = "\n".join(lines[start:end]).rstrip()
        # Refine: use AST for precise boundaries
        try:
            import ast

            tree = ast.parse(analysis.raw_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in sources:
                    start = node.lineno - 1
                    # Include decorators
                    if node.decorator_list:
                        first_deco = min(
                            d.lineno for d in node.decorator_list
                        )
                        start = first_deco - 1
                    end_line = node.end_lineno if node.end_lineno else node.lineno
                    sources[node.name] = "\n".join(
                        lines[start:end_line]
                    ).rstrip()
        except (SyntaxError, Exception):
            pass
        return sources

    def _extract_helper_sources(
        self, analysis: LocustFileAnalysis
    ) -> dict:
        """Extract source code for module-level functions."""
        sources: dict = {}
        if not analysis.module_level_functions:
            return sources
        try:
            import ast

            lines = analysis.raw_source.splitlines()
            tree = ast.parse(analysis.raw_source)
            for node in ast.iter_child_nodes(tree):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name in analysis.module_level_functions
                ):
                    start = node.lineno - 1
                    if node.decorator_list:
                        first_deco = min(
                            d.lineno for d in node.decorator_list
                        )
                        start = first_deco - 1
                    end_line = node.end_lineno if node.end_lineno else node.lineno
                    sources[node.name] = "\n".join(
                        lines[start:end_line]
                    ).rstrip()
        except (SyntaxError, Exception):
            pass
        return sources

    def _extract_class_sources(
        self, analysis: LocustFileAnalysis
    ) -> dict:
        """Extract source code for each class."""
        sources: dict = {}
        lines = analysis.raw_source.splitlines()
        for cls in analysis.user_classes:
            start = cls.line_number - 1
            end = cls.end_line_number
            sources[cls.name] = "\n".join(lines[start:end]).rstrip()
        return sources

    # ------------------------------------------------------------------
    # New workflow generation (for uncovered tags)
    # ------------------------------------------------------------------

    async def generate_new_workflow(
        self,
        tag_name: str,
        tag_endpoints: list,
        custom_requirement: str,
        swagger_url: Optional[str] = None,
        reference_workflow_source: Optional[str] = None,
    ) -> EnhanceResult:
        """Generate a brand-new workflow file for an uncovered API tag.

        Uses the AI to create a complete Locust workflow module covering
        all endpoints for the given tag, influenced by the custom
        requirement and styled after an existing reference workflow.

        Args:
            tag_name: The API tag to generate a workflow for.
            tag_endpoints: List of ``Endpoint`` objects for this tag.
            custom_requirement: Natural-language requirement driving
                the test scenario generation.
            swagger_url: Optional OpenAPI/Swagger spec URL for
                additional API context.
            reference_workflow_source: Optional source code of an
                existing workflow file to use as a style reference.

        Returns:
            An ``EnhanceResult`` whose ``enhanced_source`` contains the
            complete workflow file content.  ``original_source`` will
            be empty since this is a new file.
        """
        try:
            if self._verbose:
                self._log_verbose_workflow_start(
                    tag_name, tag_endpoints, reference_workflow_source
                )

            api_schema_summary = await self._fetch_api_schema_context(
                swagger_url
            )
            existing_imports = self._extract_reference_imports(
                reference_workflow_source
            )

            # Render the prompt
            template = self._template_env.get_template(
                "enhance_new_workflow.j2"
            )
            prompt = template.render(
                custom_requirement=custom_requirement,
                tag_name=tag_name,
                tag_endpoints=tag_endpoints,
                api_schema_summary=api_schema_summary,
                reference_workflow_source=reference_workflow_source or "",
                existing_imports_in_suite=existing_imports,
            )

            if self._verbose:
                self._logger.info(
                    "[new-workflow] rendered prompt: %d chars (~%d tokens)",
                    len(prompt),
                    len(prompt) // 4,
                )

            # Call the AI
            async with TogetherAIClient(
                self._api_key, self._ai_config
            ) as client:
                response = await client.call(self.SYSTEM_PROMPT, prompt)

            if not response or not response.strip():
                self._logger.warning(
                    "[new-workflow] AI returned empty response for tag '%s'",
                    tag_name,
                )
                return EnhanceResult(
                    success=False,
                    enhanced_source="",
                    original_source="",
                    error=f"AI returned empty response for tag '{tag_name}'",
                )

            if self._verbose:
                self._logger.info(
                    "[new-workflow] AI response: %d chars", len(response)
                )

            # Clean and format
            source = TogetherAIClient.extract_code_from_response(response)
            source = TogetherAIClient.clean_response(source)
            formatted = self._format_with_black(source)

            if self._verbose:
                self._logger.info(
                    "[new-workflow] formatted output: %d lines, %d chars",
                    formatted.count("\n") + 1,
                    len(formatted),
                )

            return EnhanceResult(
                success=True,
                enhanced_source=formatted,
                original_source="",
                added_tasks=[],
                added_classes=[tag_name],
                warnings=[],
            )

        except Exception as e:
            self._logger.error(
                "Failed to generate workflow for tag '%s': %s",
                tag_name, e,
            )
            return EnhanceResult(
                success=False,
                enhanced_source="",
                original_source="",
                error=str(e),
            )

    # ------------------------------------------------------------------
    # New workflow helpers
    # ------------------------------------------------------------------

    def _log_verbose_workflow_start(
        self,
        tag_name: str,
        tag_endpoints: list,
        reference_workflow_source: Optional[str],
    ) -> None:
        """Log the start of a new workflow generation."""
        ep_summaries = [
            f"{getattr(ep, 'method', '?').upper()} {getattr(ep, 'path', '?')}"
            for ep in tag_endpoints
        ]
        self._logger.info(
            "[new-workflow] generating for tag '%s' — %d endpoint(s): %s",
            tag_name,
            len(tag_endpoints),
            ", ".join(ep_summaries),
        )
        self._logger.info(
            "[new-workflow] reference workflow: %s",
            f"{len(reference_workflow_source)} chars"
            if reference_workflow_source
            else "none",
        )

    def _extract_reference_imports(
        self, reference_workflow_source: Optional[str]
    ) -> List[str]:
        """Extract import statements from a reference workflow source.

        Returns an empty list when no source is given or parsing fails.
        """
        if not reference_workflow_source:
            return []
        try:
            ref_analysis = self._analyze_source(reference_workflow_source)
            imports = [imp.statement for imp in ref_analysis.imports]
            if self._verbose:
                self._logger.info(
                    "[new-workflow] extracted %d import(s) from reference",
                    len(imports),
                )
            return imports
        except ValueError:
            return []
