"""
Debug Recorder for capturing intermediate states during generation.

When --debug flag is passed to the generate command, this module records
all intermediate inputs, outputs, and transformations for auditing and debugging.
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles  # type: ignore[import-untyped]


@dataclass
class ScenarioDebugInfo:
    """Debug information for a single scenario generation"""

    scenario_type: str
    context: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    llm_request: Dict[str, Any] = field(default_factory=dict)
    llm_response: str = ""
    extracted_code: str = ""
    processed_code: str = ""
    validation: Dict[str, Any] = field(default_factory=dict)
    final_code: str = ""
    retries: List[Dict[str, Any]] = field(default_factory=list)
    fallback: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointDebugInfo:
    """Debug information for a single endpoint"""

    method: str
    path: str
    operation_id: str
    tag: str
    endpoint_details: Dict[str, Any] = field(default_factory=dict)
    scenarios: Dict[str, ScenarioDebugInfo] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class DebugRecorder:
    """
    Records all intermediate states during test generation for debugging and auditing.

    Directory structure mirrors the output structure for easy navigation:

    output/
    └── .devdox_ai_locust/
        └── YYYYMMDD_HHMMSS/
            └── generate/
                ├── _manifest.json
                ├── input/
                │   ├── cli_args.json
                │   ├── openapi_raw.json
                │   ├── openapi_parsed.json
                │   └── resolved_config.json
                ├── static/
                │   └── {file_name}/
                │       ├── context.json
                │       └── rendered.py
                └── workflows/
                    └── {tag}/
                        ├── {endpoint}/
                        │   ├── _endpoint.json
                        │   └── {scenario}/
                        │       ├── context.json
                        │       ├── prompt.txt
                        │       ├── llm_request.json
                        │       ├── llm_response.txt
                        │       ├── extracted.py
                        │       ├── processed.py
                        │       ├── validation.json
                        │       ├── final.py
                        │       ├── _retry_N/
                        │       ├── _fallback/
                        │       └── _summary.json
                        ├── orchestrator/
                        │   └── ...
                        └── _summary.json
    """

    def __init__(self, output_dir: Path, enabled: bool = False):
        """
        Initialize the debug recorder.

        Args:
            output_dir: The main output directory for generated tests
            enabled: Whether debug recording is enabled (--debug flag)
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Debug directory structure
        self.debug_root = output_dir / ".devdox_ai_locust" / self.timestamp / "generate"
        self.input_dir = self.debug_root / "input"
        self.static_dir = self.debug_root / "static"
        self.workflows_dir = self.debug_root / "workflows"

        # Statistics tracking
        self.stats: Dict[str, Any] = {
            "tags": 0,
            "endpoints": 0,
            "scenarios": {"total": 0, "succeeded": 0, "fallback": 0},
            "orchestrators": {"total": 0, "succeeded": 0, "fallback": 0},
            "llm_calls": {"total": 0, "succeeded": 0, "retries": 0},
            "errors": [],
        }

        # Async lock for thread-safe writes
        self._write_lock = asyncio.Lock()

        if enabled:
            self._init_directories()

    def _init_directories(self) -> None:
        """Create the debug directory structure"""
        self.debug_root.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    async def _write_json(self, path: Path, data: Any) -> None:
        """Write JSON data to a file asynchronously"""
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._write_lock:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(data, indent=2, default=str, ensure_ascii=False)
                )

    async def _write_text(self, path: Path, content: str) -> None:
        """Write text content to a file asynchronously"""
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._write_lock:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(content)

    def _write_json_sync(self, path: Path, data: Any) -> None:
        """Write JSON data to a file synchronously"""
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    def _write_text_sync(self, path: Path, content: str) -> None:
        """Write text content to a file synchronously"""
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # =========================================================================
    # Input Recording
    # =========================================================================

    def record_cli_args(self, args: Dict[str, Any]) -> None:
        """Record CLI arguments"""
        if not self.enabled:
            return
        self._write_json_sync(self.input_dir / "cli_args.json", args)

    def record_openapi_raw(self, raw_spec: Any) -> None:
        """Record the raw OpenAPI specification as received"""
        if not self.enabled:
            return
        self._write_json_sync(self.input_dir / "openapi_raw.json", raw_spec)

    def record_openapi_parsed(
        self, endpoints: List[Any], api_info: Dict[str, Any]
    ) -> None:
        """Record parsed OpenAPI data"""
        if not self.enabled:
            return

        # Serialize endpoints to dict
        endpoints_data = []
        for ep in endpoints:
            ep_dict = {
                "method": ep.method,
                "path": ep.path,
                "operation_id": getattr(ep, "operation_id", None),
                "tags": getattr(ep, "tags", []),
                "summary": getattr(ep, "summary", None),
                "description": getattr(ep, "description", None),
                "parameters": [],
                "request_body": None,
                "responses": {},
            }

            # Serialize parameters
            if hasattr(ep, "parameters") and ep.parameters:
                for param in ep.parameters:
                    param_dict = {
                        "name": getattr(param, "name", None),
                        "location": str(
                            getattr(param, "location", getattr(param, "in_", "query"))
                        ),
                        "required": getattr(param, "required", False),
                        "type": getattr(param, "type", "string"),
                        "format": getattr(param, "format", None),
                        "enum": getattr(param, "enum", None),
                        "description": getattr(param, "description", None),
                    }
                    ep_dict["parameters"].append(param_dict)

            # Serialize request body
            if hasattr(ep, "request_body") and ep.request_body:
                rb = ep.request_body
                ep_dict["request_body"] = {
                    "required": getattr(rb, "required", False),
                    "content_type": getattr(rb, "content_type", None),
                    "schema": getattr(rb, "schema", None),
                }

            # Serialize responses (can be dict or list depending on parser)
            if hasattr(ep, "responses") and ep.responses:
                if isinstance(ep.responses, dict):
                    for status, resp in ep.responses.items():
                        ep_dict["responses"][str(status)] = {
                            "description": (
                                getattr(resp, "description", None)
                                if hasattr(resp, "description")
                                else str(resp)
                            ),
                        }
                elif isinstance(ep.responses, list):
                    for resp in ep.responses:
                        status = getattr(resp, "status_code", "unknown")
                        ep_dict["responses"][str(status)] = {
                            "description": (
                                getattr(resp, "description", None)
                                if hasattr(resp, "description")
                                else str(resp)
                            ),
                        }

            endpoints_data.append(ep_dict)

        parsed_data = {
            "api_info": api_info,
            "endpoints": endpoints_data,
            "endpoint_count": len(endpoints),
        }
        self._write_json_sync(self.input_dir / "openapi_parsed.json", parsed_data)
        self.stats["endpoints"] = len(endpoints)

    def record_resolved_config(self, config: Dict[str, Any]) -> None:
        """Record the resolved configuration"""
        if not self.enabled:
            return
        self._write_json_sync(self.input_dir / "resolved_config.json", config)

    # =========================================================================
    # Static File Recording
    # =========================================================================

    def record_static_file(
        self,
        file_name: str,
        context: Dict[str, Any],
        rendered_content: str,
    ) -> None:
        """Record a statically generated file (no LLM)"""
        if not self.enabled:
            return

        # Remove .py extension for directory name
        dir_name = file_name.replace(".py", "").replace(".txt", "").replace(".md", "")
        file_dir = self.static_dir / dir_name
        file_dir.mkdir(parents=True, exist_ok=True)

        # Don't include full base_workflow/test_data in context to save space
        # These are already recorded elsewhere
        filtered_context = {
            k: v
            for k, v in context.items()
            if k not in ("base_workflow_content", "test_data_content")
        }

        self._write_json_sync(file_dir / "context.json", filtered_context)

        # Determine file extension
        if file_name.endswith(".py"):
            self._write_text_sync(file_dir / "rendered.py", rendered_content)
        elif file_name.endswith(".txt"):
            self._write_text_sync(file_dir / "rendered.txt", rendered_content)
        elif file_name.endswith(".md"):
            self._write_text_sync(file_dir / "rendered.md", rendered_content)
        else:
            self._write_text_sync(
                file_dir / f"rendered{Path(file_name).suffix}", rendered_content
            )

    # =========================================================================
    # Endpoint/Scenario Recording
    # =========================================================================

    def _get_scenario_dir(
        self, tag: str, endpoint_dir_name: str, scenario_type: str
    ) -> Path:
        """Get the directory path for a scenario"""
        return self.workflows_dir / tag / endpoint_dir_name / scenario_type

    async def record_endpoint_details(
        self,
        tag: str,
        endpoint_dir_name: str,
        endpoint: Any,
        formatted_details: str,
    ) -> None:
        """Record endpoint details"""
        if not self.enabled:
            return

        endpoint_dir = self.workflows_dir / tag / endpoint_dir_name

        endpoint_data = {
            "method": endpoint.method,
            "path": endpoint.path,
            "operation_id": getattr(endpoint, "operation_id", None),
            "tag": tag,
            "formatted_details": formatted_details,
        }

        await self._write_json(endpoint_dir / "_endpoint.json", endpoint_data)

    async def record_scenario_context(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        context: Dict[str, Any],
    ) -> None:
        """Record the context passed to the template for prompt rendering"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_json(scenario_dir / "context.json", context)

    async def record_scenario_prompt(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        prompt: str,
    ) -> None:
        """Record the rendered prompt sent to LLM"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_text(scenario_dir / "prompt.txt", prompt)

    async def record_llm_request(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        request_data: Dict[str, Any],
    ) -> None:
        """Record the LLM API request"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_json(scenario_dir / "llm_request.json", request_data)
        self.stats["llm_calls"]["total"] += 1

    async def record_llm_response(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        response: str,
    ) -> None:
        """Record the raw LLM response"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_text(scenario_dir / "llm_response.txt", response)

    async def record_extracted_code(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        code: str,
    ) -> None:
        """Record the code extracted from LLM response"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_text(scenario_dir / "extracted.py", code)

    async def record_processed_code(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        code: str,
    ) -> None:
        """Record the code after post-processing (unicode, class name, bytes, regex fixes)"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_text(scenario_dir / "processed.py", code)

    async def record_validation_result(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        is_valid: bool,
        error: Optional[str] = None,
        checks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Record the validation result"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        validation_data = {
            "valid": is_valid,
            "error": error,
            "checks": checks or [],
        }
        await self._write_json(scenario_dir / "validation.json", validation_data)

    async def record_final_code(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        code: str,
    ) -> None:
        """Record the final code that will be written to output"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_text(scenario_dir / "final.py", code)
        self.stats["llm_calls"]["succeeded"] += 1

    async def record_scenario_summary(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        summary: Dict[str, Any],
    ) -> None:
        """Record the scenario summary"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        await self._write_json(scenario_dir / "_summary.json", summary)
        self.stats["scenarios"]["total"] += 1
        if summary.get("success"):
            self.stats["scenarios"]["succeeded"] += 1
        if summary.get("used_fallback"):
            self.stats["scenarios"]["fallback"] += 1

    # =========================================================================
    # Retry Recording
    # =========================================================================

    async def record_retry_attempt(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        attempt: int,
        error: str,
        bad_code: str,
        fix_prompt: str,
        llm_response: str,
        extracted_code: str,
        validation_result: Dict[str, Any],
    ) -> None:
        """Record a retry attempt"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        retry_dir = scenario_dir / f"_retry_{attempt}"
        retry_dir.mkdir(parents=True, exist_ok=True)

        await self._write_text(retry_dir / "error.txt", error)
        await self._write_text(retry_dir / "bad_code.py", bad_code)
        await self._write_text(retry_dir / "fix_prompt.txt", fix_prompt)
        await self._write_text(retry_dir / "llm_response.txt", llm_response)
        await self._write_text(retry_dir / "extracted.py", extracted_code)
        await self._write_json(retry_dir / "validation.json", validation_result)

        self.stats["llm_calls"]["retries"] += 1
        self.stats["llm_calls"]["total"] += 1

    # =========================================================================
    # Fallback Recording
    # =========================================================================

    async def record_fallback(
        self,
        tag: str,
        endpoint_dir_name: str,
        scenario_type: str,
        reason: str,
        all_errors: List[str],
        fallback_template: str,
    ) -> None:
        """Record a fallback usage"""
        if not self.enabled:
            return

        scenario_dir = self._get_scenario_dir(tag, endpoint_dir_name, scenario_type)
        fallback_dir = scenario_dir / "_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        await self._write_text(fallback_dir / "reason.txt", reason)
        await self._write_json(fallback_dir / "all_errors.json", all_errors)
        await self._write_text(fallback_dir / "template.py", fallback_template)

    # =========================================================================
    # Orchestrator Recording
    # =========================================================================

    async def record_orchestrator_context(
        self,
        tag: str,
        context: Dict[str, Any],
    ) -> None:
        """Record orchestrator context"""
        if not self.enabled:
            return

        orchestrator_dir = self.workflows_dir / tag / "orchestrator"
        await self._write_json(orchestrator_dir / "context.json", context)

    async def record_orchestrator_prompt(
        self,
        tag: str,
        prompt: str,
    ) -> None:
        """Record orchestrator prompt"""
        if not self.enabled:
            return

        orchestrator_dir = self.workflows_dir / tag / "orchestrator"
        await self._write_text(orchestrator_dir / "prompt.txt", prompt)

    async def record_orchestrator_llm_request(
        self,
        tag: str,
        request_data: Dict[str, Any],
    ) -> None:
        """Record orchestrator LLM request"""
        if not self.enabled:
            return

        orchestrator_dir = self.workflows_dir / tag / "orchestrator"
        await self._write_json(orchestrator_dir / "llm_request.json", request_data)
        self.stats["llm_calls"]["total"] += 1

    async def record_orchestrator_llm_response(
        self,
        tag: str,
        response: str,
    ) -> None:
        """Record orchestrator LLM response"""
        if not self.enabled:
            return

        orchestrator_dir = self.workflows_dir / tag / "orchestrator"
        await self._write_text(orchestrator_dir / "llm_response.txt", response)

    async def record_orchestrator_final(
        self,
        tag: str,
        code: str,
        summary: Dict[str, Any],
    ) -> None:
        """Record orchestrator final code and summary"""
        if not self.enabled:
            return

        orchestrator_dir = self.workflows_dir / tag / "orchestrator"
        await self._write_text(orchestrator_dir / "final.py", code)
        await self._write_json(orchestrator_dir / "_summary.json", summary)

        self.stats["orchestrators"]["total"] += 1
        if summary.get("success"):
            self.stats["orchestrators"]["succeeded"] += 1
            self.stats["llm_calls"]["succeeded"] += 1
        if summary.get("used_fallback"):
            self.stats["orchestrators"]["fallback"] += 1

    # =========================================================================
    # Summary Recording
    # =========================================================================

    async def record_tag_summary(
        self,
        tag: str,
        summary: Dict[str, Any],
    ) -> None:
        """Record tag-level summary"""
        if not self.enabled:
            return

        tag_dir = self.workflows_dir / tag
        await self._write_json(tag_dir / "_summary.json", summary)
        self.stats["tags"] += 1

    async def record_endpoint_summary(
        self,
        tag: str,
        endpoint_dir_name: str,
        summary: Dict[str, Any],
    ) -> None:
        """Record endpoint-level summary"""
        if not self.enabled:
            return

        endpoint_dir = self.workflows_dir / tag / endpoint_dir_name
        await self._write_json(endpoint_dir / "_summary.json", summary)

    async def finalize(self) -> None:
        """Finalize debug recording and write manifest"""
        if not self.enabled:
            return

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # Write workflows summary
        workflows_summary = {
            "tags": self.stats["tags"],
            "endpoints": self.stats["endpoints"],
            "scenarios": self.stats["scenarios"],
            "orchestrators": self.stats["orchestrators"],
        }
        await self._write_json(self.workflows_dir / "_summary.json", workflows_summary)

        # Write manifest
        manifest = {
            "operation": "generate",
            "started_at": self.start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": duration,
            "debug_directory": str(self.debug_root),
            "totals": {
                "tags": self.stats["tags"],
                "endpoints": self.stats["endpoints"],
                "scenarios": self.stats["scenarios"]["total"],
                "scenarios_succeeded": self.stats["scenarios"]["succeeded"],
                "scenarios_fallback": self.stats["scenarios"]["fallback"],
                "orchestrators": self.stats["orchestrators"]["total"],
                "orchestrators_succeeded": self.stats["orchestrators"]["succeeded"],
                "llm_calls": self.stats["llm_calls"]["total"],
                "llm_calls_succeeded": self.stats["llm_calls"]["succeeded"],
                "llm_retries": self.stats["llm_calls"]["retries"],
            },
            "errors": self.stats["errors"],
        }
        await self._write_json(self.debug_root / "_manifest.json", manifest)

    def record_error(
        self, error: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an error"""
        if not self.enabled:
            return
        self.stats["errors"].append(
            {
                "error": error,
                "context": context or {},
                "timestamp": datetime.now().isoformat(),
            }
        )
