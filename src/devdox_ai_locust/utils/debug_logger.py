"""
Debug Logger for DevDox AI Locust

Captures and saves artifacts at each step of the generation pipeline
for debugging and troubleshooting purposes.

Structure:
.devdox-ai-locust/
└── debug/
    └── 2024-01-19_14-30-00/
        └── {tag}/
            └── {operation_id}/
                ├── positive/
                │   ├── 1_endpoint_info.json
                │   ├── 2_rendered_prompt.txt
                │   ├── 3_llm_response_raw.txt
                │   ├── 4_extracted_code.py
                │   ├── 5_after_fixes.py
                │   ├── 6_validation_result.json
                │   └── generation.log
                ├── negative/
                └── security/
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DebugLogger:
    """
    Handles saving debug artifacts for the generation pipeline.

    Each generation (endpoint + scenario type) gets its own directory
    with artifacts saved at each step.
    """

    def __init__(self, base_dir: Path, session_id: str):
        """
        Initialize the debug logger.

        Args:
            base_dir: The .devdox-ai-locust directory
            session_id: Timestamp-based session identifier
        """
        self.base_dir = base_dir
        self.session_id = session_id
        self.session_dir = base_dir / "debug" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Write session info
        session_info = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress"
        }
        self._write_json(self.session_dir / "session_info.json", session_info)

        logger.info(f"Debug session started: {self.session_dir}")

    def get_endpoint_dir(self, tag: str, operation_id: str, scenario_type: str) -> Path:
        """Get the directory for a specific endpoint/scenario combination."""
        endpoint_dir = self.session_dir / self._sanitize(tag) / self._sanitize(operation_id) / scenario_type
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        return endpoint_dir

    def log_endpoint_info(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        endpoint_data: Dict[str, Any]
    ) -> None:
        """Step 1: Log the endpoint information being processed."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)
        self._write_json(dir_path / "1_endpoint_info.json", endpoint_data)

    def log_rendered_prompt(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        prompt: str
    ) -> None:
        """Step 2: Log the rendered prompt sent to the LLM."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)
        self._write_text(dir_path / "2_rendered_prompt.txt", prompt)

    def log_llm_response(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        response: str,
        attempt: int = 1
    ) -> None:
        """Step 3: Log the raw LLM response."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)
        filename = f"3_llm_response_raw_attempt{attempt}.txt"
        self._write_text(dir_path / filename, response)

    def log_extracted_code(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        code: str,
        attempt: int = 1
    ) -> None:
        """Step 4: Log the code after extraction from LLM response."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)
        filename = f"4_extracted_code_attempt{attempt}.py"
        self._write_text(dir_path / filename, code)

    def log_after_fixes(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        code: str,
        fixes_applied: list,
        attempt: int = 1
    ) -> None:
        """Step 5: Log the code after post-processing fixes."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)

        # Log the fixed code
        filename = f"5_after_fixes_attempt{attempt}.py"
        self._write_text(dir_path / filename, code)

        # Log what fixes were applied
        fixes_filename = f"5_fixes_applied_attempt{attempt}.json"
        self._write_json(dir_path / fixes_filename, {"fixes": fixes_applied})

    def log_validation_result(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        is_valid: bool,
        error: Optional[str],
        code: str,
        attempt: int = 1
    ) -> None:
        """Step 6: Log the validation result."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)

        result = {
            "attempt": attempt,
            "is_valid": is_valid,
            "error": error,
        }

        filename = f"6_validation_result_attempt{attempt}.json"
        self._write_json(dir_path / filename, result)

        # If validation failed, save the failing code with line numbers for easy debugging
        if not is_valid:
            numbered_code = self._add_line_numbers(code)
            fail_filename = f"6_FAILED_code_attempt{attempt}.py"
            self._write_text(dir_path / fail_filename, numbered_code)

    def log_final_outcome(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        success: bool,
        used_fallback: bool,
        final_code: Optional[str],
        error_message: Optional[str] = None
    ) -> None:
        """Step 7: Log the final outcome of generation."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)

        outcome = {
            "success": success,
            "used_fallback": used_fallback,
            "error_message": error_message,
        }
        self._write_json(dir_path / "7_final_outcome.json", outcome)

        if final_code:
            self._write_text(dir_path / "7_final_output.py", final_code)

    def log_generation_event(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        event: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append an event to the generation log."""
        dir_path = self.get_endpoint_dir(tag, operation_id, scenario_type)
        log_file = dir_path / "generation.log"

        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {event}"
        if details:
            log_entry += f" | {json.dumps(details)}"
        log_entry += "\n"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def finalize_session(self, total_endpoints: int, successful: int, failed: int) -> None:
        """Mark the session as complete with summary stats."""
        session_info = {
            "session_id": self.session_id,
            "completed_at": datetime.now().isoformat(),
            "status": "completed",
            "summary": {
                "total_endpoints": total_endpoints,
                "successful": successful,
                "failed": failed,
                "success_rate": f"{(successful / total_endpoints * 100):.1f}%" if total_endpoints > 0 else "N/A"
            }
        }
        self._write_json(self.session_dir / "session_info.json", session_info)
        logger.info(f"Debug session completed: {self.session_dir}")

    def _sanitize(self, name: str) -> str:
        """Sanitize a name for use as a directory name."""
        import re
        name = name.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
        name = re.sub(r'[^a-z0-9_]', '', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name or "unnamed"

    def _add_line_numbers(self, code: str) -> str:
        """Add line numbers to code for easy debugging."""
        lines = code.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"{i:4d} | {line}")
        return '\n'.join(numbered_lines)

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to a file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write debug JSON {path}: {e}")

    def _write_text(self, path: Path, content: str) -> None:
        """Write text content to a file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to write debug file {path}: {e}")


def create_debug_logger(output_dir: Path) -> DebugLogger:
    """
    Create a DebugLogger with a new session.

    Args:
        output_dir: The output directory (debug files go in .devdox-ai-locust relative to cwd)

    Returns:
        Configured DebugLogger instance
    """
    # .devdox-ai-locust lives in the current working directory
    internal_dir = Path.cwd() / ".devdox-ai-locust"
    internal_dir.mkdir(parents=True, exist_ok=True)

    # Create session ID from timestamp
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return DebugLogger(internal_dir, session_id)


def ensure_internal_dir_exists() -> Path:
    """
    Ensure the .devdox-ai-locust directory exists.
    Called regardless of whether --debug is used.

    Returns:
        Path to the internal directory
    """
    internal_dir = Path.cwd() / ".devdox-ai-locust"
    internal_dir.mkdir(parents=True, exist_ok=True)
    return internal_dir
