"""
Debug Logger for DevDox AI Locust

Captures and saves artifacts at each step of the generation pipeline
for debugging and troubleshooting purposes.

Structure:
.devdox-ai-locust/
└── debug/
    └── {timestamp}/
        ├── session.json
        └── {tag}/
            └── {operation_id}/
                └── {scenario}/
                    ├── endpoint.json      # Input: endpoint info
                    ├── prompt.txt         # What we sent to LLM
                    ├── attempts/
                    │   ├── 1/
                    │   │   ├── llm_response.txt
                    │   │   ├── extracted.py
                    │   │   ├── processed.py
                    │   │   ├── validation.json
                    │   │   └── FAILED.py (with line numbers)
                    │   └── 2/
                    │       └── ...
                    ├── final.py           # Final output
                    ├── outcome.json       # Summary
                    └── events.log         # Timeline
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

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
        self.has_failures = False
        self.failure_count = 0
        self.success_count = 0

        # Write session info
        session_info = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress"
        }
        self._write_json(self.session_dir / "session.json", session_info)

        logger.info(f"Debug session started: {self.session_dir}")

    def get_scenario_dir(self, tag: str, operation_id: str, scenario_type: str) -> Path:
        """Get the directory for a specific endpoint/scenario combination."""
        scenario_dir = self.session_dir / self._sanitize(tag) / self._sanitize(operation_id) / scenario_type
        scenario_dir.mkdir(parents=True, exist_ok=True)
        return scenario_dir

    def get_attempt_dir(self, tag: str, operation_id: str, scenario_type: str, attempt: int) -> Path:
        """Get the directory for a specific attempt."""
        attempt_dir = self.get_scenario_dir(tag, operation_id, scenario_type) / "attempts" / str(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        return attempt_dir

    def log_endpoint_info(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        endpoint_data: Dict[str, Any]
    ) -> None:
        """Log the endpoint information being processed."""
        dir_path = self.get_scenario_dir(tag, operation_id, scenario_type)
        self._write_json(dir_path / "endpoint.json", endpoint_data)

    def log_rendered_prompt(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        prompt: str
    ) -> None:
        """Log the rendered prompt sent to the LLM."""
        dir_path = self.get_scenario_dir(tag, operation_id, scenario_type)
        self._write_text(dir_path / "prompt.txt", prompt)

    def log_llm_response(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        response: str,
        attempt: int = 1
    ) -> None:
        """Log the raw LLM response."""
        dir_path = self.get_attempt_dir(tag, operation_id, scenario_type, attempt)
        self._write_text(dir_path / "llm_response.txt", response)

    def log_extracted_code(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        code: str,
        attempt: int = 1
    ) -> None:
        """Log the code after extraction from LLM response."""
        dir_path = self.get_attempt_dir(tag, operation_id, scenario_type, attempt)
        self._write_text(dir_path / "extracted.py", code)

    def log_after_fixes(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        code: str,
        fixes_applied: List[str],
        attempt: int = 1
    ) -> None:
        """Log the code after post-processing fixes."""
        dir_path = self.get_attempt_dir(tag, operation_id, scenario_type, attempt)
        self._write_text(dir_path / "processed.py", code)

        # Also log what fixes were applied in the validation.json later
        # Store fixes temporarily
        self._current_fixes = fixes_applied

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
        """Log the validation result."""
        dir_path = self.get_attempt_dir(tag, operation_id, scenario_type, attempt)

        result = {
            "valid": is_valid,
            "error": error,
            "fixes_applied": getattr(self, '_current_fixes', []),
        }
        self._write_json(dir_path / "validation.json", result)

        # If validation failed, save the failing code with line numbers
        if not is_valid:
            numbered_code = self._add_line_numbers(code)
            self._write_text(dir_path / "FAILED.py", numbered_code)

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
        """Log the final outcome of generation."""
        dir_path = self.get_scenario_dir(tag, operation_id, scenario_type)

        # Track success/failure counts
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            self.has_failures = True

        outcome = {
            "success": success,
            "used_fallback": used_fallback,
            "error": error_message,
        }
        self._write_json(dir_path / "outcome.json", outcome)

        if final_code:
            self._write_text(dir_path / "final.py", final_code)

    def log_generation_event(
        self,
        tag: str,
        operation_id: str,
        scenario_type: str,
        event: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append an event to the generation log."""
        dir_path = self.get_scenario_dir(tag, operation_id, scenario_type)
        log_file = dir_path / "events.log"

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
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
            "started_at": self._get_session_start_time(),
            "completed_at": datetime.now().isoformat(),
            "status": "completed",
            "summary": {
                "total_endpoints": total_endpoints,
                "successful": successful,
                "failed": failed,
                "success_rate": f"{(successful / total_endpoints * 100):.1f}%" if total_endpoints > 0 else "N/A"
            }
        }
        self._write_json(self.session_dir / "session.json", session_info)

    def _get_session_start_time(self) -> str:
        """Get the session start time from session.json."""
        try:
            session_file = self.session_dir / "session.json"
            if session_file.exists():
                with open(session_file, "r") as f:
                    data = json.load(f)
                    return data.get("started_at", datetime.now().isoformat())
        except Exception:
            pass
        return datetime.now().isoformat()

    def delete_session(self) -> bool:
        """Delete the entire debug session directory."""
        try:
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
                logger.info(f"Deleted debug session: {self.session_dir}")
                return True
        except Exception as e:
            logger.warning(f"Failed to delete debug session: {e}")
        return False

    def get_session_path(self) -> Path:
        """Get the path to the session directory."""
        return self.session_dir

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
        max_line_num = len(lines)
        width = len(str(max_line_num))
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"{i:>{width}} | {line}")
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
