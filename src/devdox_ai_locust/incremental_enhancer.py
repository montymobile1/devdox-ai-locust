"""Incremental augmentation pipeline for existing Locust suites.

This module keeps the CLI orchestrator thin by delegating IO, AI prompting,
and file merging into cohesive components. It intentionally avoids mutating
existing logic; only append/create operations are permitted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from together import AsyncTogether

from devdox_ai_locust.utils.open_ai_parser import Endpoint

logger = logging.getLogger(__name__)


@dataclass
class SuiteSnapshot:
    """In-memory view of an existing Locust suite."""

    root: Path
    main_file: Path
    workflows: Dict[Path, str]

    def file_map(self) -> Dict[Path, str]:
        """Return a mapping of all known files to their content."""

        files: Dict[Path, str] = {self.main_file: self.main_file.read_text()}
        files.update({path: content for path, content in self.workflows.items()})
        return files


@dataclass
class AugmentationUpdate:
    """Instruction for how to update a specific file."""

    path: Path
    action: str
    content: str


@dataclass
class AugmentationPlan:
    """Result from the AI prompt describing the updates to apply."""

    updates: List[AugmentationUpdate]
    raw_response: Dict[str, str]


class LocustSuiteLoader:
    """Loads an existing Locust suite from disk."""

    def __init__(self, suite_path: Path):
        self.suite_path = suite_path

    def load(self) -> SuiteSnapshot:
        root = self._resolve_root()
        main_file = root if root.is_file() else root / "locustfile.py"
        if main_file.is_dir():
            raise FileNotFoundError("locustfile.py path points to a directory")

        if not main_file.exists():
            raise FileNotFoundError(f"No locustfile.py found at {main_file}")

        workflows_dir = root / "workflows"
        workflows: Dict[Path, str] = {}
        if workflows_dir.exists():
            for path in workflows_dir.glob("*.py"):
                workflows[path] = path.read_text(encoding="utf-8")

        logger.debug("[augment] Loaded suite with %d workflow files", len(workflows))
        return SuiteSnapshot(root=root if root.is_dir() else root.parent, main_file=main_file, workflows=workflows)

    def _resolve_root(self) -> Path:
        if self.suite_path.is_file():
            return self.suite_path
        if not self.suite_path.exists():
            raise FileNotFoundError(f"Suite path does not exist: {self.suite_path}")
        return self.suite_path


class SuiteBackupManager:
    """Creates timestamped archives of an existing suite."""

    def __init__(self, clock: Optional[datetime] = None) -> None:
        self.clock = clock

    def create_backup(self, suite_root: Path, backup_root: Optional[Path] = None) -> Path:
        timestamp = (self.clock or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = backup_root or suite_root / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        archive_name = f"locust_suite_{timestamp}.tar.gz"
        archive_path = backup_dir / archive_name

        with tarfile.open(archive_path, "w:gz") as tar:
            for item in suite_root.iterdir():
                # Avoid recursive backups
                if item.resolve() == backup_dir.resolve():
                    continue
                tar.add(item, arcname=item.name)

        logger.info("[augment] Backup created at %s", archive_path)
        return archive_path


class ChangelogWriter:
    """Writes diff-style changelog reports."""

    def __init__(self, suite_root: Path):
        self.suite_root = suite_root

    def write(self, before: Dict[Path, str], after: Dict[Path, str], requirement: str) -> Path:
        from difflib import unified_diff

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        diff_dir = self.suite_root / ".diff"
        diff_dir.mkdir(parents=True, exist_ok=True)
        report_path = diff_dir / f"augment_{timestamp}.md"

        lines: List[str] = ["# Augmentation Changelog", "", f"Requirement: {requirement}", ""]

        all_paths = set(before.keys()) | set(after.keys())
        for path in sorted(all_paths):
            before_content = before.get(path, "")
            after_content = after.get(path, "")
            rel_path = path.relative_to(self.suite_root)
            lines.append(f"## {rel_path}")
            diff = unified_diff(
                before_content.splitlines(),
                after_content.splitlines(),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                lineterm="",
            )
            diff_lines = list(diff)
            if not diff_lines:
                lines.append("No changes.")
            else:
                lines.append("```diff")
                lines.extend(diff_lines)
                lines.append("```")
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("[augment] Changelog written to %s", report_path)
        return report_path


class SuiteAugmentationWriter:
    """Applies augmentation updates to disk while preserving existing content."""

    def __init__(self, suite_root: Path):
        self.suite_root = suite_root.resolve()

    def apply(self, snapshot: SuiteSnapshot, updates: List[AugmentationUpdate]) -> Dict[Path, str]:
        before_state = snapshot.file_map()

        for update in updates:
            target_path = self._resolve_target(update.path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            if update.action not in {"append", "create"}:
                raise ValueError(f"Unsupported action: {update.action}")

            payload = self._build_payload(existing, update)
            target_path.write_text(payload, encoding="utf-8")
            logger.debug("[augment] Applied %s to %s", update.action, target_path)

        tracked_paths = set(before_state.keys()) | {self._resolve_target(u.path) for u in updates}
        after_state: Dict[Path, str] = {}
        for path in tracked_paths:
            after_state[path] = path.read_text(encoding="utf-8")
            if path == snapshot.main_file:
                snapshot.main_file = path
            elif path in snapshot.workflows or path.parent.name == "workflows":
                snapshot.workflows[path] = after_state[path]
        return after_state

    def _resolve_target(self, relative_path: Path) -> Path:
        target = (self.suite_root / relative_path).resolve()
        if self.suite_root not in target.parents and target != self.suite_root:
            raise ValueError(f"Attempted to write outside suite root: {relative_path}")
        return target

    def _build_payload(self, existing: str, update: AugmentationUpdate) -> str:
        if update.action == "append" and existing:
            spacer = "\n\n# --- Augmented Scenario ---\n"
            return f"{existing.rstrip()}\n{spacer}{update.content.strip()}\n"
        return f"{existing}{update.content.strip()}\n"


class IncrementalEnhancer:
    """Translate custom requirements into append-only Locust updates."""

    def __init__(
        self,
        ai_client: AsyncTogether,
        prompt_dir: Path,
    ) -> None:
        self.ai_client = ai_client
        self.prompt_dir = prompt_dir
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    async def plan_augmentation(
        self,
        snapshot: SuiteSnapshot,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: str,
    ) -> AugmentationPlan:
        locust_prompt = self._build_locust_prompt(
            snapshot=snapshot,
            endpoints=endpoints,
            api_info=api_info,
            custom_requirement=custom_requirement,
        )
        logger.debug("[augment] Locust prompt prepared (%d chars)", len(locust_prompt))
        locust_raw = await self._call_ai(locust_prompt)
        locust_updates = self._parse_updates(locust_raw)

        workflow_prompt = self._build_workflow_prompt(
            snapshot=snapshot,
            endpoints=endpoints,
            api_info=api_info,
            custom_requirement=custom_requirement,
        )
        logger.debug(
            "[augment] Workflow prompt prepared (%d chars)", len(workflow_prompt)
        )
        workflow_raw = await self._call_ai(workflow_prompt)
        workflow_updates = self._parse_updates(workflow_raw, allow_empty=True)

        updates = locust_updates + workflow_updates
        if not updates:
            raise ValueError("No augmentation updates were generated")
        return AugmentationPlan(
            updates=updates, raw_response={"locust": locust_raw, "workflows": workflow_raw}
        )

    def _build_locust_prompt(
        self,
        snapshot: SuiteSnapshot,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: str,
    ) -> str:
        template = self.jinja_env.get_template("augment_locust.j2")
        return template.render(
            custom_requirement=custom_requirement,
            api_info=api_info,
            endpoints_for_prompt=self._format_endpoints(endpoints),
            locust_content=snapshot.main_file.read_text(encoding="utf-8"),
        )

    def _build_workflow_prompt(
        self,
        snapshot: SuiteSnapshot,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        custom_requirement: str,
    ) -> str:
        template = self.jinja_env.get_template("augment_workflow.j2")
        return template.render(
            custom_requirement=custom_requirement,
            api_info=api_info,
            endpoints_for_prompt=self._format_endpoints(endpoints),
            workflows=[
                {"name": path.name, "content": content}
                for path, content in snapshot.workflows.items()
            ],
        )

    def _format_endpoints(self, endpoints: List[Endpoint]) -> str:
        formatted = []
        for ep in endpoints:
            params = f"{len(ep.parameters)} params" if ep.parameters else "no params"
            body = "body" if ep.request_body else "no body"
            formatted.append(
                f"- {ep.method.upper()} {ep.path} ({params}, {body}) : {ep.summary or 'No summary'}"
            )
        return "\n".join(formatted[:15])

    async def _call_ai(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You augment existing Locust suites. Respond with JSON only inside <code></code> tags. "
                    "Never delete existing code; only append or create new classes/tasks."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.ai_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
        )

        if not response.choices or not response.choices[0].message:
            raise RuntimeError("AI response missing choices")

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("AI response is empty")

        return self._extract_code_block(content)

    def _extract_code_block(self, content: str) -> str:
        import re

        matches = re.findall(r"<code>(.*?)</code>", content, re.DOTALL)
        extracted = matches[0] if matches else content
        extracted = extracted.strip()
        if extracted.startswith("```"):
            extracted = extracted.strip("`\n ")
        return extracted

    def _parse_updates(self, payload: str, allow_empty: bool = False) -> List[AugmentationUpdate]:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse AI response: {exc}") from exc

        updates_json = parsed.get("updates") if isinstance(parsed, dict) else None
        if not updates_json or not isinstance(updates_json, list):
            raise ValueError("AI response missing 'updates' list")

        updates: List[AugmentationUpdate] = []
        for update in updates_json:
            path_value = update.get("path")
            content = update.get("content", "")
            action = update.get("action", "append")
            if not path_value:
                continue
            updates.append(
                AugmentationUpdate(
                    path=Path(path_value),
                    action=action,
                    content=content,
                )
            )

        if not updates and not allow_empty:
            raise ValueError("AI response did not include any updates")
        return updates
