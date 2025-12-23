"""
Unit tests for incremental augmentation components using isolated filesystems.
"""

import asyncio
import json
import tarfile
from pathlib import Path
from typing import List

import pytest

from devdox_ai_locust.incremental_enhancer import (
    AugmentationPlan,
    AugmentationUpdate,
    ChangelogWriter,
    IncrementalEnhancer,
    LocustSuiteLoader,
    SuiteAugmentationWriter,
    SuiteBackupManager,
    SuiteSnapshot,
)
from devdox_ai_locust.utils.open_ai_parser import Endpoint


def _sample_endpoints() -> List[Endpoint]:
    return [
        Endpoint(
            path="/users",
            method="get",
            operation_id="getUsers",
            summary="List users",
            parameters=[],
            request_body=None,
            responses={},
            description="",
            tags=["users"],
        )
    ]


@pytest.mark.unit
def test_suite_loader_reads_main_and_workflows(fs):
    root = Path("/suite")
    fs.create_dir(root)
    fs.create_file(root / "locustfile.py", contents="class UserTasks: pass\n")
    fs.create_dir(root / "workflows")
    fs.create_file(root / "workflows" / "flow_a.py", contents="def flow():\n    pass\n")

    snapshot = LocustSuiteLoader(root).load()

    assert snapshot.root == root
    assert snapshot.main_file == root / "locustfile.py"
    assert "flow_a.py" in {p.name for p in snapshot.workflows}
    assert "class UserTasks" in snapshot.main_file.read_text()


@pytest.mark.unit
def test_backup_manager_creates_tar_without_nested_backup(fs):
    root = Path("/suite")
    fs.create_dir(root)
    fs.create_file(root / "locustfile.py", contents="# main\n")
    fs.create_dir(root / ".backups")
    fs.create_file(root / ".backups" / "old.tar.gz", contents="placeholder")

    archive = SuiteBackupManager().create_backup(root)

    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        assert "locustfile.py" in names
        assert ".backups" not in names


@pytest.mark.unit
def test_augmentation_writer_append_and_create(fs):
    root = Path("/suite")
    fs.create_dir(root / "workflows")
    main_path = root / "locustfile.py"
    wf_path = root / "workflows" / "existing.py"
    fs.create_file(main_path, contents="# base\n")
    fs.create_file(wf_path, contents="def existing():\n    return True\n")

    snapshot = SuiteSnapshot(
        root=root, main_file=main_path, workflows={wf_path: wf_path.read_text()}
    )
    writer = SuiteAugmentationWriter(root)
    updates = [
        AugmentationUpdate(path=Path("locustfile.py"), action="append", content="def extra():\n    return 1"),
        AugmentationUpdate(path=Path("workflows/new_flow.py"), action="create", content="def new_flow():\n    return 2"),
    ]

    after_state = writer.apply(snapshot, updates)

    main_content = (root / "locustfile.py").read_text()
    assert "# --- Augmented Scenario ---" in main_content
    assert "def extra" in main_content
    assert "workflows/new_flow.py" in {str(p.relative_to(root)) for p in after_state}
    assert (root / "workflows" / "new_flow.py").exists()
    assert "new_flow" in (root / "workflows" / "new_flow.py").read_text()


@pytest.mark.unit
def test_changelog_writer_outputs_diff(fs):
    root = Path("/suite")
    fs.create_dir(root)
    before = {root / "locustfile.py": "# before\n"}
    after = {root / "locustfile.py": "# after\n"}

    changelog = ChangelogWriter(root)
    path = changelog.write(before, after, requirement="add scenarios")

    assert path.exists()
    content = path.read_text()
    assert "Requirement: add scenarios" in content
    assert "locustfile.py" in content
    assert "-# before" in content or "# before" in content


@pytest.mark.unit
def test_incremental_enhancer_builds_combined_updates(fs, monkeypatch):
    root = Path("/suite")
    fs.create_dir(root / "workflows")
    main_path = root / "locustfile.py"
    wf_path = root / "workflows" / "existing.py"
    fs.create_file(main_path, contents="# locust\n")
    fs.create_file(wf_path, contents="# wf\n")

    snapshot = SuiteSnapshot(
        root=root, main_file=main_path, workflows={wf_path: wf_path.read_text()}
    )

    prompts = []

    async def fake_call_ai(prompt: str) -> str:
        prompts.append(prompt)
        if "Existing workflows" in prompt:
            payload = {"updates": [{"path": "workflows/new.py", "action": "create", "content": "def workflow_added():\n    pass"}]}
        else:
            payload = {"updates": [{"path": "locustfile.py", "action": "append", "content": "def locust_added():\n    pass"}]}
        return json.dumps(payload)

    fs.add_real_directory("src/devdox_ai_locust/prompt")
    enhancer = IncrementalEnhancer(ai_client=None, prompt_dir=Path("src/devdox_ai_locust/prompt"))
    monkeypatch.setattr(enhancer, "_call_ai", fake_call_ai)

    plan: AugmentationPlan = asyncio.run(
        enhancer.plan_augmentation(
            snapshot=snapshot,
            endpoints=_sample_endpoints(),
            api_info={"title": "API"},
            custom_requirement="add scenarios",
        )
    )

    assert len(prompts) == 2
    assert any("Current locustfile.py" in prompt for prompt in prompts)
    assert any("Existing workflows" in prompt for prompt in prompts)
    assert len(plan.updates) == 2
    assert {u.path for u in plan.updates} == {Path("locustfile.py"), Path("workflows/new.py")}
    assert set(plan.raw_response.keys()) == {"locust", "workflows"}
