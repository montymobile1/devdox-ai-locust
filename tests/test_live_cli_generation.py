"""
Live (opt-in) tests that exercise generate and augment end-to-end against a real API spec.

These are skipped by default to avoid consuming API credits or requiring secrets in CI.
Set TOGETHER_API_KEY to run them locally.
"""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from devdox_ai_locust.cli import cli


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("TOGETHER_API_KEY"),
        reason="Requires TOGETHER_API_KEY and external network access for live run.",
    ),
]


PETSTORE_SPEC = "https://petstore3.swagger.io/api/v3/openapi.json"


def _run_cli(args: list[str], env: dict[str, str]) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, args, env=env)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0, result.output


def _assert_suite_outputs(base_dir: Path) -> None:
    assert (base_dir / "locustfile.py").exists(), "locustfile.py not generated"
    workflows_dir = base_dir / "workflows"
    assert workflows_dir.exists(), "workflows dir missing after generation"


def _latest_backup_path(base_dir: Path) -> Path:
    backups = sorted((base_dir / ".backups").glob("*.tar.gz"))
    assert backups, "No backup archive produced by augment"
    return backups[-1]


@pytest.mark.live
def test_live_generate_and_augment_changes_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Runs `generate` then `augment` with a real Together API key to ensure end-to-end flow.
    """

    env = {"TOGETHER_API_KEY": os.environ["TOGETHER_API_KEY"]}
    suite_dir = tmp_path / "suite"

    _run_cli(
        [
            "generate",
            PETSTORE_SPEC,
            "--output",
            str(suite_dir),
            "--custom-requirement",
            "Add smoke tasks for pet operations",
        ],
        env=env,
    )

    _assert_suite_outputs(suite_dir)

    _run_cli(
        [
            "augment",
            PETSTORE_SPEC,
            "--suite-path",
            str(suite_dir),
            "--custom-requirement",
            "Append new tasks that cover user login and order creation",
        ],
        env=env,
    )

    backup_path = _latest_backup_path(suite_dir)
    assert backup_path.exists()
    diff_dir = suite_dir / ".diff"
    assert diff_dir.exists()


@pytest.mark.live
def test_live_augment_can_handle_noop_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Executes augment with a no-op style requirement; accepts natural ValueError if the model returns no updates.
    """

    env = {"TOGETHER_API_KEY": os.environ["TOGETHER_API_KEY"]}
    suite_dir = tmp_path / "suite"

    _run_cli(
        [
            "generate",
            PETSTORE_SPEC,
            "--output",
            str(suite_dir),
            "--custom-requirement",
            "Generate baseline smoke tests",
        ],
        env=env,
    )

    _assert_suite_outputs(suite_dir)

    try:
        _run_cli(
            [
                "augment",
                PETSTORE_SPEC,
                "--suite-path",
                str(suite_dir),
                "--custom-requirement",
                "If no additional coverage is needed, respond with an empty updates list.",
            ],
            env=env,
        )
    except ValueError as exc:
        # The pipeline raises if the model elects to return no updates; treat that as an acceptable outcome.
        assert "did not include any updates" in str(exc)
