import subprocess
from pathlib import Path

import pytest

from code_assistant.source import (
    SOURCE_SCOPES,
    SourceDiscoveryError,
    discover_source,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


@pytest.fixture
def source_checkout(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q")
    for scope in SOURCE_SCOPES:
        (tmp_path / scope).mkdir(parents=True)
    (tmp_path / SOURCE_SCOPES[0] / "model.py").write_text(
        "class Model:\n    def train(self):\n        return 1\n"
    )
    (tmp_path / SOURCE_SCOPES[1] / "engine.py").write_text(
        "async def run():\n    return 2\n"
    )
    (tmp_path / SOURCE_SCOPES[2] / "data.py").write_text("VALUE = 3\n")
    (tmp_path / SOURCE_SCOPES[2] / "ignore.txt").write_text("not Python")
    (tmp_path / "ultralytics/nn").mkdir()
    (tmp_path / "ultralytics/nn/tasks.py").write_text("OUTSIDE = 0\n")
    _git(tmp_path, "add", "ultralytics")
    _git(
        tmp_path,
        "-c",
        "user.name=Source Test",
        "-c",
        "user.email=source-test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD")


def test_discovers_only_pinned_python_scope(source_checkout: tuple[Path, str]) -> None:
    repo, revision = source_checkout
    snapshot = discover_source(repo, expected_revision=revision)
    assert snapshot.files == (
        "ultralytics/models/model.py",
        "ultralytics/engine/engine.py",
        "ultralytics/data/data.py",
    )
    assert snapshot.revision == revision


def test_rejects_wrong_revision(source_checkout: tuple[Path, str]) -> None:
    repo, _ = source_checkout
    with pytest.raises(SourceDiscoveryError, match="revision"):
        discover_source(repo, expected_revision="0" * 40)


def test_rejects_modified_source(source_checkout: tuple[Path, str]) -> None:
    repo, revision = source_checkout
    (repo / SOURCE_SCOPES[2] / "data.py").write_text("VALUE = 4\n")
    with pytest.raises(SourceDiscoveryError, match="local changes"):
        discover_source(repo, expected_revision=revision)
