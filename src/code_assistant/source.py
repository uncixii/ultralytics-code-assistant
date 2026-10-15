"""Discover the exact Ultralytics Python source snapshot used for indexing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from code_assistant.errors import PublicError

SOURCE_REPOSITORY = "https://github.com/ultralytics/ultralytics.git"
SOURCE_REVISION = "3d75949c41f78a2c6763eb347ac7a94793098e89"
SOURCE_SCOPES = (
    "ultralytics/models",
    "ultralytics/engine",
    "ultralytics/data",
)


class SourceDiscoveryError(PublicError):
    """The checkout cannot be used as the pinned source snapshot."""


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    revision: str
    files: tuple[str, ...]


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SourceDiscoveryError("Could not inspect the Ultralytics Git checkout")
    return result.stdout.strip()


def discover_source(
    repo: Path,
    *,
    expected_revision: str = SOURCE_REVISION,
) -> SourceSnapshot:
    """Reject a different or edited checkout, then inspect all required .py files."""
    root = repo.resolve()
    if not root.is_dir():
        raise SourceDiscoveryError(
            "Ultralytics checkout does not exist. Run uv run python scripts/fetch_source.py first."
        )
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise SourceDiscoveryError("Path must be the Ultralytics repository root")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise SourceDiscoveryError(
            "Checkout revision differs from the pinned source revision. "
            f"Use a separate clean checkout at {expected_revision}."
        )
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_SCOPES):
        raise SourceDiscoveryError(
            "Required source directories contain local changes. Use a separate clean checkout."
        )

    files: list[str] = []
    for scope in SOURCE_SCOPES:
        directory = root / scope
        if not directory.is_dir():
            raise SourceDiscoveryError(f"Required source directory is missing: {scope}")
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise SourceDiscoveryError(f"Source path escapes checkout: {relative}")
            try:
                path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise SourceDiscoveryError(
                    f"Could not read Python source: {relative}"
                ) from exc
            files.append(relative)
    return SourceSnapshot(root=root, revision=revision, files=tuple(files))
