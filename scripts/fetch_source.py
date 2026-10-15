"""Download the pinned official Ultralytics source without touching existing work."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from code_assistant.source import (
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SourceDiscoveryError,
    discover_source,
)

DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / ".source" / "ultralytics"


def run_git(*arguments: str) -> None:
    result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    if result.returncode:
        raise SourceDiscoveryError("Could not fetch the pinned Ultralytics source")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    destination = args.destination.resolve()

    try:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            run_git(
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                SOURCE_REPOSITORY,
                str(destination),
            )
            run_git("-C", str(destination), "fetch", "--depth", "1", "origin", SOURCE_REVISION)
            run_git("-C", str(destination), "checkout", "--detach", SOURCE_REVISION)
            action = "fetched"
        else:
            action = "reused"
        snapshot = discover_source(destination)
    except SourceDiscoveryError as exc:
        print(f"source_fetch_failed: {exc}", file=sys.stderr)
        return 1

    print("source_checkout:", destination)
    print("source_revision:", snapshot.revision)
    print("source_action:", action)
    print("python_files:", len(snapshot.files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
