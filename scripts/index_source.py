"""Index the pinned Ultralytics Python source into MongoDB Atlas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from code_assistant.indexer import index_repository
from code_assistant.errors import error_message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path,
        default=Path(__file__).resolve().parents[1] / ".source" / "ultralytics",
    )
    parser.add_argument("--force", action="store_true", help="Re-encode all source files even when their manifests match")
    args = parser.parse_args()
    try:
        stats = index_repository(args.repo, force=args.force)
    except Exception as exc:
        print(f"index_failed: {error_message(exc)}", file=sys.stderr)
        return 1
    print(
        f"files={stats.files_seen} indexed={stats.files_indexed} "
        f"skipped={stats.files_skipped} deleted={stats.files_deleted} "
        f"chunks_written={stats.chunks_written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
