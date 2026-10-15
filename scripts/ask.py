"""Ask a source-grounded question using the existing Atlas index."""

from __future__ import annotations

import argparse
import sys

from code_assistant.answer import UNVERIFIED_NOTICE, OpenRouterClient, answer_question
from code_assistant.config import load_settings
from code_assistant.embedder import CodeEmbedder
from code_assistant.errors import error_message
from code_assistant.retrieval import select_evidence
from code_assistant.source import SOURCE_REVISION
from code_assistant.vector_store import AtlasVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    try:
        settings = load_settings()
        settings.require_mongodb_uri()
        embedder = CodeEmbedder(settings)
        store = AtlasVectorStore(settings, dimensions=embedder.dimensions, code_filters=True)
    except Exception as exc:
        print(f"initialization_failed: {error_message(exc)}", file=sys.stderr)
        return 1
    try:
        evidence = select_evidence(
            args.question, store=store, embedder=embedder,
            source_revision=SOURCE_REVISION,
        )
        client = (
            OpenRouterClient(settings.openrouter_api_key or "", settings.openrouter_model or "")
            if evidence else None
        )
        answer = answer_question(args.question, evidence, client)
        if answer.status == "unverified":
            print(f"Warning: {UNVERIFIED_NOTICE}", file=sys.stderr)
        print(answer.text)
        for item in evidence:
            print(f"[{item.number}] {item.citation} {item.source_url}")
        return 0
    except Exception as exc:
        print(f"question_failed: {error_message(exc)}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
