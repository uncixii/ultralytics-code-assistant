"""Read back the current Atlas index and check the source selected for 15 questions.

Uses the production chunker and retrieval path; never writes to Atlas. The
default run replaces evaluation/results.json without LLM calls. --compare-llms
instead makes 18 balanced streaming requests and saves evaluation/llm-comparison.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

from code_assistant.answer import OpenRouterClient, _SYSTEM, answer_question, source_prompt
from code_assistant.chunker import CHUNKING_VERSION, REPO_ID
from code_assistant.config import load_settings
from code_assistant.embedder import CodeEmbedder
from code_assistant.errors import PublicError, error_message
from code_assistant.indexer import chunks_for_file
from code_assistant.retrieval import select_evidence
from code_assistant.source import SOURCE_SCOPES, discover_source
from code_assistant.vector_store import AtlasVectorStore

PROJECT = Path(__file__).resolve().parents[1]
LLM_MODELS = ("poolside/laguna-s-2.1:free", "dots-studio/dots-3-note-preview:free")
LLM_CASES = (
    {
        "id": "ram-cache",
        "question": "Explain the RAM-cache, disk .npy-cache and original-image loading order in BaseDataset.load_image. What does im is None mean? Does a RAM cache hit read or resize the image? Include the disk-cache freshness boundary and invalid/corrupt-cache fallback.",
        "file": "ultralytics/data/base.py", "lines": (249, 321),
    },
    {
        "id": "batchnorm-condition",
        "question": "Does BaseTrainer._model_train put every BatchNorm2d module into eval mode? Explain the execution order and the exact role of freeze_layer_names, including what happens to nonmatching BatchNorm2d modules and matching modules of other types.",
        "file": "ultralytics/engine/trainer.py", "lines": (756, 762),
    },
    {
        "id": "preprocess-boundaries",
        "question": "Explain how BasePredictor.preprocess differs for a one-image NumPy list, a multi-image NumPy list and a torch.Tensor. When are images stacked, channels reversed, values divided by 255, and values converted to float16?",
        "file": "ultralytics/engine/predictor.py", "lines": (183, 207),
    },
)


def compare_llms(snapshot, settings, embedder, store, source_lines) -> None:
    """Three English questions x three trials x two models; identical frozen inputs."""
    output = PROJECT / "evaluation/llm-comparison.json"
    if output.exists():
        raise PublicError("Comparison already exists; archive it before deliberately spending 18 more requests")
    with urlopen(Request("https://openrouter.ai/api/v1/key", headers={
        "Authorization": "Bearer " + (settings.openrouter_api_key or ""),
    }), timeout=20) as response:
        quota = json.load(response)["data"].get("free_model_daily_requests", {})
    if quota.get("remaining", 0) < 18:
        raise PublicError("At least 18 confirmed free requests required; no generation calls made")
    with urlopen("https://openrouter.ai/api/v1/models", timeout=20) as response:
        catalogue = {model["id"]: model for model in json.load(response)["data"]}
    for model_id in LLM_MODELS:
        pricing = catalogue.get(model_id, {}).get("pricing", {})
        if str(pricing.get("prompt")) != "0" or str(pricing.get("completion")) != "0":
            raise PublicError("Both endpoints must have zero prompt/completion pricing")
    sources_by_case = {}
    cases = []
    for fixture in LLM_CASES:
        evidence = select_evidence(fixture["question"], store=store, embedder=embedder,
                                   source_revision=snapshot.revision)
        for hit in evidence:
            if hit.code != "".join(source_lines[hit.file_path][hit.start_line - 1:hit.end_line]):
                raise PublicError("Selected evidence differs from pinned source")
        first, last = fixture["lines"]
        required = {n for n in range(first, last + 1) if source_lines[fixture["file"]][n - 1].strip()}
        supplied = {n for hit in evidence if hit.file_path == fixture["file"]
                    for n in range(hit.start_line, hit.end_line + 1)}
        if required - supplied:
            raise PublicError(f"Required source missing for {fixture['id']}; no generation calls made")
        prompt = source_prompt(fixture["question"], evidence)
        sources_by_case[fixture["id"]] = evidence
        cases.append({"id": fixture["id"], "question": fixture["question"], "user_message": prompt})
    report = {
        "source_revision": snapshot.revision,
        "embedding_model": embedder.model_key, "system_prompt": _SYSTEM,
        "protocol": {"models": LLM_MODELS, "trials_per_question_model": 3,
                     "temperature": 0, "max_tokens": 2000,
                     "reasoning_enabled": False, "stream": True, "timeout_seconds": 60},
        "cases": cases, "attempts": [],
    }

    def save() -> None:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    last_start = None
    for trial in range(1, 4):
        for case_index, case in enumerate(cases):
            order = LLM_MODELS if (trial + case_index) % 2 else tuple(reversed(LLM_MODELS))
            for model_id in order:
                if last_start is not None:
                    time.sleep(max(0, 4 - (time.perf_counter() - last_start)))
                client = OpenRouterClient(settings.openrouter_api_key or "", model_id, stream=True)
                attempt = {"case_id": case["id"], "trial": trial, "requested_model": model_id}
                print(f"Request {len(report['attempts']) + 1}/18: {case['id']}/trial-{trial}/{model_id}", flush=True)
                last_start = time.perf_counter()
                try:
                    result = answer_question(case["question"], sources_by_case[case["id"]], client)
                    attempt["app_status"] = result.status
                except Exception as exc:
                    attempt["app_status"] = "error"
                    attempt["error"] = error_message(exc)
                attempt["request_seconds"] = round(time.perf_counter() - last_start, 4)
                attempt["response"] = client.last_response
                report["attempts"].append(attempt)
                save()
                print(f"Recorded: {attempt['app_status']}, {attempt['request_seconds']}s", flush=True)
                if client.last_response.get("rate_limit_scope") == "free-models-per-day":
                    return
    print(f"Saved 18 attempts: {output}", flush=True)


def line_ranges(numbers: set[int]) -> list[list[int]]:
    """Compact consecutive missing lines into inclusive ranges."""
    ranges = []
    for number in sorted(numbers):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ranges


def audit_index(snapshot, settings, embedder, store, source_lines) -> dict:
    expected = {}
    issues = []
    directories = {
        scope: {"files": 0, "nonempty_lines": 0, "uncovered_lines": 0}
        for scope in SOURCE_SCOPES
    }
    max_tokens = 0
    for path in snapshot.files:
        lines = source_lines[path]
        covered = set()
        for chunk in chunks_for_file(snapshot, path, embedder):
            expected[chunk.chunk_id] = chunk
            max_tokens = max(max_tokens, embedder.count_tokens(chunk.embedding_text))
            if chunk.content != "".join(lines[chunk.start_line - 1:chunk.end_line]):
                issues.append({"source_span_mismatch": chunk.chunk_id})
            else:
                covered.update(range(chunk.start_line, chunk.end_line + 1))
        missing = {n for n, line in enumerate(lines, 1) if line.strip() and n not in covered}
        if missing:
            issues.append({"uncovered_file": path, "lines": line_ranges(missing)})
        scope = next(s for s in SOURCE_SCOPES if path.startswith(s + "/"))
        directories[scope]["files"] += 1
        directories[scope]["nonempty_lines"] += sum(bool(line.strip()) for line in lines)
        directories[scope]["uncovered_lines"] += len(missing)

    # Read every document for this repository, including stale revisions/models.
    actual = {doc["_id"]: doc for doc in store.collection.find({"repo_id": REPO_ID})}
    missing_ids = sorted(expected.keys() - actual.keys())
    extra_ids = sorted(actual.keys() - expected.keys())
    if missing_ids or extra_ids:
        issues.append({"missing_chunk_ids": missing_ids, "unexpected_chunk_ids": extra_ids})
    fields = (
        "file_path", "source_revision", "file_sha256", "symbol_full", "symbol_short",
        "symbol_type", "start_line", "end_line", "part", "content", "embedding_text",
    )
    for identifier in sorted(expected.keys() & actual.keys()):
        doc, chunk = actual[identifier], expected[identifier]
        differing = [key for key in fields if doc.get(key) != getattr(chunk, key)]
        differing.extend(key for key in ("synthetic", "fallback_reason") if key in doc)
        if doc.get("embedding_model") != embedder.model_key:
            differing.append("embedding_model")
        if differing:
            issues.append({"chunk_id": identifier, "different_fields": differing})
        vector = doc.get("embedding", [])
        if (not isinstance(vector, list) or len(vector) != embedder.dimensions
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)):
            issues.append({"invalid_vector": identifier})

    manifest = store.client[settings.mongodb_db][settings.mongodb_collection + "_manifest"]
    manifests = {doc["file_path"]: doc for doc in manifest.find({"repo_id": REPO_ID})}
    expected_paths = set(snapshot.files)
    if set(manifests) != expected_paths:
        issues.append({"missing_manifests": sorted(expected_paths - manifests.keys()),
                       "unexpected_manifests": sorted(manifests.keys() - expected_paths)})
    for path in sorted(expected_paths & manifests.keys()):
        doc = manifests[path]
        chunks = [chunk for chunk in expected.values() if chunk.file_path == path]
        if (set(doc.get("chunk_ids", [])) != {c.chunk_id for c in chunks}
                or doc.get("file_sha256") != chunks[0].file_sha256
                or doc.get("source_revision") != snapshot.revision
                or doc.get("embedding_model") != embedder.model_key
                or doc.get("chunking_version") != CHUNKING_VERSION):
            issues.append({"manifest_mismatch": path})
    return {
        "passed": not issues,
        "directories": directories,
        "source_files": len(snapshot.files),
        "stored_chunks": len(actual),
        "chunk_types": dict(Counter(chunk.symbol_type for chunk in expected.values())),
        "embedding_dimensions": embedder.dimensions,
        "max_embedding_input_tokens": max_tokens,
        "issues": issues,
    }


def evaluate_questions(questions, snapshot, embedder, store, source_lines) -> dict:
    cases = []
    for item in questions:
        evidence = select_evidence(
            item["question"], store=store, embedder=embedder,
            source_revision=snapshot.revision,
        )
        # Check the exact text selected for generation, including any trimming.
        for hit in evidence:
            if hit.code != "".join(source_lines[hit.file_path][hit.start_line - 1:hit.end_line]):
                raise PublicError(f"Selected evidence differs from source: {hit.citation}")
        answerable = item.get("answerable", True)
        file_rank = symbol_rank = None
        missing = set()
        if answerable:
            path = item["expected_file"]
            symbols = item.get("expected_symbols") or [item["expected_symbol"]]
            file_rank = next((i for i, hit in enumerate(evidence, 1) if hit.file_path == path), None)
            symbol_rank = next((i for i, hit in enumerate(evidence, 1)
                                if hit.file_path == path and hit.symbol in symbols), None)
            required = set()
            for first, last in item["required_lines"]:
                if not 1 <= first <= last <= len(source_lines[path]):
                    raise PublicError(f"Invalid required source range: {item['id']}")
                required.update(n for n in range(first, last + 1) if source_lines[path][n - 1].strip())
            selected = {n for hit in evidence if hit.file_path == path
                        for n in range(hit.start_line, hit.end_line + 1)}
            missing = required - selected
            outcome = "sufficient_source" if symbol_rank and not missing else "partial_source" if symbol_rank else "miss"
        else:
            outcome = "expected_empty" if not evidence else "unexpected_source"
        case = {
            "id": item["id"], "outcome": outcome,
            "file_rank": file_rank, "symbol_rank": symbol_rank,
            "missing_required_lines": line_ranges(missing),
            "selected_sources": [f"{hit.symbol} — {hit.citation}" for hit in evidence],
        }
        if outcome in ("miss", "partial_source"):
            # Diagnose only failed questions. Preserve actual vector ranks rather
            # than inferring them from the final five selected source ranges.
            search_limit = 50 if outcome == "miss" else 12
            candidates = store.search_code(
                embedder.encode_query(item["question"]),
                source_revision=snapshot.revision,
                model_key=embedder.model_key, limit=search_limit,
            )
            case["vector_diagnostic"] = {
                "limit": search_limit,
                "returned": len(candidates),
                "expected_method_matches": [
                    {"rank": rank, "source": f"{hit['file_path']}:L{hit['start_line']}-L{hit['end_line']}"}
                    for rank, hit in enumerate(candidates, 1)
                    if hit.get("file_path") == path and hit.get("symbol_full") in symbols
                ],
            }
        cases.append(case)
        print(f"{item['id']}: {outcome}", flush=True)
    positive = [case for case, question in zip(cases, questions) if question.get("answerable", True)]
    negative = [case for case, question in zip(cases, questions) if not question.get("answerable", True)]
    return {
        "positive_questions": len(positive),
        "file_hits_at_5": sum(case["file_rank"] is not None for case in positive),
        "method_hits_at_5": sum(case["symbol_rank"] is not None for case in positive),
        "sufficient_source": sum(case["outcome"] == "sufficient_source" for case in positive),
        "partial_source": sum(case["outcome"] == "partial_source" for case in positive),
        "misses": sum(case["outcome"] == "miss" for case in positive),
        "negative_questions": len(negative),
        "correct_empty_results": sum(case["outcome"] == "expected_empty" for case in negative),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT / ".source/ultralytics")
    parser.add_argument("--dataset", type=Path, default=PROJECT / "evaluation/questions.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT / "evaluation/results.json")
    parser.add_argument("--compare-llms", action="store_true", help="Spend 18 balanced reasoning-off requests; save a JSON comparison")
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    settings = load_settings()
    settings.require_mongodb_uri()
    snapshot = discover_source(args.repo)
    source_lines = {path: (snapshot.root / path).read_text(encoding="utf-8").splitlines(keepends=True)
                    for path in snapshot.files}
    embedder = CodeEmbedder(settings)
    store = AtlasVectorStore(settings, dimensions=embedder.dimensions, code_filters=True)
    report = {
        "source_revision": snapshot.revision,
        "embedding_model": embedder.model_key,
    }
    try:
        store.ping()
        if args.compare_llms:
            compare_llms(snapshot, settings, embedder, store, source_lines)
            return
        report["index"] = audit_index(snapshot, settings, embedder, store, source_lines)
        if report["index"]["passed"]:
            report["retrieval"] = evaluate_questions(questions, snapshot, embedder, store, source_lines)
    finally:
        store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved current evaluation: {args.output}")
    if not report["index"]["passed"]:
        raise SystemExit("Index integrity failed; review index.issues before testing retrieval")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"evaluation_failed: {error_message(exc)}", file=sys.stderr)
        raise SystemExit(1) from None
