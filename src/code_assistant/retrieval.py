"""Fuse vector and exact symbol results into bounded, citable source evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

from code_assistant.chunker import REPO_ID, SOURCE_CHUNK_TYPES
from code_assistant.embedder import CodeEmbedder
from code_assistant.source import SOURCE_SCOPES
from code_assistant.vector_store import AtlasVectorStore


_BACKTICK = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)`")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_STRUCTURED = re.compile(r"\b(?:[A-Za-z_]*_[A-Za-z_0-9]+|[A-Z][a-z0-9]+[A-Z]\w*)\b")


@dataclass(frozen=True)
class Evidence:
    number: int
    chunk_id: str
    file_path: str
    source_revision: str
    symbol: str
    start_line: int
    end_line: int
    code: str

    @property
    def citation(self) -> str:
        return f"{self.file_path}:L{self.start_line}-L{self.end_line}"

    @property
    def source_url(self) -> str:
        return (
            f"https://github.com/{REPO_ID}/blob/{self.source_revision}/"
            f"{quote(self.file_path)}#L{self.start_line}-L{self.end_line}"
        )


def explicit_symbols(question: str) -> list[str]:
    """Require code-like syntax; ordinary prose words are too ambiguous."""
    quoted = _BACKTICK.findall(question)
    dotted = _DOTTED.findall(question)
    structured = [
        symbol for symbol in _STRUCTURED.findall(question)
        if not any(symbol in name.split(".") for name in dotted)
    ]
    found = [*quoted, *dotted, *structured]
    return list(dict.fromkeys(found))[:3]


def _valid_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and not path.startswith("/")
        and ".." not in path.split("/")
        and path.endswith(".py")
        and any(path.startswith(f"{scope}/") for scope in SOURCE_SCOPES)
    )


def _ranked_ids(
    dense: list[dict],
    lexical: list[dict],
    *,
    limit: int,
    pinned_id: str | None = None,
) -> list[str]:
    dense_ids = [item["_id"] for item in dense]
    symbol_ids = list(dict.fromkeys(item["_id"] for item in lexical))
    if not symbol_ids:
        return dense_ids[:limit]
    # A single exact qualified symbol is a strong navigational match.
    pinned = [pinned_id] if pinned_id in symbol_ids else []
    scores: dict[str, float] = {}
    for ranked in (dense_ids, symbol_ids):
        for rank, chunk_id in enumerate(ranked, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (60 + rank)
    fused = sorted(scores, key=lambda key: (-scores[key], key))
    return (pinned + [item for item in fused if item not in pinned])[:limit]


def select_evidence(
    question: str,
    *,
    store: AtlasVectorStore,
    embedder: CodeEmbedder,
    source_revision: str,
    limit: int = 5,
    token_budget: int = 2600,
) -> list[Evidence]:
    """Search current source only; return the exact code sent to generation."""
    if not question.strip():
        return []
    if any(f"ultralytics/{outside}/" in question for outside in ("nn", "utils", "solutions")):
        return []
    dense = store.search_code(
        embedder.encode_query(question),
        source_revision=source_revision,
        model_key=embedder.model_key,
        limit=12,
    )
    symbol_results: list[dict] = []
    for symbol in explicit_symbols(question):
        symbol_results.extend(
            store.lookup_symbol(
                symbol, source_revision=source_revision,
                model_key=embedder.model_key, limit=5,
            )
        )
    by_id = {item["_id"]: item for item in [*dense, *symbol_results]}
    exact = [
        item["_id"] for item in symbol_results
        if item.get("symbol_full") in explicit_symbols(question)
        and "." in item.get("symbol_full", "")
    ]
    ordered = _ranked_ids(
        dense, symbol_results, limit=15,
        pinned_id=exact[0] if len(set(exact)) == 1 else None,
    )
    accepted: list[Evidence] = []
    used = 0
    for chunk_id in ordered:
        item = by_id[chunk_id]
        path = item.get("file_path")
        if (
            not _valid_path(path)
            or item.get("source_revision") != source_revision
            or item.get("symbol_type") not in SOURCE_CHUNK_TYPES
        ):
            continue
        start, end = item.get("start_line"), item.get("end_line")
        code = item.get("content")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            continue
        if not isinstance(code, str) or not code.strip():
            continue
        if any(
            prior.file_path == path
            and not (end < prior.start_line or start > prior.end_line)
            for prior in accepted
        ):
            continue
        # Trim at whole source lines and update the cited span to match.
        code_lines = code.splitlines(keepends=True)
        budget_left = token_budget - used
        if budget_left <= 0:
            break
        chosen = []
        for line in code_lines:
            candidate = "".join([*chosen, line])
            if embedder.count_tokens(candidate) > budget_left:
                break
            chosen.append(line)
        if not chosen:
            continue
        trimmed = "".join(chosen)
        actual_end = min(end, start + len(chosen) - 1)
        cost = embedder.count_tokens(trimmed)
        evidence = Evidence(
            number=len(accepted) + 1,
            chunk_id=chunk_id,
            file_path=path,
            source_revision=source_revision,
            symbol=item.get("symbol_full") or path,
            start_line=start,
            end_line=actual_end,
            code=trimmed,
        )
        accepted.append(evidence)
        used += cost
        if len(accepted) >= limit:
            break
    return accepted
