"""Structure-aware Python chunks with exact source spans."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Callable

from code_assistant.errors import PublicError

REPO_ID = "ultralytics/ultralytics"
CHUNKING_VERSION = 2
SOURCE_CHUNK_TYPES = ("function", "method", "class_context", "module", "window")
TokenCount = Callable[[str], int]


def approximate_tokens(text: str) -> int:
    """Cheap local fallback; indexing should pass the embedding model tokenizer."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_path: str
    source_revision: str
    file_sha256: str
    symbol_full: str
    symbol_short: str
    symbol_type: str
    start_line: int
    end_line: int
    part: int
    content: str
    embedding_text: str

    @property
    def citation(self) -> str:
        return f"{self.file_path}:L{self.start_line}-L{self.end_line}"


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *(item.lineno for item in decorators)])


def _id(path: str, kind: str, symbol: str, start: int, end: int, part: int) -> str:
    fields = (REPO_ID, path, kind, symbol, str(start), str(end), str(part))
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def _header(path: str, symbol: str, kind: str, signature: str = "") -> str:
    items = [f"File: {path}", f"Symbol: {symbol}", f"Type: {kind}"]
    if signature:
        items.append(f"Signature: {signature}")
    return "\n".join(items)


def _signature(node: ast.AST, lines: list[str]) -> str:
    """Keep a short, source-derived signature without reconstructing Python."""
    body = getattr(node, "body", ())
    end = body[0].lineno - 1 if body else node.lineno
    source = "".join(lines[node.lineno - 1 : max(node.lineno, end)])
    if not source.strip():
        source = lines[node.lineno - 1]
    return " ".join(source.strip().split())[:350]


def _make_chunk(
    path: str,
    revision: str,
    file_hash: str,
    symbol: str,
    kind: str,
    start: int,
    end: int,
    part: int,
    content: str,
    signature: str = "",
) -> Chunk:
    header = _header(path, symbol, kind, signature)
    return Chunk(
        chunk_id=_id(path, kind, symbol, start, end, part),
        file_path=path,
        source_revision=revision,
        file_sha256=file_hash,
        symbol_full=symbol,
        symbol_short=symbol.rsplit(".", 1)[-1],
        symbol_type=kind,
        start_line=start,
        end_line=end,
        part=part,
        content=content,
        embedding_text=f"{header}\n\n{content}",
    )


def _split_lines(
    lines: list[str],
    start: int,
    end: int,
    budget: int,
    count: TokenCount,
    boundaries: set[int] | None = None,
    *,
    file_path: str,
) -> list[tuple[int, int, str]]:
    """Keep complete source lines, preferring statement boundaries."""
    if start > end:
        return []
    boundaries = boundaries or set()
    result: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = start

    def flush(last: int) -> None:
        nonlocal current, current_start
        if current:
            result.append((current_start, last, "".join(current)))
            current = []

    for lineno in range(start, end + 1):
        line = lines[lineno - 1]
        if current and count("".join(current) + line) > budget:
            flush(lineno - 1)
            current_start = lineno
        if count(line) > budget:
            raise PublicError(
                f"Cannot split {file_path}:L{lineno}: a complete source line exceeds "
                f"the {budget}-token content budget. Indexing stopped to preserve exact source citations."
            )
        current.append(line)
        if lineno in boundaries and count("".join(current)) >= budget * 0.65:
            flush(lineno)
            current_start = lineno + 1
    flush(end)
    return result


def chunk_python(
    file_path: str,
    source: str,
    source_revision: str,
    *,
    max_tokens: int = 700,
    count_tokens: TokenCount = approximate_tokens,
    fallback_lines: int = 60,
) -> list[Chunk]:
    """Return bounded, source-grounded chunks for one repository-relative .py file."""
    if max_tokens < 80:
        raise PublicError("max_tokens must be at least 80")
    if not file_path.endswith(".py") or not file_path.startswith(
        ("ultralytics/models/", "ultralytics/engine/", "ultralytics/data/")
    ):
        raise PublicError("file_path is outside the allowed Python source scope")
    if ".." in file_path.split("/") or file_path.startswith("/"):
        raise PublicError("file_path must be a safe relative path")
    if not source.strip():
        return []
    lines = source.splitlines(keepends=True)
    file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    chunks: list[Chunk] = []

    def add_span(
        symbol: str,
        kind: str,
        start: int,
        end: int,
        signature: str = "",
        boundaries: set[int] | None = None,
    ) -> None:
        header_cost = count_tokens(_header(file_path, symbol, kind, signature)) + 4
        budget = max_tokens - header_cost
        if budget < 1:
            raise PublicError(f"Metadata exceeds chunk token budget: {symbol}")
        spans = _split_lines(lines, start, end, budget, count_tokens, boundaries, file_path=file_path)
        for part, (first, last, code) in enumerate(spans, 1):
            if code.strip():
                chunks.append(
                    _make_chunk(
                        file_path, source_revision, file_hash, symbol, kind,
                        first, last, part, code, signature,
                    )
                )

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        for first in range(1, len(lines) + 1, fallback_lines):
            add_span(file_path, "window", first, min(first + fallback_lines - 1, len(lines)))
        return chunks

    def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        if isinstance(node, ast.ClassDef):
            symbol = ".".join((*parents, node.name))
            # Store the real class header, docstring, assignments and comments
            # in the gaps around methods/nested classes, without duplicating them.
            cursor = _start(node)
            for child in node.body:
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_span(symbol, "class_context", cursor, _start(child) - 1)
                    visit(child, (*parents, node.name))
                    cursor = child.end_lineno + 1
            add_span(symbol, "class_context", cursor, node.end_lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol = ".".join((*parents, node.name))
            kind = "method" if parents else "function"
            boundaries = {child.end_lineno for child in node.body if hasattr(child, "end_lineno")}
            add_span(symbol, kind, _start(node), node.end_lineno, _signature(node, lines), boundaries)

    definitions = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, definitions):
            visit(node)

    # Include imports, constants, comments and other module-level material
    # without duplicating the bodies of definitions.
    excluded: set[int] = set()
    for node in tree.body:
        if isinstance(node, definitions):
            excluded.update(range(_start(node), node.end_lineno + 1))
    run_start: int | None = None
    for lineno in range(1, len(lines) + 2):
        included = lineno <= len(lines) and lineno not in excluded
        if included and run_start is None:
            run_start = lineno
        elif not included and run_start is not None:
            add_span(file_path, "module", run_start, lineno - 1)
            run_start = None

    return sorted(chunks, key=lambda item: (item.start_line, item.end_line, item.symbol_type, item.part))
