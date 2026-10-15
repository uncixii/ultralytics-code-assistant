"""Idempotent Atlas indexing for one pinned Ultralytics source snapshot."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from code_assistant.chunker import CHUNKING_VERSION, Chunk, REPO_ID, chunk_python
from code_assistant.config import Settings, load_settings
from code_assistant.embedder import CodeEmbedder
from code_assistant.errors import PublicError
from code_assistant.source import SourceSnapshot, discover_source
from code_assistant.vector_store import AtlasVectorStore


@dataclass
class IndexStats:
    files_seen: int = 0
    files_skipped: int = 0
    files_indexed: int = 0
    files_deleted: int = 0
    chunks_written: int = 0


def chunks_for_file(snapshot: SourceSnapshot, path: str, embedder: CodeEmbedder) -> list[Chunk]:
    source = (snapshot.root / path).read_text(encoding="utf-8")
    try:
        chunks = chunk_python(path, source, snapshot.revision)
    except PublicError:
        # An estimate can overstate the size of a physical line or its header.
        chunks = None
    # The inexpensive first pass is conservative. Check with the actual model
    # tokenizer and redo only exceptional files with precise token counts.
    if chunks is None or any(embedder.count_tokens(chunk.embedding_text) > 800 for chunk in chunks):
        chunks = chunk_python(
            path, source, snapshot.revision,
            max_tokens=700, count_tokens=embedder.count_tokens,
        )
    oversized = [chunk for chunk in chunks if embedder.count_tokens(chunk.embedding_text) > 1024]
    if oversized:
        raise PublicError(f"Chunks exceed embedding model input limit: {path}")
    return chunks


def index_snapshot(
    snapshot: SourceSnapshot,
    settings: Settings,
    embedder: CodeEmbedder,
    store: AtlasVectorStore,
    *,
    force: bool = False,
) -> IndexStats:
    if store.dimensions != embedder.dimensions:
        raise PublicError("Embedding dimensions do not match Atlas index. Check the embedding model and Atlas index configuration.")
    if not store.filter_fields:
        raise PublicError("Atlas code filters are not configured")
    store.ping()
    # A regular index creates the collection on a fresh Atlas deployment.
    store.ensure_code_indexes()
    store.ensure_vector_index()
    store.wait_until_queryable()
    manifest = store.client[settings.mongodb_db][f"{settings.mongodb_collection}_manifest"]
    manifest.create_index([("repo_id", 1), ("file_path", 1)], unique=True)
    stats = IndexStats()
    active_paths = set(snapshot.files)
    for path in snapshot.files:
        stats.files_seen += 1
        source = (snapshot.root / path).read_text(encoding="utf-8")
        file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        key = {"repo_id": REPO_ID, "file_path": path}
        previous = manifest.find_one(key)
        if (
            not force
            and previous
            and all((
                previous.get("file_sha256") == file_hash,
                previous.get("source_revision") == snapshot.revision,
                previous.get("embedding_model") == embedder.model_key,
                previous.get("chunking_version") == CHUNKING_VERSION,
            ))
            and previous.get("chunk_ids") is not None
            and store.contains_chunks(
                previous["chunk_ids"], source_revision=snapshot.revision,
                model_key=embedder.model_key,
            )
        ):
            stats.files_skipped += 1
            continue
        chunks = chunks_for_file(snapshot, path, embedder)
        vectors = embedder.encode([chunk.embedding_text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise PublicError(f"Embedding count mismatch: {path}")
        store.upsert_chunks(chunks, vectors, embedder.model_key)
        old_ids = set(previous.get("chunk_ids", ())) if previous else set()
        new_ids = {chunk.chunk_id for chunk in chunks}
        store.delete_ids(sorted(old_ids - new_ids))
        manifest.replace_one(
            key,
            {
                **key,
                "file_sha256": file_hash,
                "source_revision": snapshot.revision,
                "embedding_model": embedder.model_key,
                "chunking_version": CHUNKING_VERSION,
                "chunk_ids": sorted(new_ids),
            },
            upsert=True,
        )
        stats.files_indexed += 1
        stats.chunks_written += len(chunks)

    for previous in manifest.find({"repo_id": REPO_ID}, {"file_path": 1, "chunk_ids": 1}):
        if previous["file_path"] not in active_paths:
            store.delete_ids(previous.get("chunk_ids", ()))
            manifest.delete_one({"_id": previous["_id"]})
            stats.files_deleted += 1
    # Also upgrade unchanged files; removing metadata does not require re-embedding.
    store.remove_legacy_chunk_metadata()
    return stats


def index_repository(repo: Path, settings: Settings | None = None, *, force: bool = False) -> IndexStats:
    settings = settings or load_settings()
    settings.require_mongodb_uri()
    snapshot = discover_source(repo)
    embedder = CodeEmbedder(settings)
    store = AtlasVectorStore(settings, dimensions=embedder.dimensions, code_filters=True)
    try:
        return index_snapshot(snapshot, settings, embedder, store, force=force)
    finally:
        store.close()
