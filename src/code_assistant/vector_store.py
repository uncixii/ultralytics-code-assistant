"""Small MongoDB Atlas Vector Search adapter with explicit index checks."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from pymongo import MongoClient
from pymongo.operations import ReplaceOne, SearchIndexModel

from code_assistant.config import Settings
from code_assistant.errors import PublicError
from code_assistant.chunker import Chunk, REPO_ID, SOURCE_CHUNK_TYPES

VECTOR_FIELD = "embedding"


CODE_FILTER_FIELDS = ("repo_id", "source_revision", "embedding_model")


def vector_index_definition(
    dimensions: int, *, filter_fields: Sequence[str] = ()
) -> dict[str, Any]:
    if dimensions <= 0:
        raise PublicError("Vector dimensions must be positive")
    return {
        "fields": [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": dimensions,
                "similarity": "cosine",
            },
            *({"type": "filter", "path": field} for field in filter_fields),
        ]
    }


def checked_vector(vector: Sequence[float], dimensions: int) -> list[float]:
    if len(vector) != dimensions:
        raise PublicError("Vector length does not match the Atlas index dimensions")
    result = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in result):
        raise PublicError("Vector values must be finite")
    return result


class AtlasVectorStore:
    def __init__(
        self,
        settings: Settings,
        *,
        dimensions: int,
        collection_name: str | None = None,
        client: MongoClient[Any] | None = None,
        code_filters: bool = False,
    ) -> None:
        vector_index_definition(dimensions)
        self.dimensions = dimensions
        self.index_name = settings.mongodb_vector_index
        self.filter_fields = CODE_FILTER_FIELDS if code_filters else ()
        self._owns_client = client is None
        self.client = (
            client
            if client is not None
            else MongoClient(
                settings.require_mongodb_uri(),
                serverSelectionTimeoutMS=15_000,
                connectTimeoutMS=10_000,
            )
        )
        self.collection = self.client[settings.mongodb_db][
            collection_name or settings.mongodb_collection
        ]

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def ping(self) -> None:
        self.client.admin.command("ping")

    def _existing_index(self) -> dict[str, Any] | None:
        return next(
            (
                index
                for index in self.collection.list_search_indexes()
                if index.get("name") == self.index_name
            ),
            None,
        )

    def ensure_vector_index(self) -> str:
        """Create the index, or migrate its vector dimension for a new encoder."""
        existing = self._existing_index()
        if existing is not None:
            definition = existing.get("latestDefinition") or existing.get("definition") or {}
            fields = definition.get("fields", [])
            expected = vector_index_definition(
                self.dimensions, filter_fields=getattr(self, "filter_fields", ())
            )["fields"]
            if existing.get("type") != "vectorSearch" or not all(
                field in fields for field in expected
            ):
                # The normal build path also handles changing embedding models.
                # Only migrate dimension-only differences; reject unrelated indexes.
                old_vector = next((field for field in fields if field.get("type") == "vector"), {})
                migrated = [
                    {**field, "numDimensions": self.dimensions}
                    if field is old_vector else field for field in fields
                ]
                if (existing.get("type") == "vectorSearch" and old_vector
                        and len(migrated) == len(expected)
                        and all(field in migrated for field in expected)):
                    self.collection.update_search_index(
                        self.index_name, {"fields": expected}
                    )
                    return "updated"
                raise PublicError("Existing Atlas vector index definition does not match")
            return "existing"

        self.collection.create_search_index(
            model=SearchIndexModel(
                definition=vector_index_definition(
                    self.dimensions, filter_fields=self.filter_fields
                ),
                name=self.index_name,
                type="vectorSearch",
            )
        )
        return "created"

    def wait_until_queryable(self, *, timeout_seconds: float = 240) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            index = self._existing_index()
            if index is not None:
                if index.get("status") == "FAILED":
                    raise PublicError("Atlas vector index build failed. Check its status in Atlas before rerunning the indexer.")
                if index.get("queryable") is True and index.get("status") == "READY":
                    return
            time.sleep(3)
        raise PublicError("Atlas vector index is not ready yet. Check its build status and rerun the indexer when it is queryable.")

    def delete_ids(self, document_ids: Sequence[str]) -> None:
        if document_ids:
            self.collection.delete_many({"_id": {"$in": list(document_ids)}})

    def ensure_code_indexes(self) -> None:
        """Exact-match indexes are distinct from the Atlas vectorSearch index."""
        self.collection.create_index(
            [("repo_id", 1), ("source_revision", 1), ("embedding_model", 1), ("symbol_full", 1)]
        )
        self.collection.create_index(
            [("repo_id", 1), ("source_revision", 1), ("embedding_model", 1), ("symbol_short", 1)]
        )
        self.collection.create_index(
            [("repo_id", 1), ("source_revision", 1), ("embedding_model", 1), ("file_path", 1)]
        )

    def count_code_chunks(self, *, source_revision: str, model_key: str) -> int:
        return self.collection.count_documents({
            "repo_id": REPO_ID,
            "source_revision": source_revision,
            "embedding_model": model_key,
        })

    def contains_chunks(
        self, chunk_ids: Sequence[str], *, source_revision: str, model_key: str
    ) -> bool:
        """Check a file's recorded chunks before skipping its rebuild."""
        identifiers = list(dict.fromkeys(chunk_ids))
        if not identifiers:
            return True
        return self.collection.count_documents({
            "_id": {"$in": identifiers},
            "repo_id": REPO_ID,
            "source_revision": source_revision,
            "embedding_model": model_key,
        }) == len(identifiers)

    def remove_legacy_chunk_metadata(self) -> None:
        """Clean old source documents without changing their text or vectors."""
        self.collection.update_many(
            {"repo_id": REPO_ID, "symbol_type": {"$in": list(SOURCE_CHUNK_TYPES)},
             "$or": [{"synthetic": {"$exists": True}}, {"fallback_reason": {"$exists": True}}]},
            {"$unset": {"synthetic": "", "fallback_reason": ""}},
        )

    def upsert_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], model_key: str
    ) -> None:
        if len(chunks) != len(embeddings):
            raise PublicError("Chunk and embedding counts differ")
        if not chunks:
            return
        operations = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            document = {
                "_id": chunk.chunk_id,
                "repo_id": REPO_ID,
                "source_revision": chunk.source_revision,
                "file_sha256": chunk.file_sha256,
                "file_path": chunk.file_path,
                "symbol_full": chunk.symbol_full,
                "symbol_short": chunk.symbol_short,
                "symbol_type": chunk.symbol_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "part": chunk.part,
                "content": chunk.content,
                "embedding_text": chunk.embedding_text,
                "embedding_model": model_key,
                VECTOR_FIELD: checked_vector(embedding, self.dimensions),
            }
            operations.append(ReplaceOne({"_id": chunk.chunk_id}, document, upsert=True))
        self.collection.bulk_write(operations, ordered=True)

    def search_code(
        self,
        embedding: Sequence[float],
        *,
        source_revision: str,
        model_key: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not self.filter_fields:
            raise PublicError("Code index requires vector filter fields")
        if limit <= 0:
            raise PublicError("Search limit must be positive")
        scope = {
            "repo_id": REPO_ID,
            "source_revision": source_revision,
            "embedding_model": model_key,
        }
        pipeline = [
            {"$vectorSearch": {
                "index": self.index_name,
                "path": VECTOR_FIELD,
                "queryVector": checked_vector(embedding, self.dimensions),
                "numCandidates": max(100, 20 * limit),
                "limit": limit,
                "filter": scope,
            }},
            {"$project": {
                "_id": 1,
                "repo_id": 1,
                "source_revision": 1,
                "file_path": 1,
                "symbol_full": 1,
                "symbol_short": 1,
                "symbol_type": 1,
                "start_line": 1,
                "end_line": 1,
                "part": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }},
        ]
        return list(self.collection.aggregate(pipeline))

    def lookup_symbol(
        self, symbol: str, *, source_revision: str, model_key: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        base = {
            "repo_id": REPO_ID,
            "source_revision": source_revision,
            "embedding_model": model_key,
            "symbol_type": {"$in": list(SOURCE_CHUNK_TYPES)},
        }
        result = []
        for field in ("symbol_full", "symbol_short"):
            for document in self.collection.find(
                {**base, field: symbol}, {VECTOR_FIELD: 0}
            ).limit(limit):
                if document["_id"] not in {item["_id"] for item in result}:
                    result.append(document)
                if len(result) >= limit:
                    return result
        return result
