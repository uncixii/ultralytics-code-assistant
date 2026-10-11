"""Small MongoDB Atlas Vector Search adapter with explicit index checks."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from code_assistant.config import Settings

VECTOR_FIELD = "embedding"


def vector_index_definition(dimensions: int) -> dict[str, Any]:
    if dimensions <= 0:
        raise ValueError("Vector dimensions must be positive")
    return {
        "fields": [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": dimensions,
                "similarity": "cosine",
            }
        ]
    }


def checked_vector(vector: Sequence[float], dimensions: int) -> list[float]:
    if len(vector) != dimensions:
        raise ValueError("Vector length does not match the Atlas index dimensions")
    result = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Vector values must be finite")
    return result


class AtlasVectorStore:
    def __init__(
        self,
        settings: Settings,
        *,
        dimensions: int,
        collection_name: str | None = None,
        client: MongoClient[Any] | None = None,
    ) -> None:
        vector_index_definition(dimensions)
        self.dimensions = dimensions
        self.index_name = settings.mongodb_vector_index
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
        """Create the index if absent; reject a reused name with a different shape."""
        existing = self._existing_index()
        if existing is not None:
            definition = existing.get("latestDefinition") or existing.get("definition") or {}
            fields = definition.get("fields", [])
            expected = vector_index_definition(self.dimensions)["fields"][0]
            if existing.get("type") != "vectorSearch" or expected not in fields:
                raise ValueError("Existing Atlas vector index definition does not match")
            return "existing"

        self.collection.create_search_index(
            model=SearchIndexModel(
                definition=vector_index_definition(self.dimensions),
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
                    raise RuntimeError("Atlas vector index build failed")
                if index.get("queryable") is True:
                    return
            time.sleep(3)
        raise TimeoutError("Atlas vector index was not queryable before the timeout")

    def upsert(self, document_id: str, text: str, embedding: Sequence[float]) -> None:
        self.collection.replace_one(
            {"_id": document_id},
            {
                "_id": document_id,
                "text": text,
                VECTOR_FIELD: checked_vector(embedding, self.dimensions),
            },
            upsert=True,
        )

    def search(self, embedding: Sequence[float], *, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("Search limit must be positive")
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": VECTOR_FIELD,
                    "queryVector": checked_vector(embedding, self.dimensions),
                    "numCandidates": max(100, 20 * limit),
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self.collection.aggregate(pipeline))

    def delete_ids(self, document_ids: Sequence[str]) -> None:
        self.collection.delete_many({"_id": {"$in": list(document_ids)}})
