"""Read local configuration without logging or mutating process secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True, repr=False)
class Settings:
    mongodb_uri: str | None
    mongodb_db: str
    mongodb_collection: str
    mongodb_vector_index: str
    openrouter_api_key: str | None
    openrouter_model: str | None
    embedding_model: str

    def require_mongodb_uri(self) -> str:
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI is required for Atlas operations")
        return self.mongodb_uri


def load_settings(env_file: Path | None = None) -> Settings:
    """Load an optional dotenv file, then let process environment take precedence."""
    path = env_file if env_file is not None else Path(".env")
    file_values = dotenv_values(path) if path.is_file() else {}
    values = {**file_values, **os.environ}

    def value(name: str, default: str = "") -> str:
        return str(values.get(name) or default).strip()

    return Settings(
        mongodb_uri=value("MONGODB_URI") or value("MONGODB_CONN_URI") or None,
        mongodb_db=value("MONGODB_DB", "ultralytics_code_assistant"),
        mongodb_collection=value("MONGODB_COLLECTION", "code_chunks"),
        mongodb_vector_index=value("MONGODB_VECTOR_INDEX", "code_vector_index"),
        openrouter_api_key=value("OPENROUTER_API_KEY") or None,
        openrouter_model=value("OPENROUTER_MODEL") or None,
        embedding_model=value("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code"),
    )
