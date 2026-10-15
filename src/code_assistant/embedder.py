"""One pinned local embedding model for both source and questions."""

from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from code_assistant.config import Settings
from code_assistant.errors import PublicError


class CodeEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model_key = f"{settings.embedding_model}@{settings.embedding_model_revision}"
        self.model = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_model_revision,
            trust_remote_code=True,
        )
        self.model.max_seq_length = 1024
        self.dimensions = self.model.get_embedding_dimension()
        if self.dimensions is None or self.dimensions <= 0:
            raise PublicError(
                "Embedding model did not report a valid dimension. "
                "Check EMBEDDING_MODEL and its revision."
            )
        self.tokenizer = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(list(texts), batch_size=16, normalize_embeddings=True)
        if len(vectors) != len(texts):
            raise PublicError("Embedding model returned the wrong number of vectors.")
        return [vector.tolist() for vector in vectors]

    def encode_query(self, question: str) -> list[float]:
        return self.encode([question])[0]
