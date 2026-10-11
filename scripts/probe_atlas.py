"""Run a real, isolated Jina-embedding round trip through MongoDB Atlas."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from uuid import uuid4

from sentence_transformers import SentenceTransformer

from code_assistant.config import load_settings
from code_assistant.vector_store import AtlasVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Local dotenv file outside Git")
    args = parser.parse_args()
    settings = load_settings(args.env_file)

    try:
        model = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_model_revision,
            trust_remote_code=True,
        )
        model.max_seq_length = 1024
        question = "Where does YOLO start model training?"
        samples = [
            "def train(model, data):\n    # Start YOLO model training.\n    return model.train(data=data)",
            "def predict(model, image):\n    # Run YOLO inference on an image.\n    return model.predict(source=image)",
        ]
        vectors = model.encode(
            [question, *samples],
            normalize_embeddings=True,
        )
        dimensions = len(vectors[0])
        if dimensions != model.get_embedding_dimension():
            raise ValueError("Jina reported a different dimension from its actual output")
        print("embedding_model:", settings.embedding_model)
        print("embedding_revision:", settings.embedding_model_revision)
        print("measured_dimensions:", dimensions)

        # A separate collection keeps probe documents out of the source-code index.
        store = AtlasVectorStore(
            settings,
            dimensions=dimensions,
            collection_name=f"{settings.mongodb_collection}_probe",
        )
        run_id = uuid4().hex
        ids = [f"probe:{run_id}:{name}" for name in ("train", "predict")]
        try:
            store.ping()
            print("atlas_ping: ok")
            for document_id, sample, vector in zip(ids, samples, vectors[1:]):
                store.upsert(document_id, sample, vector)
            print("probe_documents_written:", len(ids))
            print("vector_index:", store.ensure_vector_index())
            store.wait_until_queryable()
            print("vector_index_queryable: yes")
            # A queryable index can still be catching up with recent writes.
            deadline = time.monotonic() + 45
            results = []
            matched = False
            while time.monotonic() < deadline:
                results = store.search(vectors[0], limit=2)
                matched = bool(results) and results[0].get("_id") == ids[0]
                if matched:
                    break
                time.sleep(3)
            print("retrieved_expected_document_first:", matched)
            print("retrieved_count:", len(results))
            return 0 if matched else 1
        finally:
            try:
                store.delete_ids(ids)
            finally:
                store.close()
    except Exception as exc:
        # PyMongo errors can include connection details; report only the class.
        print(f"probe_failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
