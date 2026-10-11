import pytest

from code_assistant.vector_store import AtlasVectorStore, checked_vector, vector_index_definition


def test_vector_index_uses_measured_dimension() -> None:
    definition = vector_index_definition(768)
    assert definition["fields"] == [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,
            "similarity": "cosine",
        }
    ]


def test_rejects_wrong_or_nonfinite_vectors() -> None:
    with pytest.raises(ValueError, match="length"):
        checked_vector([0.1], 768)
    with pytest.raises(ValueError, match="finite"):
        checked_vector([float("nan")], 1)


def test_rejects_existing_index_with_wrong_dimension() -> None:
    class ExistingIndexCollection:
        def list_search_indexes(self) -> list[dict]:
            return [
                {
                    "name": "code_vector_index",
                    "type": "vectorSearch",
                    "latestDefinition": vector_index_definition(384),
                }
            ]

    store = object.__new__(AtlasVectorStore)
    store.index_name = "code_vector_index"
    store.dimensions = 768
    store.collection = ExistingIndexCollection()
    with pytest.raises(ValueError, match="does not match"):
        store.ensure_vector_index()
