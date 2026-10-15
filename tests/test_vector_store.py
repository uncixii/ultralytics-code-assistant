import pytest

from code_assistant.vector_store import CODE_FILTER_FIELDS, AtlasVectorStore, vector_index_definition


@pytest.mark.parametrize(("count", "expected"), [(1, False), (2, True)])
def test_chunk_presence_checks_all_ids_in_the_current_source_and_model(count, expected):
    class Collection:
        query = None

        def count_documents(self, query):
            self.query = query
            return count

    store = object.__new__(AtlasVectorStore)
    store.collection = Collection()
    assert store.contains_chunks(["a", "b", "a"], source_revision="revision", model_key="model") is expected
    assert store.collection.query == {
        "_id": {"$in": ["a", "b"]}, "repo_id": "ultralytics/ultralytics",
        "source_revision": "revision", "embedding_model": "model",
    }


def test_index_and_query_use_model_dimension_and_source_filters():
    dimensions = 1024
    fields = vector_index_definition(dimensions, filter_fields=CODE_FILTER_FIELDS)["fields"]
    assert next(field for field in fields if field["type"] == "vector") == {
        "type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine",
    }
    assert {field["path"] for field in fields if field["type"] == "filter"} == {
        "repo_id", "source_revision", "embedding_model",
    }

    class Collection:
        pipeline = None

        def aggregate(self, pipeline):
            self.pipeline = pipeline
            return []

    store = object.__new__(AtlasVectorStore)
    store.dimensions = dimensions
    store.index_name = "code_vector_index"
    store.filter_fields = CODE_FILTER_FIELDS
    store.collection = Collection()
    vector = [0.1] * dimensions
    store.search_code(vector, source_revision="revision", model_key="model", limit=5)
    query = store.collection.pipeline[0]["$vectorSearch"]
    assert query["queryVector"] == vector
    assert query["filter"] == {
        "repo_id": "ultralytics/ultralytics", "source_revision": "revision", "embedding_model": "model",
    }
    assert query["limit"] == 5


def test_model_dimension_change_updates_index_but_invalid_vectors_are_rejected():
    class Collection:
        updated = None

        def list_search_indexes(self):
            return [{
                "name": "code_vector_index", "type": "vectorSearch",
                "latestDefinition": vector_index_definition(384, filter_fields=CODE_FILTER_FIELDS),
            }]

        def update_search_index(self, name, definition):
            self.updated = (name, definition)

    store = object.__new__(AtlasVectorStore)
    store.index_name = "code_vector_index"
    store.dimensions = 768
    store.filter_fields = CODE_FILTER_FIELDS
    store.collection = Collection()
    assert store.ensure_vector_index() == "updated"
    assert store.collection.updated == (
        store.index_name, vector_index_definition(768, filter_fields=CODE_FILTER_FIELDS)
    )
    with pytest.raises(ValueError, match="length"):
        store.search_code([0.1], source_revision="revision", model_key="model")

    store.filter_fields = (*CODE_FILTER_FIELDS, "other")
    with pytest.raises(ValueError, match="does not match"):
        store.ensure_vector_index()


def test_fresh_index_uses_model_dimension_and_waits_for_ready(monkeypatch):
    class Collection:
        created = None

        def list_search_indexes(self):
            return []

        def create_search_index(self, model):
            self.created = model.document

    store = object.__new__(AtlasVectorStore)
    store.index_name = "code_vector_index"
    store.dimensions = 768
    store.filter_fields = CODE_FILTER_FIELDS
    store.collection = Collection()
    assert store.ensure_vector_index() == "created"
    assert store.collection.created["definition"] == vector_index_definition(768, filter_fields=CODE_FILTER_FIELDS)

    statuses = iter([{"queryable": True, "status": "PENDING"}, {"queryable": True, "status": "READY"}])
    monkeypatch.setattr(store, "_existing_index", lambda: next(statuses))
    sleeps = []
    monkeypatch.setattr("code_assistant.vector_store.time.sleep", sleeps.append)
    store.wait_until_queryable()
    assert sleeps == [3]


def test_name_lookup_accepts_source_without_legacy_flags_and_excludes_summaries():
    source = {
        "_id": "source", "repo_id": "ultralytics/ultralytics",
        "source_revision": "revision", "embedding_model": "model",
        "symbol_full": "Runner", "symbol_short": "Runner", "symbol_type": "class_context",
    }
    summary = {**source, "_id": "old-summary", "symbol_type": "class_summary"}

    class Cursor(list):
        def limit(self, count):
            return self[:count]

    class Collection:
        def find(self, query, projection):
            return Cursor(
                doc for doc in (source, summary)
                if all(doc.get(key) in value["$in"] if isinstance(value, dict)
                       else doc.get(key) == value for key, value in query.items())
            )

    store = object.__new__(AtlasVectorStore)
    store.collection = Collection()
    assert store.lookup_symbol("Runner", source_revision="revision", model_key="model") == [source]
