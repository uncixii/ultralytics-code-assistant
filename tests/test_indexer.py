from types import SimpleNamespace

import pytest

from code_assistant.chunker import CHUNKING_VERSION
from code_assistant.indexer import chunks_for_file, index_snapshot
from code_assistant.source import SourceSnapshot


PATH = "ultralytics/engine/example.py"
REVISION = "a" * 40


class FakeManifest:
    def __init__(self):
        self.documents = {}

    def create_index(self, *args, **kwargs):
        pass

    def find_one(self, key):
        return self.documents.get(key["file_path"])

    def replace_one(self, key, document, upsert=False):
        self.documents[key["file_path"]] = document

    def find(self, *args, **kwargs):
        return list(self.documents.values())


class FakeStore:
    dimensions = 2
    filter_fields = ("repo_id", "source_revision", "embedding_model")

    def __init__(self, manifest):
        self.client = {"db": {"chunks_manifest": manifest}}
        self.written = []
        self.deleted = []
        self.metadata_cleanups = 0
        self.chunk_ids = set()

    def ping(self):
        pass

    def ensure_vector_index(self):
        pass

    def wait_until_queryable(self):
        pass

    def ensure_code_indexes(self):
        pass

    def upsert_chunks(self, chunks, vectors, model_key):
        self.written.extend(chunk.chunk_id for chunk in chunks)
        self.chunk_ids.update(chunk.chunk_id for chunk in chunks)

    def contains_chunks(self, chunk_ids, **kwargs):
        return set(chunk_ids) <= self.chunk_ids

    def delete_ids(self, ids):
        self.deleted.extend(ids)
        self.chunk_ids.difference_update(ids)

    def remove_legacy_chunk_metadata(self):
        self.metadata_cleanups += 1


class FakeEmbedder:
    dimensions = 2
    model_key = "model@revision"

    def count_tokens(self, text):
        return len(text.split())

    def encode(self, texts):
        return [[0.1, 0.2] for _ in texts]


@pytest.fixture
def index_case(tmp_path):
    file = tmp_path / PATH
    file.parent.mkdir(parents=True)
    file.write_text("class Runner:\n    enabled = True\n    def run(self):\n        return self.enabled\n")
    snapshot = SourceSnapshot(
        tmp_path, REVISION, (PATH,),
    )
    store = FakeStore(FakeManifest())
    settings = SimpleNamespace(mongodb_db="db", mongodb_collection="chunks")
    return snapshot, settings, store


def test_unchanged_source_rerun_writes_no_chunks(index_case):
    snapshot, settings, store = index_case
    first = index_snapshot(snapshot, settings, FakeEmbedder(), store)
    written = list(store.written)
    assert first.files_indexed == 1
    assert written
    second = index_snapshot(snapshot, settings, FakeEmbedder(), store)
    assert second.files_skipped == 1
    assert second.chunks_written == 0
    assert store.written == written
    assert store.metadata_cleanups == 2


@pytest.mark.parametrize("change", ["chunking_version", "embedding_model", "force"])
def test_changed_configuration_or_force_rebuilds_source(index_case, change):
    snapshot, settings, store = index_case
    index_snapshot(snapshot, settings, FakeEmbedder(), store)
    embedder = FakeEmbedder()
    previous = store.client["db"]["chunks_manifest"].documents[PATH]
    if change == "chunking_version":
        previous.pop("chunking_version")
    elif change == "embedding_model":
        embedder.model_key = "new-model@revision"
    result = index_snapshot(snapshot, settings, embedder, store, force=change == "force")
    current = store.client["db"]["chunks_manifest"].documents[PATH]
    assert result.files_indexed == 1 and result.files_skipped == 0
    assert result.chunks_written == len(current["chunk_ids"])
    assert current["embedding_model"] == embedder.model_key
    assert current["chunking_version"] == CHUNKING_VERSION


@pytest.mark.parametrize("damage", ["missing_chunk", "missing_chunk_ids"])
def test_rerun_repairs_only_the_incomplete_file(index_case, damage):
    snapshot, settings, store = index_case
    other = "ultralytics/engine/other.py"
    (snapshot.root / other).write_text("OTHER = 1\n")
    snapshot = SourceSnapshot(snapshot.root, snapshot.revision, (*snapshot.files, other))
    index_snapshot(snapshot, settings, FakeEmbedder(), store)
    previous = store.client["db"]["chunks_manifest"].documents[PATH]
    expected_ids = set(previous["chunk_ids"])
    if damage == "missing_chunk":
        store.chunk_ids.remove(previous["chunk_ids"][0])
    else:
        previous.pop("chunk_ids")
    result = index_snapshot(snapshot, settings, FakeEmbedder(), store)
    assert result.files_indexed == 1 and result.files_skipped == 1
    assert expected_ids <= store.chunk_ids
    assert set(store.client["db"]["chunks_manifest"].documents[PATH]["chunk_ids"]) == expected_ids


def test_rebuild_deletes_obsolete_chunks(index_case):
    snapshot, settings, store = index_case
    index_snapshot(snapshot, settings, FakeEmbedder(), store)
    previous = store.client["db"]["chunks_manifest"].documents[PATH]
    previous["chunk_ids"].append("obsolete-chunk")
    store.chunk_ids.add("obsolete-chunk")
    index_snapshot(snapshot, settings, FakeEmbedder(), store, force=True)
    assert "obsolete-chunk" not in store.chunk_ids
    assert store.deleted == ["obsolete-chunk"]


def test_oversized_source_line_is_rejected_before_embedding_or_writes(tmp_path):
    file = tmp_path / PATH
    file.parent.mkdir(parents=True)
    # The approximate pass fits, but exact counting finds an oversized line.
    file.write_text('VALUE = "' + 'a ' * 500 + '"\n')
    snapshot = SourceSnapshot(tmp_path, REVISION, (PATH,))

    class ExactCounter(FakeEmbedder):
        def count_tokens(self, text):
            return len(text)

        def encode(self, texts):
            pytest.fail("Rejected source must not be embedded")

    store = FakeStore(FakeManifest())
    settings = SimpleNamespace(mongodb_db="db", mongodb_collection="chunks")
    with pytest.raises(ValueError, match=f"{PATH}:L1"):
        index_snapshot(snapshot, settings, ExactCounter(), store)
    assert store.written == [] and store.deleted == []
    assert store.client["db"]["chunks_manifest"].documents == {}


def test_estimated_line_overflow_is_rechecked_with_exact_counter(tmp_path):
    file = tmp_path / PATH
    file.parent.mkdir(parents=True)
    # Punctuation is expensive to the estimator but cheap to this counter.
    source = 'VALUE = "' + '#' * 1000 + '"\n'
    file.write_text(source)
    snapshot = SourceSnapshot(tmp_path, REVISION, (PATH,))
    chunks = chunks_for_file(snapshot, PATH, FakeEmbedder())
    assert len(chunks) == 1 and chunks[0].content == source
