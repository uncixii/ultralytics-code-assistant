from code_assistant.chunker import REPO_ID
from code_assistant.retrieval import select_evidence


def test_evaluation_records_vector_ranks_without_changing_selected_source(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from code_assistant.retrieval import Evidence

    path = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluation_script", path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    source_path = "ultralytics/data/base.py"
    question = {"id": "image", "question": "Where is the image resized?",
                "expected_file": source_path, "expected_symbol": "BaseDataset.load_image",
                "required_lines": [[1, 2]]}
    first = Evidence(1, "first", source_path, REVISION, "BaseDataset.load_image", 1, 1, "read()\n")
    second = Evidence(2, "second", source_path, REVISION, "BaseDataset.load_image", 2, 2, "resize()\n")

    class DiagnosticStore:
        def __init__(self):
            self.limits = []

        def search_code(self, vector, **kwargs):
            self.limits.append(kwargs["limit"])
            return [
                {"file_path": "ultralytics/data/other.py", "symbol_full": first.symbol},
                {"file_path": source_path, "symbol_full": first.symbol, "start_line": 2, "end_line": 2},
            ]

    for selected, expected_limit, outcome in (([first], 12, "partial_source"),
                                               ([], 50, "miss"),
                                               ([first, second], None, "sufficient_source")):
        monkeypatch.setattr(evaluation, "select_evidence", lambda *args, **kwargs: selected)
        store = DiagnosticStore()
        report = evaluation.evaluate_questions(
            [question], SimpleNamespace(revision=REVISION), FakeEmbedder(), store,
            {source_path: ["read()\n", "resize()\n"]},
        )
        case = report["cases"][0]
        assert case["outcome"] == outcome
        assert len(case["selected_sources"]) == len(selected)
        assert store.limits == ([expected_limit] if expected_limit else [])
        if expected_limit:
            assert case["vector_diagnostic"] == {
                "limit": expected_limit, "returned": 2,
                "expected_method_matches": [{"rank": 2, "source": f"{source_path}:L2-L2"}],
            }
        else:
            assert "vector_diagnostic" not in case


REVISION = "a" * 40


class FakeEmbedder:
    model_key = "example@revision"

    def encode_query(self, question):
        return [0.1, 0.2]

    def count_tokens(self, text):
        return len(text.split())


class FakeStore:
    def __init__(self, dense, symbols=None):
        self.dense = dense
        self.symbols = symbols or {}

    def search_code(self, vector, **kwargs):
        return self.dense

    def lookup_symbol(self, symbol, **kwargs):
        return self.symbols.get(symbol, [])


def row(identifier, **extra):
    return {
        "_id": identifier,
        "repo_id": REPO_ID,
        "source_revision": REVISION,
        "file_path": "ultralytics/engine/predictor.py",
        "symbol_full": "BasePredictor.preprocess",
        "start_line": 10,
        "end_line": 11,
        "content": "    normalized = images\n    return normalized\n",
        "symbol_type": "method",
    } | extra


def test_symbol_match_is_selected_and_cited_within_budget():
    match = row("match")
    store = FakeStore(
        [row("other", symbol_full="Other.run", start_line=30, end_line=31), match],
        {"BasePredictor.preprocess": [match]},
    )
    embedder = FakeEmbedder()
    result = select_evidence(
        "What does `BasePredictor.preprocess` do?", store=store,
        embedder=embedder, source_revision=REVISION, token_budget=4,
    )
    assert [item.chunk_id for item in result] == ["match"]
    assert result[0].code == "    normalized = images\n"
    assert result[0].citation == "ultralytics/engine/predictor.py:L10-L10"
    assert result[0].source_url.endswith(f"/{REVISION}/ultralytics/engine/predictor.py#L10-L10")
    assert sum(embedder.count_tokens(item.code) for item in result) <= 4


def test_repeated_name_hits_do_not_change_fused_ranking():
    first = row("A", symbol_full="Dataset.load_image")
    second = row("B", symbol_full="Dataset.load_image", start_line=30, end_line=31)
    store = FakeStore(
        [first, second],
        {"Dataset.load_image": [second, first], "load_image": [second]},
    )
    kwargs = {"store": store, "embedder": FakeEmbedder(), "source_revision": REVISION}
    baseline = select_evidence("Explain `Dataset.load_image`.", **kwargs)
    repeated = select_evidence("Explain `Dataset.load_image` and `load_image`.", **kwargs)
    assert [item.chunk_id for item in baseline] == ["A", "B"]
    assert repeated == baseline


def test_invalid_sources_are_discarded_without_losing_valid_evidence():
    valid = row("valid")
    store = FakeStore([
        row("outside", file_path="ultralytics/nn/tasks.py"),
        row("unsafe", file_path="ultralytics/models/../secrets.py"),
        row("stale", source_revision="b" * 40),
        row("old-summary", symbol_type="class_summary"),
        valid,
    ])
    kwargs = {"store": store, "embedder": FakeEmbedder(), "source_revision": REVISION}
    result = select_evidence("Explain preprocessing", **kwargs)
    assert [item.chunk_id for item in result] == ["valid"]
    assert result[0].code == valid["content"]
    assert select_evidence("How does ultralytics/nn/tasks.py work?", **kwargs) == []
