from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.delta_generator import DeltaGenerator

from code_assistant.retrieval import Evidence


@pytest.fixture
def app_environment(monkeypatch):
    settings = SimpleNamespace(
        require_mongodb_uri=lambda: "test-uri", embedding_model="test-embedding",
        openrouter_api_key="test-key", openrouter_model="test/model:free",
    )
    store = SimpleNamespace(ping=lambda: None, count_code_chunks=lambda **kwargs: 1)
    source = Evidence(1, "test", "ultralytics/engine/example.py", "a" * 40,
                      "run", 1, 2, "def run():\n    return 1\n")
    monkeypatch.setattr("code_assistant.config.load_settings", lambda: settings)
    monkeypatch.setattr("code_assistant.embedder.CodeEmbedder", lambda _: SimpleNamespace(dimensions=2, model_key="test"))
    monkeypatch.setattr("code_assistant.vector_store.AtlasVectorStore", lambda *args, **kwargs: store)
    monkeypatch.setattr("code_assistant.retrieval.select_evidence", lambda *args, **kwargs: [source])
    return source


def test_status_fallback_displays_gentle_note_in_answer_and_history(monkeypatch, app_environment):
    source = app_environment
    responses = iter(["Returns one. [1]", "ANSWERED: Still returns one. [1]"])
    calls = []

    class Client:
        model = "test/model:free"

        def __init__(self, *args, stream, on_token):
            assert stream is True
            self.on_token = on_token

        def generate(self, system, user):
            calls.append(user)
            response = next(responses)
            self.on_token(response)
            return response

    monkeypatch.setattr("code_assistant.answer.OpenRouterClient", Client)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.chat_input[0].set_value("What does run return?").run()
    assert not app.exception
    assert not app.warning
    assert len(app.info) == 1
    assert "does not mean the answer is incorrect" in app.info[0].value
    assert app.session_state.messages[-1]["status"] == "unverified"
    assert any(item.value == "Returns one. [1]" for item in app.markdown)
    assert any(item.value == source.code.rstrip("\n") for item in app.code)

    app.chat_input[0].set_value("Explain it again.").run()
    assert not app.exception
    assert not app.warning
    assert any(item.value == "Returns one. [1]" for item in app.markdown)
    assert any(item.value == "Still returns one. [1]" for item in app.markdown)
    assert len(app.info) == 1
    assert "source references below" in app.info[0].value
    assert app.session_state.messages[-1]["status"] == "answered"
    assert len(calls) == 2


@pytest.mark.parametrize("outcome", ["success", "invalid_citation", "disconnect", "truncated"])
def test_live_stream_updates_and_only_valid_answers_enter_history(monkeypatch, app_environment, outcome):
    import json

    rendered = []
    markdown = DeltaGenerator.markdown

    def record_markdown(self, body, *args, **kwargs):
        rendered.append(body)
        return markdown(self, body, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "markdown", record_markdown)
    reference = "99" if outcome == "invalid_citation" else "1"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for content in ["STA", "TUS: ANSWERED\nReturns ", f"one. [{reference}]"]:
                yield ("data: " + json.dumps({"choices": [{"delta": {"content": content}}]}) + "\n").encode()
                yield b"\n"
                if content.endswith("Returns "):
                    # This runs before the next network frame is delivered.
                    assert "Returns ▌" in rendered
                    assert not any("STATUS:" in text or text == "STA ▌" for text in rendered)
            if outcome != "disconnect":
                finish = "length" if outcome == "truncated" else "stop"
                yield ("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}) + "\n\n").encode()
                yield b"\n"
                yield b"data: [DONE]\n"
                yield b"\n"

    def open_response(request, **kwargs):
        assert json.loads(request.data)["stream"] is True
        return Response()

    monkeypatch.setattr("code_assistant.answer.urlopen", open_response)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.chat_input[0].set_value("What does run return?").run()
    assert not app.exception
    assert "Returns ▌" in rendered
    assert app.code
    if outcome == "success":
        assert not app.error
        assert app.session_state.messages[-1]["content"] == "Returns one. [1]"
        assert any(item.value == "Returns one. [1]" for item in app.markdown)
    else:
        assert len(app.error) == 1
        assert [message["role"] for message in app.session_state.messages] == ["user"]
        assert not any("Returns" in item.value for item in app.markdown)
    assert not any("▌" in item.value or "STATUS:" in item.value for item in app.markdown)


def test_no_evidence_does_not_contact_model(monkeypatch, app_environment):
    monkeypatch.setattr("code_assistant.retrieval.select_evidence", lambda *args, **kwargs: [])

    def unexpected_request(*args, **kwargs):
        pytest.fail("No evidence should not trigger an API request")

    monkeypatch.setattr("code_assistant.answer.urlopen", unexpected_request)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.chat_input[0].set_value("What happens?").run()
    assert not app.exception
    assert app.info
    assert app.session_state.messages[-1]["status"] == "insufficient_evidence"
