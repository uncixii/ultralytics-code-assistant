import importlib.util
import io
import json
from pathlib import Path
import sys

from pymongo.errors import ConnectionFailure
import pytest

from code_assistant.answer import OpenRouterClient
from code_assistant.config import Settings
from code_assistant.errors import OpenRouterError, PublicError, error_message
from code_assistant.source import SourceDiscoveryError


@pytest.mark.parametrize("error", [
    PublicError("Set MONGODB_URI in .env, then retry."),
    SourceDiscoveryError("Checkout revision differs from the pinned source revision"),
    OpenRouterError("OpenRouter request failed (HTTP 429)"),
])
def test_authored_error_messages_remain_actionable(error):
    assert error_message(error) == str(error)


@pytest.mark.parametrize("error", [
    ValueError("mongodb://username:secret@private-host"),
    RuntimeError("Bearer private-api-key"),
    ConnectionFailure("mongodb://username:secret@private-host"),
])
def test_external_errors_do_not_reveal_raw_messages(error):
    message = error_message(error)
    assert type(error).__name__ in message
    assert "Check" in message
    assert str(error) not in message


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ask_missing_uri_shows_fix_before_loading_model(monkeypatch, capsys):
    script = _script("ask")
    settings = Settings(None, "db", "chunks", "vector", None, None, "model", "revision")
    monkeypatch.setattr(script, "load_settings", lambda: settings)
    monkeypatch.setattr(script, "CodeEmbedder", lambda _: pytest.fail("model must not load"))
    monkeypatch.setattr(sys, "argv", ["ask.py", "Explain preprocessing"])
    assert script.main() == 1
    assert "MONGODB_URI is required" in capsys.readouterr().err


def test_index_command_preserves_source_validation_reason(monkeypatch, capsys):
    script = _script("index_source")

    def fail(_, **kwargs):
        raise SourceDiscoveryError("Checkout revision differs from the pinned source revision")

    monkeypatch.setattr(script, "index_repository", fail)
    monkeypatch.setattr(sys, "argv", ["index_source.py"])
    assert script.main() == 1
    assert "Checkout revision differs" in capsys.readouterr().err


def test_provider_error_code_cannot_expose_private_response(monkeypatch):
    payload = {"error": {"code": "private-api-key", "message": "private-provider-detail"}}
    monkeypatch.setattr(
        "code_assistant.answer.urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    client = OpenRouterClient("test-key", "test/model:free")
    with pytest.raises(OpenRouterError) as raised:
        client.generate("system", "question")
    assert "API error" in error_message(raised.value)
    assert "private" not in error_message(raised.value)
    assert "private" not in json.dumps(client.last_response)
