from pathlib import Path

import pytest

from code_assistant.config import load_settings


def test_load_settings_prefers_process_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / ".env"
    config.write_text("MONGODB_URI=mongodb://from-file\nOPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("MONGODB_URI", "mongodb://from-process")
    settings = load_settings(config)
    assert settings.mongodb_uri == "mongodb://from-process"
    assert settings.openrouter_api_key == "from-file"
    assert settings.embedding_model == "jinaai/jina-embeddings-v2-base-code"
    assert "from-file" not in repr(settings)


def test_missing_mongodb_uri_has_safe_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_CONN_URI", raising=False)
    with pytest.raises(ValueError, match="MONGODB_URI is required"):
        load_settings(tmp_path / "absent.env").require_mongodb_uri()
