from code_assistant.config import load_settings


def test_environment_overrides_file_without_exposing_secret(tmp_path, monkeypatch):
    config = tmp_path / ".env"
    config.write_text("MONGODB_URI=mongodb://from-file\nOPENROUTER_API_KEY=file-key\n")
    monkeypatch.setenv("MONGODB_URI", "mongodb://from-process")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = load_settings(config)
    assert settings.mongodb_uri == "mongodb://from-process"
    assert settings.openrouter_api_key == "file-key"
    assert "file-key" not in repr(settings)


def test_first_run_defaults_and_example_use_the_same_pinned_jina(tmp_path, monkeypatch):
    from pathlib import Path
    from dotenv import dotenv_values

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_REVISION", raising=False)
    settings = load_settings(tmp_path / "absent.env")
    example = dotenv_values(Path(__file__).resolve().parents[1] / ".env.example")
    assert settings.embedding_model == example["EMBEDDING_MODEL"] == "jinaai/jina-embeddings-v2-base-code"
    assert settings.embedding_model_revision == example["EMBEDDING_MODEL_REVISION"] == "516f4baf13dec4ddddda8631e019b5737c8bc250"
