import os
import socket

import pytest


@pytest.mark.parametrize("iteration", range(2))
def test_fixtures_reset_between_tests(client, monkeypatch, iteration):
    import config
    import services.classification_service as classification
    import services.ner_service as ner

    assert config.settings.port == 8080
    assert classification._ZSC_PIPE is None
    assert ner._NER_PIPE is None
    assert not hasattr(client.app.state, "leaked")
    assert "ISOLATION_MARKER" not in os.environ
    monkeypatch.setenv("ISOLATION_MARKER", "set")
    # Direct mutations deliberately test the fixture's reset, not monkeypatch.
    config.settings.port = 9999
    classification._ZSC_PIPE = object()
    ner._NER_PIPE = object()
    client.app.state.leaked = True


def test_cache_and_lifespan(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import create_app
    from config import Settings

    cache = str(tmp_path / "lifespan-cache")
    monkeypatch.delenv("HF_HOME")
    application = create_app(Settings(hf_home=cache))
    assert "HF_HOME" not in os.environ
    with TestClient(application) as client:
        assert os.environ["HF_HOME"] == cache
        assert client.get("/healthz").status_code == 200


def test_real_pipeline_and_network_are_blocked():
    import services.classification_service as classification
    import services.ner_service as ner

    for loader in (classification._get_pipeline, ner._get_pipeline):
        with pytest.raises(AssertionError, match="Real model construction"):
            loader()
    with pytest.raises(AssertionError, match="Network access"):
        socket.create_connection(("example.invalid", 443))


def test_local_launcher(monkeypatch):
    import uvicorn

    import app
    from config import Settings

    monkeypatch.setenv("NDM_HOST", "127.0.0.2")
    monkeypatch.setenv("NDM_PORT", "9123")
    monkeypatch.setenv("PORT", "9124")
    monkeypatch.setattr(app, "settings", Settings())
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    app.run()
    assert calls == [(("app:app",), {"host": "127.0.0.2", "port": 9124, "workers": 1})]
