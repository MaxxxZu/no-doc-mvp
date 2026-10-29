import os
import socket

import pytest


@pytest.mark.parametrize("iteration", range(2))
def test_fixtures_reset_between_tests(client, monkeypatch, iteration):
    import config

    assert config.settings.port == 8080
    assert client.app.state.runtime.ready
    assert not hasattr(client.app.state, "leaked")
    assert "ISOLATION_MARKER" not in os.environ
    monkeypatch.setenv("ISOLATION_MARKER", "set")
    # Direct mutations deliberately test the fixture's reset, not monkeypatch.
    config.settings.port = 9999
    client.app.state.runtime.classifier = object()
    client.app.state.runtime.ner = object()
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
        assert "HF_HOME" not in os.environ
        assert application.state.runtime.settings.hf_home == cache
        assert client.get("/healthz").status_code == 200


def test_real_pipeline_and_network_are_blocked():

    from config import Settings
    from services.model_loader import load_pipeline

    with pytest.raises((AssertionError, ImportError)):
        load_pipeline("classify", Settings(_env_file=None))
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
