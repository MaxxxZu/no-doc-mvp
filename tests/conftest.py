import os
import socket
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

LEGACY_KEYS = {
    "PORT", "ZSC_MODEL", "NER_MODEL", "ZSC_MULTI_LABEL", "ZSC_MAX_LEN", "HF_HOME",
}


def forbidden_pipeline(*args, **kwargs):
    raise AssertionError("Real model construction is forbidden in the fast suite")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    # Run before importing application modules: neither .env nor HF libraries may
    # inspect the developer's configuration/cache during collection or startup.
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.upper().startswith("NDM_") or key.upper() in LEGACY_KEYS:
            monkeypatch.delenv(key)
    for key in (
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE", "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the fast suite")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    transformers = ModuleType("transformers")
    transformers.pipeline = forbidden_pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    import config
    import services.classification_service as classification
    import services.ner_service as ner

    test_settings = config.Settings(_env_file=None)
    monkeypatch.setattr(config, "settings", test_settings)
    for module, singleton in ((classification, "_ZSC_PIPE"), (ner, "_NER_PIPE")):
        monkeypatch.setattr(module, "settings", test_settings)
        monkeypatch.setattr(module, singleton, None)
        monkeypatch.setattr(module, "pipeline", forbidden_pipeline)
    yield test_settings


@pytest.fixture
def client(isolated_environment):
    from app import create_app

    with TestClient(create_app(isolated_environment)) as test_client:
        yield test_client
