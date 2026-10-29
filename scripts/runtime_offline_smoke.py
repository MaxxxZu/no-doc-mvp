"""Offline-only runtime/API smoke, separate from fake-model pytest fixtures."""

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Existing artifact cache itself (not its parent)")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.cache_dir.is_dir():
        raise RuntimeError("Existing artifact cache is required; downloads are forbidden")

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def no_network(*args, **kwargs):
        raise RuntimeError("Network is forbidden in this smoke")

    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    socket.getaddrinfo = no_network

    from fastapi.testclient import TestClient

    from app import create_app
    from config import Settings
    from model_specs import NER_MODEL, NER_REVISION, ZSC_MODEL, ZSC_REVISION
    from services.model_loader import load_pipeline
    from services.runtime import ModelRuntime

    settings = Settings(
        _env_file=None, hf_home=str(args.cache_dir.resolve()), offline=True, cpu_threads=2,
        zsc_model=ZSC_MODEL, zsc_revision=ZSC_REVISION,
        ner_model=NER_MODEL, ner_revision=NER_REVISION,
    )
    calls = []

    def tracked_loader(operation, app_settings):
        calls.append(operation)
        return load_pipeline(operation, app_settings)

    application = create_app(settings)
    runtime = ModelRuntime(settings, tracked_loader)
    application.state.runtime = runtime
    assert not runtime.ready
    with TestClient(application) as client:
        assert runtime.ready
        classifier, ner = runtime.classifier, runtime.ner
        assert calls == ["classify", "ner"]
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        for _ in range(2):
            classified = client.post("/classify", json={
                "text": "Oslo kommune behandler en søknad om byggetillatelse.",
                "labels": ["byggesøknad", "sport"], "multi_label": False,
            })
            entities = client.post("/ner", json={
                "text": "Ola Nordmann arbeider i Oslo kommune i Norge.",
            })
            assert classified.status_code == entities.status_code == 200
            assert set(classified.json()["labels"]) == {"byggesøknad", "sport"}
            assert len(classified.json()["scores"]) == 2
            assert entities.json()["entities"]
            assert runtime.classifier is classifier and runtime.ner is ner
            assert calls == ["classify", "ner"]
        for pipe, revision in ((classifier, ZSC_REVISION), (ner, NER_REVISION)):
            assert pipe.model.config._commit_hash == revision
            assert next(pipe.model.parameters()).device.type == "cpu"
            assert pipe.tokenizer.is_fast
    assert not runtime.ready
    assert runtime.classifier is runtime.ner is None
    report = {
        "result": "passed", "network": "blocked sockets and DNS; local_files_only=True",
        "cache_dir": str(args.cache_dir.resolve()), "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "numpy", "transformers", "tokenizers", "huggingface-hub", "safetensors",
            "fastapi", "httpx", "anyio",
        )},
        "models": {ZSC_MODEL: ZSC_REVISION, NER_MODEL: NER_REVISION},
        "load_order": calls, "requests_per_endpoint": 2, "pipeline_identity_reused": True,
        "startup_readiness_shutdown": "passed", "accuracy": "not measured",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.report}")


if __name__ == "__main__":
    main()
