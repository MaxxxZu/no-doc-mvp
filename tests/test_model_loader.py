import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("operation,prefix,task", [
    ("classify", "zsc", "zero-shot-classification"),
    ("ner", "ner", "token-classification"),
])
@pytest.mark.parametrize("offline", [False, True])
def test_explicit_loader_arguments(monkeypatch, operation, prefix, task, offline):
    from config import Settings
    from services.model_loader import load_pipeline

    calls = []
    tokenizer = SimpleNamespace(is_fast=True)

    class Model:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))

    model = Model()
    result = object()

    def tokenizer_load(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return tokenizer

    def model_load(name, **kwargs):
        calls.append(("model", name, kwargs))
        return model

    def pipe(task, **kwargs):
        calls.append(("pipeline", task, kwargs))
        return result

    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model_load),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=model_load),
        pipeline=pipe,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    settings = Settings(_env_file=None, hf_home="/explicit/artifacts", offline=offline,
                        zsc_revision="a" * 40, ner_revision="b" * 40)
    assert load_pipeline(operation, settings) is result
    common = dict(cache_dir="/explicit/artifacts",
                  revision=getattr(settings, prefix + "_revision"),
                  local_files_only=offline, trust_remote_code=False)
    model_id = getattr(settings, prefix + "_model")
    assert calls[:4] == [
        ("tokenizer", model_id, dict(common, use_fast=True)),
        ("model", model_id, dict(common, use_safetensors=True)),
        ("to", "cpu"), ("eval",),
    ]
    extra = {} if operation == "classify" else {"aggregation_strategy": "simple"}
    expected = dict(model=model, tokenizer=tokenizer, device=-1, **extra)
    assert calls[4] == ("pipeline", task, expected)
