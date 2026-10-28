def test_classify_service_unit(monkeypatch):
    # Patch transformers.pipeline to avoid downloading models
    import services.classification_service as svc

    class FakeZSC:
        def __call__(
            self, sequences, candidate_labels, multi_label,
            hypothesis_template
        ):
            # Return scores sorted desc to mimic pipeline behavior
            return {
                "labels": candidate_labels,
                "scores": [0.8] + [0.1] * (len(candidate_labels) - 1),
            }

    monkeypatch.setattr(svc, "_ZSC_PIPE", None, raising=False)
    monkeypatch.setattr(svc, "pipeline", lambda *a, **k: FakeZSC())

    labels, scores = svc.classify_text_raw(
        text="Dette er en test",
        labels=["A", "B", "C"],
        multi_label_override=False,
    )
    assert labels == ["A", "B", "C"]
    assert scores[0] >= scores[1] >= 0


def test_ner_service_unit(monkeypatch):
    import services.ner_service as svc

    class FakeNER:
        def __call__(self, text):
            return [{
                "entity_group": "PER",
                "word": "Ola Nordmann",
                "score": 0.95,
                "start": 0,
                "end": 12
            }]

    monkeypatch.setattr(svc, "_NER_PIPE", None, raising=False)
    monkeypatch.setattr(svc, "pipeline", lambda *a, **k: FakeNER())

    out = svc.extract_entities_raw("Ola Nordmann")
    assert isinstance(out, list)
    assert out[0]["entity_group"] == "PER"


def test_loaders_receive_model_and_tokenizer_revision(monkeypatch):
    import services.classification_service as classification
    import services.ner_service as ner

    for service, prefix in ((classification, "zsc"), (ner, "ner")):
        calls = []
        sentinel = object()

        def loader(**kwargs):
            calls.append(kwargs)
            return sentinel

        monkeypatch.setattr(service.settings, f"{prefix}_revision", "a" * 40)
        monkeypatch.setattr(service, "pipeline", loader)
        assert service._get_pipeline() is sentinel
        assert service._get_pipeline() is sentinel
        assert len(calls) == 1
        assert calls[0]["model"] == getattr(service.settings, f"{prefix}_model")
        assert calls[0]["tokenizer"] == calls[0]["model"]
        assert calls[0]["revision"] == "a" * 40
        assert calls[0]["trust_remote_code"] is False
        assert calls[0]["model_kwargs"] == {"use_safetensors": True}
        assert calls[0]["device"] == -1
