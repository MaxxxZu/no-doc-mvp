def test_classify_endpoint_happy_path(client, monkeypatch):
    import app as app_module

    def fake_classify_text_raw(text, labels=None, multi_label_override=None, **kwargs):
        used_labels = labels or ["A", "B", "C"]
        scores = [0.9, 0.07, 0.03][: len(used_labels)]
        return used_labels, scores

    monkeypatch.setattr(app_module, "classify_text_raw", fake_classify_text_raw)

    payload = {
        "text": "Søknad om igangsettingstillatelse...",
        "labels": ["rammetillatelse", "igangsettingstillatelse", "annet"],
        "multi_label": False,
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["top_label"] == "rammetillatelse"
    assert data["labels"] == payload["labels"]
    assert len(data["scores"]) == len(payload["labels"])


def test_classify_endpoint_empty_text(client):
    resp = client.post("/classify", json={"text": "   "})
    assert resp.status_code == 400
