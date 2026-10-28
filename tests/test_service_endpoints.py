def test_endpoints_with_real_services_and_fake_pipelines(client, monkeypatch):
    import services.classification_service as classification
    import services.ner_service as ner

    calls = []

    def fake_zsc(**kwargs):
        calls.append(kwargs)
        return {"labels": kwargs["candidate_labels"], "scores": [0.9, 0.1]}

    def fake_ner(text):
        calls.append(text)
        return [{
            "entity_group": "LOC", "word": "Oslo", "score": 0.95, "start": 0, "end": 4,
        }]

    monkeypatch.setattr(classification, "pipeline", lambda **kwargs: fake_zsc)
    monkeypatch.setattr(ner, "pipeline", lambda **kwargs: fake_ner)
    response = client.post("/classify", json={
        "text": "Dette er en test", "labels": ["A", "B"], "multi_label": False,
    })
    assert response.status_code == 200
    assert response.json() == {
        "labels": ["A", "B"], "scores": [0.9, 0.1], "top_label": "A", "top_score": 0.9,
    }
    assert calls[0]["sequences"] == "Dette er en test"
    assert calls[0]["multi_label"] is False
    response = client.post("/ner", json={"text": "Oslo"})
    assert response.status_code == 200
    assert response.json() == {"entities": [{
        "entity_group": "LOC", "word": "Oslo", "score": 0.95, "start": 0, "end": 4,
    }]}
    assert calls[1] == "Oslo"
