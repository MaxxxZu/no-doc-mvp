def test_ner_endpoint_happy_path(client, monkeypatch):
    import app as app_module

    def fake_extract_entities_raw(text: str, **kwargs):
        return [
            {
                "entity_group": "ORG",
                "word": "AF Gruppen",
                "score": 0.99,
                "start": 0, "end": 10
            },
            {
                "entity_group": "LOC",
                "word": "Oslo",
                "score": 0.98,
                "start": 11,
                "end": 15
            },
        ]

    monkeypatch.setattr(app_module, "extract_entities_raw", fake_extract_entities_raw)

    resp = client.post("/ner", json={"text": "AF Gruppen ... Oslo"})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert len(entities) == 2
    assert entities[0]["entity_group"] == "ORG"
    assert entities[1]["entity_group"] == "LOC"


def test_ner_endpoint_empty_text(client):
    resp = client.post("/ner", json={"text": ""})
    assert resp.status_code == 422
