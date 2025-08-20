def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"
    # presence of model keys
    assert "zsc_model" in body
    assert "ner_model" in body
