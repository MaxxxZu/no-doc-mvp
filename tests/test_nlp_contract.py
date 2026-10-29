from types import SimpleNamespace

import pytest


class FakePipeline:
    def __init__(self, limit=100, result=None):
        self.model = SimpleNamespace(config=SimpleNamespace(
            model_type="bert", max_position_embeddings=limit))
        self.tokenizer = self.tokenize
        self.result = result
        self.calls = []

    def tokenize(self, text, pair=None, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        return {"input_ids": [0] * (len(text) + (len(pair) if pair else 0) + (3 if pair else 2))}

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def real_preflight(monkeypatch):
    import services.classification_service as classification
    import services.ner_service as ner
    from services.validation import check_tokens

    monkeypatch.setattr(classification, "check_tokens", check_tokens)
    monkeypatch.setattr(ner, "check_tokens", check_tokens)


def test_ner_unicode_and_token_boundary(client, real_preflight):
    text = " \nÅ😀e\u0301"
    pipe = FakePipeline(len(text) + 2, [{"entity_group": "X", "word": "different",
                                         "score": 0.5, "start": 2, "end": len(text)}])
    client.app.state.runtime.ner = pipe
    response = client.post("/ner", json={"text": text})
    assert response.json()["entities"][0]["word"] == text[2:]
    assert pipe.calls == [((text,), {})]
    assert client.post("/ner", json={"text": text + "x"}).status_code == 422
    assert len(pipe.calls) == 1


def test_classification_pair_boundary_and_top(client, real_preflight):
    text, label = " Å", "long label"
    hypothesis = "Dette dokumentet handler om {}.".format(label)
    pipe = FakePipeline(len(text) + len(hypothesis) + 3,
                        {"labels": ["x", label], "scores": [0.1, 0.9]})
    client.app.state.runtime.classifier = pipe
    response = client.post("/classify", json={"text": text, "labels": [label, "x"]})
    assert response.status_code == 200
    assert response.json()["top_label"] == label
    assert pipe.calls[0][1]["sequences"] == text
    response = client.post("/classify", json={"text": text + "x", "labels": [label, "x"]})
    assert response.status_code == 422
    assert len(pipe.calls) == 1


@pytest.mark.parametrize("labels", [[], [" "], ["a", " a "], ["x" * 129], ["x"] * 33])
def test_bad_labels(client, labels):
    assert client.post("/classify", json={"text": "x", "labels": labels}).status_code == 422


@pytest.mark.parametrize("result", [
    ([], []), (["a"], []), (["a"], [float("nan")]),
    (["a"], [float("inf")]), (["b"], [0.5]),
])
def test_bad_classification(client, monkeypatch, result):
    import app
    monkeypatch.setattr(app, "classify_text_raw", lambda **kwargs: result)
    response = client.post("/classify", json={"text": "PRIVATE_DOCUMENT", "labels": ["a"]})
    assert response.status_code == 500
    assert "PRIVATE_DOCUMENT" not in response.text


@pytest.mark.parametrize("span", [(0, 0), (-1, 1), (0, 2), (0.1, 1), (True, 1)])
def test_bad_spans(client, monkeypatch, span):
    import app
    monkeypatch.setattr(app, "extract_entities_raw", lambda **kwargs: [dict(
        entity_group="X", word="x", score=0.5, start=span[0], end=span[1])])
    assert client.post("/ner", json={"text": "x"}).status_code == 500


def test_empty_ner_and_openapi(client):
    assert client.post("/ner", json={"text": "x"}).json() == {"entities": []}
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/ner"]["post"]["responses"]["422"]["content"]["application/json"][
        "schema"]["$ref"].endswith("/ErrorResponse")


def test_character_limit_before_inference(client, monkeypatch):
    import app
    client.app.state.settings.zsc_max_len = 3
    calls = []
    monkeypatch.setattr(app, "classify_text_raw",
                        lambda **kw: (calls.append(kw) or (["a"], [0.5])))
    assert client.post("/classify", json={"text": "abc", "labels": ["a"]}).status_code == 200
    assert client.post("/classify", json={"text": "abcd", "labels": ["a"]}).status_code == 422
    assert len(calls) == 1


@pytest.mark.parametrize("with_length", [True, False])
def test_body_limit_chunked(with_length):
    import asyncio

    from services.http_errors import BodyLimitMiddleware

    async def run():
        events = iter([{"type": "http.request", "body": b"123", "more_body": True},
                       {"type": "http.request", "body": b"456", "more_body": True}])
        sent = []

        async def receive():
            return next(events)

        async def send(event):
            sent.append(event)

        async def app(scope, receive, send):
            raise AssertionError("Oversized body reached application")

        headers = [(b"content-length", b"6")] if with_length else []
        scope = {"type": "http", "method": "POST", "path": "/ner", "headers": headers}
        await BodyLimitMiddleware(app, 5)(scope, receive, send)
        assert sent[0]["status"] == 413
        assert sum(event["type"] == "http.response.start" for event in sent) == 1

    asyncio.run(run())


def test_default_labels_and_trim(client, monkeypatch):
    import app
    from services.classification_service import DEFAULT_LABELS

    seen = []

    def fake(**kwargs):
        seen.append(kwargs["labels"])
        labels = DEFAULT_LABELS if kwargs["labels"] is None else kwargs["labels"]
        return labels, [0.5] * len(labels)

    monkeypatch.setattr(app, "classify_text_raw", fake)
    for data in ({}, {"labels": None}, {"labels": ["  Å  ", "a"]}):
        assert client.post("/classify", json={"text": "x", **data}).status_code == 200
    assert seen == [None, None, ["Å", "a"]]


def test_body_error_cors_and_malformed_json():
    from fastapi.testclient import TestClient

    from app import create_app
    from config import Settings

    app = create_app(Settings(_env_file=None, body_max_bytes=10,
                              cors_origins=["https://example.com"]))
    with TestClient(app) as client:
        headers = {"Origin": "https://example.com"}
        response = client.post("/ner", content=b"x" * 11, headers=headers)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "body_too_large"
        assert response.headers["access-control-allow-origin"] == headers["Origin"]
        response = client.post("/ner", content=b"{", headers=headers)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"
        assert response.headers["access-control-allow-origin"] == headers["Origin"]


@pytest.mark.parametrize("result", [
    {"labels": ["a", "b"], "scores": [True, False]},
    {"labels": ["a", "b"], "scores": ["0.1", "0.9"]},
    {"labels": "ab", "scores": [0.1, 0.9]},
    ["a", "b"],
])
def test_pipeline_result_is_checked_before_conversion(client, real_preflight, caplog, result):
    pipe = FakePipeline(result=result)
    client.app.state.runtime.classifier = pipe
    response = client.post("/classify", json={"text": "PRIVATE_DOCUMENT", "labels": ["a", "b"]})
    assert len(pipe.calls) == 1
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "inference_failed", "message": "Inference failed."},
    }
    assert "PRIVATE_DOCUMENT" not in response.text + caplog.text


@pytest.mark.parametrize("endpoint", ["/classify", "/ner"])
@pytest.mark.parametrize("body", [b'{"text":"PRIVATE_DOCUMENT',
                                  b'{"text":"PRIVATE_DOCUMENT\xff"}'])
def test_invalid_json_and_utf8_are_safe_and_skip_forward(endpoint, body, caplog):
    from fastapi.testclient import TestClient

    from app import create_app
    from config import Settings

    application = create_app(Settings(_env_file=None, cors_origins=["https://example.com"]))
    with TestClient(application) as client:
        pipe = FakePipeline()
        application.state.runtime.classifier = application.state.runtime.ner = pipe
        response = client.post(endpoint, content=body, headers={
            "content-type": "application/json", "origin": "https://example.com",
        })
        assert response.status_code == 422
        assert response.json() == {
            "error": {"code": "invalid_input", "message": "Invalid request."},
        }
        assert response.headers["access-control-allow-origin"] == "https://example.com"
        assert not pipe.calls
        assert "utf-8" not in response.text.lower() + caplog.text.lower()
        assert "PRIVATE_DOCUMENT" not in response.text + caplog.text


def test_other_http_errors_keep_their_status():
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from app import create_app
    from config import Settings

    application = create_app(Settings(_env_file=None))

    @application.get("/test-http-error")
    def endpoint():
        try:
            b"\xff".decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="ordinary error") from exc

    with TestClient(application) as client:
        response = client.get("/test-http-error")
        assert response.status_code == 400
        assert response.json() == {"detail": "ordinary error"}


def test_pipeline_valid_numpy_scores(client, real_preflight):
    import numpy as np

    pipe = FakePipeline(result={"labels": ["b", "a"],
                                "scores": [np.float32(0.9), np.float32(0.1)]})
    client.app.state.runtime.classifier = pipe
    response = client.post("/classify", json={"text": "x", "labels": ["a", "b"]})
    assert response.status_code == 200
    assert response.json()["top_label"] == "b"
    assert response.json()["top_score"] == pytest.approx(0.9)
