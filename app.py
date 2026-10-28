import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import Settings, settings
from models import ClassifyRequest, ClassifyResponse, NerEntity, NerRequest, NerResponse
from services.classification_service import classify_text_raw
from services.ner_service import extract_entities_raw


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-configure HF caches at startup (improves first request latency)
    if app.state.settings.hf_home:
        os.environ.setdefault("HF_HOME", app.state.settings.hf_home)
    yield


def create_app(app_settings: Settings = settings) -> FastAPI:
    application = FastAPI(
        title="no-doc-mvp — Classification & NER API",
        version="0.1.0",
        description=(
            "FastAPI service for zero-shot classification and NER on Norwegian text."
            "Open '/docs' to test requests directly in the browser."
        ),
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/healthz", tags=["health"])
    def health():
        return {
            "status": "ok",
            "zsc_model": app_settings.zsc_model,
            "ner_model": app_settings.ner_model,
        }

    application.add_api_route(
        "/classify", classify_endpoint, methods=["POST"],
        response_model=ClassifyResponse, tags=["classification"],
    )
    application.add_api_route(
        "/ner", ner_endpoint, methods=["POST"], response_model=NerResponse, tags=["ner"],
    )
    return application


def classify_endpoint(req: ClassifyRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    labels, scores = classify_text_raw(
        text=req.text,
        labels=req.labels,
        multi_label_override=req.multi_label,
    )
    if not labels:
        raise HTTPException(status_code=400, detail="classification failed")

    top_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
    return ClassifyResponse(
        labels=labels,
        scores=scores,
        top_label=labels[top_idx],
        top_score=scores[top_idx],
    )


def ner_endpoint(req: NerRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    preds = extract_entities_raw(req.text)
    entities = [
        NerEntity(
            entity_group=p["entity_group"],
            word=p["word"],
            score=float(p["score"]),
            start=int(p["start"]),
            end=int(p["end"]),
        )
        for p in preds
    ]
    return NerResponse(entities=entities)


app = create_app()


def run():
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    run()
