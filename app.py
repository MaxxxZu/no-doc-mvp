import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import ClassifyRequest, ClassifyResponse, NerEntity, NerRequest, NerResponse
from services.classification_service import classify_text_raw
from services.ner_service import extract_entities_raw


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-configure HF caches at startup (improves first request latency)
    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    yield

app = FastAPI(
    title="no-doc-mvp — Classification & NER API",
    version="0.1.0",
    description=(
        "FastAPI service for zero-shot classification and NER on Norwegian text."
        "Open '/docs' to test requests directly in the browser."
    ),
    lifespan=lifespan,
)

# CORS (from env list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["health"])
def health():
    return {
        "status": "ok",
        "zsc_model": settings.zsc_model,
        "ner_model": settings.ner_model,
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["classification"])
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


@app.post("/ner", response_model=NerResponse, tags=["ner"])
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


# Local run (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=settings.port, reload=True)
