import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable, TypeVar

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from config import Settings, settings
from models import ClassifyRequest, ClassifyResponse, NerEntity, NerRequest, NerResponse
from services.classification_service import classify_text_raw
from services.model_loader import Operation
from services.ner_service import extract_entities_raw
from services.runtime import ModelRuntime, RuntimeBusy, RuntimeUnavailable


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = app.state.runtime
    try:
        await run_in_threadpool(runtime.initialize)
        yield
    finally:
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(runtime.close)


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
    application.state.runtime = ModelRuntime(app_settings)
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

    @application.get("/readyz", tags=["health"])
    def ready():
        if not application.state.runtime.ready:
            raise HTTPException(status_code=503, detail="models are not ready")
        return {"status": "ready"}

    application.add_api_route(
        "/classify", classify_endpoint, methods=["POST"],
        response_model=ClassifyResponse, tags=["classification"],
    )
    application.add_api_route(
        "/ner", ner_endpoint, methods=["POST"], response_model=NerResponse, tags=["ner"],
    )
    return application


async def classify_endpoint(req: ClassifyRequest, request: Request):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    runtime = request.app.state.runtime
    labels, scores = await infer(
        runtime, "classify", classify_text_raw,
        text=req.text, pipeline=runtime.classifier, settings=runtime.settings,
        labels=req.labels, multi_label_override=req.multi_label,
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


async def ner_endpoint(req: NerRequest, request: Request):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    runtime = request.app.state.runtime
    preds = await infer(runtime, "ner", extract_entities_raw,
                        text=req.text, pipeline=runtime.ner)
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


Result = TypeVar("Result")


async def infer(
    runtime: ModelRuntime, operation: Operation, function: Callable[..., Result], **kwargs: Any,
) -> Result:
    # Admit before scheduling a worker: busy requests never queue for CPU work.
    try:
        with runtime.inference(operation):
            # Cancellation/HTTP disconnect is not a way to stop PyTorch. Keep the
            # slot until the worker returns, even if the caller stops waiting.
            with anyio.CancelScope(shield=True):
                worker = asyncio.create_task(run_in_threadpool(function, **kwargs))
                cancelled = False
                while True:
                    try:
                        result = await asyncio.shield(worker)
                        break
                    except asyncio.CancelledError:
                        # Also handle direct asyncio Task.cancel(), which bypasses
                        # AnyIO shielding. Do not release a running worker's slot.
                        if worker.cancelled():
                            raise
                        cancelled = True
                if cancelled:
                    raise asyncio.CancelledError
                return result
    except RuntimeBusy:
        raise HTTPException(503, "inference is busy", headers={"Retry-After": "1"}) from None
    except RuntimeUnavailable:
        raise HTTPException(503, "models are not ready", headers={"Retry-After": "1"}) from None
    except Exception:
        # Preserve HTTP 500 without logging untrusted exception text or documents.
        raise HTTPException(500, "inference failed") from None


app = create_app()


def run():
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    run()
