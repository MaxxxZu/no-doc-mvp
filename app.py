import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import anyio
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from config import Settings, settings
from models import ClassifyRequest, ClassifyResponse, NerEntity, NerRequest, NerResponse
from services.classification_service import DEFAULT_LABELS, classify_text_raw
from services.http_errors import ApiError, BodyLimitMiddleware, error_response
from services.model_loader import Operation
from services.ner_service import extract_entities_raw
from services.runtime import ModelRuntime, RuntimeBusy, RuntimeUnavailable
from services.validation import InputTooLong, validate_classification, validate_entities


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


ERROR_RESPONSES = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in (
        (413, "Request body too large"), (422, "Invalid input"),
        (500, "Inference failed"), (503, "Runtime unavailable"),
    )
}


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
    application.add_middleware(BodyLimitMiddleware, limit=app_settings.body_max_bytes)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(ApiError)
    async def api_error_handler(request, exc):
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @application.exception_handler(HTTPException)
    async def http_error_handler(request, exc):
        # FastAPI wraps non-JSONDecodeError body parsing failures in HTTP 400.
        if (exc.status_code == 400 and exc.detail == "There was an error parsing the body"
                and isinstance(exc.__cause__, UnicodeDecodeError)):
            return error_response(422, "invalid_input", "Invalid request.")
        return await http_exception_handler(request, exc)

    @application.exception_handler(RequestValidationError)
    async def request_error_handler(request, exc):
        return error_response(422, "invalid_input", "Invalid request.")

    @application.exception_handler(ResponseValidationError)
    async def response_error_handler(request, exc):
        return error_response(500, "inference_failed", "Inference failed.")

    @application.get("/healthz", tags=["health"])
    def health():
        return {
            "status": "ok",
            "zsc_model": app_settings.zsc_model,
            "ner_model": app_settings.ner_model,
        }

    @application.get("/readyz", tags=["health"], responses={503: ERROR_RESPONSES[503]})
    def ready():
        if not application.state.runtime.ready:
            raise ApiError(503, "runtime_unavailable", "Models are not ready.",
                           {"Retry-After": "1"})
        return {"status": "ready"}

    application.add_api_route(
        "/classify", classify_endpoint, methods=["POST"],
        response_model=ClassifyResponse, tags=["classification"], responses=ERROR_RESPONSES,
    )
    application.add_api_route(
        "/ner", ner_endpoint, methods=["POST"], response_model=NerResponse,
        tags=["ner"], responses=ERROR_RESPONSES,
    )
    return application


async def classify_endpoint(req: ClassifyRequest, request: Request):
    if len(req.text) > request.app.state.settings.zsc_max_len:
        raise ApiError(422, "text_too_long", "Text exceeds the supported length.")

    runtime = request.app.state.runtime
    result = await infer(
        runtime, "classify", classify_text_raw,
        text=req.text, pipeline=runtime.classifier, settings=runtime.settings,
        labels=req.labels, multi_label_override=req.multi_label,
    )
    try:
        categories = DEFAULT_LABELS if req.labels is None else req.labels
        labels, scores = validate_classification(result, categories)
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
    except (ValueError, TypeError, KeyError, IndexError):
        raise ApiError(500, "inference_failed", "Inference failed.") from None
    return ClassifyResponse(
        labels=labels,
        scores=scores,
        top_label=labels[top_idx],
        top_score=scores[top_idx],
    )


async def ner_endpoint(req: NerRequest, request: Request):
    if len(req.text) > request.app.state.settings.ner_max_len:
        raise ApiError(422, "text_too_long", "Text exceeds the supported length.")

    runtime = request.app.state.runtime
    preds = await infer(runtime, "ner", extract_entities_raw,
                        text=req.text, pipeline=runtime.ner)
    try:
        entities = [NerEntity(**entity) for entity in validate_entities(preds, req.text)]
    except (ValueError, TypeError, KeyError, IndexError):
        raise ApiError(500, "inference_failed", "Inference failed.") from None
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
        raise ApiError(503, "inference_busy", "Inference is busy.", {"Retry-After": "1"}) from None
    except RuntimeUnavailable:
        raise ApiError(503, "runtime_unavailable", "Models are not ready.",
                       {"Retry-After": "1"}) from None
    except InputTooLong:
        raise ApiError(422, "text_too_long", "Text exceeds the supported length.") from None
    except Exception:  # noqa: BLE001 - inference is an untrusted boundary.
        raise ApiError(500, "inference_failed", "Inference failed.") from None


app = create_app()


def run():
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    run()
