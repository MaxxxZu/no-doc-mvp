"""Explicit artifact loading; importing this module does not import Transformers."""

from typing import Any, Literal, Protocol

from config import Settings

Operation = Literal["classify", "ner"]


class Pipeline(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


def load_pipeline(operation: Operation, settings: Settings) -> Pipeline:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    is_zsc = operation == "classify"
    model_id = settings.zsc_model if is_zsc else settings.ner_model
    common = dict(
        cache_dir=settings.hf_home,
        revision=settings.zsc_revision if is_zsc else settings.ner_revision,
        local_files_only=settings.offline,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **common)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required")
    loader = AutoModelForSequenceClassification if is_zsc else AutoModelForTokenClassification
    model = loader.from_pretrained(model_id, use_safetensors=True, **common).to("cpu")
    model.eval()
    extra = {} if is_zsc else {"aggregation_strategy": "simple"}
    return pipeline(
        "zero-shot-classification" if is_zsc else "token-classification",
        model=model, tokenizer=tokenizer, device=-1, **extra,
    )
