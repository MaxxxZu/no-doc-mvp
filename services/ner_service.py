import os
from typing import List, Dict, Any
from transformers import pipeline
from config import settings

_NER_PIPE = None  # lazy singleton


def _ensure_hf_cache_env():
    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    if settings.transformers_cache:
        os.environ.setdefault("TRANSFORMERS_CACHE", settings.transformers_cache)


def _get_pipeline():
    global _NER_PIPE
    if _NER_PIPE is None:
        _ensure_hf_cache_env()
        _NER_PIPE = pipeline(
            task="token-classification",
            model=settings.ner_model,
            aggregation_strategy="simple",
            device=-1,  # CPU
        )
    return _NER_PIPE


def extract_entities_raw(text: str) -> List[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return []
    ner = _get_pipeline()
    return ner(t)
