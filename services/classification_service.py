from typing import List, Optional, Tuple
import os
from transformers import pipeline
from config import settings

DEFAULT_LABELS = [
    "rammetillatelse",
    "igangsettingstillatelse",
    "tiltakssøknad",
    "endringssøknad",
    "ferdigattest",
    "midlertidig brukstillatelse",
    "tilsyn/kontroll",
    "klage",
    "teknisk tegning",
    "følgebrev",
    "annet",
]

_ZSC_PIPE = None  # lazy singleton


def _ensure_hf_cache_env():
    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    if settings.transformers_cache:
        os.environ.setdefault("TRANSFORMERS_CACHE", settings.transformers_cache)


def _get_pipeline():
    global _ZSC_PIPE
    if _ZSC_PIPE is None:
        _ensure_hf_cache_env()
        _ZSC_PIPE = pipeline(
            task="zero-shot-classification",
            model=settings.zsc_model,
            device=-1,  # CPU
        )
    return _ZSC_PIPE


def classify_text_raw(
    text: str,
    labels: Optional[List[str]] = None,
    multi_label_override: Optional[bool] = None,
) -> Tuple[List[str], List[float]]:
    """Return (labels, scores) sorted by descending score."""
    t = (text or "").strip()
    if not t:
        return [], []

    zsc = _get_pipeline()
    used_labels = labels or DEFAULT_LABELS
    multi = settings.zsc_multi_label if multi_label_override is None else multi_label_override
    if len(t) > settings.zsc_max_len:
        t = t[: settings.zsc_max_len]

    res = zsc(
        sequences=t,
        candidate_labels=used_labels,
        multi_label=multi,
        hypothesis_template="Dette dokumentet handler om {}.",
    )
    out_labels = list(res["labels"])
    out_scores = [float(s) for s in res["scores"]]
    return out_labels, out_scores
