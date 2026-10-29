from typing import List, Optional, Tuple

from config import Settings
from services.model_loader import Pipeline

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


def classify_text_raw(
    text: str,
    pipeline: Pipeline,
    settings: Settings,
    labels: Optional[List[str]] = None,
    multi_label_override: Optional[bool] = None,
) -> Tuple[List[str], List[float]]:
    """Return (labels, scores) sorted by descending score."""
    t = (text or "").strip()
    if not t:
        return [], []

    used_labels = labels or DEFAULT_LABELS
    multi = settings.zsc_multi_label if multi_label_override is None else multi_label_override
    if len(t) > settings.zsc_max_len:
        t = t[: settings.zsc_max_len]

    res = pipeline(
        sequences=t,
        candidate_labels=used_labels,
        multi_label=multi,
        hypothesis_template="Dette dokumentet handler om {}.",
    )
    out_labels = list(res["labels"])
    out_scores = [float(s) for s in res["scores"]]
    return out_labels, out_scores
