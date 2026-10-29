from collections.abc import Mapping
from typing import List, Optional, Tuple

from config import Settings
from services.model_loader import Pipeline
from services.validation import check_tokens, validate_classification

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
    used_labels = DEFAULT_LABELS if labels is None else labels
    multi = settings.zsc_multi_label if multi_label_override is None else multi_label_override
    for label in used_labels:
        check_tokens(pipeline, text, "Dette dokumentet handler om {}.".format(label))

    res = pipeline(
        sequences=text,
        candidate_labels=used_labels,
        multi_label=multi,
        hypothesis_template="Dette dokumentet handler om {}.",
    )
    if not isinstance(res, Mapping):
        raise ValueError("Invalid classification result")
    labels_result, scores_result = validate_classification(
        (res["labels"], res["scores"]), used_labels,
    )
    out_labels = list(labels_result)
    out_scores = [float(score) for score in scores_result]
    return out_labels, out_scores
