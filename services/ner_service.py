from typing import List, TypedDict

from services.model_loader import Pipeline


class NerPrediction(TypedDict):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


def extract_entities_raw(text: str, pipeline: Pipeline) -> List[NerPrediction]:
    t = (text or "").strip()
    if not t:
        return []
    return pipeline(t)
