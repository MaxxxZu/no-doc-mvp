from typing import List, TypedDict

from services.model_loader import Pipeline
from services.validation import check_tokens


class NerPrediction(TypedDict):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


def extract_entities_raw(text: str, pipeline: Pipeline) -> List[NerPrediction]:
    check_tokens(pipeline, text)
    return pipeline(text)
