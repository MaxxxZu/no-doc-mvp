from typing import List

from pydantic import BaseModel


class NerRequest(BaseModel):
    text: str


class NerEntity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


class NerResponse(BaseModel):
    entities: List[NerEntity]
