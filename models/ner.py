from typing import List

from pydantic import BaseModel, Field


class NerRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text to extract named entities from",
        example="AF Gruppen er arkitekt i Oslo. BRA 120 m², tomteareal 900 m².",
    )


class NerEntity(BaseModel):
    entity_group: str = Field(..., example="ORG")
    word: str = Field(..., example="AF Gruppen")
    score: float = Field(..., example=0.99)
    start: int = Field(..., example=0)
    end: int = Field(..., example=10)


class NerResponse(BaseModel):
    entities: List[NerEntity]
