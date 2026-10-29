from typing import List

from pydantic import BaseModel, Field, field_validator


class NerRequest(BaseModel):
    text: str = Field(..., description="Text to extract named entities from")

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


class NerEntity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


class NerResponse(BaseModel):
    entities: List[NerEntity]
