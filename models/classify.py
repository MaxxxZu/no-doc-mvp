from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Full document text to classify")
    labels: Optional[List[str]] = Field(None, description="Candidate labels; null uses defaults")
    multi_label: Optional[bool] = Field(None, description="Overrides ZSC_MULTI_LABEL")

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value

    @field_validator("labels")
    @classmethod
    def valid_labels(cls, value):
        if value is None:
            return value
        if not 1 <= len(value) <= 32:
            raise ValueError("Invalid number of labels")
        labels = [label.strip() for label in value]
        if any(not label or len(label) > 128 for label in labels):
            raise ValueError("Invalid label")
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate labels")
        return labels


class ClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[float]
    top_label: str
    top_score: float
