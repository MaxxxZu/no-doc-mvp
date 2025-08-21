from typing import List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        description="Full document text to classify",
        example="Søknad om igangsettingstillatelse for nybygg av kontorbygg i Oslo.",
    )
    labels: Optional[List[str]] = Field(
        None,
        description="List of labels; if missing — default set is used",
        example=["rammetillatelse", "igangsettingstillatelse", "annet"],
    )
    multi_label: Optional[bool] = Field(
        None,
        description="Overrides config ZSC_MULTI_LABEL",
        example=False,
    )


class ClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[float]
    top_label: str
    top_score: float
