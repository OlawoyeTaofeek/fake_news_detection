"""Request and response shapes for the API. Pydantic validates every request automatically."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

DISCLAIMER = (
    "This model detects misinformation-style writing patterns. "
    "It does not check facts, so always verify with trusted sources."
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        max_length=20000,
        description="Headline or article text to check",
        examples=["SHOCKING: Scientists finally admit what they have been hiding from you!!!"],
    )

    @field_validator("text")
    @classmethod
    def not_just_whitespace(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("Text must contain at least 20 non-space characters")
        return v


class PredictResponse(BaseModel):
    request_id: str
    label: Literal["fake", "real"]
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted label")
    prob_fake: float = Field(..., ge=0, le=1)
    model_version: str
    latency_ms: float
    disclaimer: str = DISCLAIMER


class FeedbackRequest(BaseModel):
    request_id: str
    correct: bool = Field(..., description="Was the prediction correct?")


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class StatsResponse(BaseModel):
    total_predictions: int
    fake_ratio: float | None
    avg_confidence: float | None
    avg_latency_ms: float | None
    feedback_count: int
    feedback_accuracy: float | None
