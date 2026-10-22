"""Optional integration contract with nullable scientific quantities."""
from datetime import date
from typing import Literal,Any
from pydantic import BaseModel,ConfigDict,Field


class AnalysisRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    study_id: str = Field(min_length=1,max_length=100)
    location_id: str = Field(min_length=1,max_length=100)
    intervention_date: date
    horizon_months: int = Field(6,ge=1,le=24)


class AnalysisResponse(BaseModel):
    model_config=ConfigDict(extra='forbid')
    study_id: str
    location_id: str
    intervention_type: str
    effect_absolute: float | None = None
    effect_percent: float | None = None
    interval_lower: float | None = None
    interval_upper: float | None = None
    reliability_score: float | None = Field(None,ge=0,le=1)
    decision: Literal['SUPPORTED_ESTIMATE','UNCERTAIN_REVIEW_REQUIRED','INSUFFICIENT_EVIDENCE']
    model_weights: dict[str,float] = Field(default_factory=dict)
    diagnostics: dict[str,Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trajectories: list[dict[str,Any]] = Field(default_factory=list)
    exploratory_contrast: dict[str,Any] | None = None
    explanation: dict[str,Any] = Field(default_factory=dict)
    provenance: dict[str,Any] | None = None
