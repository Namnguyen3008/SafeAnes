from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class CaseSummary(BaseModel):
    case_id: int
    duration_sec: float
    source: str


class CasesResponse(BaseModel):
    cases: list[CaseSummary]


class SignalInfo(BaseModel):
    name: str
    sample_rate_hz: float
    status: str
    units: str = ""
    track_name: str | None = None
    reason: str | None = None


class NumericInfo(SignalInfo):
    value: float | None = None


class ReplayCreatedResponse(BaseModel):
    replay_id: str
    case_id: int
    source: str
    duration_sec: float
    case_time_sec: float
    playing: bool
    speed: float
    waveforms: list[SignalInfo]
    numerics: list[SignalInfo]


class WaveformWindow(SignalInfo):
    display_rate_hz: float
    display_range: tuple[float, float] | None = None
    samples: list[float | None]


class NumericValue(NumericInfo):
    pass


class ModelPredictionResponse(BaseModel):
    model_id: str
    status: str
    timestamp_sec: float
    scores: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    score_kind: str = ""
    reason: str = ""
    notes: list[str] = Field(default_factory=list)


class PredictionPoint(BaseModel):
    timestamp_sec: float
    models: list[ModelPredictionResponse]


class ReplayFrameResponse(BaseModel):
    replay_id: str
    case_id: int
    source: str
    case_time_sec: float
    duration_sec: float
    playing: bool
    speed: float
    waveforms: list[WaveformWindow]
    numerics: dict[str, NumericValue]
    prediction_history: list[PredictionPoint]


class ReplayControlRequest(BaseModel):
    action: Literal["play", "pause", "seek", "speed", "restart"]
    position_sec: float | None = Field(default=None, ge=0)
    speed: float | None = Field(default=None, gt=0)


class ReplayControlResponse(BaseModel):
    replay_id: str
    case_id: int
    case_time_sec: float
    duration_sec: float
    playing: bool
    speed: float


class PredictionRequest(BaseModel):
    case_time_sec: float = Field(ge=0)


class PredictionResponse(BaseModel):
    replay_id: str
    case_id: int
    timestamp_sec: float
    models: list[ModelPredictionResponse]
    history: list[PredictionPoint]
