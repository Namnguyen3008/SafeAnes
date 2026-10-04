"""Versioned, provenance-bearing UC01 observations and signal segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class DataQuality(str, Enum):
    VALID = "valid"
    SUSPECT = "suspect"
    ARTIFACT = "artifact"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class ClinicalEventStatus(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"


def _nonblank(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    subject_id: str
    observed_at: datetime
    name: str
    value: float | None
    unit: str
    source_id: str
    quality: DataQuality = DataQuality.VALID

    def __post_init__(self) -> None:
        for field in ("observation_id", "subject_id", "name", "unit", "source_id"):
            object.__setattr__(self, field, _nonblank(getattr(self, field), field))
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        quality = DataQuality(self.quality)
        object.__setattr__(self, "quality", quality)

        if self.value is None:
            if quality not in (DataQuality.MISSING, DataQuality.UNAVAILABLE):
                raise ValueError("A null observation value requires missing or unavailable quality")
            return
        if quality in (DataQuality.MISSING, DataQuality.UNAVAILABLE):
            raise ValueError("Missing or unavailable observations cannot carry a value")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("Observation value must be numeric or null")
        numeric_value = float(self.value)
        if not math.isfinite(numeric_value):
            raise ValueError("Observation value must be finite")
        object.__setattr__(self, "value", numeric_value)


@dataclass(frozen=True)
class SignalSegment:
    segment_id: str
    subject_id: str
    name: str
    started_at: datetime
    sample_rate_hz: float
    samples: tuple[float | None, ...]
    unit: str
    source_id: str
    quality: DataQuality = DataQuality.VALID

    def __post_init__(self) -> None:
        for field in ("segment_id", "subject_id", "name", "unit", "source_id"):
            object.__setattr__(self, field, _nonblank(getattr(self, field), field))
        object.__setattr__(self, "started_at", _utc(self.started_at, "started_at"))
        rate = float(self.sample_rate_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        object.__setattr__(self, "sample_rate_hz", rate)

        converted: list[float | None] = []
        for sample in self.samples:
            if sample is None:
                converted.append(None)
                continue
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise TypeError("Signal samples must be numeric or null")
            value = float(sample)
            if not math.isfinite(value):
                raise ValueError("Signal samples must be finite or null")
            converted.append(value)
        object.__setattr__(self, "samples", tuple(converted))

        quality = DataQuality(self.quality)
        object.__setattr__(self, "quality", quality)
        if quality in (DataQuality.MISSING, DataQuality.UNAVAILABLE) and any(
            sample is not None for sample in converted
        ):
            raise ValueError("Missing or unavailable signal segments cannot carry samples")
        if quality is DataQuality.VALID and (
            not converted or all(sample is None for sample in converted)
        ):
            raise ValueError("A valid signal segment requires at least one present sample")

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class ClinicalEvent:
    event_id: str
    subject_id: str
    event_type: str
    started_at: datetime
    source_id: str
    ended_at: datetime | None = None
    status: ClinicalEventStatus = ClinicalEventStatus.OBSERVED
    evidence_ids: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self) -> None:
        for field in ("event_id", "subject_id", "event_type", "source_id"):
            object.__setattr__(self, field, _nonblank(getattr(self, field), field))
        started_at = _utc(self.started_at, "started_at")
        object.__setattr__(self, "started_at", started_at)
        if self.ended_at is not None:
            ended_at = _utc(self.ended_at, "ended_at")
            if ended_at < started_at:
                raise ValueError("ended_at cannot precede started_at")
            object.__setattr__(self, "ended_at", ended_at)

        status = ClinicalEventStatus(self.status)
        object.__setattr__(self, "status", status)
        evidence = tuple(_nonblank(item, "evidence_id") for item in self.evidence_ids)
        if status is ClinicalEventStatus.DERIVED and not evidence:
            raise ValueError("Derived clinical events require evidence_ids")
        object.__setattr__(self, "evidence_ids", evidence)

        normalized: list[tuple[str, str | int | float | bool]] = []
        seen: set[str] = set()
        for key, value in self.attributes:
            name = _nonblank(key, "attribute name")
            if name in seen:
                raise ValueError(f"Duplicate event attribute: {name}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Event attribute {name} must be finite")
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"Event attribute {name} must be a scalar")
            normalized.append((name, value))
            seen.add(name)
        object.__setattr__(self, "attributes", tuple(sorted(normalized)))
