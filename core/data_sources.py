"""Timestamped patient signals shared by replay, display, and inference code."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SignalStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_SUPPORTED = "NOT SUPPORTED"


@dataclass(frozen=True)
class TimelineSignal:
    """A regular sampled signal whose sample zero is at case time zero."""

    values: np.ndarray | Sequence[float] | None
    sample_rate_hz: float
    status: SignalStatus = SignalStatus.AVAILABLE
    track_name: str | None = None
    units: str = ""
    reason: str | None = None

    def __post_init__(self) -> None:
        rate = float(self.sample_rate_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("sample rate must be a positive finite number")
        object.__setattr__(self, "sample_rate_hz", rate)
        status = SignalStatus(self.status)
        object.__setattr__(self, "status", status)
        if self.values is None:
            return
        array = np.asarray(self.values, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "values", array)

    def window(
        self,
        end_sec: float,
        seconds: float,
        target_rate_hz: float,
    ) -> np.ndarray:
        """Return a right-aligned, fixed-size window, padding unavailable time with NaN."""

        end_sec = float(end_sec)
        seconds = float(seconds)
        target_rate_hz = float(target_rate_hz)
        if not all(math.isfinite(x) for x in (end_sec, seconds, target_rate_hz)):
            raise ValueError("window times and target sample rate must be finite")
        if seconds <= 0 or target_rate_hz <= 0:
            raise ValueError("window duration and target sample rate must be positive")
        count = math.ceil(seconds * target_rate_hz)
        result = np.full(count, np.nan, dtype=np.float32)
        if self.status is not SignalStatus.AVAILABLE or self.values is None:
            return result

        values = np.asarray(self.values, dtype=np.float32)
        if values.size == 0:
            return result

        start_sec = end_sec - seconds
        target_times = start_sec + np.arange(count, dtype=np.float64) / target_rate_hz
        source_positions = target_times * self.sample_rate_hz
        in_range = (source_positions >= 0) & (source_positions < values.size)
        if not np.any(in_range):
            return result

        positions = source_positions[in_range]
        lower = np.floor(positions).astype(np.int64)
        fraction = positions - lower
        upper = np.minimum(lower + 1, values.size - 1)
        exact = np.isclose(fraction, 0.0, atol=1e-7)

        sampled = np.full(positions.shape, np.nan, dtype=np.float32)
        exact_valid = exact & np.isfinite(values[lower])
        sampled[exact_valid] = values[lower[exact_valid]]
        between = (~exact) & np.isfinite(values[lower]) & np.isfinite(values[upper])
        if np.any(between):
            sampled[between] = (
                values[lower[between]] * (1.0 - fraction[between])
                + values[upper[between]] * fraction[between]
            )
        result[np.flatnonzero(in_range)] = sampled
        return result


@dataclass
class PatientTimeline:
    """One real or synthetic case on a common case-time axis."""

    case_id: int | str
    duration_sec: float
    source_label: str
    waveforms: Mapping[str, TimelineSignal] = field(default_factory=dict)
    numerics: Mapping[str, TimelineSignal] = field(default_factory=dict)
    static_features: Mapping[str, float] = field(default_factory=dict)
    generation: int = 0

    def __post_init__(self) -> None:
        duration = float(self.duration_sec)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("case duration must be positive and finite")
        self.duration_sec = duration
        self.waveforms = {str(k).lower(): v for k, v in self.waveforms.items()}
        self.numerics = {str(k).upper(): v for k, v in self.numerics.items()}
        self.static_features = dict(self.static_features)

    def waveform_window(
        self,
        name: str,
        end_sec: float,
        seconds: float,
        target_rate_hz: float,
    ) -> np.ndarray:
        signal = self.waveforms.get(str(name).lower())
        if signal is None:
            return _nan_window(seconds, target_rate_hz)
        return signal.window(end_sec, seconds, target_rate_hz)

    def numeric_history(
        self,
        names: Sequence[str],
        end_sec: float,
        seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered 1 Hz features and a finite-value mask."""

        count = math.ceil(float(seconds))
        if count <= 0 or not math.isfinite(float(seconds)):
            raise ValueError("numeric history duration must be positive and finite")
        columns: list[np.ndarray] = []
        for name in names:
            signal = self.numerics.get(str(name).upper())
            if signal is None:
                column = np.full(count, np.nan, dtype=np.float32)
            else:
                column = signal.window(end_sec, seconds, 1.0)
            columns.append(column)
        values = (
            np.column_stack(columns).astype(np.float32, copy=False)
            if columns
            else np.empty((count, 0), dtype=np.float32)
        )
        mask = np.isfinite(values).astype(np.float32)
        return values, mask


def _nan_window(seconds: float, target_rate_hz: float) -> np.ndarray:
    seconds = float(seconds)
    rate = float(target_rate_hz)
    if not math.isfinite(seconds) or not math.isfinite(rate) or seconds <= 0 or rate <= 0:
        raise ValueError("window duration and target sample rate must be positive and finite")
    return np.full(math.ceil(seconds * rate), np.nan, dtype=np.float32)
