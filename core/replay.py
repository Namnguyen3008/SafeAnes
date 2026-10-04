"""Deterministic case-time replay clock and monitor-buffer writer."""

from __future__ import annotations

import math
import time

import numpy as np

from core.buffer import RingBuffer
from core.data_sources import PatientTimeline

REPLAY_CHANNELS = ("ecg", "resp", "pleth", "abp", "co2", "awp")
_TIMELINE_CHANNELS = {
    "ecg": "ecg",
    "resp": "resp",
    "pleth": "ppg",
    "abp": "art",
    "co2": "capno",
    "awp": "awp",
}


class ReplaySession:
    """Play one patient timeline into a ring buffer using one shared clock."""

    def __init__(self, timeline: PatientTimeline, buffer: RingBuffer):
        missing = set(REPLAY_CHANNELS) - set(buffer.channels)
        if missing:
            raise ValueError(f"replay buffer is missing channels: {sorted(missing)}")
        if buffer.sample_rate <= 0:
            raise ValueError("replay buffer sample rate must be positive")
        self.timeline = timeline
        self.buffer = buffer
        self.sample_rate = float(buffer.sample_rate)
        self._position_sec = 0.0
        self._anchor_position_sec = 0.0
        self._anchor_wall_sec = 0.0
        self._next_sample_index = 0
        self._speed = 1.0
        self._playing = False

    @property
    def position_sec(self) -> float:
        return self._position_sec

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def speed(self) -> float:
        return self._speed

    def play(self, now: float | None = None) -> None:
        if self._position_sec >= self.timeline.duration_sec:
            return
        if now is None:
            now = time.monotonic()
        self._anchor_position_sec = self._position_sec
        self._anchor_wall_sec = float(now)
        self._playing = True

    def pause(self, now: float | None = None) -> None:
        if self._playing:
            self.advance(now)
            self._playing = False
            self._anchor_position_sec = self._position_sec

    def seek(self, position_sec: float, now: float | None = None) -> float:
        position = float(position_sec)
        if not math.isfinite(position):
            raise ValueError("seek position must be finite")
        if now is None:
            now = time.monotonic()
        was_playing = self._playing
        self._position_sec = min(max(position, 0.0), self.timeline.duration_sec)
        self._anchor_position_sec = self._position_sec
        self._anchor_wall_sec = float(now)
        self.buffer.clear()
        end_index = math.floor(self._position_sec * self.sample_rate + 1e-9)
        count = min(end_index, self.buffer.capacity)
        start_index = end_index - count
        if count:
            self.buffer.write(self.sample_block(start_index, count))
        self._next_sample_index = end_index
        self._playing = was_playing and self._position_sec < self.timeline.duration_sec
        return self._position_sec

    def set_speed(self, speed: float, now: float | None = None) -> None:
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("replay speed must be positive and finite")
        if self._playing:
            self.advance(now)
        self._speed = speed
        if now is None:
            now = time.monotonic()
        self._anchor_position_sec = self._position_sec
        self._anchor_wall_sec = float(now)

    def advance(self, now: float | None = None) -> int:
        """Advance to wall time ``now`` and write all newly elapsed samples."""

        if not self._playing:
            return 0
        if now is None:
            now = time.monotonic()
        elapsed = max(0.0, float(now) - self._anchor_wall_sec)
        position = min(
            self.timeline.duration_sec,
            self._anchor_position_sec + elapsed * self._speed,
        )
        end_index = math.floor(position * self.sample_rate + 1e-9)
        count = max(0, end_index - self._next_sample_index)
        if count:
            self.buffer.write(self.sample_block(self._next_sample_index, count))
            self._next_sample_index = end_index
        self._position_sec = position
        if position >= self.timeline.duration_sec:
            self._playing = False
            self._anchor_position_sec = position
        return count

    def sample_block(self, start_index: int, count: int) -> np.ndarray:
        """Sample every monitor channel at one shared set of case timestamps."""

        start_index = int(start_index)
        count = int(count)
        if start_index < 0 or count < 0:
            raise ValueError("sample block indices must be non-negative")
        block = np.full((len(REPLAY_CHANNELS), count), np.nan, dtype=np.float64)
        if count == 0:
            return block
        end_sec = (start_index + count) / self.sample_rate
        duration_sec = count / self.sample_rate
        for channel_index, buffer_name in enumerate(REPLAY_CHANNELS):
            timeline_name = _TIMELINE_CHANNELS[buffer_name]
            samples = self.timeline.waveform_window(
                timeline_name,
                end_sec=end_sec,
                seconds=duration_sec,
                target_rate_hz=self.sample_rate,
            )
            block[channel_index] = samples[:count]
        return block


def sanitize_recorded_vitals(values: dict[str, float | None]) -> dict[str, float | None]:
    """Hide internally inconsistent arterial pressure readings in the UI."""
    cleaned = dict(values)
    raw_sbp = cleaned.get("SBP")
    raw_dbp = cleaned.get("DBP")

    def finite_pressure(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            pressure = float(value)
        except (TypeError, ValueError):
            return None
        return pressure if np.isfinite(pressure) and pressure > 0.0 else None

    sbp = finite_pressure(raw_sbp)
    dbp = finite_pressure(raw_dbp)
    if raw_sbp is not None and raw_dbp is not None and (
        sbp is None or dbp is None or sbp < dbp
    ):
        cleaned.update({"SBP": None, "DBP": None, "MAP": None})
        return cleaned
    cleaned["SBP"] = sbp
    cleaned["DBP"] = dbp

    mean = finite_pressure(cleaned.get("MAP"))
    systolic = sbp
    diastolic = dbp
    if mean is not None:
        if (
            (diastolic is not None and mean < diastolic - 10.0)
            or (systolic is not None and mean > systolic + 10.0)
        ):
            mean = None
    cleaned["MAP"] = mean
    return cleaned
