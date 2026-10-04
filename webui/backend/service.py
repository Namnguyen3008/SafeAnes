from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal
from core.model_inference import InputStatus, ModelPrediction, SafeAnesInference
from core.replay import sanitize_recorded_vitals
from core.vitaldb_source import VitalDBSource

MAX_WINDOW_SECONDS = 30.0
MAX_DISPLAY_RATE_HZ = 250.0
ALLOWED_SPEEDS = frozenset({0.5, 1.0, 2.0, 5.0, 10.0})
PREDICTION_STRIDE_SECONDS = 30.0


CHANNEL_ORDER = ("ecg", "pleth", "ppg", "art", "abp", "resp", "capno", "co2", "awp", "flow")


class ReplayServiceError(Exception):
    """Base class for errors safe to map to local API responses."""


class CaseNotFound(ReplayServiceError):
    pass


class CaseSourceError(ReplayServiceError):
    pass


class ReplayNotFound(ReplayServiceError):
    pass


class ReplayValidationError(ReplayServiceError):
    pass


@dataclass
class _Replay:
    replay_id: str
    timeline: PatientTimeline
    display_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    position_sec: float = 0.0
    playing: bool = False
    speed: float = 1.0
    anchor_wall_sec: float = 0.0
    prediction_history: dict[int, tuple[ModelPrediction, ModelPrediction]] = field(
        default_factory=dict
    )
    prediction_lock: threading.Lock = field(default_factory=threading.Lock)


class ReplayService:
    """Keep real patient timelines in Python and expose bounded case-time windows."""

    def __init__(
        self,
        source: VitalDBSource | None = None,
        inference: SafeAnesInference | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.source = source or VitalDBSource()
        self.inference = inference or SafeAnesInference()
        self.clock = clock or time.monotonic
        self._replays: dict[str, _Replay] = {}
        self._lock = threading.RLock()

    def available_cases(self, limit: int = 50) -> dict:
        if limit < 1 or limit > 500:
            raise ReplayValidationError("Case limit must be between 1 and 500.")
        try:
            cases = self.source.available_cases(limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise CaseSourceError(f"VitalDB case lookup failed: {type(exc).__name__}: {exc}") from exc
        return {
            "cases": [
                {
                    "case_id": int(case.case_id),
                    "duration_sec": float(case.duration_sec),
                    "source": "VitalDB",
                }
                for case in cases
            ]
        }

    @staticmethod
    def _display_ranges(timeline: PatientTimeline) -> dict[str, tuple[float, float]]:
        """Keep each replay channel on one robust, case-wide display scale."""
        ranges: dict[str, tuple[float, float]] = {}
        for name, signal in timeline.waveforms.items():
            if signal.values is None:
                continue
            values = np.asarray(signal.values, dtype=np.float64).reshape(-1)
            if values.size > 100_000:
                stride = max(1, (values.size + 99_999) // 100_000)
                values = values[::stride]
            values = values[np.isfinite(values)]
            if name.lower() in {"art", "abp", "capno", "co2"}:
                values = values[values >= 0.0]
            if values.size < 2:
                continue

            low, high = np.percentile(values, (1.0, 99.0))
            span = float(high - low)
            if span <= 1e-6:
                padding = max(abs(float((high + low) * 0.5)) * 0.1, 1.0)
            else:
                padding = max(span * 0.05, 0.05)
            low = float(low) - padding
            high = float(high) + padding
            if name.lower() in {"art", "abp", "capno", "co2"}:
                low = max(0.0, low)
            if high <= low:
                high = low + 1.0
            ranges[name.lower()] = (low, high)
        return ranges

    def load_replay(self, case_id: int | str) -> dict:
        try:
            normalized_id = int(case_id)
        except (TypeError, ValueError) as exc:
            raise ReplayValidationError("VitalDB case ID must be a positive integer.") from exc
        if normalized_id <= 0:
            raise ReplayValidationError("VitalDB case ID must be a positive integer.")
        try:
            timeline = self.source.load_timeline(normalized_id)
        except (KeyError, ValueError) as exc:
            raise CaseNotFound(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise CaseSourceError(
                f"VitalDB case {normalized_id} could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        replay_id = uuid.uuid4().hex
        replay = _Replay(
            replay_id=replay_id,
            timeline=timeline,
            display_ranges=self._display_ranges(timeline),
            anchor_wall_sec=self.clock(),
        )
        with self._lock:
            self._replays[replay_id] = replay
        return self._created(replay)

    def frame(self, replay_id: str, window_seconds: float = 10.0) -> dict:
        replay = self._get_replay(replay_id)
        try:
            window = float(window_seconds)
        except (TypeError, ValueError) as exc:
            raise ReplayValidationError("Waveform window must be numeric.") from exc
        if not math.isfinite(window) or not 1.0 <= window <= MAX_WINDOW_SECONDS:
            raise ReplayValidationError(
                f"Waveform window must be between 1 and {MAX_WINDOW_SECONDS:g} seconds."
            )

        position = self._position(replay)
        order = {name: index for index, name in enumerate(CHANNEL_ORDER)}
        rows = []
        ordered_waveforms = sorted(
            replay.timeline.waveforms.items(),
            key=lambda item: (order.get(item[0].lower(), len(order)), item[0].lower()),
        )
        for name, signal in ordered_waveforms:
            display_rate = min(signal.sample_rate_hz, MAX_DISPLAY_RATE_HZ)
            values = signal.window(position, window, display_rate)
            rows.append(
                {
                    **self._signal_info(name, signal),
                    "display_rate_hz": display_rate,
                    "display_range": replay.display_ranges.get(name.lower()),
                    "samples": [self._finite_or_none(value) for value in values],
                }
            )

        raw_numerics: dict[str, float | None] = {}
        for name, signal in replay.timeline.numerics.items():
            raw_numerics[name] = self._numeric_at(signal, position)
        cleaned_numerics = sanitize_recorded_vitals(raw_numerics)
        numerics = {
            name: {
                **self._signal_info(name, signal),
                "value": cleaned_numerics.get(name),
            }
            for name, signal in sorted(replay.timeline.numerics.items())
        }
        history = [
            self._prediction_point(stride * PREDICTION_STRIDE_SECONDS, predictions)
            for stride, predictions in sorted(replay.prediction_history.items())
            if stride * PREDICTION_STRIDE_SECONDS <= position
        ]
        return {
            "replay_id": replay.replay_id,
            "case_id": int(replay.timeline.case_id),
            "source": replay.timeline.source_label,
            "case_time_sec": position,
            "duration_sec": replay.timeline.duration_sec,
            "playing": replay.playing,
            "speed": replay.speed,
            "waveforms": rows,
            "numerics": numerics,
            "prediction_history": history,
        }

    def control(
        self,
        replay_id: str,
        action: str,
        position_sec: float | None = None,
        speed: float | None = None,
    ) -> dict:
        replay = self._get_replay(replay_id)
        now = self.clock()
        position = self._position(replay, now)
        duration = replay.timeline.duration_sec

        if action == "play":
            if position < duration:
                replay.position_sec = position
                replay.anchor_wall_sec = now
                replay.playing = True
        elif action == "pause":
            replay.position_sec = position
            replay.anchor_wall_sec = now
            replay.playing = False
        elif action == "seek":
            if position_sec is None or not math.isfinite(float(position_sec)):
                raise ReplayValidationError("Seek requires a finite case-time position.")
            target = float(position_sec)
            if target < 0 or target > duration:
                raise ReplayValidationError(f"Seek position must be between 0 and {duration:g} seconds.")
            replay.position_sec = target
            replay.anchor_wall_sec = now
            replay.playing = False
        elif action == "speed":
            if speed is None or not math.isfinite(float(speed)):
                raise ReplayValidationError("Speed control requires a finite replay speed.")
            target_speed = float(speed)
            if target_speed not in ALLOWED_SPEEDS:
                allowed = ", ".join(f"{value:g}×" for value in sorted(ALLOWED_SPEEDS))
                raise ReplayValidationError(f"Replay speed must be one of: {allowed}.")
            replay.position_sec = position
            replay.anchor_wall_sec = now
            replay.speed = target_speed
        elif action == "restart":
            replay.position_sec = 0.0
            replay.anchor_wall_sec = now
            replay.playing = False
        else:
            raise ReplayValidationError(f"Unsupported replay action: {action}.")

        return {
            "replay_id": replay.replay_id,
            "case_id": int(replay.timeline.case_id),
            "case_time_sec": self._position(replay, now),
            "duration_sec": duration,
            "playing": replay.playing,
            "speed": replay.speed,
        }

    def predict(self, replay_id: str, end_sec: float) -> dict:
        replay = self._get_replay(replay_id)
        try:
            requested_time = float(end_sec)
        except (TypeError, ValueError) as exc:
            raise ReplayValidationError("Prediction time must be numeric.") from exc
        if not math.isfinite(requested_time):
            raise ReplayValidationError("Prediction time must be finite.")
        cursor = self._position(replay)
        if requested_time < 0 or requested_time > cursor + 1e-6:
            raise ReplayValidationError("Prediction time cannot be ahead of the replay cursor.")
        if requested_time > replay.timeline.duration_sec:
            raise ReplayValidationError("Prediction time is beyond the case duration.")
        stride_index = int(requested_time // PREDICTION_STRIDE_SECONDS)
        timestamp = float(stride_index * PREDICTION_STRIDE_SECONDS)
        with replay.prediction_lock:
            predictions = replay.prediction_history.get(stride_index)
            if predictions is None:
                try:
                    predictions = self.inference.predict_all(replay.timeline, timestamp)
                except Exception as exc:  # noqa: BLE001
                    reason = f"Inference failed: {type(exc).__name__}: {exc}"
                    predictions = (
                        ModelPrediction("UC04", InputStatus.ERROR, timestamp, reason=reason),
                        ModelPrediction("UC05", InputStatus.ERROR, timestamp, reason=reason),
                    )
                replay.prediction_history[stride_index] = predictions
        history = [
            self._prediction_point(stride * PREDICTION_STRIDE_SECONDS, pair)
            for stride, pair in sorted(replay.prediction_history.items())
            if stride * PREDICTION_STRIDE_SECONDS <= cursor
        ]
        return {
            "replay_id": replay.replay_id,
            "case_id": int(replay.timeline.case_id),
            "timestamp_sec": timestamp,
            "models": [self._prediction(model) for model in predictions],
            "history": history,
        }

    def release(self, replay_id: str) -> None:
        with self._lock:
            if self._replays.pop(replay_id, None) is None:
                raise ReplayNotFound(f"Replay {replay_id} was not found or has expired.")

    def _get_replay(self, replay_id: str) -> _Replay:
        with self._lock:
            replay = self._replays.get(replay_id)
        if replay is None:
            raise ReplayNotFound(f"Replay {replay_id} was not found or has expired.")
        return replay

    def _position(self, replay: _Replay, now: float | None = None) -> float:
        if replay.playing:
            elapsed = max(0.0, (self.clock() if now is None else now) - replay.anchor_wall_sec)
            current = replay.position_sec + elapsed * replay.speed
            if current >= replay.timeline.duration_sec:
                replay.position_sec = replay.timeline.duration_sec
                replay.anchor_wall_sec = self.clock() if now is None else now
                replay.playing = False
                return replay.position_sec
            return current
        return replay.position_sec

    def _created(self, replay: _Replay) -> dict:
        return {
            "replay_id": replay.replay_id,
            "case_id": int(replay.timeline.case_id),
            "source": replay.timeline.source_label,
            "duration_sec": replay.timeline.duration_sec,
            "case_time_sec": 0.0,
            "playing": False,
            "speed": replay.speed,
            "waveforms": [
                self._signal_info(name, signal)
                for name, signal in sorted(replay.timeline.waveforms.items())
            ],
            "numerics": [
                self._signal_info(name, signal)
                for name, signal in sorted(replay.timeline.numerics.items())
            ],
        }

    @staticmethod
    def _signal_info(name: str, signal: TimelineSignal) -> dict:
        return {
            "name": name,
            "sample_rate_hz": signal.sample_rate_hz,
            "status": signal.status.value,
            "units": signal.units,
            "track_name": signal.track_name,
            "reason": signal.reason,
        }

    @staticmethod
    def _finite_or_none(value: float | np.floating) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    @staticmethod
    def _numeric_at(signal: TimelineSignal, position_sec: float) -> float | None:
        if signal.status is not SignalStatus.AVAILABLE or signal.values is None:
            return None
        values = np.asarray(signal.values, dtype=np.float32)
        index = min(int(math.floor(position_sec * signal.sample_rate_hz)), values.size - 1)
        if index < 0 or values.size == 0:
            return None
        value = float(values[index])
        return value if math.isfinite(value) else None

    @classmethod
    def _prediction(cls, prediction: ModelPrediction) -> dict:
        return {
            "model_id": prediction.model_id,
            "status": prediction.status.value,
            "timestamp_sec": float(prediction.timestamp_sec),
            "scores": cls._finite_map(prediction.scores),
            "thresholds": cls._finite_map(prediction.thresholds),
            "score_kind": prediction.score_kind,
            "reason": prediction.reason,
            "notes": list(prediction.notes),
        }

    @classmethod
    def _prediction_point(
        cls, timestamp: int, predictions: tuple[ModelPrediction, ModelPrediction]
    ) -> dict:
        return {
            "timestamp_sec": float(timestamp),
            "models": [cls._prediction(model) for model in predictions],
        }

    @staticmethod
    def _finite_map(values: dict | object) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in dict(values).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                output[str(key)] = number
        return output
