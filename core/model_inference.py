"""Artifact-compatible UC04/UC05 preprocessing and local model execution."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from core.data_sources import PatientTimeline, SignalStatus
from core.vitaldb_source import (
    UC04_NUMERIC_FEATURES,
    UC04_STATIC_FEATURES,
    UC05_NUMERIC_FEATURES,
    UC05_STATIC_FEATURES,
)


class InputStatus(str, Enum):
    READY = "READY"
    WAITING_FOR_HISTORY = "WAITING FOR HISTORY"
    MISSING_REQUIRED_INPUT = "MISSING REQUIRED INPUT"
    NOT_ELIGIBLE = "NOT ELIGIBLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PreparedModelInput:
    model_id: str
    status: InputStatus
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    status: InputStatus
    timestamp_sec: float
    scores: Mapping[str, float] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    score_kind: str = ""
    reason: str = ""
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is InputStatus.READY


@dataclass(frozen=True)
class ModelPaths:
    uc04_artifact_dir: Path
    uc05_artifact_dir: Path
    source_dir: Path

    @classmethod
    def discover(cls) -> ModelPaths:
        project_root = Path(__file__).resolve().parents[1]
        workspace_root = project_root.parent
        downloads_root = workspace_root.parent

        source_candidates = (
            project_root / "safeanes",
            workspace_root / "safeanes",
            downloads_root / "uc4" / "safeanes",
        )
        source_dir = next((p for p in source_candidates if p.is_dir()), source_candidates[0])

        uc04_candidates = (
            project_root / "model_output_v12" / "package",
            workspace_root / "model_output_v12" / "package",
            workspace_root / "UC04" / "package_extracted",
        )
        uc04_dir = next((p for p in uc04_candidates if p.is_dir()), uc04_candidates[0])

        uc05_candidates = (
            project_root / "model_output_v12" / "package",
            workspace_root / "model_output_v12" / "package",
            workspace_root / "UC05" / "package_extracted",
        )
        uc05_dir = next((p for p in uc05_candidates if p.is_dir()), uc05_candidates[0])

        return cls(
            uc04_artifact_dir=Path(os.environ.get("SAFEANES_UC04_ARTIFACT_DIR", uc04_dir)),
            uc05_artifact_dir=Path(os.environ.get("SAFEANES_UC05_ARTIFACT_DIR", uc05_dir)),
            source_dir=Path(os.environ.get("SAFEANES_MODEL_SOURCE_DIR", source_dir)),
        )


def build_uc04_input(timeline: PatientTimeline, end_sec: float) -> PreparedModelInput:
    """Build one 5-minute numeric / 30-second waveform UC04 input."""

    if end_sec < 300.0:
        return PreparedModelInput("UC04", InputStatus.WAITING_FOR_HISTORY,
                                  reason="UC04 requires 300 seconds of numeric history.")
    current_sec = int(end_sec)
    timing_error = _training_timing_reason("UC04", timeline, current_sec, 300, 15 * 60)
    if timing_error:
        return PreparedModelInput("UC04", InputStatus.NOT_ELIGIBLE, reason=timing_error)

    cleaned_map = _clean_training_numeric(timeline, "MAP", current_sec, (20.0, 180.0))
    if cleaned_map is None or current_sec >= cleaned_map.size or not np.isfinite(cleaned_map[current_sec]):
        return PreparedModelInput("UC04", InputStatus.MISSING_REQUIRED_INPUT,
                                  reason="Current recorded MAP value is unavailable.")
    if cleaned_map[current_sec] <= 65.0:
        return PreparedModelInput("UC04", InputStatus.NOT_ELIGIBLE,
                                  reason="UC04 candidate time requires current MAP above 65 mmHg.")
    map_history = cleaned_map[current_sec - 300:current_sec]
    if map_history.size < 300 or float(np.isfinite(map_history).mean()) < 0.80:
        return PreparedModelInput("UC04", InputStatus.MISSING_REQUIRED_INPUT,
                                  reason="Recorded MAP history does not meet the 80% validity gate.")
    if _in_recent_event_recovery(cleaned_map, current_sec, 65.0, 60, inclusive=True):
        return PreparedModelInput("UC04", InputStatus.NOT_ELIGIBLE,
                                  reason="UC04 excludes the 120-second recovery after hypotension.")

    specs = (
        ("art", 30.0, 100.0, (-20.0, 300.0), 0.90, 10),
        ("ecg", 30.0, 250.0, (-20.0, 20.0), 0.80, 10),
        ("ppg", 30.0, 100.0, None, 0.80, 10),
        ("capno", 30.0, 62.5, (-5.0, 100.0), 0.80, 10),
    )
    arrays: dict[str, np.ndarray] = {}
    masks = []
    for name, seconds, target_hz, plausible, min_fraction, min_samples in specs:
        processed, present = _wave_input(
            timeline, name, end_sec, seconds, target_hz, plausible, min_fraction, min_samples
        )
        arrays[name] = processed
        masks.append(int(present))
    if masks[0] == 0:
        return PreparedModelInput("UC04", InputStatus.MISSING_REQUIRED_INPUT,
                                  reason="Recorded ART waveform does not meet the 90% validity gate.")

    numeric, _ = timeline.numeric_history(UC04_NUMERIC_FEATURES, end_sec, 300.0)
    numeric = _interpolate_matrix(numeric, limit=5)
    arrays["numeric"] = numeric.astype(np.float32, copy=False)
    arrays["numeric_mask"] = np.isfinite(numeric).astype(np.float32)
    arrays["static"] = _static_vector(timeline, UC04_STATIC_FEATURES)
    arrays["modality_mask"] = np.asarray(masks, dtype=np.float32)
    return PreparedModelInput("UC04", InputStatus.READY, arrays)


def build_uc05_input(timeline: PatientTimeline, end_sec: float) -> PreparedModelInput:
    """Build one 10-minute numeric / 60-second waveform UC05 input."""

    if end_sec < 600.0:
        return PreparedModelInput("UC05", InputStatus.WAITING_FOR_HISTORY,
                                  reason="UC05 requires 600 seconds of numeric history.")
    current_sec = int(end_sec)
    timing_error = _training_timing_reason("UC05", timeline, current_sec, 600, 5 * 60)
    if timing_error:
        return PreparedModelInput("UC05", InputStatus.NOT_ELIGIBLE, reason=timing_error)

    cleaned_spo2 = _clean_training_numeric(timeline, "SPO2", current_sec, (50.0, 100.0))
    if cleaned_spo2 is None or current_sec >= cleaned_spo2.size or not np.isfinite(cleaned_spo2[current_sec]):
        return PreparedModelInput("UC05", InputStatus.MISSING_REQUIRED_INPUT,
                                  reason="Current recorded SpO₂ value is unavailable.")
    if cleaned_spo2[current_sec] < 90.0:
        return PreparedModelInput("UC05", InputStatus.NOT_ELIGIBLE,
                                  reason="UC05 candidate time requires current SpO₂ at or above 90%.")
    spo2_history = cleaned_spo2[current_sec - 600:current_sec]
    if spo2_history.size < 600 or float(np.isfinite(spo2_history).mean()) < 0.80:
        return PreparedModelInput("UC05", InputStatus.MISSING_REQUIRED_INPUT,
                                  reason="Recorded SpO₂ history does not meet the 80% validity gate.")
    if _in_recent_event_recovery(cleaned_spo2, current_sec, 90.0, 60, inclusive=False):
        return PreparedModelInput("UC05", InputStatus.NOT_ELIGIBLE,
                                  reason="UC05 excludes the 120-second recovery after hypoxemia.")

    specs = (
        ("capno", 60.0, 62.5, 62.5, (-5.0, 120.0), False),
        ("awp", 60.0, 62.5, 62.5, (-20.0, 120.0), False),
        ("ppg", 60.0, 500.0, 125.0, None, False),
        ("flow", 60.0, 62.5, 62.5, None, True),
        ("resp", 60.0, 62.5, 62.5, None, True),
    )
    arrays: dict[str, np.ndarray] = {}
    masks: list[int] = []
    reasons = []
    for name, seconds, source_hz, target_hz, plausible, optional in specs:
        processed, present = _wave_input(
            timeline, name, end_sec, seconds, target_hz, plausible, 0.80, 8,
            source_rate_hz=source_hz,
        )
        if not present:
            signal = timeline.waveforms.get(name)
            unsupported = signal is None or signal.status is SignalStatus.NOT_SUPPORTED
            if unsupported and optional:
                processed = np.zeros(round(seconds * target_hz), dtype=np.float32)
            else:
                reasons.append(name.upper())
        arrays[name] = processed
        masks.append(int(present))
    if reasons:
        return PreparedModelInput(
            "UC05", InputStatus.MISSING_REQUIRED_INPUT,
            reason=f"Required UC05 waveforms failed validity checks: {', '.join(reasons)}.",
        )

    feature_keys = tuple(
        "ETCO2_UC05" if feature == "ETCO2" else feature
        for feature in UC05_NUMERIC_FEATURES
    )
    numeric, _ = timeline.numeric_history(feature_keys, end_sec, 600.0)
    numeric = _interpolate_matrix(numeric, limit=5)
    arrays["numeric"] = numeric.astype(np.float32, copy=False)
    arrays["numeric_mask"] = np.isfinite(numeric).astype(np.float32)
    arrays["static"] = _static_vector(timeline, UC05_STATIC_FEATURES)
    arrays["wave_mask"] = np.asarray(masks, dtype=np.float32)
    return PreparedModelInput("UC05", InputStatus.READY, arrays)


class SafeAnesInference:
    """Lazy CPU loader for the two released weight packages and their wrappers."""

    def __init__(self, paths: ModelPaths | None = None):
        self.paths = paths or ModelPaths.discover()
        self._models: dict[str, Any] = {}
        self._modules: dict[str, Any] = {}
        self._torch = None

    def predict_all(self, timeline: PatientTimeline, end_sec: float) -> tuple[ModelPrediction, ModelPrediction]:
        return self.predict_uc04(timeline, end_sec), self.predict_uc05(timeline, end_sec)

    def predict_uc04(self, timeline: PatientTimeline, end_sec: float) -> ModelPrediction:
        prepared = build_uc04_input(timeline, end_sec)
        if prepared.status is not InputStatus.READY:
            return ModelPrediction("UC04", prepared.status, end_sec, reason=prepared.reason,
                                   notes=self._uc04_notes())
        try:
            model = self._load_model("UC04")
            torch = self._torch_module()
            with torch.inference_mode():
                logits = model(**_to_tensors(prepared.arrays, torch))
            scores = _sigmoid(logits.detach().cpu().numpy().reshape(-1))
            thresholds = self._thresholds("UC04")
            return ModelPrediction(
                "UC04", InputStatus.READY, end_sec,
                scores={f"{h}m": float(p) for h, p in zip((5, 10, 15), scores, strict=True)},
                thresholds=thresholds,
                score_kind="raw model score (uncalibrated)",
                notes=self._uc04_notes(),
            )
        except Exception as exc:  # noqa: BLE001
            return ModelPrediction("UC04", InputStatus.ERROR, end_sec,
                                   reason=f"UC04 inference failed: {type(exc).__name__}: {exc}",
                                   notes=self._uc04_notes())

    def predict_uc05(self, timeline: PatientTimeline, end_sec: float) -> ModelPrediction:
        prepared = build_uc05_input(timeline, end_sec)
        if prepared.status is not InputStatus.READY:
            return ModelPrediction("UC05", prepared.status, end_sec, reason=prepared.reason,
                                   notes=self._uc05_notes())
        try:
            model = self._load_model("UC05")
            torch = self._torch_module()
            args = _to_tensors(prepared.arrays, torch)
            with torch.inference_mode():
                hypox_logits, vent_logit = model(**args)
            calibrated = self._calibrate_uc05(hypox_logits.detach().cpu().numpy().reshape(-1))
            vent_score = float(_sigmoid(np.asarray(vent_logit.detach().cpu().numpy()).reshape(-1))[0])
            return ModelPrediction(
                "UC05", InputStatus.READY, end_sec,
                scores={
                    **{f"{h}m": float(p) for h, p in zip((1, 3, 5), calibrated, strict=True)},
                    "ventilation pattern (weak-label head)": vent_score,
                },
                thresholds=self._thresholds("UC05"),
                score_kind="temperature-scaled model score",
                notes=self._uc05_notes(),
            )
        except Exception as exc:  # noqa: BLE001
            return ModelPrediction("UC05", InputStatus.ERROR, end_sec,
                                   reason=f"UC05 inference failed: {type(exc).__name__}: {exc}",
                                   notes=self._uc05_notes())

    def _load_model(self, model_id: str):
        if model_id in self._models:
            return self._models[model_id]
        self._torch_module()
        source_file = self.paths.source_dir / f"SafeAnes_{model_id}_model.py"
        artifact_dir = getattr(self.paths, f"{model_id.lower()}_artifact_dir")
        if not source_file.is_file():
            raise FileNotFoundError(f"Model wrapper not found: {source_file}")
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Model artifact package not found: {artifact_dir}")
        module_name = f"safeanes_{model_id.lower()}_model_runtime"
        spec = importlib.util.spec_from_file_location(module_name, source_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load model wrapper from {source_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loader = getattr(module, f"load_{model_id.lower()}_model")
        model = loader(artifact_dir, device="cpu")
        self._modules[model_id] = module
        self._models[model_id] = model
        return model

    def _torch_module(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Model inference requires the optional CPU PyTorch package") from exc
            self._torch = torch
        return self._torch

    def _thresholds(self, model_id: str) -> dict[str, float]:
        artifact_dir = getattr(self.paths, f"{model_id.lower()}_artifact_dir")
        candidates = (
            (artifact_dir / "thresholds.json",)
            if model_id == "UC04"
            else (artifact_dir / "calibration_thresholds_metrics.json",)
        )
        for path in candidates:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                key = "5m" if model_id == "UC04" else "threshold_5m"
                value = data.get(key)
                if value is not None:
                    return {"5m": float(value)}
        return {}

    def _calibrate_uc05(self, logits: np.ndarray) -> np.ndarray:
        path = self.paths.uc05_artifact_dir / "calibration_thresholds_metrics.json"
        temperatures = {"1": 1.0, "3": 1.0, "5": 1.0}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            temperatures.update({str(k): float(v) for k, v in data.get("temperature", {}).items()})
        adjusted = np.asarray([
            float(logit) / max(temperatures.get(str(h), 1.0), 1e-6)
            for h, logit in zip((1, 3, 5), logits, strict=True)
        ])
        return _sigmoid(adjusted)

    def _uc04_notes(self) -> tuple[str, ...]:
        path = self.paths.uc04_artifact_dir / "config.json"
        if not path.is_file():
            return ("Research artifact; clinical validity is not established.",)
        config = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"Package provenance: SMOKE_TEST={config.get('SMOKE_TEST')}, EPOCHS={config.get('EPOCHS')}; stream validation disabled.",
            "Research only; not for clinical decisions.",
        )

    def _uc05_notes(self) -> tuple[str, ...]:
        return (
            "The released validation event sensitivity was 1/14; the test event sensitivity was 0/1.",
            "The ventilation-pattern head uses weak signal-derived labels; research only, not a diagnosis or clinical alert.",
        )


def _wave_input(
    timeline: PatientTimeline,
    name: str,
    end_sec: float,
    seconds: float,
    target_hz: float,
    plausible: tuple[float, float] | None,
    min_fraction: float,
    min_samples: int,
    source_rate_hz: float | None = None,
) -> tuple[np.ndarray, bool]:
    signal = timeline.waveforms.get(name)
    output_count = round(seconds * target_hz)
    if signal is None:
        return np.zeros(output_count, dtype=np.float32), False
    source_hz = float(source_rate_hz or signal.sample_rate_hz)
    raw = timeline.waveform_window(name, end_sec, seconds, source_hz)
    raw = _fixed_length(raw, round(seconds * source_hz))
    if abs(source_hz - target_hz) > 1e-6:
        ratio = Fraction(target_hz / source_hz).limit_denominator(1000)
        raw = scipy_signal.resample_poly(raw, ratio.numerator, ratio.denominator).astype(np.float32)
    raw = _fixed_length(raw, output_count)
    if plausible is not None:
        low, high = plausible
        raw[(raw < low) | (raw > high)] = np.nan
    valid = np.isfinite(raw)
    if float(valid.mean()) < min_fraction or int(valid.sum()) < min_samples:
        return np.zeros(output_count, dtype=np.float32), False
    indices = np.arange(len(raw))
    raw[~valid] = np.interp(indices[~valid], indices[valid], raw[valid])
    median = float(np.median(raw))
    q25, q75 = np.percentile(raw, [25, 75])
    scale = max(float(q75 - q25) / 1.349, float(np.std(raw)), 1e-3)
    normalized = np.clip((raw - median) / scale, -8.0, 8.0).astype(np.float32)
    return normalized, True


def _interpolate_matrix(values: np.ndarray, limit: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 2:
        raise ValueError("numeric model history must be a two-dimensional matrix")
    for column in range(result.shape[1]):
        data = result[:, column]
        valid = np.flatnonzero(np.isfinite(data))
        for left, right in pairwise(valid):
            gap = int(right - left - 1)
            if gap <= 0:
                continue
            fill_count = min(gap, limit)
            positions = np.arange(left + 1, left + fill_count + 1)
            ratio = (positions - left) / (right - left)
            data[positions] = data[left] + ratio * (data[right] - data[left])
    return result



def _training_time_bounds(timeline: PatientTimeline) -> tuple[int, int] | None:
    """Return VitalDB anesthesia bounds required by the released candidate-time rules."""

    case_end = max(0, int(timeline.duration_sec))
    try:
        anesthesia_start = int(float(timeline.static_features["anestart_sec"]))
        anesthesia_end = int(float(timeline.static_features["aneend_sec"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(anesthesia_start) or not np.isfinite(anesthesia_end):
        return None
    if anesthesia_end <= anesthesia_start:
        anesthesia_start, anesthesia_end = 0, case_end
    return anesthesia_start, min(case_end, anesthesia_end)


def _training_timing_reason(
    model_id: str,
    timeline: PatientTimeline,
    end_sec: float,
    history_sec: int,
    horizon_sec: int,
) -> str | None:
    bounds = _training_time_bounds(timeline)
    if bounds is None:
        return f"{model_id} requires VitalDB anesthesia timing metadata for its training window."
    anesthesia_start, analysis_end = bounds
    current_sec = int(end_sec)
    if current_sec < max(history_sec, anesthesia_start):
        return f"{model_id} is not eligible before its history window and anesthesia start."
    if current_sec + horizon_sec > analysis_end:
        return f"{model_id} requires {horizon_sec // 60} minutes remaining before anesthesia/case end."
    return None


def _clean_training_numeric(
    timeline: PatientTimeline,
    feature: str,
    current_sec: int,
    plausible: tuple[float, float],
) -> np.ndarray | None:
    signal = timeline.numerics.get(feature)
    if signal is None or signal.status is not SignalStatus.AVAILABLE or signal.values is None:
        return None
    count = current_sec + 1
    values, _ = timeline.numeric_history((feature,), float(count), float(count))
    cleaned = values[:, 0]
    cleaned[(cleaned < plausible[0]) | (cleaned > plausible[1])] = np.nan
    return _interpolate_matrix(cleaned[:, None], limit=5)[:, 0]


def _in_recent_event_recovery(
    cleaned: np.ndarray,
    current_sec: int,
    threshold: float,
    minimum_duration_sec: int,
    inclusive: bool,
) -> bool:
    below = np.isfinite(cleaned) & (
        cleaned <= threshold if inclusive else cleaned < threshold
    )
    padded = np.concatenate(([False], below, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    qualifies = (ends - starts) >= minimum_duration_sec
    return bool(np.any(qualifies & (ends <= current_sec) & (current_sec < ends + 120)))


def _fixed_length(values: np.ndarray, count: int) -> np.ndarray:
    result = np.full(count, np.nan, dtype=np.float32)
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    n = min(count, source.size)
    result[:n] = source[:n]
    return result


def _static_vector(timeline: PatientTimeline, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([timeline.static_features.get(name, np.nan) for name in names], dtype=np.float32)


def _to_tensors(arrays: Mapping[str, np.ndarray], torch: Any) -> dict[str, Any]:
    return {name: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).unsqueeze(0)
            for name, value in arrays.items()}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
