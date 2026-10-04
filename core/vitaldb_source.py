"""VitalDB real-case access and an atomic, local cache for replay timelines."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal


@dataclass(frozen=True)
class TrackSpec:
    sample_rate_hz: float
    track_candidates: tuple[str, ...]
    units: str = ""


UC04_NUMERIC_FEATURES = ("MAP", "SBP", "DBP", "HR", "SPO2", "RR", "ETCO2")
UC05_NUMERIC_FEATURES = (
    "SPO2", "ETCO2", "RR_CO2", "TV", "MV", "PIP", "PEEP", "PPLAT",
    "MAWP", "FIO2", "COMPLIANCE", "VENT_LEAK", "SET_FIO2", "SET_TV",
    "SET_PIP", "SET_RR", "HR", "MAC",
)

UC04_WAVEFORMS = {
    "art": TrackSpec(100.0, ("SNUADC/ART",), "mmHg"),
    "ecg": TrackSpec(250.0, ("SNUADC/ECG_II",), "mV"),
    "ppg": TrackSpec(500.0, ("SNUADC/PLETH",), "a.u."),
    "capno": TrackSpec(62.5, ("Primus/CO2",), "mmHg"),
}

UC05_WAVEFORMS = {
    "capno": TrackSpec(62.5, ("Primus/CO2",), "mmHg"),
    "awp": TrackSpec(62.5, ("Primus/AWP",), "cmH2O"),
    "ppg": TrackSpec(500.0, ("SNUADC/PLETH",), "a.u."),
    # The released training track map does not define VitalDB tracks for these inputs.
    "flow": TrackSpec(62.5, (), ""),
    "resp": TrackSpec(62.5, (), ""),
}

UC04_NUMERIC_TRACKS = {
    "MAP": ("Solar8000/ART_MBP",),
    "SBP": ("Solar8000/ART_SBP",),
    "DBP": ("Solar8000/ART_DBP",),
    "HR": ("Solar8000/HR",),
    "SPO2": ("Solar8000/PLETH_SPO2",),
    "RR": ("Solar8000/RR_CO2",),
    "ETCO2": ("Solar8000/ETCO2",),
}

UC05_NUMERIC_TRACKS = {
    "SPO2": ("Solar8000/PLETH_SPO2",),
    "ETCO2": ("Primus/ETCO2", "Solar8000/ETCO2"),
    "RR_CO2": ("Primus/RR_CO2", "Solar8000/RR_CO2"),
    "TV": ("Primus/TV", "Solar8000/VENT_TV"),
    "MV": ("Primus/MV", "Solar8000/VENT_MV"),
    "PIP": ("Primus/PIP_MBAR", "Solar8000/VENT_PIP"),
    "PEEP": ("Primus/PEEP_MBAR", "Solar8000/VENT_MEAS_PEEP"),
    "PPLAT": ("Primus/PPLAT_MBAR", "Solar8000/VENT_PPLAT"),
    "MAWP": ("Primus/MAWP_MBAR", "Solar8000/VENT_MAWP"),
    "FIO2": ("Primus/FIO2", "Solar8000/FIO2"),
    "COMPLIANCE": ("Primus/COMPLIANCE", "Solar8000/VENT_COMPL"),
    "VENT_LEAK": ("Primus/VENT_LEAK",),
    "SET_FIO2": ("Primus/SET_FIO2", "Solar8000/VENT_SET_FIO2"),
    "SET_TV": ("Primus/SET_TV_L", "Solar8000/VENT_SET_TV"),
    "SET_PIP": ("Primus/SET_PIP", "Primus/SET_INSP_PRES", "Solar8000/VENT_SET_PCP"),
    "SET_RR": ("Primus/SET_RR_IPPV", "Solar8000/VENT_RR"),
    "HR": ("Solar8000/HR",),
    "MAC": ("Primus/MAC",),
}

UC04_NUMERIC_ALIASES = {feature: feature for feature in UC04_NUMERIC_FEATURES}
UC05_NUMERIC_ALIASES = {
    feature: (
        f"{feature}_UC05"
        if feature in UC04_NUMERIC_TRACKS
        and UC04_NUMERIC_TRACKS[feature] != UC05_NUMERIC_TRACKS.get(feature)
        else feature
    )
    for feature in UC05_NUMERIC_FEATURES
}

UC04_STATIC_FEATURES = ("age", "bmi", "asa", "emop", "sex_male")
UC05_STATIC_FEATURES = ("age", "bmi", "asa", "sex_male", "weight", "height", "emop")


@dataclass(frozen=True)
class VitalDBCase:
    case_id: int
    age: float | None
    duration_sec: float
    sex: str = ""


class LocalTimelineCache:
    """Store public case timelines in compressed NPZ files without pickle."""

    def __init__(self, directory: str | Path | None = None):
        if directory is None:
            directory = Path.home() / "AppData" / "Local" / "SafeAnes" / "cache" / "vitaldb"
        self.directory = Path(directory)

    def _path_for(self, case_id: int | str) -> Path:
        key = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:20]
        return self.directory / f"case-{key}.npz"

    def save(self, timeline: PatientTimeline) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {
            "case_id": timeline.case_id,
            "duration_sec": timeline.duration_sec,
            "source_label": timeline.source_label,
            "generation": timeline.generation,
            "static_features": _json_float_map(timeline.static_features),
            "signals": {"waveforms": {}, "numerics": {}},
        }
        for category, signals in (("waveforms", timeline.waveforms), ("numerics", timeline.numerics)):
            for index, (name, signal) in enumerate(signals.items()):
                key = f"{category}_{index}"
                arrays[key] = (
                    np.asarray(signal.values, dtype=np.float32)
                    if signal.values is not None
                    else np.empty(0, dtype=np.float32)
                )
                metadata["signals"][category][name] = {
                    "array_key": key,
                    "has_values": signal.values is not None,
                    "sample_rate_hz": signal.sample_rate_hz,
                    "status": signal.status.value,
                    "track_name": signal.track_name,
                    "units": signal.units,
                    "reason": signal.reason,
                }
        arrays["metadata"] = np.asarray(json.dumps(metadata, allow_nan=False))
        target = self._path_for(timeline.case_id)
        temp = target.with_suffix(".npz.tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp, target)
        return target

    def load(self, case_id: int | str) -> PatientTimeline | None:
        path = self._path_for(case_id)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            signal_groups: dict[str, dict[str, TimelineSignal]] = {"waveforms": {}, "numerics": {}}
            for category, signal_records in metadata["signals"].items():
                for name, record in signal_records.items():
                    values = archive[record["array_key"]].copy() if record["has_values"] else None
                    signal_groups[category][name] = TimelineSignal(
                        values=values,
                        sample_rate_hz=record["sample_rate_hz"],
                        status=SignalStatus(record["status"]),
                        track_name=record["track_name"],
                        units=record["units"],
                        reason=record["reason"],
                    )
        static_features = {
            name: (float(value) if value is not None else math.nan)
            for name, value in metadata["static_features"].items()
        }
        return PatientTimeline(
            case_id=metadata["case_id"],
            duration_sec=metadata["duration_sec"],
            source_label=metadata["source_label"],
            waveforms=signal_groups["waveforms"],
            numerics=signal_groups["numerics"],
            static_features=static_features,
            generation=metadata["generation"],
        )


class VitalDBSource:
    """Load synchronized UC04/UC05 tracks through VitalDB's open-data reader."""

    source_label = "VitalDB Real Replay"

    def __init__(
        self,
        cache: LocalTimelineCache | None = None,
        vitaldb_module: Any | None = None,
        metadata_loader=None,
    ):
        self.cache = cache or LocalTimelineCache()
        self._vitaldb = vitaldb_module
        self._metadata_loader = metadata_loader or self._read_case_metadata

    def available_cases(
        self,
        limit: int = 100,
        minimum_age: float = 18.0,
        minimum_duration_sec: float = 1800.0,
    ) -> list[VitalDBCase]:
        """Find adult cases with the core channels needed for both model inputs."""

        if limit <= 0:
            return []
        client = self._client()
        required = (
            "SNUADC/ART", "SNUADC/ECG_II", "SNUADC/PLETH", "Primus/CO2",
            "Primus/AWP", "Solar8000/ART_MBP", "Solar8000/PLETH_SPO2",
        )
        case_ids = {int(value) for value in client.find_cases(list(required))}
        rows = self._metadata_loader()
        selected: list[VitalDBCase] = []
        for row in rows:
            case_id = _int_or_none(row.get("caseid"))
            if case_id is None or case_id not in case_ids:
                continue
            age = _float_or_none(row.get("age"))
            duration = _float_or_none(row.get("caseend")) or 0.0
            if age is None or age < minimum_age or duration < minimum_duration_sec:
                continue
            selected.append(VitalDBCase(case_id, age, duration, str(row.get("sex") or "")))
        selected.sort(key=lambda case: case.case_id)
        return selected[:limit]

    def load_timeline(self, case_id: int | str, use_cache: bool = True) -> PatientTimeline:
        case_id = int(case_id)
        if case_id <= 0:
            raise ValueError("VitalDB case id must be a positive integer")
        if use_cache:
            cached = self.cache.load(case_id)
            if cached is not None and "ETCO2_UC05" in cached.numerics:
                if "anestart_sec" not in cached.static_features or "aneend_sec" not in cached.static_features:
                    try:
                        rows = self._metadata_loader()
                    except Exception:
                        return cached
                    case_row = next(
                        (row for row in rows if _int_or_none(row.get("caseid")) == case_id), None
                    )
                    if case_row is not None:
                        cached.static_features.update(_static_features(case_row))
                        self.cache.save(cached)
                return cached

        rows = self._metadata_loader()
        case_row = next((row for row in rows if _int_or_none(row.get("caseid")) == case_id), None)
        if case_row is None:
            raise ValueError(f"VitalDB case {case_id} is not present in the public case index")

        client = self._client()
        all_specs = {**UC04_WAVEFORMS, **UC05_WAVEFORMS}
        track_names: set[str] = set()
        for spec in all_specs.values():
            track_names.update(spec.track_candidates)
        numeric_bindings: dict[str, tuple[str, ...]] = {}
        for feature, candidates in UC04_NUMERIC_TRACKS.items():
            numeric_bindings[UC04_NUMERIC_ALIASES[feature]] = candidates
        for feature, candidates in UC05_NUMERIC_TRACKS.items():
            numeric_bindings[UC05_NUMERIC_ALIASES[feature]] = candidates
        for candidates in numeric_bindings.values():
            track_names.update(candidates)

        reader = client.VitalFile(case_id, track_names=sorted(track_names))
        available = set(reader.get_track_names())
        waveform_values: dict[tuple[str, float], np.ndarray] = {}
        for (track_name, rate_hz), names in _group_wave_tracks(all_specs, available).items():
            sampled = _to_samples(reader.to_numpy(names, interval=1.0 / rate_hz), len(names))
            waveform_values[(track_name, rate_hz)] = sampled[:, 0]

        waveforms: dict[str, TimelineSignal] = {}
        for name, spec in all_specs.items():
            selected = next((candidate for candidate in spec.track_candidates if candidate in available), None)
            if not spec.track_candidates:
                waveforms[name] = TimelineSignal(
                    None, spec.sample_rate_hz, SignalStatus.NOT_SUPPORTED, units=spec.units,
                    reason="No source track is defined by the released training map.",
                )
                continue
            if selected is None:
                waveforms[name] = TimelineSignal(
                    None, spec.sample_rate_hz, SignalStatus.MISSING, units=spec.units,
                    reason="No matching track is present in this case.",
                )
                continue
            values = waveform_values.get((selected, spec.sample_rate_hz))
            if values is None or not np.isfinite(values).any():
                waveforms[name] = TimelineSignal(
                    None, spec.sample_rate_hz, SignalStatus.MISSING, track_name=selected,
                    units=spec.units, reason="The track has no finite samples in this case.",
                )
            else:
                waveforms[name] = TimelineSignal(
                    values, spec.sample_rate_hz, SignalStatus.AVAILABLE,
                    track_name=selected, units=spec.units,
                )

        numeric_names = sorted({
            track for candidates in numeric_bindings.values()
            for track in candidates if track in available
        })
        numeric_data: dict[str, np.ndarray] = {}
        if numeric_names:
            sampled = _to_samples(reader.to_numpy(numeric_names, interval=1.0), len(numeric_names))
            numeric_data = {name: sampled[:, index] for index, name in enumerate(numeric_names)}
        numerics: dict[str, TimelineSignal] = {}
        for alias, candidates in numeric_bindings.items():
            selected = next((candidate for candidate in candidates if candidate in numeric_data
                             and np.isfinite(numeric_data[candidate]).any()), None)
            if selected is None:
                numerics[alias] = TimelineSignal(
                    None, 1.0, SignalStatus.MISSING,
                    reason="No mapped numeric track has finite values in this case.",
                )
            else:
                numerics[alias] = TimelineSignal(
                    numeric_data[selected], 1.0, SignalStatus.AVAILABLE,
                    track_name=selected,
                )

        metadata = _static_features(case_row)
        duration = _float_or_none(case_row.get("caseend")) or 0.0
        if duration <= 0:
            raise ValueError(f"VitalDB case {case_id} has no valid duration")
        timeline = PatientTimeline(
            case_id=case_id,
            duration_sec=duration,
            source_label=self.source_label,
            waveforms=waveforms,
            numerics=numerics,
            static_features=metadata,
        )
        self.cache.save(timeline)
        return timeline

    def _client(self):
        if self._vitaldb is None:
            try:
                import vitaldb
            except ImportError as exc:
                raise RuntimeError("VitalDB replay requires the optional 'vitaldb' package") from exc
            self._vitaldb = vitaldb
        return self._vitaldb

    @staticmethod
    def _read_case_metadata() -> list[dict[str, str]]:
        try:
            import vitaldb
        except ImportError as exc:
            raise RuntimeError("VitalDB replay requires the optional 'vitaldb' package") from exc
        base_url = (getattr(vitaldb.api, "API_URL", None) or "https://api.vitaldb.net").rstrip("/")
        req = Request(
            f"{base_url}/cases",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        )
        with urlopen(req, timeout=30) as response:
            body = response.read()
            encoding = getattr(response, "headers", {}).get("Content-Encoding", "").lower()
        if encoding == "gzip" or body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        text = body.decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))


def _group_wave_tracks(
    specs: Mapping[str, TrackSpec],
    available: set[str],
) -> dict[tuple[str, float], list[str]]:
    grouped: dict[tuple[str, float], list[str]] = {}
    for spec in specs.values():
        selected = next((track for track in spec.track_candidates if track in available), None)
        if selected is not None:
            grouped.setdefault((selected, spec.sample_rate_hz), [selected])
    return grouped


def _to_samples(data: Any, expected_channels: int) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        if expected_channels != 1:
            raise ValueError("VitalDB reader returned one dimension for multiple tracks")
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("VitalDB reader must return a two-dimensional time-by-track array")
    if array.shape[1] == expected_channels:
        return array
    if array.shape[0] == expected_channels:
        return array.T
    raise ValueError(f"VitalDB returned {array.shape} for {expected_channels} requested tracks")


def _int_or_none(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _static_features(row: Mapping[str, Any]) -> dict[str, float]:
    features = {}
    for name in ("age", "bmi", "asa", "emop", "weight", "height"):
        value = _float_or_none(row.get(name))
        if value is not None:
            features[name] = value
    sex = str(row.get("sex") or "").strip().lower()
    if sex:
        features["sex_male"] = float(sex in {"m", "male"})

    case_end = _float_or_none(row.get("caseend"))
    if case_end is not None:
        anesthesia_start = _float_or_none(row.get("anestart"))
        anesthesia_end = _float_or_none(row.get("aneend"))
        anesthesia_start = anesthesia_start if anesthesia_start is not None else 0.0
        anesthesia_end = anesthesia_end if anesthesia_end is not None else case_end
        if anesthesia_end <= anesthesia_start:
            anesthesia_start, anesthesia_end = 0.0, case_end
        features["anestart_sec"] = anesthesia_start
        features["aneend_sec"] = anesthesia_end
    return features


def _json_float_map(values: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, value in values.items():
        number = _float_or_none(value)
        result[str(name)] = number
    return result
