"""Exact preprocessing contracts for the released UC04 and UC05 artifacts."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np

from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal


def _require_builder():
    spec = importlib.util.find_spec("core.model_inference")
    assert spec is not None, "core.model_inference must define the SafeAnes input contract"
    return importlib.import_module("core.model_inference")


def _timeline(duration_sec: int = 1800):
    duration = int(duration_sec)
    waves = {
        "art": TimelineSignal(np.full(duration * 100, 75.0), 100.0, SignalStatus.AVAILABLE),
        "ecg": TimelineSignal(np.full(duration * 250, 0.1), 250.0, SignalStatus.AVAILABLE),
        "ppg": TimelineSignal(np.full(duration * 500, 1.0), 500.0, SignalStatus.AVAILABLE),
        "capno": TimelineSignal(np.full(int(duration * 62.5), 35.0), 62.5, SignalStatus.AVAILABLE),
        "awp": TimelineSignal(np.full(int(duration * 62.5), 8.0), 62.5, SignalStatus.AVAILABLE),
        "flow": TimelineSignal(None, 62.5, SignalStatus.NOT_SUPPORTED),
        "resp": TimelineSignal(None, 62.5, SignalStatus.NOT_SUPPORTED),
    }
    numeric_values = {
        "MAP": 75.0, "SBP": 120.0, "DBP": 65.0, "HR": 75.0, "SPO2": 98.0,
        "RR": 14.0, "RR_CO2": 14.0, "ETCO2": 35.0, "ETCO2_UC05": 35.0,
        "TV": 450.0, "MV": 6.0, "PIP": 18.0, "PEEP": 5.0, "PPLAT": 15.0,
        "MAWP": 10.0, "FIO2": 45.0, "COMPLIANCE": 50.0, "VENT_LEAK": 0.0,
        "SET_FIO2": 45.0, "SET_TV": 450.0, "SET_PIP": 18.0, "SET_RR": 12.0,
        "MAC": 1.0,
    }
    numerics = {
        name: TimelineSignal(np.full(duration, value), 1.0, SignalStatus.AVAILABLE)
        for name, value in numeric_values.items()
    }
    return PatientTimeline(
        case_id=1,
        duration_sec=float(duration),
        source_label="VitalDB Real Replay",
        waveforms=waves,
        numerics=numerics,
        static_features={
            "age": 77.0, "bmi": 26.3, "asa": 2.0, "emop": 0.0,
            "sex_male": 1.0, "weight": 67.5, "height": 160.2,
            "anestart_sec": 0.0, "aneend_sec": float(duration),
        },
    )


def test_uc04_preprocessing_matches_release_tensor_shapes_and_masks():
    module = _require_builder()

    result = module.build_uc04_input(_timeline(), end_sec=300.0)

    assert result.status is module.InputStatus.READY
    assert result.arrays["art"].shape == (3000,)
    assert result.arrays["ecg"].shape == (7500,)
    assert result.arrays["ppg"].shape == (3000,)
    assert result.arrays["capno"].shape == (1875,)
    assert result.arrays["numeric"].shape == (300, 7)
    assert result.arrays["numeric_mask"].shape == (300, 7)
    assert result.arrays["static"].shape == (5,)
    np.testing.assert_array_equal(result.arrays["modality_mask"], [1, 1, 1, 1])


def test_uc05_preprocessing_keeps_optional_missing_modalities_masked():
    module = _require_builder()

    result = module.build_uc05_input(_timeline(), end_sec=600.0)

    assert result.status is module.InputStatus.READY
    assert result.arrays["capno"].shape == (3750,)
    assert result.arrays["awp"].shape == (3750,)
    assert result.arrays["ppg"].shape == (7500,)
    assert result.arrays["flow"].shape == (3750,)
    assert result.arrays["resp"].shape == (3750,)
    assert result.arrays["numeric"].shape == (600, 18)
    assert result.arrays["numeric_mask"].shape == (600, 18)
    assert result.arrays["static"].shape == (7,)
    np.testing.assert_array_equal(result.arrays["wave_mask"], [1, 1, 1, 0, 0])
    assert np.all(result.arrays["flow"] == 0)
    assert np.all(result.arrays["resp"] == 0)


def test_models_wait_until_their_full_training_history_is_available():
    module = _require_builder()
    timeline = _timeline()

    assert module.build_uc04_input(timeline, end_sec=299.0).status is module.InputStatus.WAITING_FOR_HISTORY
    assert module.build_uc05_input(timeline, end_sec=599.0).status is module.InputStatus.WAITING_FOR_HISTORY


def test_uc04_blocks_when_recorded_map_history_does_not_meet_training_quality_gate():
    module = _require_builder()
    timeline = _timeline()
    timeline.numerics["MAP"] = TimelineSignal(
        np.full(600, np.nan), 1.0, SignalStatus.AVAILABLE, track_name="Solar8000/ART_MBP"
    )

    result = module.build_uc04_input(timeline, end_sec=300.0)

    assert result.status is module.InputStatus.MISSING_REQUIRED_INPUT
    assert "MAP" in result.reason



def test_uc04_obeys_anesthesia_window_and_15_minute_forecast_horizon():
    module = _require_builder()
    timeline = _timeline(duration_sec=1800)
    timeline.static_features["anestart_sec"] = 360.0
    timeline.static_features["aneend_sec"] = 1300.0

    before_anesthesia = module.build_uc04_input(timeline, end_sec=300.0)
    too_late_for_horizon = module.build_uc04_input(timeline, end_sec=420.0)

    assert before_anesthesia.status.value == "NOT ELIGIBLE"
    assert too_late_for_horizon.status.value == "NOT ELIGIBLE"


def test_uc04_requires_current_map_above_threshold_and_excludes_recovery():
    module = _require_builder()
    timeline = _timeline(duration_sec=2400)
    timeline.numerics["MAP"].values[1000:1060] = 60.0

    currently_hypotensive = module.build_uc04_input(timeline, end_sec=1000.0)
    recovering = module.build_uc04_input(timeline, end_sec=1179.0)
    recovery_complete = module.build_uc04_input(timeline, end_sec=1180.0)

    assert currently_hypotensive.status.value == "NOT ELIGIBLE"
    assert recovering.status.value == "NOT ELIGIBLE"
    assert recovery_complete.status is module.InputStatus.READY


def test_uc05_requires_normal_current_spo2_and_the_full_forecast_horizon():
    module = _require_builder()
    low_spo2 = _timeline(duration_sec=1800)
    low_spo2.numerics["SPO2"].values[600] = 85.0
    short_case = _timeline(duration_sec=899)

    low_current = module.build_uc05_input(low_spo2, end_sec=600.0)
    insufficient_future = module.build_uc05_input(short_case, end_sec=600.0)

    assert low_current.status.value == "NOT ELIGIBLE"
    assert insufficient_future.status.value == "NOT ELIGIBLE"


def test_uc05_requires_a_valid_current_spo2_sample():
    module = _require_builder()
    timeline = _timeline(duration_sec=1800)
    timeline.numerics["SPO2"].values[600] = np.nan

    result = module.build_uc05_input(timeline, end_sec=600.0)

    assert result.status is module.InputStatus.MISSING_REQUIRED_INPUT
    assert "current" in result.reason.lower()


def test_uc05_excludes_120_seconds_after_hypoxemia_episode():
    module = _require_builder()
    timeline = _timeline(duration_sec=1200)
    timeline.numerics["SPO2"].values[490:550] = 85.0

    recovering = module.build_uc05_input(timeline, end_sec=669.0)
    recovery_complete = module.build_uc05_input(timeline, end_sec=670.0)

    assert recovering.status.value == "NOT ELIGIBLE"
    assert recovery_complete.status is module.InputStatus.READY
