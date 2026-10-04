"""Tests for the shared, timestamped patient-data representation."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np


def _require_contract():
    spec = importlib.util.find_spec("core.data_sources")
    assert spec is not None, "core.data_sources must define the timeline contract"
    return importlib.import_module("core.data_sources")


def _timeline(module):
    return module.PatientTimeline(
        case_id=1,
        duration_sec=4.0,
        source_label="VitalDB Real Replay",
        waveforms={
            "ecg": module.TimelineSignal(
                values=np.array([10.0, 11.0, 12.0, 13.0]),
                sample_rate_hz=2.0,
                status=module.SignalStatus.AVAILABLE,
                track_name="SNUADC/ECG_II",
            )
        },
        numerics={
            "HR": module.TimelineSignal(
                values=np.array([60.0, 61.0, 62.0, 63.0]),
                sample_rate_hz=1.0,
                status=module.SignalStatus.AVAILABLE,
                track_name="Solar8000/HR",
            ),
            "ETCO2": module.TimelineSignal(
                values=None,
                sample_rate_hz=1.0,
                status=module.SignalStatus.MISSING,
            ),
        },
        static_features={"age": 50.0, "sex_male": 1.0},
    )


def test_waveform_window_is_case_time_aligned_and_nan_padded_before_case_start():
    module = _require_contract()
    timeline = _timeline(module)

    window = timeline.waveform_window(
        "ecg", end_sec=1.0, seconds=2.0, target_rate_hz=2.0
    )

    np.testing.assert_allclose(window, [np.nan, np.nan, 10.0, 11.0], equal_nan=True)


def test_numeric_history_preserves_feature_order_and_marks_missing_values():
    module = _require_contract()
    timeline = _timeline(module)

    values, mask = timeline.numeric_history(
        ("HR", "ETCO2"), end_sec=3.0, seconds=3.0
    )

    np.testing.assert_allclose(
        values,
        [[60.0, np.nan], [61.0, np.nan], [62.0, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_array_equal(mask, [[1, 0], [1, 0], [1, 0]])


def test_missing_waveform_returns_nan_instead_of_a_flat_fake_signal():
    module = _require_contract()
    timeline = _timeline(module)

    window = timeline.waveform_window(
        "co2", end_sec=2.0, seconds=1.0, target_rate_hz=62.5
    )

    assert window.shape == (63,)
    assert np.isnan(window).all()


def test_timeline_rejects_a_signal_with_an_invalid_sample_rate():
    module = _require_contract()

    try:
        module.TimelineSignal(
            values=np.array([1.0]),
            sample_rate_hz=0.0,
            status=module.SignalStatus.AVAILABLE,
        )
    except ValueError as exc:
        assert "sample rate" in str(exc).lower()
    else:
        raise AssertionError("zero-rate signals must be rejected")
