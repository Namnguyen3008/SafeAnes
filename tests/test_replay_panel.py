"""UI contract tests for the research-only real replay controls."""

from __future__ import annotations

import importlib
import importlib.util
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal
from core.model_inference import InputStatus, ModelPrediction


def _require_panel():
    spec = importlib.util.find_spec("ui.replay_panel")
    assert spec is not None, "ui.replay_panel must provide the real replay controls"
    return importlib.import_module("ui.replay_panel")


def _app():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_panel_exposes_missing_tracks_raw_values_and_nonclinical_model_scores():
    module = _require_panel()
    app = _app()
    panel = module.RealReplayPanel()
    timeline = PatientTimeline(
        case_id=1,
        duration_sec=1200.0,
        source_label="VitalDB Real Replay",
        waveforms={
            "ecg": TimelineSignal(np.arange(10), 250.0, SignalStatus.AVAILABLE,
                                  track_name="SNUADC/ECG_II"),
            "art": TimelineSignal(np.arange(10), 100.0, SignalStatus.AVAILABLE,
                                  track_name="SNUADC/ART"),
            "ppg": TimelineSignal(np.arange(10), 500.0, SignalStatus.AVAILABLE,
                                  track_name="SNUADC/PLETH"),
            "capno": TimelineSignal(np.arange(10), 62.5, SignalStatus.AVAILABLE,
                                    track_name="Primus/CO2"),
            "awp": TimelineSignal(np.arange(10), 62.5, SignalStatus.AVAILABLE,
                                  track_name="Primus/AWP"),
            "flow": TimelineSignal(None, 62.5, SignalStatus.NOT_SUPPORTED),
            "resp": TimelineSignal(None, 62.5, SignalStatus.NOT_SUPPORTED),
        },
    )
    panel.set_timeline(timeline)
    panel.set_recorded_vitals({"HR": 88, "SPO2": 97, "MAP": 75, "SBP": 121,
                               "DBP": 64, "RR": 14, "ETCO2": 35})
    panel.set_predictions((
        ModelPrediction("UC04", InputStatus.READY, 1020.0,
                        scores={"5m": 0.41}, thresholds={"5m": 0.5},
                        score_kind="raw model score (uncalibrated)"),
        ModelPrediction("UC05", InputStatus.READY, 1020.0,
                        scores={"5m": 0.15}, thresholds={"5m": 0.05},
                        score_kind="temperature-scaled model score"),
    ))

    assert "VitalDB Real Replay" in panel.source_status_label.text()
    assert "NOT SUPPORTED" in panel.track_status_label.text()
    assert "MAP 75" in panel.recorded_vitals_label.text()
    assert "UC04" in panel.model_results_label.text()
    assert "UC05" in panel.model_results_label.text()
    assert "Research only" in panel.research_note_label.text()

    panel.stop()
    panel.close()
    assert app is not None


def test_numeric_lookup_holds_recent_samples_across_missing_seconds():
    _require_panel()
    app = _app()
    from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal
    from ui.main_window import MainWindow

    window = MainWindow()
    signal = TimelineSignal(
        [88.0, float("nan"), 90.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
        1.0,
        SignalStatus.AVAILABLE,
    )
    window._replay_timeline = PatientTimeline(
        case_id=1,
        duration_sec=8.0,
        source_label="test",
        numerics={"HR": signal},
    )

    assert window._numeric_at("HR", 1.5) == 88.0
    assert window._numeric_at("HR", 2.0) == 90.0
    assert window._numeric_at("HR", 7.0) == 90.0
    assert window._numeric_at("HR", 7.1) is None

    window.close()
    assert app is not None
