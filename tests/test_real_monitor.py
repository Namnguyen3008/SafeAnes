"""Monitor integration tests for additional recorded respiratory traces."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.buffer import RingBuffer
from core.replay import REPLAY_CHANNELS
from ui.monitor import MonitorView


def test_co2_and_awp_rows_are_hidden_for_synthetic_and_draw_real_buffer_data():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    monitor = MonitorView()
    assert monitor.row_widgets["co2"].isHidden()
    assert monitor.row_widgets["awp"].isHidden()

    monitor.set_replay_mode(True)
    assert not monitor.row_widgets["co2"].isHidden()
    assert not monitor.row_widgets["awp"].isHidden()

    buffer = RingBuffer(
        capacity=monitor.n_samples,
        channels=REPLAY_CHANNELS,
        sample_rate=monitor.sample_rate,
    )
    samples = np.arange(monitor.n_samples, dtype=np.float64)
    buffer.write({name: samples + index * 100.0 for index, name in enumerate(REPLAY_CHANNELS)})
    monitor.attach(buffer)
    monitor.refresh()

    _, co2 = monitor.curves["co2"].getData()
    _, awp = monitor.curves["awp"].getData()
    assert np.isfinite(co2).any()
    assert np.isfinite(awp).any()
    assert monitor.labels["co2"].toPlainText().lower().find("co") >= 0

    monitor.set_replay_mode(False)
    assert monitor.row_widgets["co2"].isHidden()
    monitor.close()
    assert app is not None


def test_replay_monitor_autoscales_raw_tracks_and_gaps_outliers():
    from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal

    app = QApplication.instance() or QApplication(sys.argv[:1])
    monitor = MonitorView()
    count = monitor.n_samples
    phase = np.linspace(0.0, 20.0 * np.pi, count, endpoint=False)
    ecg = np.sin(phase)
    ecg[:20] = 5.0
    ecg[20:40] = -5.0
    ppg = 30.0 + 10.0 * np.sin(phase)
    art = np.full(count, 100.0)
    art[: count // 10] = -8.0
    capno = 30.0 * np.maximum(0.0, np.sin(phase))
    awp = 5.0 + 2.0 * np.sin(phase)

    timeline = PatientTimeline(
        case_id=1,
        duration_sec=count / monitor.sample_rate,
        source_label="test",
        waveforms={
            "ecg": TimelineSignal(ecg, monitor.sample_rate, SignalStatus.AVAILABLE, units="mV"),
            "ppg": TimelineSignal(ppg, monitor.sample_rate, SignalStatus.AVAILABLE, units="a.u."),
            "art": TimelineSignal(art, monitor.sample_rate, SignalStatus.AVAILABLE, units="mmHg"),
            "capno": TimelineSignal(capno, monitor.sample_rate, SignalStatus.AVAILABLE, units="mmHg"),
            "awp": TimelineSignal(awp, monitor.sample_rate, SignalStatus.AVAILABLE, units="cmH2O"),
        },
    )
    monitor.set_replay_mode(True)
    monitor.set_replay_display_ranges(timeline)
    monitor.update_traces(ecg=ecg, resp=np.zeros(count), pleth=ppg, abp=art, co2=capno, awp=awp)

    _, ecg_drawn = monitor.curves["ecg"].getData()
    assert np.isnan(ecg_drawn[:40]).all()
    assert np.isfinite(ecg_drawn[40:]).any()

    _, ppg_drawn = monitor.curves["pleth"].getData()
    assert np.isfinite(ppg_drawn).any()
    assert np.nanmax(ppg_drawn) > 1.6
    ppg_low, ppg_high = monitor.pleth_plot.getPlotItem().viewRange()[1]
    assert ppg_low > 0.0 and ppg_high > 1.6

    _, art_drawn = monitor.curves["abp"].getData()
    art_decimation = monitor.decimation["abp"]
    assert np.isnan(art_drawn[: (count // 10) // art_decimation]).all()
    assert np.nanmin(art_drawn) >= 0.0

    abp_range = monitor.abp_plot.getPlotItem().viewRange()[1]
    monitor.set_abp_scale_for(150.0)
    assert monitor.abp_plot.getPlotItem().viewRange()[1] == abp_range

    monitor.set_replay_mode(False)
    assert monitor._replay_display_ranges == {}
    assert np.allclose(monitor.pleth_plot.getPlotItem().viewRange()[1], (-0.3, 1.6))
    monitor.close()
    assert app is not None


def test_recorded_vital_tiles_explain_when_no_recent_sample_exists():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    monitor = MonitorView()
    monitor.show_recorded_vitals({
        "hr": None,
        "spo2": None,
        "sbp": None,
        "dbp": None,
        "map": None,
        "rr": None,
        "etco2": None,
        "awp": None,
    })

    assert monitor.hr_tile.value_text == "---"
    assert monitor.hr_tile.sub_label.text() == "No recent sample"
    assert monitor.abp_tile.sub_label.text() == "No valid reading"
    assert monitor.rr_tile.sub_label.text() == "No recent sample"
    assert monitor.co2_tile.sub_label.text() == "No recent sample"

    monitor.show_recorded_vitals({
        "hr": 90.0,
        "spo2": 97.0,
        "sbp": 150.0,
        "dbp": 66.0,
        "map": 103.0,
        "rr": 12.0,
        "etco2": 34.0,
        "awp": 5.0,
    })
    assert monitor.hr_tile.sub_label.text() == "Recorded"
    assert monitor.abp_tile.sub_label.text() == "MAP 103"
    assert monitor.rr_tile.sub_label.text() == "Recorded"
    assert monitor.co2_tile.sub_label.text() == "Recorded"

    monitor.close()
    assert app is not None
