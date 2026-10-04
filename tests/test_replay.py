"""Case-time playback tests independent of Qt and wall-clock scheduling."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np

from core.buffer import RingBuffer
from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal


def _require_replay():
    spec = importlib.util.find_spec("core.replay")
    assert spec is not None, "core.replay must define the real replay contract"
    return importlib.import_module("core.replay")


def _timeline():
    return PatientTimeline(
        case_id=9,
        duration_sec=8.0,
        source_label="VitalDB Real Replay",
        waveforms={
            "ecg": TimelineSignal(np.arange(16) * 2, 2.0, SignalStatus.AVAILABLE),
            "art": TimelineSignal(np.arange(16) + 100, 2.0, SignalStatus.AVAILABLE),
            "ppg": TimelineSignal(np.arange(16) + 200, 2.0, SignalStatus.AVAILABLE),
            "capno": TimelineSignal(np.arange(16) + 300, 2.0, SignalStatus.AVAILABLE),
            "awp": TimelineSignal(np.arange(16) + 400, 2.0, SignalStatus.AVAILABLE),
            "resp": TimelineSignal(None, 62.5, SignalStatus.NOT_SUPPORTED),
        },
    )


def test_replay_emits_all_modalities_on_one_shared_case_time_axis():
    replay = _require_replay()
    buffer = RingBuffer(capacity=8, channels=replay.REPLAY_CHANNELS, sample_rate=4)
    session = replay.ReplaySession(_timeline(), buffer)

    block = session.sample_block(start_index=0, count=4)

    np.testing.assert_allclose(block[buffer.channel_index("ecg")], [0, 1, 2, 3])
    np.testing.assert_allclose(block[buffer.channel_index("abp")], [100, 100.5, 101, 101.5])
    np.testing.assert_allclose(block[buffer.channel_index("pleth")], [200, 200.5, 201, 201.5])
    np.testing.assert_allclose(block[buffer.channel_index("co2")], [300, 300.5, 301, 301.5])
    np.testing.assert_allclose(block[buffer.channel_index("awp")], [400, 400.5, 401, 401.5])
    assert np.isnan(block[buffer.channel_index("resp")]).all()


def test_replay_clock_supports_pause_seek_speed_and_end_of_case():
    replay = _require_replay()
    buffer = RingBuffer(capacity=8, channels=replay.REPLAY_CHANNELS, sample_rate=4)
    session = replay.ReplaySession(_timeline(), buffer)

    session.set_speed(2.0, now=100.0)
    session.play(now=100.0)
    session.advance(now=101.0)
    assert session.position_sec == 2.0

    session.pause(now=101.5)
    assert session.position_sec == 3.0
    session.advance(now=102.0)
    assert session.position_sec == 3.0

    session.seek(6.0, now=103.0)
    assert session.position_sec == 6.0
    session.play(now=103.0)
    session.advance(now=104.0)
    assert session.position_sec == 8.0
    assert not session.playing


def test_sanitize_recorded_vitals_hides_inconsistent_arterial_pressures():
    replay = _require_replay()
    sanitize = getattr(replay, "sanitize_recorded_vitals")

    artifact = sanitize({
        "HR": 90.0, "SPO2": 97.0, "SBP": 8.0, "DBP": -12.0, "MAP": -9.0,
        "RR": None, "ETCO2": None,
    })
    assert artifact["HR"] == 90.0
    assert artifact["SPO2"] == 97.0
    assert artifact["SBP"] is None
    assert artifact["DBP"] is None
    assert artifact["MAP"] is None

    valid = sanitize({"SBP": 150.0, "DBP": 66.0, "MAP": 103.0})
    assert valid == {"SBP": 150.0, "DBP": 66.0, "MAP": 103.0}

    bad_map = sanitize({"SBP": 150.0, "DBP": 66.0, "MAP": -9.0})
    assert bad_map == {"SBP": 150.0, "DBP": 66.0, "MAP": None}
