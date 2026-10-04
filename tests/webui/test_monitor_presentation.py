from __future__ import annotations

import numpy as np

from core.data_sources import TimelineSignal
from webui.backend.service import ReplayService


def test_default_frame_matches_ten_second_sweep_and_preserves_ecg_detail(
    source, inference, clock
):
    source.timeline.waveforms["ecg"] = TimelineSignal(
        np.arange(300 * 900, dtype=np.float32), 300, units="mV"
    )
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="seek", position_sec=30)

    ecg = next(row for row in service.frame(replay_id)["waveforms"] if row["name"] == "ecg")

    assert ecg["display_rate_hz"] == 250
    assert len(ecg["samples"]) == 2500


def test_frame_api_defaults_to_ten_second_sweep(client, replay_id):
    client.patch(
        f"/api/replays/{replay_id}",
        json={"action": "seek", "position_sec": 30},
    )

    frame = client.get(f"/api/replays/{replay_id}/frame").json()
    ecg = next(row for row in frame["waveforms"] if row["name"] == "ecg")

    assert ecg["display_rate_hz"] == 4
    assert len(ecg["samples"]) == 40
