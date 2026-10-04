from __future__ import annotations

import pytest

from webui.backend.service import ReplayNotFound, ReplayService


from core.model_inference import InputStatus, ModelPrediction

def test_replay_clock_uses_case_time_and_speed(source, inference, clock):
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay = service.load_replay(101)
    replay_id = replay["replay_id"]

    service.control(replay_id, action="play")
    clock.advance(5)
    assert service.frame(replay_id, 12)["case_time_sec"] == pytest.approx(5)

    service.control(replay_id, action="speed", speed=2)
    clock.advance(4)
    assert service.frame(replay_id, 12)["case_time_sec"] == pytest.approx(13)


def test_pause_and_seek_keep_monitor_cursor_deterministic(source, inference, clock):
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="play")
    clock.advance(14)
    service.control(replay_id, action="pause")
    clock.advance(20)
    assert service.frame(replay_id, 12)["case_time_sec"] == pytest.approx(14)

    service.control(replay_id, action="seek", position_sec=450)
    assert service.frame(replay_id, 12)["case_time_sec"] == pytest.approx(450)
    assert service.frame(replay_id, 12)["playing"] is False


def test_inference_is_timestamped_and_deduplicated_by_30_second_stride(
    source, inference, clock
):
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="seek", position_sec=600)

    result = service.predict(replay_id, end_sec=600)
    again = service.predict(replay_id, end_sec=600)

    assert result["timestamp_sec"] == 600
    assert [model["status"] for model in result["models"]] == ["READY", "READY"]
    assert result["models"][0]["timestamp_sec"] == 600
    assert again["timestamp_sec"] == 600
    assert len(inference.calls) == 1


def test_inference_history_is_filtered_after_seek_back(source, inference, clock):
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="seek", position_sec=600)
    service.predict(replay_id, end_sec=600)
    service.control(replay_id, action="seek", position_sec=300)

    frame = service.frame(replay_id, 12)

    assert [point["timestamp_sec"] for point in frame["prediction_history"]] == []


def test_unknown_replay_has_a_specific_error(source, inference, clock):
    service = ReplayService(source=source, inference=inference, clock=clock)
    with pytest.raises(ReplayNotFound):
        service.frame("missing", 12)



def test_frame_orders_waveforms_like_the_ecg_simulator(source, inference, clock):
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]

    rows = service.frame(replay_id, 12)["waveforms"]

    assert [row["name"] for row in rows] == ["ecg", "pleth", "resp", "capno", "flow"]


def test_waveform_display_range_is_case_wide_and_stable_after_seek(
    source, inference, clock
):
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="seek", position_sec=10)

    first = service.frame(replay_id, 1)["waveforms"]
    ecg = next(row for row in first if row["name"] == "ecg")
    assert "display_range" in ecg
    first_range = ecg["display_range"]
    finite = [sample for sample in ecg["samples"] if sample is not None]
    assert finite
    assert first_range[0] < min(finite)
    assert first_range[1] > max(finite)

    service.control(replay_id, action="seek", position_sec=30)
    later = service.frame(replay_id, 1)["waveforms"]
    later_ecg = next(row for row in later if row["name"] == "ecg")

    assert later_ecg["display_range"] == first_range



def test_model_missing_input_states_and_reasons_reach_the_replay_response(
    source, inference, clock
):
    inference.predict_all = lambda timeline, end_sec: (
        ModelPrediction("UC04", InputStatus.MISSING_REQUIRED_INPUT, end_sec,
                        reason="Recorded MAP history does not meet the validity gate."),
        ModelPrediction("UC05", InputStatus.NOT_ELIGIBLE, end_sec,
                        reason="Current SpO₂ is below the training candidate threshold."),
    )
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="seek", position_sec=600)

    response = service.predict(replay_id, end_sec=600)

    assert [model["status"] for model in response["models"]] == [
        "MISSING REQUIRED INPUT", "NOT ELIGIBLE"
    ]
    assert [model["reason"] for model in response["models"]] == [
        "Recorded MAP history does not meet the validity gate.",
        "Current SpO₂ is below the training candidate threshold.",
    ]
    assert all(model["scores"] == {} for model in response["models"])


def test_inference_errors_are_returned_without_fabricating_scores(
    source, inference, clock
):
    inference.raise_error = True
    service = ReplayService(source=source, inference=inference, clock=clock)
    replay_id = service.load_replay(101)["replay_id"]
    service.control(replay_id, action="seek", position_sec=600)

    response = service.predict(replay_id, end_sec=600)

    assert [model["status"] for model in response["models"]] == ["ERROR", "ERROR"]
    assert all("test inference failure" in model["reason"] for model in response["models"])
    assert all(model["scores"] == {} for model in response["models"])
