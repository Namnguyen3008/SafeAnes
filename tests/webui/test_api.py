from __future__ import annotations


def test_health_and_case_listing_hide_patient_demographics(client):
    assert client.get("/api/health").json()["status"] == "ok"
    response = client.get("/api/cases?limit=10")
    assert response.status_code == 200
    case = response.json()["cases"][0]
    assert case["case_id"] == 101
    assert case["duration_sec"] == 900
    assert "age" not in case
    assert "sex" not in case


def test_case_load_reports_all_signal_availability_without_exposing_values(client):
    response = client.post("/api/replays", json={"case_id": 101})
    assert response.status_code == 201
    body = response.json()
    states = {signal["name"]: signal["status"] for signal in body["waveforms"]}
    assert states == {
        "ecg": "AVAILABLE",
        "pleth": "AVAILABLE",
        "capno": "MISSING",
        "flow": "NOT SUPPORTED",
        "resp": "INVALID",
    }
    assert "samples" not in body["waveforms"][0]


def test_unknown_case_returns_structured_not_found_error(client):
    response = client.post("/api/replays", json={"case_id": 999})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "CASE_NOT_FOUND"


def test_frame_includes_every_waveform_bounded_samples_and_null_for_nan(
    client, replay_id, clock
):
    client.patch(f"/api/replays/{replay_id}", json={"action": "play"})
    clock.advance(3)
    response = client.get(f"/api/replays/{replay_id}/frame?window_seconds=30")

    assert response.status_code == 200
    body = response.json()
    assert {row["name"] for row in body["waveforms"]} == {
        "ecg", "pleth", "capno", "flow", "resp"
    }
    ecg = next(row for row in body["waveforms"] if row["name"] == "ecg")
    assert len(ecg["samples"]) <= 1500
    assert ecg["samples"][-1] is None
    missing = next(row for row in body["waveforms"] if row["name"] == "capno")
    assert missing["status"] == "MISSING"
    assert all(value is None for value in missing["samples"])
    assert body["numerics"]["HR"]["value"] == 72
    assert body["numerics"]["SBP"]["value"] == 120


def test_play_pause_seek_speed_and_restart_share_one_cursor(client, replay_id, clock):
    assert client.patch(f"/api/replays/{replay_id}", json={"action": "play"}).status_code == 200
    clock.advance(4)
    client.patch(f"/api/replays/{replay_id}", json={"action": "pause"})
    clock.advance(10)
    frame = client.get(f"/api/replays/{replay_id}/frame").json()
    assert frame["case_time_sec"] == 4
    assert frame["playing"] is False

    client.patch(f"/api/replays/{replay_id}", json={"action": "seek", "position_sec": 200})
    client.patch(f"/api/replays/{replay_id}", json={"action": "speed", "speed": 2})
    client.patch(f"/api/replays/{replay_id}", json={"action": "play"})
    clock.advance(3)
    frame = client.get(f"/api/replays/{replay_id}/frame").json()
    assert frame["case_time_sec"] == 206
    assert frame["speed"] == 2

    client.patch(f"/api/replays/{replay_id}", json={"action": "restart"})
    frame = client.get(f"/api/replays/{replay_id}/frame").json()
    assert frame["case_time_sec"] == 0
    assert frame["playing"] is False


def test_replay_controls_validate_bounds_and_unknown_ids(client, replay_id):
    invalid = client.patch(
        f"/api/replays/{replay_id}", json={"action": "speed", "speed": 3}
    )
    assert invalid.status_code == 422
    out_of_range = client.patch(
        f"/api/replays/{replay_id}", json={"action": "seek", "position_sec": 901}
    )
    assert out_of_range.status_code == 422
    assert client.get("/api/replays/not-a-replay/frame").status_code == 404


def test_predictions_preserve_waiting_and_ready_model_outputs(client, replay_id):
    waiting = client.post(
        f"/api/replays/{replay_id}/predictions", json={"case_time_sec": 0}
    )
    assert waiting.status_code == 200
    assert [m["status"] for m in waiting.json()["models"]] == [
        "WAITING FOR HISTORY", "WAITING FOR HISTORY"
    ]

    client.patch(
        f"/api/replays/{replay_id}", json={"action": "seek", "position_sec": 600}
    )
    ready = client.post(
        f"/api/replays/{replay_id}/predictions", json={"case_time_sec": 600}
    )
    assert ready.status_code == 200
    assert ready.json()["timestamp_sec"] == 600
    assert all(m["timestamp_sec"] == 600 for m in ready.json()["models"])
    assert all(m["status"] == "READY" for m in ready.json()["models"])
    assert ready.json()["models"][0]["scores"] == {"5m": 0.23}


def test_delete_expires_replay(client, replay_id):
    assert client.delete(f"/api/replays/{replay_id}").status_code == 204
    assert client.get(f"/api/replays/{replay_id}/frame").status_code == 404
