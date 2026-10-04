from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal
from core.model_inference import InputStatus, ModelPrediction
from core.vitaldb_source import VitalDBCase
from webui.backend.app import create_app


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSource:
    def __init__(self, timeline: PatientTimeline) -> None:
        self.timeline = timeline
        self.loaded: list[int] = []

    def available_cases(self, limit: int = 100) -> list[VitalDBCase]:
        return [VitalDBCase(101, 54, self.timeline.duration_sec, "F")][:limit]

    def load_timeline(self, case_id: int) -> PatientTimeline:
        if case_id != self.timeline.case_id:
            raise ValueError(f"VitalDB case {case_id} is not available")
        self.loaded.append(case_id)
        return self.timeline


class FakeInference:
    def __init__(self) -> None:
        self.calls: list[tuple[PatientTimeline, float]] = []
        self.raise_error = False

    def predict_all(
        self, timeline: PatientTimeline, end_sec: float
    ) -> tuple[ModelPrediction, ModelPrediction]:
        self.calls.append((timeline, end_sec))
        if self.raise_error:
            raise RuntimeError("test inference failure")
        uc04 = (
            ModelPrediction("UC04", InputStatus.WAITING_FOR_HISTORY, end_sec,
                            reason="UC04 requires 300 seconds of numeric history.")
            if end_sec < 300
            else ModelPrediction("UC04", InputStatus.READY, end_sec,
                                 scores={"5m": 0.23}, thresholds={"5m": 0.5},
                                 score_kind="test-only score")
        )
        uc05 = (
            ModelPrediction("UC05", InputStatus.WAITING_FOR_HISTORY, end_sec,
                            reason="UC05 requires 600 seconds of numeric history.")
            if end_sec < 600
            else ModelPrediction("UC05", InputStatus.READY, end_sec,
                                 scores={"event": 0.41}, thresholds={"event": 0.5},
                                 score_kind="test-only score")
        )
        return uc04, uc05


@pytest.fixture
def timeline() -> PatientTimeline:
    ecg = np.arange(3600, dtype=np.float32) / 100
    ecg[11] = np.nan
    numerics = {
        "HR": TimelineSignal(np.full(900, 72, dtype=np.float32), 1, units="bpm"),
        "SPO2": TimelineSignal(np.full(900, 98, dtype=np.float32), 1, units="%"),
        "SBP": TimelineSignal(np.full(900, 120, dtype=np.float32), 1, units="mmHg"),
        "DBP": TimelineSignal(np.full(900, 70, dtype=np.float32), 1, units="mmHg"),
        "MAP": TimelineSignal(np.full(900, 90, dtype=np.float32), 1, units="mmHg"),
        "RR": TimelineSignal(np.full(900, 12, dtype=np.float32), 1, units="/min"),
        "ETCO2": TimelineSignal(np.full(900, 35, dtype=np.float32), 1, units="mmHg"),
        "ETCO2_UC05": TimelineSignal(np.full(900, 36, dtype=np.float32), 1, units="mmHg"),
    }
    return PatientTimeline(
        case_id=101,
        duration_sec=900,
        source_label="VitalDB",
        waveforms={
            "ecg": TimelineSignal(ecg, 4, units="mV", track_name="SNUADC/ECG_II"),
            "pleth": TimelineSignal(np.ones(1800, dtype=np.float32), 2, units="a.u."),
            "capno": TimelineSignal(None, 62.5, SignalStatus.MISSING,
                                    units="mmHg", reason="No matching track."),
            "flow": TimelineSignal(None, 62.5, SignalStatus.NOT_SUPPORTED,
                                    reason="No source track is defined."),
            "resp": TimelineSignal(None, 62.5, SignalStatus.INVALID,
                                    reason="Track contains no valid samples."),
        },
        numerics=numerics,
        static_features={},
    )


@pytest.fixture
def source(timeline: PatientTimeline) -> FakeSource:
    return FakeSource(timeline)


@pytest.fixture
def inference() -> FakeInference:
    return FakeInference()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(source: FakeSource, inference: FakeInference, clock: FakeClock):
    with TestClient(create_app(source=source, inference=inference, clock=clock)) as client:
        yield client


@pytest.fixture
def replay_id(client: TestClient) -> str:
    response = client.post("/api/replays", json={"case_id": 101})
    assert response.status_code == 201
    return response.json()["replay_id"]
