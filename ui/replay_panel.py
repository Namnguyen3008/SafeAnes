"""Real VitalDB replay controls and research-only SafeAnes output display."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.data_sources import PatientTimeline
from core.model_inference import ModelPrediction, SafeAnesInference
from core.vitaldb_source import VitalDBCase, VitalDBSource


class VitalDBLoadWorker(QThread):
    cases_ready = pyqtSignal(object)
    timeline_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, action: str, case_id: int | None = None, parent=None):
        super().__init__(parent)
        self.action = action
        self.case_id = case_id

    def run(self) -> None:
        try:
            source = VitalDBSource()
            if self.action == "cases":
                self.cases_ready.emit(source.available_cases(limit=100))
            elif self.action == "load" and self.case_id is not None:
                self.timeline_ready.emit(source.load_timeline(self.case_id))
            else:
                raise ValueError(f"Unsupported VitalDB action: {self.action}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class InferenceWorker(QThread):
    predictions_ready = pyqtSignal(object, float, int)
    failed = pyqtSignal(str, float, int)

    def __init__(
        self,
        runner: SafeAnesInference,
        timeline: PatientTimeline,
        at_sec: float,
        generation: int,
        parent=None,
    ):
        super().__init__(parent)
        self.runner = runner
        self.timeline = timeline
        self.at_sec = float(at_sec)
        self.generation = int(generation)

    def run(self) -> None:
        try:
            predictions = self.runner.predict_all(self.timeline, self.at_sec)
            self.predictions_ready.emit(predictions, self.at_sec, self.generation)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}", self.at_sec, self.generation)


class RealReplayPanel(QWidget):
    """Choose an adult case, control replay, and inspect model research scores."""

    timeline_loaded = pyqtSignal(object)
    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    seek_requested = pyqtSignal(float)
    speed_requested = pyqtSignal(float)
    synthetic_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("real-replay-panel")
        self._timeline: PatientTimeline | None = None
        self._loader: VitalDBLoadWorker | None = None
        self._playing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        intro = QLabel("Replay recorded, de-identified VitalDB cases on their original case-time axis.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        choose = QVBoxLayout()
        self.case_combo = QComboBox(objectName="replay-case")
        self.case_combo.addItem("1", 1)
        choose.addWidget(self.case_combo)
        choose_buttons = QHBoxLayout()
        self.find_cases_button = QPushButton("Find cases", objectName="find-vitaldb-cases")
        self.load_case_button = QPushButton("Load case", objectName="load-vitaldb-case")
        choose_buttons.addWidget(self.find_cases_button)
        choose_buttons.addWidget(self.load_case_button)
        choose.addLayout(choose_buttons)
        root.addLayout(choose)

        transport = QVBoxLayout()
        transport_controls = QHBoxLayout()
        self.play_button = QPushButton("Play", objectName="replay-play")
        self.play_button.setEnabled(False)
        self.position_slider = QSlider(Qt.Orientation.Horizontal, objectName="replay-position")
        self.position_slider.setRange(0, 1)
        self.position_slider.setEnabled(False)
        self.position_label = QLabel("00:00:00 / 00:00:00", objectName="replay-time")
        self.speed_combo = QComboBox(objectName="replay-speed")
        for speed, label in ((0.5, "0.5×"), (1.0, "1×"), (2.0, "2×"), (4.0, "4×")):
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setEnabled(False)
        transport_controls.addWidget(self.play_button)
        transport_controls.addWidget(self.position_slider, stretch=1)
        transport_controls.addWidget(self.speed_combo)
        transport.addLayout(transport_controls)
        transport.addWidget(self.position_label)
        root.addLayout(transport)

        self.source_status_label = QLabel("Synthetic monitor active. Load a case to switch to real replay.",
                                          objectName="replay-source-status")
        self.source_status_label.setWordWrap(True)
        root.addWidget(self.source_status_label)

        signals_box = QGroupBox("Recorded signal status")
        signals_layout = QVBoxLayout(signals_box)
        self.track_status_label = QLabel("Case signals have not been loaded.", objectName="replay-track-status")
        self.track_status_label.setWordWrap(True)
        signals_layout.addWidget(self.track_status_label)
        root.addWidget(signals_box)

        self.recorded_vitals_label = QLabel("Recorded numerics: —", objectName="recorded-vitals")
        self.recorded_vitals_label.setWordWrap(True)
        root.addWidget(self.recorded_vitals_label)

        results_box = QGroupBox("SafeAnes package outputs")
        results_layout = QVBoxLayout(results_box)
        self.model_results_label = QLabel("Waiting for a complete model input window.",
                                          objectName="safeanes-model-results")
        self.model_results_label.setWordWrap(True)
        results_layout.addWidget(self.model_results_label)
        root.addWidget(results_box)

        self.research_note_label = QLabel(
            "Research only. These prototype scores are not clinical alarms, diagnoses, or treatment advice. "
            "Model package validation is limited.",
            objectName="research-only-note",
        )
        self.research_note_label.setWordWrap(True)
        root.addWidget(self.research_note_label)

        actions = QHBoxLayout()
        self.synthetic_button = QPushButton("Use Synthetic", objectName="use-synthetic")
        actions.addStretch(1)
        actions.addWidget(self.synthetic_button)
        root.addLayout(actions)
        root.addStretch(1)

        self.find_cases_button.clicked.connect(self.find_cases)
        self.load_case_button.clicked.connect(self.load_case)
        self.play_button.clicked.connect(self._toggle_play)
        self.position_slider.sliderMoved.connect(lambda value: self.seek_requested.emit(float(value)))
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.synthetic_button.clicked.connect(self.synthetic_requested.emit)

    def find_cases(self) -> None:
        if self._loader is not None and self._loader.isRunning():
            return
        self.source_status_label.setText("Checking VitalDB for adult cases with mapped UC04/UC05 tracks…")
        self._start_loader(VitalDBLoadWorker("cases", parent=self))

    def load_case(self) -> None:
        if self._loader is not None and self._loader.isRunning():
            return
        case_id = self.case_combo.currentData()
        if not isinstance(case_id, int):
            case_id = None
        if case_id is None or case_id <= 0:
            self.source_status_label.setText("Choose a valid VitalDB case, then load it.")
            return
        self.source_status_label.setText(f"Loading VitalDB case {case_id}…")
        self._start_loader(VitalDBLoadWorker("load", case_id, self))

    def _start_loader(self, worker: VitalDBLoadWorker) -> None:
        self._loader = worker
        worker.cases_ready.connect(self._cases_loaded)
        worker.timeline_ready.connect(self.set_timeline)
        worker.failed.connect(self._load_failed)
        worker.finished.connect(lambda: self._loader_finished(worker))
        self.find_cases_button.setEnabled(False)
        self.load_case_button.setEnabled(False)
        worker.start()

    def _loader_finished(self, worker: VitalDBLoadWorker) -> None:
        if self._loader is worker:
            self._loader = None
        self.find_cases_button.setEnabled(True)
        self.load_case_button.setEnabled(True)

    def _cases_loaded(self, cases: list[VitalDBCase]) -> None:
        self.case_combo.clear()
        for case in cases:
            self.case_combo.addItem(
                f"{case.case_id} · {case.age:.0f}y · {_clock_text(case.duration_sec)}",
                case.case_id,
            )
        if cases:
            self.source_status_label.setText(f"Found {len(cases)} adult cases with mapped core tracks.")
        else:
            self.case_combo.addItem("1", 1)
            self.source_status_label.setText("No eligible cases were returned by the current VitalDB search.")

    def _load_failed(self, message: str) -> None:
        self.source_status_label.setText(f"VitalDB load failed: {message}")

    def set_timeline(self, timeline: PatientTimeline) -> None:
        self._timeline = timeline
        self.position_slider.setRange(0, max(1, int(timeline.duration_sec)))
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.set_playing(False)
        self.source_status_label.setText(
            f"{timeline.source_label} · case {timeline.case_id} · {_clock_text(timeline.duration_sec)} · paused at 00:00:00"
        )
        rows = []
        for label, key in (
            ("ART", "art"),
            ("ECG II", "ecg"),
            ("Pleth", "ppg"),
            ("CO₂", "capno"),
            ("AWP", "awp"),
            ("Flow", "flow"),
            ("Resp", "resp"),
        ):
            signal = timeline.waveforms.get(key)
            if signal is None:
                rows.append(f"{label}: MISSING")
            else:
                track = f" · {signal.track_name}" if signal.track_name else ""
                rows.append(
                    f"{label}: {signal.status.value} · {signal.sample_rate_hz:g} Hz{track}"
                )
        rows.append("Plots auto-scale; out-of-range waveform samples appear as gaps.")
        self.track_status_label.setText("\n".join(rows))
        self.recorded_vitals_label.setText("Recorded numerics: waiting for playback position.")
        self.model_results_label.setText("UC04 requires 5 min and UC05 requires 10 min of case history.")
        self.timeline_loaded.emit(timeline)

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self.play_button.setText("Pause" if self._playing else "Play")

    def set_position(self, position_sec: float, duration_sec: float) -> None:
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(int(position_sec))
        self.position_label.setText(f"{_clock_text(position_sec)} / {_clock_text(duration_sec)}")
        if self._timeline is not None:
            self.source_status_label.setText(
                f"{self._timeline.source_label} · case {self._timeline.case_id} · "
                f"{_clock_text(position_sec)} / {_clock_text(duration_sec)}"
            )

    def set_recorded_vitals(self, values: dict[str, float | None]) -> None:
        units = {"HR": "bpm", "SPO2": "%", "MAP": "mmHg", "SBP": "mmHg",
                 "DBP": "mmHg", "RR": "/min", "ETCO2": "mmHg"}
        parts = []
        for key in ("HR", "SPO2", "SBP", "DBP", "MAP", "RR", "ETCO2"):
            value = values.get(key)
            shown = "—" if value is None or not math.isfinite(float(value)) else f"{float(value):g}"
            parts.append(f"{key} {shown} {units[key]}")
        self.recorded_vitals_label.setText("Recorded values · " + "   |   ".join(parts))

    def set_predictions(self, predictions: tuple[ModelPrediction, ...]) -> None:
        lines = []
        for result in predictions:
            if result.status.value == "READY":
                values = []
                for horizon, score in result.scores.items():
                    threshold = result.thresholds.get(horizon)
                    suffix = f" · package threshold {threshold:.3f}" if threshold is not None else ""
                    values.append(f"{horizon}: {score:.3f}{suffix}")
                content = ", ".join(values)
                lines.append(f"{result.model_id} · {result.score_kind}: {content}")
            else:
                detail = f" — {result.reason}" if result.reason else ""
                lines.append(f"{result.model_id} · {result.status.value}{detail}")
            lines.extend(f"  {note}" for note in result.notes)
        self.model_results_label.setText("\n".join(lines) or "Waiting for a complete model input window.")

    def set_inference_busy(self, busy: bool) -> None:
        if busy:
            self.model_results_label.setText("Running local UC04/UC05 research inference…")

    def stop(self) -> None:
        worker = self._loader
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait()
        self._loader = None

    def _toggle_play(self) -> None:
        if self._playing:
            self.pause_requested.emit()
        else:
            self.play_requested.emit()

    def _speed_changed(self, index: int) -> None:
        speed = self.speed_combo.itemData(index)
        if speed is not None:
            self.speed_requested.emit(float(speed))


def _clock_text(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
