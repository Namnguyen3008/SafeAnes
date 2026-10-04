"""Main window: header, tabbed control panel, alarm banner and the monitor.

The panel emits intent keyed by *state field name* rather than by widget, so
:meth:`MainWindow.bind` connects the whole thing in one hop.  It owns no
simulation state of its own, which is what lets it be built and tested without
a running generator.

Twice a second the window measures vitals from the buffer, checks them against
the alarm limits, advances any running scenario, and logs whatever changed -
from whichever source changed it: a dropdown, a scenario step, or a shock.

The panel collapses (Ctrl+H, F9, or the rail button) to give the monitor the
full width; the toggle deliberately lives outside the panel so it survives the
panel being hidden, and the alarm banner sits above the monitor for the same
reason.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.alarms import LIMIT_RANGES, AlarmLimits, AlarmTracker, by_parameter, evaluate
from core.buffer import RingBuffer
from core.data_sources import PatientTimeline
from core.model_inference import SafeAnesInference
from core.pathology import (
    RESP_SPECS,
    RHYTHMS,
    SHOCK_KINDS,
    THERAPY_CARDIOVERT,
    THERAPY_DEFIB,
    URGENCY_LETHAL,
    URGENCY_URGENT,
    therapy_for,
)
from core.replay import REPLAY_CHANNELS, ReplaySession, sanitize_recorded_vitals
from core.scenarios import SCENARIOS, ScenarioPlayer
from core.state import (
    BASELINE_RANGE,
    CARDIAC_RHYTHMS,
    DIASTOLIC_RANGE,
    GAUSSIAN_RANGE,
    HR_RANGE,
    MAINS_RANGE,
    RESP_PATTERNS,
    RR_RANGE,
    SPO2_RANGE,
    SYSTOLIC_RANGE,
    StateSnapshot,
)
from core.vitals import VitalsAnalyzer

from . import icons
from .monitor import ECG_COLOR, GAINS, RESP_COLOR, MonitorView
from .replay_panel import InferenceWorker, RealReplayPanel
from .theme import (
    ACCENT,
    DANGER,
    STYLESHEET,
    TEXT,
    TEXT_MUTED,
    WARNING,
    numeric_font,
    ui_font,
)
from .widgets import (
    AlarmBanner,
    Card,
    EventLog,
    LabeledSlider,
    SegmentedControl,
    ToggleSwitch,
)

COLLAPSE_GLYPH = "‹"     # single left angle quote
EXPAND_GLYPH = "›"       # single right angle quote

ALARM_FLASH_MS = 550     # slow enough to read, fast enough to read as an alarm
VITALS_MS = 500          # numerics update twice a second, as on a bedside monitor
ACKNOWLEDGE_S = 60.0     # an acknowledged alarm stops flashing for this long

SHOCK_MESSAGES = {
    "converted": "Shock delivered - rhythm converted to sinus.",
    "persists": "Shock delivered - rhythm unchanged. Resume CPR and recharge.",
    "no_effect": "Shock delivered - no effect. This rhythm is not shockable.",
    "induced_vf": "Shock landed on the T wave - VENTRICULAR FIBRILLATION induced.",
    "not_delivered": "Not delivered - no R wave to synchronise to. Use unsynchronised.",
}
SHOCK_SEVERITY = {"converted": "ok", "persists": "warn", "no_effect": "warn",
                  "induced_vf": "alarm", "not_delivered": "warn"}

# Guidance for the vital-sign alarms that reach the banner (high priority only).
PHYSIOLOGICAL_ADVICE = {
    "asystole": "Start CPR. Check leads and the patient.",
    "hr_unreadable": "Check the patient and the ECG leads.",
    "hr_low": "Assess perfusion; consider atropine or pacing.",
    "hr_high": "Assess stability and treat the cause.",
    "no_pulse": "Check for a pulse; start CPR if absent.",
    "map_low": "Assess perfusion: fluids or vasopressors.",
    "spo2_low": "Check the probe, airway and oxygen supply.",
    "apnoea": "Open the airway and ventilate.",
}


def _stamp(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class ControlPanel(QWidget):
    """Left pane.  Emits intent; owns no simulation state."""

    parameter_changed = pyqtSignal(str, float)     # (state field name, value)
    motion_requested = pyqtSignal()
    pvc_requested = pyqtSignal()
    export_requested = pyqtSignal()
    shock_requested = pyqtSignal(str)              # 'defibrillate' | 'cardiovert'
    alarm_toggled = pyqtSignal(bool)
    vital_alarms_toggled = pyqtSignal(bool)
    cpr_toggled = pyqtSignal(bool)
    gain_changed = pyqtSignal(float)
    grid_toggled = pyqtSignal(bool)
    limits_changed = pyqtSignal(object)            # AlarmLimits
    acknowledge_requested = pyqtSignal()
    scenario_requested = pyqtSignal(str)
    scenario_stop_requested = pyqtSignal()
    clear_log_requested = pyqtSignal()

    def __init__(self, defaults: StateSnapshot | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        defaults = defaults or StateSnapshot()
        self.sliders: dict[str, LabeledSlider] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._patient_tab(defaults), "Patient")
        self.tabs.addTab(self._signal_tab(defaults), "Signal")
        self.tabs.addTab(self._therapy_tab(defaults), "Therapy")
        self.tabs.addTab(self._alarms_tab(defaults), "Alarms")
        self.tabs.addTab(self._scenarios_tab(), "Scenarios")
        self.tabs.addTab(self._log_tab(), "Log")
        tips = ("Rhythm, breathing and baseline vitals",
                "Noise, display and injected events",
                "Defibrillation, cardioversion and CPR",
                "Alarm sources and limits",
                "Scripted clinical scenarios",
                "Timestamped event history")
        for index, tip in enumerate(tips):
            self.tabs.setTabToolTip(index, tip)
        layout.addWidget(self.tabs, stretch=1)

        self.status_label = QLabel("", objectName="status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        for combo in (self.rhythm_combo, self.resp_combo):
            combo.currentTextChanged.connect(self._refresh_summary)
        self._refresh_summary()

    # -- tabs -------------------------------------------------------------------
    @staticmethod
    def _page() -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("panel")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
        return page, layout

    def _patient_tab(self, d: StateSnapshot) -> QWidget:
        page, layout = self._page()

        rhythm = Card("Pathology", "activity")
        rhythm.add(QLabel("ECG rhythm", objectName="controlName"))
        self.rhythm_combo = QComboBox()
        self.rhythm_combo.addItems(CARDIAC_RHYTHMS)
        self.rhythm_combo.setCurrentText(d.cardiac_rhythm)
        self.rhythm_combo.setAccessibleName("ECG rhythm")
        rhythm.add(self.rhythm_combo)
        rhythm.add(QLabel("Breathing pattern", objectName="controlName"))
        self.resp_combo = QComboBox()
        self.resp_combo.addItems(RESP_PATTERNS)
        self.resp_combo.setCurrentText(d.resp_pattern)
        self.resp_combo.setAccessibleName("Breathing pattern")
        rhythm.add(self.resp_combo)
        # What the current selection actually changes.  Reads the catalogue, so
        # a rhythm added to core.pathology documents itself here for free.
        self.summary_label = QLabel(objectName="summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        rhythm.add(self.summary_label)
        layout.addWidget(rhythm)

        vitals = Card("Baseline vitals", "heart")
        self._slider(vitals, "heart_rate", "Heart rate", HR_RANGE, d.heart_rate, 170, "bpm")
        self._slider(vitals, "respiratory_rate", "Respiratory rate", RR_RANGE,
                     d.respiratory_rate, 22, "rpm")
        self._slider(vitals, "spo2_target", "SpO₂", SPO2_RANGE, d.spo2_target, 40, "%")
        self._slider(vitals, "systolic", "Systolic", SYSTOLIC_RANGE, d.systolic, 150, "mmHg")
        self._slider(vitals, "diastolic", "Diastolic", DIASTOLIC_RANGE, d.diastolic, 100, "mmHg")
        hint = QLabel("Pressures are the patient's baseline. Arrhythmias, filling "
                      "time and CPR change what the arterial line actually reads.",
                      objectName="hint")
        hint.setWordWrap(True)
        vitals.add(hint)
        layout.addWidget(vitals)
        layout.addStretch(1)
        return page

    def _signal_tab(self, d: StateSnapshot) -> QWidget:
        page, layout = self._page()

        noise = Card("Interference", "sliders")
        self._slider(noise, "mains_amplitude", "Mains 50 Hz", MAINS_RANGE,
                     d.mains_amplitude, 100, "mV", 2)
        self._slider(noise, "baseline_amplitude", "Baseline wander", BASELINE_RANGE,
                     d.baseline_amplitude, 100, "mV", 2)
        self._slider(noise, "gaussian_sigma", "Sensor noise", GAUSSIAN_RANGE,
                     d.gaussian_sigma, 100, "mV", 3)
        layout.addWidget(noise)

        display = Card("Display", "grid")
        display.add(QLabel("ECG gain", objectName="controlName"))
        self.gain_control = SegmentedControl(list(GAINS), "x1")
        self.gain_control.changed.connect(lambda label: self.gain_changed.emit(GAINS[label]))
        display.add(self.gain_control)
        self.grid_switch = ToggleSwitch("ECG paper grid")
        self.grid_switch.toggled.connect(self.grid_toggled)
        display.add(self.grid_switch)
        layout.addWidget(display)

        events = Card("Inject", "alert")
        self.motion_button = QPushButton("  Motion artifact")
        self.motion_button.setIcon(icons.icon("activity", WARNING, 16))
        self.motion_button.clicked.connect(self.motion_requested)
        self.pvc_button = QPushButton("  PVC")
        self.pvc_button.setAccessibleName("Premature ventricular contraction")
        self.pvc_button.setIcon(icons.icon("heart", DANGER, 16))
        self.pvc_button.setToolTip("One PVC, coupled to the preceding beat")
        self.pvc_button.clicked.connect(self.pvc_requested)
        for button in (self.motion_button, self.pvc_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            events.add(button)
        layout.addWidget(events)

        data = Card("Data", "download")
        self.export_button = QPushButton("  Export buffer to CSV")
        self.export_button.setIcon(icons.icon("download", TEXT, 16))
        self.export_button.setToolTip("Ctrl+E")
        self.export_button.clicked.connect(self.export_requested)
        data.add(self.export_button)
        hint = QLabel("The visible 10 seconds of every channel at 1 kHz.", objectName="hint")
        hint.setWordWrap(True)
        data.add(hint)
        layout.addWidget(data)
        layout.addStretch(1)
        return page

    def _therapy_tab(self, d: StateSnapshot) -> QWidget:
        page, layout = self._page()

        shock = Card("Electrical therapy", "bolt")
        self.defib_button = QPushButton("  Defib · unsync", objectName="danger")
        self.defib_button.setAccessibleName("Defibrillate, unsynchronised")
        self.defib_button.setIcon(icons.icon("bolt", TEXT, 17))
        self.defib_button.setToolTip(
            "Unsynchronised shock - for pulseless VF/VT. "
            "On an organised rhythm this can land on the T wave and cause VF.")
        self.defib_button.clicked.connect(lambda: self.shock_requested.emit(THERAPY_DEFIB))
        self.cardiovert_button = QPushButton("  Cardiovert · sync", objectName="warning")
        self.cardiovert_button.setAccessibleName("Cardiovert, synchronised")
        self.cardiovert_button.setIcon(icons.icon("sync", TEXT, 17))
        self.cardiovert_button.setToolTip(
            "Synchronised shock - timed to the R wave. "
            "Cannot be delivered without an organised QRS.")
        self.cardiovert_button.clicked.connect(
            lambda: self.shock_requested.emit(THERAPY_CARDIOVERT))
        for button in (self.defib_button, self.cardiovert_button):
            button.setMinimumHeight(40)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            shock.add(button)
        self.shock_label = QLabel("No shocks delivered.", objectName="hint")
        self.shock_label.setWordWrap(True)
        shock.add(self.shock_label)
        layout.addWidget(shock)

        cpr = Card("Chest compressions", "compress")
        self.cpr_button = QPushButton("  Start CPR", objectName="cpr")
        self.cpr_button.setIcon(icons.icon("compress", TEXT, 17))
        self.cpr_button.setCheckable(True)
        self.cpr_button.setChecked(d.cpr_active)
        self.cpr_button.setMinimumHeight(40)
        self.cpr_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cpr_button.toggled.connect(self._on_cpr_toggled)
        cpr.add(self.cpr_button)
        hint = QLabel("110 compressions a minute. They generate a pulse on the "
                      "arterial line - and an artifact on the ECG that the rate "
                      "meter counts, which is why rhythm checks pause compressions.",
                      objectName="hint")
        hint.setWordWrap(True)
        cpr.add(hint)
        layout.addWidget(cpr)
        layout.addStretch(1)
        self._sync_cpr_text()
        return page

    def _alarms_tab(self, d: StateSnapshot) -> QWidget:
        page, layout = self._page()

        sources = Card("Alarm sources", "bell")
        self.alarm_check = ToggleSwitch("Rhythm alarms")
        self.alarm_check.setToolTip("Alert when the rhythm calls for electrical therapy.")
        self.alarm_check.setChecked(d.alarm_enabled)
        self.alarm_check.toggled.connect(self.alarm_toggled)
        self.vital_alarm_check = ToggleSwitch("Vital-sign alarms")
        self.vital_alarm_check.setToolTip("Alert when a measured numeric leaves its limits.")
        self.vital_alarm_check.setChecked(True)
        self.vital_alarm_check.toggled.connect(self.vital_alarms_toggled)
        self.acknowledge_button = QPushButton("  Acknowledge", objectName="ghost")
        self.acknowledge_button.setIcon(icons.icon("check", TEXT_MUTED, 16))
        self.acknowledge_button.setToolTip("Stop alarms flashing for 60 seconds")
        self.acknowledge_button.clicked.connect(self.acknowledge_requested)
        for widget in (self.alarm_check, self.vital_alarm_check, self.acknowledge_button):
            sources.add(widget)
        layout.addWidget(sources)

        limits = Card("Limits", "sliders")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        defaults = AlarmLimits()
        self.limit_spins: dict[str, QSpinBox] = {}
        rows = (("HR", "hr_low", "hr_high", "bpm"), ("SpO₂", "spo2_low", None, "%"),
                ("MAP", "map_low", None, "mmHg"), ("RR", "rr_low", "rr_high", "rpm"))
        grid.addWidget(QLabel("Low", objectName="hint"), 0, 1)
        grid.addWidget(QLabel("High", objectName="hint"), 0, 2)
        grid.setColumnStretch(0, 1)
        for row, (name, low, high, unit) in enumerate(rows, start=1):
            # Units go on the row label, not into the spinbox: a suffix widens
            # every box once Qt polishes it, and pushed the panel past its width.
            label = QLabel(f"{name} <span style='color:#64748B'>{unit}</span>",
                           objectName="controlName")
            label.setTextFormat(Qt.TextFormat.RichText)
            grid.addWidget(label, row, 0)
            for column, key in ((1, low), (2, high)):
                if key is None:
                    continue
                spin = QSpinBox()
                spin.setRange(*LIMIT_RANGES[key])
                spin.setValue(getattr(defaults, key))
                spin.setFixedWidth(84)
                spin.setAccessibleName(f"{name} {'low' if column == 1 else 'high'} limit")
                spin.valueChanged.connect(self._emit_limits)
                self.limit_spins[key] = spin
                grid.addWidget(spin, row, column)
        limits.body.addLayout(grid)
        layout.addWidget(limits)

        active = Card("Active alarms", "alert")
        self.active_list = QListWidget()
        self.active_list.setMinimumHeight(90)
        active.add(self.active_list)
        layout.addWidget(active)
        layout.addStretch(1)
        return page

    def _scenarios_tab(self) -> QWidget:
        page, layout = self._page()

        card = Card("Scenarios", "timeline")
        self.scenario_list = QListWidget()
        for name in SCENARIOS:
            self.scenario_list.addItem(name)
        self.scenario_list.setCurrentRow(0)
        self.scenario_list.setMinimumHeight(150)
        self.scenario_list.currentTextChanged.connect(self._describe_scenario)
        card.add(self.scenario_list)
        self.scenario_summary = QLabel(objectName="hint")
        self.scenario_summary.setWordWrap(True)
        card.add(self.scenario_summary)

        self.scenario_button = QPushButton("  Start scenario")
        self.scenario_button.setIcon(icons.icon("play", ACCENT, 16))
        self.scenario_button.setMinimumHeight(38)
        self.scenario_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scenario_button.clicked.connect(self._on_scenario_button)
        card.add(self.scenario_button)

        self.scenario_progress = QProgressBar()
        self.scenario_progress.setRange(0, 1000)
        self.scenario_progress.setValue(0)
        self.scenario_progress.setFixedHeight(8)
        card.add(self.scenario_progress)
        self.scenario_now = QLabel("", objectName="controlName")
        self.scenario_now.setWordWrap(True)
        self.scenario_next = QLabel("", objectName="hint")
        self.scenario_next.setWordWrap(True)
        card.add(self.scenario_now)
        card.add(self.scenario_next)
        layout.addWidget(card)
        layout.addStretch(1)
        self._scenario_running = False
        self._describe_scenario(self.scenario_list.currentItem().text())
        return page

    def _log_tab(self) -> QWidget:
        page, layout = self._page()
        card = Card("Event log", "list")
        self.event_log = EventLog()
        self.event_log.setMinimumHeight(320)
        card.add(self.event_log)
        clear = QPushButton("Clear", objectName="ghost")
        clear.clicked.connect(self.clear_log_requested)
        clear.clicked.connect(self.event_log.clear)
        card.add(clear)
        layout.addWidget(card, stretch=1)
        return page

    # -- construction helpers -----------------------------------------------------
    def _slider(self, card: Card, key: str, label: str, limits, value: float,
                steps: int, unit: str = "", decimals: int = 0) -> None:
        control = LabeledSlider(key, label, limits, value, steps, unit, decimals)
        control.changed.connect(self.parameter_changed)
        self.sliders[key] = control
        card.add(control)

    def _emit_limits(self, *_args) -> None:
        self.limits_changed.emit(self.limits())

    def limits(self) -> AlarmLimits:
        return AlarmLimits(**{key: spin.value() for key, spin in self.limit_spins.items()})

    def _on_cpr_toggled(self, on: bool) -> None:
        self._sync_cpr_text()
        self.cpr_toggled.emit(on)

    def _sync_cpr_text(self) -> None:
        self.cpr_button.setText("  Stop CPR" if self.cpr_button.isChecked() else "  Start CPR")

    # -- scenarios ---------------------------------------------------------------
    def _describe_scenario(self, name: str) -> None:
        scenario = SCENARIOS.get(name)
        if scenario is not None:
            self.scenario_summary.setText(
                f"{scenario.summary}  ({_stamp(scenario.duration)})")

    def _on_scenario_button(self) -> None:
        if self._scenario_running:
            self.scenario_stop_requested.emit()
        else:
            item = self.scenario_list.currentItem()
            if item is not None:
                self.scenario_requested.emit(item.text())

    def show_scenario(self, name: str | None, progress: float = 0.0,
                      now_text: str = "", next_text: str = "") -> None:
        """Reflect the player's state; ``name`` None means nothing is running."""
        self._scenario_running = name is not None
        if name is None:
            self.scenario_button.setText("  Start scenario")
            self.scenario_button.setIcon(icons.icon("play", ACCENT, 16))
        else:
            self.scenario_button.setText("  Stop scenario")
            self.scenario_button.setIcon(icons.icon("stop", DANGER, 16))
        self.scenario_list.setEnabled(name is None)
        self.scenario_progress.setValue(round(progress * 1000))
        self.scenario_now.setText(now_text)
        self.scenario_next.setText(next_text)

    # -- selection summary --------------------------------------------------------
    def describe_selection(self) -> str:
        """Plain-text account of what the two dropdowns currently change."""
        lines = []
        rhythm = self.rhythm_combo.currentText()
        spec = RHYTHMS.get(rhythm)
        if spec is not None:
            lines.append(f"ECG - {rhythm}")
            lines.append(spec.note)
            if spec.rate_bpm is not None:
                lines.append(f"Rate fixed at {spec.rate_bpm:.0f} bpm - "
                             "the Heart Rate slider has no effect.")
            elif spec.scheduler in ("wenckebach", "mobitz2", "complete_block"):
                lines.append("Heart Rate sets the atrial rate; some beats are "
                             "not conducted, so the pulse is slower.")
            else:
                lines.append("Rate follows the Heart Rate slider.")

        breathing = self.resp_combo.currentText()
        rspec = RESP_SPECS.get(breathing)
        if rspec is not None:
            lines.append("")
            lines.append(f"BREATHING - {breathing}")
            lines.append(rspec.note)
            if rspec.rate_bpm is not None:
                lines.append(f"Rate fixed at {rspec.rate_bpm:.0f} brpm - "
                             "the Respiratory Rate slider has no effect.")
            else:
                lines.append("Rate follows the Respiratory Rate slider.")
        return "\n".join(lines)

    def _refresh_summary(self, *_args) -> None:
        rhythm = self.rhythm_combo.currentText()
        breathing = self.resp_combo.currentText()
        spec, rspec = RHYTHMS.get(rhythm), RESP_SPECS.get(breathing)

        blocks = []
        if spec is not None:
            detail = [spec.note]
            if spec.rate_bpm is not None:
                detail.append(f"Rate fixed at <b>{spec.rate_bpm:.0f} bpm</b> &mdash; "
                              "the Heart Rate slider has no effect.")
            elif spec.scheduler in ("wenckebach", "mobitz2", "complete_block"):
                detail.append("Heart Rate sets the <i>atrial</i> rate; dropped "
                              "beats make the pulse slower.")
            else:
                detail.append("Rate follows the Heart Rate slider.")
            blocks.append(
                f"<span style='color:{ECG_COLOR}'><b>ECG &middot; {rhythm}</b></span>"
                f"<br>{'<br>'.join(detail)}")

        if rspec is not None:
            detail = [rspec.note]
            if rspec.rate_bpm is not None:
                detail.append(f"Rate fixed at <b>{rspec.rate_bpm:.0f} brpm</b> &mdash; "
                              "the Respiratory Rate slider has no effect.")
            blocks.append(
                f"<span style='color:{RESP_COLOR}'><b>BREATHING &middot; {breathing}"
                f"</b></span><br>{'<br>'.join(detail)}")

        self.summary_label.setText("<br><br>".join(blocks))

    # -- feedback -----------------------------------------------------------------
    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def sync(self, params: StateSnapshot) -> None:
        """Push state back into the widgets without re-emitting change signals."""
        for key, control in self.sliders.items():
            control.set_value(getattr(params, key))
        for combo, value in ((self.rhythm_combo, params.cardiac_rhythm),
                             (self.resp_combo, params.resp_pattern)):
            blocked = combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(blocked)
        for toggle, value in ((self.alarm_check, params.alarm_enabled),
                              (self.cpr_button, params.cpr_active)):
            blocked = toggle.blockSignals(True)
            toggle.setChecked(value)
            toggle.blockSignals(blocked)
        self._sync_cpr_text()
        # The combos were updated with their signals blocked to avoid writing
        # straight back to state, which also suppresses the summary's own
        # currentTextChanged hook - so refresh it explicitly.
        self._refresh_summary()


class MainWindow(QMainWindow):
    """Header, control panel and monitor; the panel and monitor split 1:3."""

    def __init__(self, defaults: StateSnapshot | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ECG Simulator")
        self.setWindowIcon(icons.icon("heart", ECG_COLOR, 32))
        self.setFont(ui_font())
        self.setStyleSheet(STYLESHEET)
        self.resize(1480, 900)

        header = self._build_header()

        self.controls = ControlPanel(defaults)
        self.controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # The panel is taller than a short window, so it scrolls rather than
        # clipping its own bottom controls.
        self.controls_pane = QScrollArea(objectName="panel")
        self.controls_pane.setWidget(self.controls)
        self.controls_pane.setWidgetResizable(True)
        self.controls_pane.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_pane.setMinimumWidth(356)
        self.controls_pane.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # The toggle lives outside the panel: put it inside and hiding the panel
        # would hide the only way to bring it back.
        self.toggle_button = QPushButton(COLLAPSE_GLYPH, objectName="rail")
        self.toggle_button.setFixedWidth(18)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.toggle_button.setToolTip("Hide the control panel (Ctrl+H)")
        self.toggle_button.clicked.connect(self.toggle_controls)

        self.monitor = MonitorView()
        self.alarm_banner = AlarmBanner()

        display = QWidget(objectName="root")
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(8, 8, 8, 8)
        display_layout.setSpacing(8)
        display_layout.addWidget(self.alarm_banner)
        display_layout.addWidget(self.monitor, stretch=1)

        body = QWidget(objectName="root")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self.controls_pane, stretch=1)
        body_layout.addWidget(self.toggle_button)
        body_layout.addWidget(display, stretch=3)

        root = QWidget(objectName="root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(header)
        root_layout.addWidget(body, stretch=1)
        self.setCentralWidget(root)

        # The rhythm banner flashes on its own timer; vitals, alarms, scenarios
        # and the clock share a slower one.
        self._alarm_timer = QTimer(self)
        self._alarm_timer.timeout.connect(self._refresh_alarm)
        self._alarm_timer.start(ALARM_FLASH_MS)
        self._vitals_timer = QTimer(self)
        self._vitals_timer.timeout.connect(self.refresh_vitals)

        shortcuts = (("Ctrl+H", self.toggle_controls), ("F9", self.toggle_controls),
                     ("Space", self.toggle_freeze), ("Ctrl+E", self.export_buffer),
                     ("Ctrl+G", lambda: self.controls.grid_switch.toggle()))
        for sequence, handler in shortcuts:
            QShortcut(QKeySequence(sequence), self, activated=handler)

        # Display-only controls need no simulation behind them, so they are
        # wired here rather than in bind().
        self.grid_button.toggled.connect(self._set_grid)
        self.controls.grid_toggled.connect(self._set_grid)
        self.controls.gain_changed.connect(self.monitor.set_gain)

        self._state = None
        self._buffer = None
        self._generator = None
        self.analyzer: VitalsAnalyzer | None = None
        self.player: ScenarioPlayer | None = None
        self.limits = AlarmLimits()
        self.vital_alarms_enabled = True
        self.physiological_alarms: list = []
        self.alarm_tracker = AlarmTracker()
        self._acknowledged_until = 0.0
        self._started = time.monotonic()
        self._seen: dict[str, object] = {}
        # Swapped out in tests so the export path can be supplied without a dialog.
        self.choose_export_path = self._ask_export_path
        self._init_real_replay()

    # -- header -----------------------------------------------------------------
    def _init_real_replay(self) -> None:
        self.real_replay_panel = RealReplayPanel()
        self.controls.tabs.setUsesScrollButtons(True)
        self.replay_tab_index = self.controls.tabs.addTab(self.real_replay_panel, "Replay")
        self.controls.tabs.setTabToolTip(
            self.replay_tab_index,
            "Replay de-identified VitalDB cases with UC04/UC05 research outputs",
        )
        self._synthetic_buffer: RingBuffer | None = None
        self._replay_buffer: RingBuffer | None = None
        self._replay_timeline: PatientTimeline | None = None
        self._replay_session: ReplaySession | None = None
        self._replay_generation = 0
        self._last_inference_stride = -1
        self._inference_runner = SafeAnesInference()
        self._inference_workers: list[InferenceWorker] = []

        self._replay_pump_timer = QTimer(self)
        self._replay_pump_timer.setInterval(25)
        self._replay_pump_timer.timeout.connect(self._pump_replay)

        panel = self.real_replay_panel
        panel.timeline_loaded.connect(self._on_replay_loaded)
        panel.play_requested.connect(self._on_replay_play)
        panel.pause_requested.connect(self._on_replay_pause)
        panel.seek_requested.connect(self._on_replay_seek)
        panel.speed_requested.connect(self._on_replay_speed)
        panel.synthetic_requested.connect(self._on_synthetic_requested)

    def _on_replay_loaded(self, timeline: PatientTimeline) -> None:
        if self.player is not None and self.player.running:
            self.stop_scenario()

        self._replay_generation += 1
        self._last_inference_stride = -1
        self._replay_timeline = timeline
        self._replay_buffer = RingBuffer(
            capacity=self.monitor.n_samples,
            channels=REPLAY_CHANNELS,
            sample_rate=self.monitor.sample_rate,
        )
        self._replay_session = ReplaySession(timeline, self._replay_buffer)
        self.monitor.attach(self._replay_buffer)
        self.monitor.set_replay_mode(True)
        self.monitor.set_replay_display_ranges(timeline)
        self.monitor.clear_vitals()
        self.monitor.start()

        for index in range(5):
            self.controls.tabs.setTabEnabled(index, False)
        self.controls.tabs.setTabEnabled(5, True)
        self.controls.tabs.setTabEnabled(self.replay_tab_index, True)
        self.controls.tabs.setCurrentIndex(self.replay_tab_index)

        self.alarm_banner.hide()
        self._alarm_timer.stop()
        self.alarm_tracker = AlarmTracker()
        self.physiological_alarms = []
        self.controls.active_list.clear()
        self.controls.active_list.addItem("Clinical alarms are disabled during replay.")
        self.controls.set_status(f"Real VitalDB replay active · case {timeline.case_id}")
        self.real_replay_panel.set_predictions(())
        self.real_replay_panel.set_playing(False)
        self.real_replay_panel.set_position(0.0, timeline.duration_sec)
        self._replay_pump_timer.start()
        self._vitals_timer.start(VITALS_MS)
        self._refresh_real_replay()

    def _on_synthetic_requested(self) -> None:
        if self._replay_session is None:
            self.real_replay_panel.source_status_label.setText(
                "Synthetic monitor active. Load a case to switch to real replay."
            )
            return

        self._replay_generation += 1
        self._replay_pump_timer.stop()
        self._replay_session.pause()
        self._replay_session = None
        self._replay_timeline = None
        self._replay_buffer = None
        self.real_replay_panel.set_playing(False)
        self.real_replay_panel.set_predictions(())
        self.real_replay_panel.source_status_label.setText(
            "Synthetic monitor active. Load a case to switch to real replay."
        )

        if self._synthetic_buffer is not None:
            self._buffer = self._synthetic_buffer
            self.monitor.attach(self._synthetic_buffer)
        self.monitor.set_replay_mode(False)
        self.monitor.clear_vitals()
        self.alarm_banner.show()
        self._alarm_timer.start(ALARM_FLASH_MS)
        for index in range(5):
            self.controls.tabs.setTabEnabled(index, True)
        self.controls.tabs.setTabEnabled(5, True)
        self.controls.tabs.setCurrentIndex(0)
        self.controls.set_status("Synthetic monitor active.")
        self.alarm_tracker = AlarmTracker()
        self.physiological_alarms = []
        self._vitals_timer.start(VITALS_MS)
        self.refresh_vitals()
        self._refresh_alarm()

    def _on_replay_play(self) -> None:
        if self._replay_session is None:
            return
        self._replay_session.play()
        self.real_replay_panel.set_playing(self._replay_session.playing)
        self._refresh_real_replay()

    def _on_replay_pause(self) -> None:
        if self._replay_session is None:
            return
        self._replay_session.pause()
        self.real_replay_panel.set_playing(False)
        self._refresh_real_replay()

    def _on_replay_seek(self, position_sec: float) -> None:
        if self._replay_session is None:
            return
        self._replay_generation += 1
        self._last_inference_stride = -1
        self._replay_session.seek(position_sec)
        self._refresh_real_replay()

    def _on_replay_speed(self, speed: float) -> None:
        if self._replay_session is None:
            return
        self._replay_session.set_speed(speed)
        self._refresh_real_replay()

    def _pump_replay(self) -> None:
        session = self._replay_session
        if session is None:
            self._replay_pump_timer.stop()
            return
        session.advance()
        if not session.playing:
            self.real_replay_panel.set_playing(False)
            self._refresh_real_replay()

    def _numeric_at(self, name: str, position_sec: float) -> float | None:
        """Return the latest recorded numeric, held for at most five seconds."""
        if self._replay_timeline is None or not math.isfinite(position_sec) or position_sec < 0:
            return None
        signal = self._replay_timeline.numerics.get(name.upper())
        values = None if signal is None else signal.values
        if values is None or len(values) == 0:
            return None

        sample_rate = float(signal.sample_rate_hz)
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            return None
        current_index = min(
            int(math.floor(position_sec * sample_rate)),
            len(values) - 1,
        )
        oldest_index = max(0, current_index - math.ceil(5.0 * sample_rate))
        for index in range(current_index, oldest_index - 1, -1):
            try:
                value = float(values[index])
            except (TypeError, ValueError):
                continue
            age_sec = position_sec - index / sample_rate
            if math.isfinite(value):
                return value if age_sec <= 5.0 + 1e-9 else None
        return None

    def _waveform_at(self, name: str, position_sec: float) -> float | None:
        if self._replay_timeline is None:
            return None
        signal = self._replay_timeline.waveforms.get(name.lower())
        if signal is None:
            return None
        samples = signal.window(position_sec + 1.0, 1.0, 1.0)
        value = float(samples[-1])
        return value if math.isfinite(value) else None

    def _refresh_real_replay(self) -> None:
        session = self._replay_session
        timeline = self._replay_timeline
        if session is None or timeline is None:
            return

        position = session.position_sec
        self.real_replay_panel.set_position(position, timeline.duration_sec)
        values = {
            "HR": self._numeric_at("HR", position),
            "SPO2": self._numeric_at("SPO2", position),
            "SBP": self._numeric_at("SBP", position),
            "DBP": self._numeric_at("DBP", position),
            "MAP": self._numeric_at("MAP", position),
            "RR": self._numeric_at("RR", position),
            "ETCO2": self._numeric_at("ETCO2", position),
        }
        if values["ETCO2"] is None:
            values["ETCO2"] = self._numeric_at("ETCO2_UC05", position)
        values = sanitize_recorded_vitals(values)
        self.real_replay_panel.set_recorded_vitals(values)
        self.monitor.show_recorded_vitals({
            "hr": values["HR"],
            "spo2": values["SPO2"],
            "sbp": values["SBP"],
            "dbp": values["DBP"],
            "map": values["MAP"],
            "rr": values["RR"],
            "etco2": values["ETCO2"],
            "awp": self._waveform_at("awp", position),
        })

        hours = int(position // 3600)
        minutes = int(position % 3600 // 60)
        seconds = int(position % 60)
        self.clock_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self._maybe_run_inference(position)

    def _maybe_run_inference(self, position_sec: float) -> None:
        if self._replay_timeline is None or position_sec < 300.0:
            return
        if self.real_replay_panel.position_slider.isSliderDown():
            return
        stride = int(position_sec // 30.0) * 30
        if stride <= self._last_inference_stride:
            return
        self._last_inference_stride = stride
        worker = InferenceWorker(
            self._inference_runner,
            self._replay_timeline,
            stride,
            self._replay_generation,
            parent=self,
        )
        self._inference_workers.append(worker)
        worker.predictions_ready.connect(self._on_inference_ready)
        worker.failed.connect(self._on_inference_failed)
        worker.finished.connect(lambda current=worker: self._inference_finished(current))
        self.real_replay_panel.set_inference_busy(True)
        worker.start()

    def _on_inference_ready(self, predictions, at_sec: float, generation: int) -> None:
        if generation != self._replay_generation or self._replay_session is None:
            return
        self.real_replay_panel.set_predictions(predictions)

    def _on_inference_failed(self, message: str, at_sec: float, generation: int) -> None:
        if generation != self._replay_generation or self._replay_session is None:
            return
        self.real_replay_panel.model_results_label.setText(
            f"Inference error at case time {at_sec:.0f}s: {message}"
        )

    def _inference_finished(self, worker: InferenceWorker) -> None:
        if worker in self._inference_workers:
            self._inference_workers.remove(worker)
        worker.deleteLater()
        busy = any(
            current.isRunning() and current.generation == self._replay_generation
            for current in self._inference_workers
        )
        self.real_replay_panel.set_inference_busy(busy)

    def _build_header(self) -> QFrame:
        header = QFrame(objectName="header")
        header.setFixedHeight(56)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(10)

        logo = QLabel()
        logo.setPixmap(icons.render("heart", ECG_COLOR, 24))
        layout.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel("ECG Simulator", objectName="brand"))
        titles.addWidget(QLabel("Synthetic patient monitor", objectName="brandSub"))
        layout.addLayout(titles)
        layout.addStretch(1)

        self.scenario_chip = QLabel("", objectName="chip")
        self.scenario_chip.setProperty("state", "scenario")
        self.scenario_chip.setVisible(False)
        layout.addWidget(self.scenario_chip)

        self.live_chip = QLabel("● LIVE", objectName="chip")
        self.live_chip.setProperty("state", "live")
        layout.addWidget(self.live_chip)

        self.clock_label = QLabel("00:00:00", objectName="clock")
        self.clock_label.setFont(numeric_font(15, QFont.Weight.Normal))
        self.clock_label.setToolTip("Time since monitoring started")
        layout.addWidget(self.clock_label)

        self.freeze_button = QPushButton("  Freeze", objectName="ghost")
        self.freeze_button.setIcon(icons.icon("freeze", TEXT_MUTED, 16))
        self.freeze_button.setCheckable(True)
        self.freeze_button.setToolTip("Hold the waveforms (Space). The patient keeps going.")
        self.freeze_button.toggled.connect(self.set_frozen)
        layout.addWidget(self.freeze_button)

        self.grid_button = QPushButton("  Grid", objectName="ghost")
        self.grid_button.setIcon(icons.icon("grid", TEXT_MUTED, 16))
        self.grid_button.setCheckable(True)
        self.grid_button.setToolTip("ECG paper grid (Ctrl+G)")
        layout.addWidget(self.grid_button)
        return header

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # -- control panel visibility ---------------------------------------------
    @property
    def controls_visible(self) -> bool:
        return not self.controls_pane.isHidden()

    def set_controls_visible(self, visible: bool) -> None:
        """Show or collapse the left pane; the monitor takes the freed width."""
        self.controls_pane.setVisible(visible)
        self.toggle_button.setText(COLLAPSE_GLYPH if visible else EXPAND_GLYPH)
        self.toggle_button.setToolTip(
            "Hide the control panel (Ctrl+H)" if visible
            else "Show the control panel (Ctrl+H)")

    def toggle_controls(self) -> None:
        self.set_controls_visible(not self.controls_visible)

    # -- display ----------------------------------------------------------------
    def set_frozen(self, frozen: bool) -> None:
        self.monitor.set_frozen(frozen)
        blocked = self.freeze_button.blockSignals(True)
        self.freeze_button.setChecked(frozen)
        self.freeze_button.blockSignals(blocked)
        self.live_chip.setText("❚❚ FROZEN" if frozen else "● LIVE")
        self.live_chip.setProperty("state", "frozen" if frozen else "live")
        self._restyle(self.live_chip)
        if self._state is not None:
            self.log("Display frozen" if frozen else "Display live", "info")

    def toggle_freeze(self) -> None:
        self.set_frozen(not self.monitor.frozen)

    def _set_grid(self, on: bool) -> None:
        self.monitor.set_grid(on)
        for widget in (self.grid_button, self.controls.grid_switch):
            blocked = widget.blockSignals(True)
            widget.setChecked(on)
            widget.blockSignals(blocked)

    # -- wiring -----------------------------------------------------------------
    def bind(self, state, buffer, generator=None) -> None:
        """Connect the panel to the simulation and start the sweep."""
        self._state = state
        self._buffer = buffer
        self._synthetic_buffer = buffer
        self._generator = generator
        self.analyzer = VitalsAnalyzer(buffer.sample_rate)
        self.player = ScenarioPlayer(state)
        self._started = time.monotonic()

        c = self.controls
        c.parameter_changed.connect(self._on_parameter_changed)
        c.motion_requested.connect(self._on_motion)
        c.pvc_requested.connect(self._on_pvc)
        c.export_requested.connect(self.export_buffer)
        c.shock_requested.connect(self.deliver_shock)
        c.alarm_toggled.connect(lambda on: self._state.update(alarm_enabled=on))
        c.vital_alarms_toggled.connect(self._on_vital_alarms_toggled)
        c.cpr_toggled.connect(lambda on: self._state.update(cpr_active=on))
        c.limits_changed.connect(self._on_limits_changed)
        c.acknowledge_requested.connect(self.acknowledge)
        c.scenario_requested.connect(self.start_scenario)
        c.scenario_stop_requested.connect(self.stop_scenario)
        if generator is not None:
            generator.shock_delivered.connect(self._on_shock_delivered)
        c.rhythm_combo.currentTextChanged.connect(
            lambda text: self._state.update(cardiac_rhythm=text))
        c.resp_combo.currentTextChanged.connect(
            lambda text: self._state.update(resp_pattern=text))
        c.sync(state.snapshot())
        self._remember(state.snapshot())

        self.monitor.attach(buffer)
        self.monitor.start()
        self._vitals_timer.start(VITALS_MS)
        self._refresh_alarm()
        c.set_status(f"Monitoring - {buffer.duration:.0f} s sweep.")
        self.log("Monitoring started", "info")

    def _on_parameter_changed(self, key: str, value: float) -> None:
        # The panel emits state field names, so this stays a single hop no
        # matter how many sliders the panel grows.
        self._state.update(**{key: value})

    def _on_motion(self) -> None:
        self._state.trigger_motion()
        self.log("Motion artifact injected", "neutral")

    def _on_pvc(self) -> None:
        self._state.trigger_pvc()
        self.log("PVC injected", "neutral")

    def _on_vital_alarms_toggled(self, on: bool) -> None:
        self.vital_alarms_enabled = on
        if not on:
            self.alarm_tracker.reset()
        self.log("Vital-sign alarms on" if on else "Vital-sign alarms off", "info")

    def _on_limits_changed(self, limits: AlarmLimits) -> None:
        self.limits = limits

    # -- event log ----------------------------------------------------------------
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def signal_time(self) -> float:
        """Seconds of signal produced so far - the clock alarm persistence uses.

        Equal to wall time while the producer keeps up, but it stays correct if
        it ever falls behind, and it lets tests run faster than real time.
        """
        if self._buffer is None:
            return 0.0
        return self._buffer.total_written / self._buffer.sample_rate

    def log(self, message: str, severity: str = "neutral") -> None:
        self.controls.event_log.add(_stamp(self.elapsed()), message, severity)

    def _remember(self, params: StateSnapshot) -> None:
        self._seen = {"cardiac_rhythm": params.cardiac_rhythm,
                      "resp_pattern": params.resp_pattern,
                      "cpr_active": params.cpr_active}

    def _log_state_changes(self, params: StateSnapshot) -> None:
        """Log what changed since the last tick, whatever changed it."""
        if params.cardiac_rhythm != self._seen.get("cardiac_rhythm"):
            self.log(f"Rhythm: {params.cardiac_rhythm}", "info")
        if params.resp_pattern != self._seen.get("resp_pattern"):
            self.log(f"Breathing: {params.resp_pattern}", "info")
        if params.cpr_active != self._seen.get("cpr_active"):
            self.log("CPR started" if params.cpr_active else "CPR stopped",
                     "warn" if params.cpr_active else "neutral")
        self._remember(params)

    # -- scenarios -----------------------------------------------------------------
    def start_scenario(self, name: str) -> None:
        if self.player is None:
            return
        steps = self.player.start(name, time.monotonic())
        self.log(f"Scenario started: {name}", "info")
        self._after_steps(steps)

    def stop_scenario(self) -> None:
        if self.player is not None and self.player.running:
            self.log(f"Scenario stopped: {self.player.scenario.name}", "neutral")
            self.player.stop()
        self._show_scenario()

    def _after_steps(self, steps) -> None:
        for step in steps:
            self.log(step.note, "warn")
        if steps:
            self.controls.sync(self._state.snapshot())
        self._show_scenario()

    def _show_scenario(self) -> None:
        player = self.player
        if player is None or not player.running:
            self.controls.show_scenario(None)
            self.scenario_chip.setVisible(False)
            return
        now = time.monotonic()
        scenario = player.scenario
        done = [s for s in scenario.steps if s.at <= player.elapsed(now)]
        upcoming = player.next_step
        next_text = (f"Next at {_stamp(upcoming.at)}: {upcoming.note}" if upcoming
                     else "Final state - holding.")
        self.controls.show_scenario(scenario.name, player.progress(now),
                                    f"Now: {done[-1].note}" if done else "", next_text)
        self.scenario_chip.setText(
            f"{scenario.name}   {_stamp(player.elapsed(now))} / {_stamp(scenario.duration)}")
        self.scenario_chip.setVisible(True)

    # -- vitals and alarms -----------------------------------------------------------
    def refresh_vitals(self) -> None:
        """Twice a second: scenario, numerics, vital-sign alarms, log, clock."""
        if self._replay_session is not None:
            self._refresh_real_replay()
            return
        if self._state is None or self.analyzer is None:
            return
        now = time.monotonic()
        if self.player is not None and self.player.running:
            name = self.player.scenario.name
            steps = self.player.tick(now)
            self._after_steps(steps)
            if not self.player.running:
                self.log(f"Scenario complete: {name}", "ok")
                self._show_scenario()

        params = self._state.snapshot()
        self._log_state_changes(params)

        self.analyzer.feed(self._buffer)
        vitals = self.analyzer.measure(params.spo2_target)
        evaluated = evaluate(vitals, self.limits) if self.vital_alarms_enabled else []
        # Debounced: raised only once persistent, cleared only once gone a while.
        alarms, raised, cleared = self.alarm_tracker.update(evaluated, self.signal_time())
        self.physiological_alarms = alarms
        steady = now < self._acknowledged_until
        self.monitor.show_vitals(vitals, by_parameter(alarms), self.limits)
        self.monitor.flash_tiles(steady)
        self.monitor.set_abp_scale_for(params.systolic)

        for alarm in raised:
            self.log(f"Alarm: {alarm.message}", "alarm" if alarm.is_high else "warn")
        for alarm in cleared:
            self.log(f"Cleared: {alarm.message}", "ok")

        active = self.controls.active_list
        active.clear()
        for alarm in alarms:
            active.addItem(("HIGH   " if alarm.is_high else "MEDIUM   ") + alarm.message)
        if not alarms:
            active.addItem("None")

        seconds = int(self.elapsed())
        self.clock_label.setText(f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:"
                                 f"{seconds % 60:02d}")
        self._show_scenario()

    def acknowledge(self) -> None:
        self._acknowledged_until = time.monotonic() + ACKNOWLEDGE_S
        self.log("Alarms acknowledged for 60 s", "info")

    # -- therapy ----------------------------------------------------------------------
    def deliver_shock(self, kind: str) -> None:
        """Queue one shock; the generator applies it on its next chunk."""
        if self._state is None:
            return
        if kind not in SHOCK_KINDS:
            raise ValueError(f"unknown shock kind {kind!r}")
        self._state.trigger_shock(kind)
        self.controls.set_status(
            "Unsynchronised shock charging..." if kind == THERAPY_DEFIB
            else "Synchronising to the R wave...")

    def _on_shock_delivered(self, kind: str, outcome: str, rhythm: str) -> None:
        message = SHOCK_MESSAGES.get(outcome, outcome).format(kind=kind)
        self.controls.set_status(message)
        self.controls.shock_label.setText(message)
        label = "Defibrillation" if kind == THERAPY_DEFIB else "Cardioversion"
        self.log(f"{label}: {message}", SHOCK_SEVERITY.get(outcome, "neutral"))
        self.controls.sync(self._state.snapshot())     # a shock can change the rhythm
        self._refresh_alarm()

    def _refresh_alarm(self) -> None:
        """Update the banner and advance its flash phase.

        A rhythm that needs therapy takes the banner; otherwise the most urgent
        high-priority vital-sign alarm does, so apnoea or a lost pulse is never
        confined to a tile the operator may not be looking at.
        """
        if self._state is None:
            self.alarm_banner.clear_alarm()
            return
        params = self._state.snapshot()
        spec = therapy_for(params.cardiac_rhythm)
        headline = advice = urgency = None
        if params.alarm_enabled and spec.alarm:
            headline, advice, urgency = spec.alarm, spec.advice, spec.urgency
        else:
            urgent = [a for a in self.physiological_alarms if a.is_high]
            if urgent:
                alarm = urgent[0]
                headline, urgency = alarm.message, URGENCY_LETHAL
                advice = PHYSIOLOGICAL_ADVICE.get(alarm.code, "Check the patient.")
        if headline is None:
            self.alarm_banner.clear_alarm()
            return
        if time.monotonic() < self._acknowledged_until:
            urgency = URGENCY_URGENT                   # acknowledged: steady, not flashing
        if self.alarm_banner.is_active and self.alarm_banner.headline == headline:
            self.alarm_banner.flash()
        else:
            self.alarm_banner.show_alarm(headline, advice, urgency)

    # -- export -------------------------------------------------------------------------
    def _ask_export_path(self) -> str:
        default = f"ecg_sim_{datetime.now().astimezone():%Y%m%d_%H%M%S}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export buffer to CSV", default, "CSV files (*.csv);;All files (*)")
        return path

    def export_buffer(self) -> str | None:
        """Dump the currently displayed source buffer. Returns the path or None."""
        buffer = self._replay_buffer if self._replay_session is not None else self._buffer
        if buffer is None:
            return None
        path = self.choose_export_path()
        if not path:
            return None
        try:
            rows = buffer.to_csv(path)
        except OSError as exc:
            self.controls.set_status(f"Export failed: {exc}")
            return None
        self.controls.set_status(f"Exported {rows} samples to {os.path.basename(path)}")
        self.log(f"Exported {os.path.basename(path)}", "ok")
        return path

    # -- lifecycle ------------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Stop timers and join the producer and replay workers before closing."""
        self._vitals_timer.stop()
        self._alarm_timer.stop()
        self._replay_pump_timer.stop()
        self._replay_generation += 1
        self.real_replay_panel.stop()
        for worker in tuple(self._inference_workers):
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait()
        self._inference_workers.clear()
        self.monitor.stop()
        if self._generator is not None:
            self._generator.stop()
        super().closeEvent(event)
