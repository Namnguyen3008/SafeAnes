"""UI checks: the widgets exist, are styled as specified, and report the right
values.  A screenshot shows it *looks* right; these show it *behaves* right -
sliders in engineering units, a 1:3 split, a panel that collapses without
taking its own toggle with it, and a summary that stays in step with the
catalogue.

Run directly (``python tests/test_ui.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QShortcut, QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication                  # noqa: E402

from core.pathology import RESP_SPECS, RHYTHMS            # noqa: E402
from core.state import (                                   # noqa: E402
    BUFFER_SIZE,
    CARDIAC_RHYTHMS,
    GAUSSIAN_RANGE,
    HR_RANGE,
    RESP_PATTERNS,
    RR_RANGE,
    SimulationState,
    StateSnapshot,
)
from ui.main_window import (                              # noqa: E402
    COLLAPSE_GLYPH,
    EXPAND_GLYPH,
    ControlPanel,
    LabeledSlider,
    MainWindow,
)
from ui.monitor import ECG_COLOR, ECG_RANGE, RESP_COLOR, RESP_RANGE, MonitorView  # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv[:1])


def collect(signal) -> list:
    """Record everything a signal emits, for assertions after the fact."""
    seen = []
    signal.connect(lambda *args: seen.append(args if len(args) != 1 else args[0]))
    return seen


# --- slider value mapping ---------------------------------------------------
def test_slider_spans_its_full_range():
    s = LabeledSlider("heart_rate", "Heart Rate", HR_RANGE, 72, steps=170, unit="bpm")
    s.slider.setValue(0)
    assert s.value() == 30.0
    s.slider.setValue(170)
    assert s.value() == 200.0
    s.slider.setValue(85)
    assert s.value() == 115.0                       # midpoint of 30..200


def test_slider_reports_engineering_units_not_steps():
    s = LabeledSlider("gaussian_sigma", "Gaussian", GAUSSIAN_RANGE, 0.0,
                      steps=100, unit="mV", decimals=3)
    emitted = collect(s.changed)
    s.slider.setValue(50)
    assert emitted == [("gaussian_sigma", 0.1)]     # not (…, 50)
    assert s.value_label.text() == "0.100 mV"


def test_set_value_syncs_without_emitting():
    """Step 4 pushes state into the panel; that must not loop back as a change."""
    s = LabeledSlider("respiratory_rate", "Resp", RR_RANGE, 15, steps=22, unit="brpm")
    emitted = collect(s.changed)
    s.set_value(24)
    assert emitted == []
    assert s.value() == 24.0
    assert s.value_label.text() == "24 brpm"


def test_slider_round_trips_every_state_field():
    panel = ControlPanel(StateSnapshot())
    target = StateSnapshot(heart_rate=143, respiratory_rate=27,
                           mains_amplitude=0.31, baseline_amplitude=0.12,
                           gaussian_sigma=0.077)
    panel.sync(target)
    for key, control in panel.sliders.items():
        expected = getattr(target, key)
        assert abs(control.value() - expected) < 0.01, f"{key}: {control.value()} != {expected}"


# --- control panel signals --------------------------------------------------
def test_panel_forwards_parameter_changes_keyed_by_state_field():
    panel = ControlPanel()
    emitted = collect(panel.parameter_changed)
    panel.sliders["heart_rate"].slider.setValue(0)
    panel.sliders["mains_amplitude"].slider.setValue(100)
    assert emitted == [("heart_rate", 30.0), ("mains_amplitude", 0.5)]

    # The keys must be real state fields, or Step 4's wiring raises at runtime.
    state = SimulationState()
    for key, value in emitted:
        state.update(**{key: value})


def test_buttons_emit_their_intents():
    panel = ControlPanel()
    motion, pvc, export = (collect(panel.motion_requested),
                           collect(panel.pvc_requested),
                           collect(panel.export_requested))
    panel.motion_button.click()
    panel.pvc_button.click()
    panel.export_button.click()
    assert len(motion) == 1 and len(pvc) == 1 and len(export) == 1


def test_panel_opens_at_the_state_defaults():
    panel = ControlPanel(StateSnapshot())
    assert panel.sliders["heart_rate"].value() == 72.0
    assert panel.sliders["respiratory_rate"].value() == 15.0
    assert panel.sliders["gaussian_sigma"].value() == 0.0


# --- monitor styling --------------------------------------------------------
def test_monitor_is_configured_as_a_monitor_not_a_plot():
    view = MonitorView()
    for plot in (view.ecg_plot, view.resp_plot):
        item = plot.getPlotItem()
        vb = item.getViewBox()
        assert vb.state["mouseEnabled"] == [False, False], "panning/zooming is enabled"
        assert item.ctrl.xGridCheck.isChecked() is False
        assert item.ctrl.yGridCheck.isChecked() is False
        assert vb.state["autoRange"] == [False, False], "autorange would rubber-band"
        assert plot.backgroundBrush().color().name() == "#000000"

    assert view.ecg_plot.getPlotItem().getViewBox().viewRange()[0] == [0.0, 10.0]
    assert tuple(view.ecg_plot.getPlotItem().getViewBox().viewRange()[1]) == ECG_RANGE
    assert tuple(view.resp_plot.getPlotItem().getViewBox().viewRange()[1]) == RESP_RANGE


def test_pens_match_the_specified_colours_and_width():
    view = MonitorView()
    for curve, colour in ((view.ecg_curve, ECG_COLOR), (view.resp_curve, RESP_COLOR)):
        pen = curve.opts["pen"]
        assert pen.color().name().upper() == colour.upper()
        assert pen.width() == 2


def test_traces_accept_a_full_buffer_and_keep_nan_gaps():
    view = MonitorView()
    ecg = np.zeros(BUFFER_SIZE)
    resp = np.zeros(BUFFER_SIZE)
    ecg[4000:4080] = np.nan                        # the Step 4 sweep gap
    view.update_traces(ecg, resp)

    assert view.ecg_curve.yData.size == BUFFER_SIZE
    assert np.isnan(view.ecg_curve.yData[4040]), "NaN gap was dropped or filled"
    assert view.ecg_curve.xData[-1] == (BUFFER_SIZE - 1) / view.sample_rate


# --- window layout ----------------------------------------------------------
def test_window_splits_one_to_three():
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    APP.processEvents()
    try:
        panel_w = window.controls_pane.width()
        monitor_w = window.monitor.width()
        ratio = monitor_w / panel_w
        assert abs(ratio - 3.0) < 0.15, f"panes are {panel_w}:{monitor_w} (ratio {ratio:.2f})"
    finally:
        window.close()


def test_ecg_gets_more_height_than_respiration():
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    APP.processEvents()
    try:
        assert window.monitor.ecg_plot.height() > window.monitor.resp_plot.height()
    finally:
        window.close()


# --- collapsible control panel ----------------------------------------------
def test_panel_collapses_and_gives_the_width_to_the_monitor():
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    APP.processEvents()
    try:
        assert window.controls_visible is True
        wide_open = window.monitor.width()

        window.toggle_controls()
        APP.processEvents()
        assert window.controls_visible is False
        assert window.controls_pane.isHidden()
        assert window.monitor.width() > wide_open, "monitor did not take the space"

        window.toggle_controls()
        APP.processEvents()
        assert window.controls_visible is True
        assert window.monitor.width() == wide_open, "layout did not restore"
    finally:
        window.close()


def test_toggle_stays_reachable_while_the_panel_is_hidden():
    """The only way back must not be inside the thing being hidden."""
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    APP.processEvents()
    try:
        window.set_controls_visible(False)
        APP.processEvents()
        assert window.toggle_button.isVisible(), "no way to bring the panel back"
        assert window.toggle_button.width() > 0
        assert window.toggle_button.text() == EXPAND_GLYPH

        window.toggle_button.click()
        APP.processEvents()
        assert window.controls_visible is True
        assert window.toggle_button.text() == COLLAPSE_GLYPH
    finally:
        window.close()


def test_panel_has_keyboard_shortcuts():
    window = MainWindow()
    try:
        bound = {s.key().toString() for s in window.findChildren(QShortcut)}
        assert "Ctrl+H" in bound and "F9" in bound, f"shortcuts bound: {bound}"
    finally:
        window.close()


def test_panel_scrolls_rather_than_clipping_on_a_short_window():
    window = MainWindow()
    window.resize(1200, 420)          # deliberately too short for the full panel
    window.show()
    APP.processEvents()
    try:
        assert window.controls.height() >= window.controls_pane.viewport().height()
        assert window.controls_pane.verticalScrollBar().maximum() > 0, (
            "panel is taller than the window but cannot be scrolled")
    finally:
        window.close()


def test_replay_waveform_rows_scroll_to_the_bottom():
    window = MainWindow()
    window.resize(1480, 900)
    window.monitor.set_replay_mode(True)
    window.show()
    APP.processEvents()
    try:
        scroll = window.monitor.waveform_scroll
        bar = scroll.verticalScrollBar()
        assert bar.maximum() > 0, "replay rows are clipped without a vertical scroll range"
        bar.setValue(bar.maximum())
        APP.processEvents()
        awp = window.monitor.row_widgets["awp"]
        awp_top = awp.mapTo(scroll.viewport(), awp.rect().topLeft()).y()
        assert 0 <= awp_top < scroll.viewport().height(), "the AWP row cannot be brought into view"

        bar.setValue(0)
        plot_view = window.monitor.plots["ecg"].viewport()
        point = QPointF(plot_view.rect().center())
        wheel = QWheelEvent(
            point,
            QPointF(plot_view.mapToGlobal(point.toPoint())),
            QPoint(0, 0),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(plot_view, wheel)
        APP.processEvents()
        assert bar.value() > 0, "mouse wheel over a waveform did not scroll the replay rows"
    finally:
        window.close()


def test_no_tab_is_wider_than_the_panel():
    """Qt polishes a tab's widgets the first time it is shown, which can widen
    them - so this visits every tab before measuring, as a user would."""
    window = MainWindow()
    window.resize(1480, 900)
    window.show()
    APP.processEvents()
    try:
        tabs = window.controls.tabs
        viewport = window.controls_pane.viewport().width()
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            APP.processEvents()
            needed = window.controls.minimumSizeHint().width()
            assert needed <= viewport, (
                f"'{tabs.tabText(index)}' needs {needed}px in a {viewport}px panel")
        assert window.controls.width() <= viewport, "panel content is clipped"
    finally:
        window.close()


# --- pathology summary ------------------------------------------------------
def test_summary_describes_the_current_selection():
    panel = ControlPanel()
    panel.rhythm_combo.setCurrentText("Atrial Flutter")
    panel.resp_combo.setCurrentText("Cheyne-Stokes")

    text = panel.describe_selection()
    assert "Atrial Flutter" in text and RHYTHMS["Atrial Flutter"].note in text
    assert "Cheyne-Stokes" in text and RESP_SPECS["Cheyne-Stokes"].note in text
    # ...and it is actually rendered, not just computed.
    assert "Atrial Flutter" in panel.summary_label.text()


def test_summary_warns_when_a_slider_is_overridden():
    """The most confusing thing about these rhythms is a dead slider."""
    panel = ControlPanel()
    panel.rhythm_combo.setCurrentText("V-Tach")
    text = panel.describe_selection()
    assert "160" in text and "no effect" in text, text

    panel.rhythm_combo.setCurrentText("Sinus")
    assert "no effect" not in panel.describe_selection().split("BREATHING")[0]

    panel.resp_combo.setCurrentText("Kussmaul")
    assert "28" in panel.describe_selection()


def test_summary_covers_every_catalogue_entry():
    panel = ControlPanel()
    for rhythm in CARDIAC_RHYTHMS:
        panel.rhythm_combo.setCurrentText(rhythm)
        for pattern in RESP_PATTERNS:
            panel.resp_combo.setCurrentText(pattern)
            text = panel.describe_selection()
            assert rhythm in text and pattern in text
            assert len(panel.summary_label.text()) > 40, f"{rhythm}/{pattern} empty"


def test_summary_follows_a_rhythm_changed_from_outside_the_panel():
    """A shock converts the rhythm without touching the combo; the summary
    must still follow, or it silently describes the wrong patient."""
    panel = ControlPanel(StateSnapshot(cardiac_rhythm="V-Fib"))
    assert "V-Fib" in panel.summary_label.text()

    panel.sync(StateSnapshot(cardiac_rhythm="Sinus"))
    assert panel.rhythm_combo.currentText() == "Sinus"
    assert "V-Fib" not in panel.summary_label.text(), "summary is stale"
    assert "Sinus" in panel.summary_label.text()


def test_summary_updates_when_the_dropdown_changes():
    panel = ControlPanel()
    before = panel.summary_label.text()
    panel.rhythm_combo.setCurrentText("Asystole")
    assert panel.summary_label.text() != before
    assert "Asystole" in panel.summary_label.text()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
