"""Step 4 checks: the sweep cursor, the control wiring, export and shutdown.

The sweep is verified by driving ``refresh()`` directly rather than waiting on
the timer, so the assertions are about *where the gap lands*, not about timing.

Run directly (``python tests/test_sweep.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication                   # noqa: E402

from core.buffer import RingBuffer                          # noqa: E402
from core.generator import SignalGenerator                  # noqa: E402
from core.state import BUFFER_SIZE, SAMPLE_RATE, SimulationState   # noqa: E402
from ui.main_window import MainWindow                       # noqa: E402
from ui.monitor import SWEEP_GAP_SAMPLES, MonitorView, apply_sweep_gap   # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv[:1])


def frame(ecg, resp, pleth=None, abp=None) -> dict:
    """A write for the four-channel buffer; the extra channels get distinct ramps
    so a column mix-up in the export would show."""
    ecg = np.asarray(ecg, dtype=float)
    return {"ecg": ecg, "resp": np.asarray(resp, dtype=float),
            "pleth": ecg * 3 if pleth is None else pleth,
            "abp": ecg * 4 if abp is None else abp}


def filled_buffer(n_written: int) -> RingBuffer:
    """A buffer carrying a known ramp, wound forward to a chosen write index."""
    buffer = RingBuffer()
    step = 500
    for start in range(0, n_written, step):
        count = min(step, n_written - start)
        ramp = np.arange(start, start + count, dtype=float)
        buffer.write(frame(ramp, -ramp))
    return buffer


def nan_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) spans of NaN, for locating the erase bar."""
    mask = np.isnan(x)
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(x.size)
    return list(zip(starts, stops))


# --- the erase bar ----------------------------------------------------------
def test_gap_sits_immediately_ahead_of_the_write_cursor():
    data = np.zeros((2, 1000))
    apply_sweep_gap(data, write_index=400, gap=80)
    for channel in data:
        assert nan_runs(channel) == [(400, 480)]


def test_gap_wraps_around_the_end_of_the_buffer():
    data = np.zeros((2, 1000))
    apply_sweep_gap(data, write_index=960, gap=80)
    for channel in data:
        assert nan_runs(channel) == [(0, 40), (960, 1000)]   # split across the seam
        assert np.isnan(channel).sum() == 80


def test_gap_is_identical_on_every_channel():
    data = np.zeros((2, 1000))
    apply_sweep_gap(data, write_index=970, gap=80)
    assert np.array_equal(np.isnan(data[0]), np.isnan(data[1]))


def test_gap_never_exceeds_the_buffer():
    data = np.zeros((2, 50))
    apply_sweep_gap(data, write_index=10, gap=500)
    assert np.isnan(data).all()


# --- the monitor ------------------------------------------------------------
def test_refresh_blanks_the_cursor_and_keeps_the_rest():
    view = MonitorView()
    buffer = filled_buffer(BUFFER_SIZE)          # exactly one lap: cursor back at 0
    view.attach(buffer)
    view.refresh()

    y = view.ecg_curve.yData
    assert nan_runs(y) == [(0, SWEEP_GAP_SAMPLES)]
    assert np.isfinite(y[SWEEP_GAP_SAMPLES:]).all(), "erased more than the bar"
    assert y[SWEEP_GAP_SAMPLES] == float(SWEEP_GAP_SAMPLES)   # data itself untouched


def test_cursor_advances_but_the_axis_never_moves():
    """The defining property of a sweep: the trace stays put, the gap travels."""
    view = MonitorView()
    buffer = RingBuffer()
    view.attach(buffer)

    positions, axes = [], []
    for _ in range(6):
        buffer.write(frame(np.ones(700), np.ones(700)))
        view.refresh()
        positions.append(nan_runs(view.ecg_curve.yData)[0][0])
        axes.append(view.ecg_curve.xData.copy())

    assert positions == sorted(positions), f"cursor went backwards: {positions}"
    assert positions[-1] - positions[0] == 5 * 700
    for axis in axes[1:]:
        assert np.array_equal(axis, axes[0]), "x-axis moved; this is scrolling, not sweeping"


def test_cursor_wraps_and_starts_overwriting():
    view = MonitorView()
    buffer = filled_buffer(BUFFER_SIZE - 100)
    view.attach(buffer)
    view.refresh()
    before = nan_runs(view.ecg_curve.yData)[0][0]

    buffer.write(frame(np.zeros(300), np.zeros(300)))
    view.refresh()
    after = nan_runs(view.ecg_curve.yData)[0][0]

    assert before == BUFFER_SIZE - 100
    assert after == 200, f"cursor did not wrap to 200, got {after}"


def test_attach_rejects_a_mismatched_buffer():
    view = MonitorView()
    try:
        view.attach(RingBuffer(capacity=500, channels=("ecg", "resp")))
    except ValueError:
        return
    raise AssertionError("a buffer of the wrong length should be refused")


def test_start_requires_a_buffer():
    view = MonitorView()
    try:
        view.start()
    except RuntimeError:
        return
    raise AssertionError("starting the sweep with no buffer should raise")


def test_timer_runs_and_stops():
    view = MonitorView()
    view.attach(RingBuffer())
    assert view.is_running is False
    view.start(fps=30)
    assert view.is_running is True
    view.stop()
    assert view.is_running is False


# --- control wiring ---------------------------------------------------------
def bound_window():
    state = SimulationState()
    buffer = RingBuffer()
    window = MainWindow(state.snapshot())
    window.bind(state, buffer)
    return window, state, buffer


def test_sliders_reach_the_state():
    window, state, _ = bound_window()
    try:
        window.controls.sliders["heart_rate"].slider.setValue(170)      # top of range
        window.controls.sliders["gaussian_sigma"].slider.setValue(50)
        params = state.snapshot()
        assert params.heart_rate == 200.0
        assert abs(params.gaussian_sigma - 0.1) < 1e-9
    finally:
        window.close()


def test_buttons_raise_the_one_shot_flags():
    window, state, _ = bound_window()
    try:
        window.controls.motion_button.click()
        window.controls.pvc_button.click()
        assert state.consume_motion() is True
        assert state.consume_pvc() is True
        assert state.consume_motion() is False      # consumed, not latched
    finally:
        window.close()


def test_bind_syncs_the_panel_to_existing_state():
    state = SimulationState(heart_rate=155, respiratory_rate=22)
    window = MainWindow()                            # built with defaults, not this state
    try:
        window.bind(state, RingBuffer())
        assert window.controls.sliders["heart_rate"].value() == 155.0
        assert window.controls.sliders["respiratory_rate"].value() == 22.0
        assert state.snapshot().heart_rate == 155.0, "sync must not write back to state"
    finally:
        window.close()


def test_slider_change_actually_changes_the_waveform():
    """End to end: move the slider, run the producer, measure the output rate."""
    from scipy.signal import find_peaks

    state = SimulationState()
    buffer = RingBuffer()
    window = MainWindow(state.snapshot())
    window.bind(state, buffer)
    generator = SignalGenerator(state, buffer, rng=np.random.default_rng(5))
    try:
        window.controls.sliders["heart_rate"].slider.setValue(150 - 30)   # 150 bpm
        assert state.snapshot().heart_rate == 150.0

        for _ in range(240):                       # 12 s of signal, generated flat out
            buffer.write(generator.engine.next_chunk(50))

        ecg = buffer.chronological()[0]
        peaks, _ = find_peaks(ecg, height=0.5, distance=int(0.2 * SAMPLE_RATE))
        measured = 60.0 / (np.diff(peaks).mean() / SAMPLE_RATE)
        assert abs(measured - 150) < 5, f"slider said 150 bpm, waveform shows {measured:.0f}"
    finally:
        window.close()


# --- export -----------------------------------------------------------------
def test_export_writes_the_buffer():
    window, _, buffer = bound_window()
    path = os.path.join(tempfile.gettempdir(), "ecg_sim_export_test.csv")
    window.choose_export_path = lambda: path        # stand in for the file dialog
    try:
        for _ in range(20):
            buffer.write(frame(np.arange(500.0), np.arange(500.0) * 2))
        assert window.export_buffer() == path

        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        assert lines[0] == "time_s,ecg,resp,pleth,abp"
        assert len(lines) == BUFFER_SIZE + 1
        assert "Exported 10000 samples" in window.controls.status_label.text()

        last = lines[-1].split(",")
        assert float(last[0]) == (BUFFER_SIZE - 1) / SAMPLE_RATE
        assert float(last[2]) == 2 * float(last[1])     # channels still aligned
        assert float(last[3]) == 3 * float(last[1])
        assert float(last[4]) == 4 * float(last[1])
    finally:
        window.close()
        if os.path.exists(path):
            os.remove(path)


def test_export_cancelled_writes_nothing():
    window, _, _ = bound_window()
    window.choose_export_path = lambda: ""          # user hit Cancel
    try:
        assert window.export_buffer() is None
    finally:
        window.close()


def test_export_reports_failure_instead_of_raising():
    window, _, _ = bound_window()
    window.choose_export_path = lambda: os.path.join(tempfile.gettempdir(), "no_such_dir", "x.csv")
    try:
        assert window.export_buffer() is None
        assert "failed" in window.controls.status_label.text().lower()
    finally:
        window.close()


# --- lifecycle --------------------------------------------------------------
def test_closing_the_window_stops_the_producer():
    state = SimulationState()
    buffer = RingBuffer()
    generator = SignalGenerator(state, buffer)
    window = MainWindow(state.snapshot())
    window.bind(state, buffer, generator)

    generator.start()
    time.sleep(0.3)
    assert generator.isRunning()
    assert window.monitor.is_running

    window.close()
    assert generator.isFinished(), "producer thread outlived the window"
    assert window.monitor.is_running is False


def test_live_sweep_against_the_real_producer():
    """The whole pipeline: thread writes, timer reads, gap tracks the cursor."""
    state = SimulationState(heart_rate=90)
    buffer = RingBuffer()
    generator = SignalGenerator(state, buffer, rng=np.random.default_rng(9))
    window = MainWindow(state.snapshot())
    window.bind(state, buffer, generator)
    generator.start()
    try:
        deadline = time.perf_counter() + 1.2
        seen = []
        while time.perf_counter() < deadline:
            APP.processEvents()
            window.monitor.refresh()
            written = buffer.total_written
            if written:
                seen.append(nan_runs(window.monitor.ecg_curve.yData)[0][0])
            time.sleep(0.03)

        assert len(seen) > 20, "sweep produced almost no frames"
        assert seen[-1] > seen[0], "cursor did not advance against the live producer"

        # Quiesce the producer so the frame and the counter cannot disagree.
        generator.stop()
        window.monitor.refresh()
        written = min(buffer.total_written, BUFFER_SIZE)
        y = window.monitor.ecg_curve.yData

        # Under a minute of runtime the buffer is only part full, so the region
        # ahead of the cursor is legitimately blank - the erase bar lands inside
        # it and must not eat into the samples already drawn.
        assert np.isfinite(y).sum() == written, "the erase bar consumed real samples"
        assert np.isnan(y[written: written + SWEEP_GAP_SAMPLES]).all()
        assert np.nanmax(y[:written]) > 0.5, "no QRS complexes reached the display"
    finally:
        window.close()


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
