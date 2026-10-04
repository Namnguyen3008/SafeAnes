"""The monitor's waveforms, each paired with its numeric.

The display scrolls vertically.  The x-axis remains nailed to 0..10 s and the trace
stays where it was drawn; a block of ``NaN`` samples is blanked just ahead of
the buffer's write cursor, so what moves across the screen is the *gap*.  New
samples appear at its trailing edge and ten-second-old samples vanish at its
leading edge - the erase bar of a hospital monitor.

Each row pairs a waveform with its numeric tile, the layout every bedside
monitor uses: the number you read sits beside the trace it came from.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from core.alarms import Alarm, AlarmLimits
from core.buffer import RingBuffer
from core.state import BUFFER_SECONDS, SAMPLE_RATE
from core.vitals import Vitals

from .theme import (  # noqa: F401  (colours re-exported for callers and tests)
    ABP_COLOR,
    BORDER,
    ECG_COLOR,
    INFO,
    PLETH_COLOR,
    RESP_COLOR,
    TEXT_FAINT,
    TRACE_BG,
)
from .widgets import VitalTile

BACKGROUND = TRACE_BG
AXIS = "#1E2A3C"
AXIS_TEXT = "#4B5B72"

# Fixed display windows.  A real monitor clips rather than rescaling: an
# autoranging trace would rubber-band on every artifact and make the rhythm
# impossible to read.
ECG_RANGE = (-1.2, 2.2)     # mV at gain x1; a 1.8 mV PVC fits whole, motion flat-tops
RESP_RANGE = (-1.5, 1.5)    # arbitrary units
PLETH_RANGE = (-0.3, 1.6)   # normalised
ABP_SCALES = (120, 160, 200, 240)   # mmHg; the smallest that clears the target is used

SWEEP_FPS = 30
SWEEP_GAP_SAMPLES = 80      # erase bar width, 80 ms at 1 kHz
PEN_WIDTH = 2

GAINS = {"x0.5": 0.5, "x1": 1.0, "x2": 2.0}

# Channel order top to bottom, with how much height each row gets and how far
# its trace is decimated for display.  The ECG keeps every sample - it has the
# sharp edges - while the slow channels lose nothing visible at 4x or 8x and
# the frame budget gets the savings.
ROWS = (
    ("ecg", 3, 1),
    ("pleth", 2, 4),
    ("abp", 2, 2),
    ("resp", 2, 8),
    ("co2", 2, 4),
    ("awp", 2, 4),
)


def trace_pen(colour: str, width: int = PEN_WIDTH) -> pg.mkPen:
    """A width-2 pen that does not cost 60 ms a frame to draw.

    Qt strokes a normal wide pen by building an outline polygon around every
    segment - on a 10,000-point trace that alone blows the 33 ms frame budget,
    and the monitor visibly stalls once the buffer fills.  A *cosmetic* pen is
    stroked in device space by the rasteriser instead, which is ~15x cheaper
    and looks identical here because the trace is never scaled.
    """
    pen = pg.mkPen(colour, width=width)
    pen.setCosmetic(True)
    return pen


def apply_sweep_gap(data: np.ndarray, write_index: int, gap: int = SWEEP_GAP_SAMPLES):
    """Blank ``gap`` samples starting at the write cursor, wrapping at the end.

    Operates on the last axis, so it takes the whole ``(channels, n)`` block at
    once and keeps every channel's gap in the same place.  Mutates in place -
    callers pass the copy that :meth:`RingBuffer.snapshot` already handed them.
    """
    n = data.shape[-1]
    gap = min(int(gap), n)
    end = write_index + gap
    if end <= n:
        data[..., write_index:end] = np.nan
    else:                                   # the bar is straddling the wrap point
        data[..., write_index:] = np.nan
        data[..., : end - n] = np.nan
    return data


def _paper_grid(duration: float) -> tuple[pg.PlotCurveItem, pg.PlotCurveItem]:
    """ECG paper: 40 ms / 0.1 mV small squares, 200 ms / 0.5 mV large ones."""
    def lines(step_x: float, step_y: float):
        xs, ys = [], []
        for x in np.arange(0.0, duration + 1e-9, step_x):
            xs += [x, x]
            ys += [-6.0, 6.0]
        for y in np.arange(-6.0, 6.0 + 1e-9, step_y):
            xs += [0.0, duration]
            ys += [y, y]
        return np.array(xs), np.array(ys)

    minor_pen = pg.mkPen(255, 70, 70, 26)
    minor_pen.setCosmetic(True)
    major_pen = pg.mkPen(255, 70, 70, 64)
    major_pen.setCosmetic(True)
    minor = pg.PlotCurveItem(*lines(0.04, 0.1), pen=minor_pen, connect="pairs")
    major = pg.PlotCurveItem(*lines(0.2, 0.5), pen=major_pen, connect="pairs")
    return minor, major


class MonitorView(QWidget):
    """Stacked ECG, pleth, arterial pressure and respiration, with numerics."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = BUFFER_SECONDS,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_samples = int(self.sample_rate * self.duration)

        # One shared x-axis; only the y data ever changes, which is what makes
        # the sweep cheap to redraw.
        self._x = np.arange(self.n_samples, dtype=np.float64) / self.sample_rate

        # Antialiasing measured to make no difference to frame time here, but
        # on a 10 k-point trace it buys nothing visible either.
        pg.setConfigOptions(antialias=False)

        self.gain = 1.0
        self.frozen = False
        self.abp_scale = ABP_SCALES[1]
        self.decimation: dict[str, int] = {}
        self.plots: dict[str, pg.PlotWidget] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.labels: dict[str, pg.TextItem] = {}
        self.row_widgets: dict[str, QWidget] = {}

        colours = {"ecg": ECG_COLOR, "pleth": PLETH_COLOR, "abp": ABP_COLOR,
                   "resp": RESP_COLOR, "co2": INFO, "awp": "#D4A94B"}
        ranges = {"ecg": ECG_RANGE, "pleth": PLETH_RANGE,
                  "abp": (0.0, float(self.abp_scale)), "resp": RESP_RANGE,
                  "co2": (0.0, 100.0), "awp": (-20.0, 120.0)}
        titles = {"ecg": "II", "pleth": "Pleth", "abp": "ART", "resp": "RESP",
                  "co2": "CO₂", "awp": "AWP"}

        self.hr_tile = VitalTile("HR", "bpm", ECG_COLOR, "heart", value_px=72)
        self.spo2_tile = VitalTile("SpO₂", "%", PLETH_COLOR, "droplet", value_px=60)
        self.abp_tile = VitalTile("ART", "mmHg", ABP_COLOR, "gauge", value_px=42)
        self.rr_tile = VitalTile("RR", "rpm", RESP_COLOR, "lungs", value_px=60)
        self.co2_tile = VitalTile("CO₂", "mmHg", INFO, "lungs", value_px=60)
        self.awp_tile = VitalTile("AWP", "cmH₂O", "#D4A94B", "gauge", value_px=60)
        self.tiles = {"ecg": self.hr_tile, "pleth": self.spo2_tile,
                      "abp": self.abp_tile, "resp": self.rr_tile,
                      "co2": self.co2_tile, "awp": self.awp_tile}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.waveform_scroll = QScrollArea(self)
        self.waveform_scroll.setObjectName("monitorWaveforms")
        self.waveform_scroll.setWidgetResizable(True)
        self.waveform_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.waveform_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.waveform_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.rows_content = QWidget()
        rows_layout = QVBoxLayout(self.rows_content)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(6)

        for name, stretch, decimate in ROWS:
            plot = self._make_plot(ranges[name])
            plot.viewport().installEventFilter(self)
            curve = plot.plot(self._x[::decimate], np.full(self._x[::decimate].size, np.nan),
                              pen=trace_pen(colours[name]))
            label = pg.TextItem(anchor=(0, 0))
            label.setHtml(self._label_html(titles[name], colours[name], name))
            plot.addItem(label, ignoreBounds=True)
            self.plots[name], self.curves[name], self.labels[name] = plot, curve, label
            self.decimation[name] = decimate

            row_widget = QWidget(self.rows_content)
            row_widget.setMinimumHeight(170)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(plot, stretch=1)
            row.addWidget(self.tiles[name])
            rows_layout.addWidget(row_widget, stretch=stretch)
            self.row_widgets[name] = row_widget
            if name in ("co2", "awp"):
                row_widget.hide()

        self.waveform_scroll.setWidget(self.rows_content)
        layout.addWidget(self.waveform_scroll, stretch=1)

        self.ecg_plot, self.resp_plot = self.plots["ecg"], self.plots["resp"]
        self.pleth_plot, self.abp_plot = self.plots["pleth"], self.plots["abp"]
        self.ecg_curve, self.resp_curve = self.curves["ecg"], self.curves["resp"]
        self.pleth_curve, self.abp_curve = self.curves["pleth"], self.curves["abp"]

        self._grid = _paper_grid(self.duration)
        for item in self._grid:
            item.setZValue(-10)
            item.setVisible(False)
            self.ecg_plot.addItem(item, ignoreBounds=True)
        self.frozen_label = pg.TextItem(anchor=(1, 0))
        self.frozen_label.setHtml(f"<span style='color:{INFO}; font-size:11pt;"
                                  " font-weight:600;'>FROZEN</span>")
        self.frozen_label.setVisible(False)
        self.ecg_plot.addItem(self.frozen_label, ignoreBounds=True)
        self._place_labels()

        # Consumer side of the producer/consumer split: the timer samples
        # whatever the generator thread has written, at its own rate.
        self._buffer: RingBuffer | None = None
        self._indices: dict[str, int] = {}
        self.gap_samples = SWEEP_GAP_SAMPLES

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.refresh)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            delta_y = event.pixelDelta().y()
            if not delta_y:
                pixels_per_notch = max(36, self.fontMetrics().height() * 3)
                delta_y = event.angleDelta().y() * pixels_per_notch / 120
            if delta_y:
                scrollbar = self.waveform_scroll.verticalScrollBar()
                scrollbar.setValue(scrollbar.value() - round(delta_y))
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _label_html(self, title: str, colour: str, name: str) -> str:
        extra = ""
        if name == "ecg":
            extra = f"&nbsp;&nbsp;<span style='color:{TEXT_FAINT}'>{self._gain_text()}</span>"
        elif name == "abp":
            replay_bounds = getattr(self, "_replay_display_ranges", {}).get("abp")
            if getattr(self, "_replay_mode", False) and replay_bounds is not None:
                low, high = replay_bounds
                extra = f"&nbsp;&nbsp;<span style='color:{TEXT_FAINT}'>{low:.0f}–{high:.0f} mmHg</span>"
            else:
                extra = f"&nbsp;&nbsp;<span style='color:{TEXT_FAINT}'>0–{self.abp_scale}</span>"
        return (f"<span style='color:{colour}; font-size:10pt; font-weight:600;'>{title}</span>"
                f"<span style='font-size:9pt;'>{extra}</span>")

    def _gain_text(self) -> str:
        return next((k for k, v in GAINS.items() if v == self.gain), f"x{self.gain:g}")

    def _make_plot(self, y_range) -> pg.PlotWidget:
        widget = pg.PlotWidget()
        widget.setBackground(BACKGROUND)
        widget.setMinimumHeight(60)
        item = widget.getPlotItem()
        item.showGrid(x=False, y=False)
        item.setMouseEnabled(x=False, y=False)      # a monitor is not pannable
        item.setMenuEnabled(False)
        item.hideButtons()
        item.disableAutoRange()                     # fixed window; clip, do not rescale
        item.setXRange(0.0, self.duration, padding=0.0)
        item.setYRange(*y_range, padding=0.0)
        item.hideAxis("bottom")
        axis = item.getAxis("left")
        axis.setWidth(36)
        axis.setPen(AXIS)
        axis.setTextPen(AXIS_TEXT)
        axis.setStyle(tickLength=-4)
        item.getViewBox().setBorder(None)
        return widget

    TICK_STEPS: ClassVar[dict[str, float]] = {"ecg": 1.0, "pleth": 0.5, "abp": 50.0, "resp": 1.0, "co2": 20.0, "awp": 20.0}

    def _apply_ticks(self, name: str) -> None:
        """Round-number ticks, kept clear of the plot edges where labels clip."""
        plot = self.plots[name].getPlotItem()
        (_, _), (low, high) = plot.getViewBox().viewRange()
        step = self.TICK_STEPS[name]
        if name == "ecg" and self.gain > 1.0:
            step = 0.5
        if name == "abp" and self.abp_scale <= 120:
            step = 40.0
        margin = 0.04 * (high - low)
        first = np.ceil((low + margin) / step) * step
        values = np.arange(first, high - margin + 1e-9, step)
        # "+ 0.0" folds IEEE negative zero into zero, which would print as "-0".
        ticks = [(float(v), f"{v + 0.0:g}") for v in np.round(values, 6)]
        plot.getAxis("left").setTicks([ticks, []])

    def _place_labels(self) -> None:
        for name in self.plots:
            self._apply_ticks(name)
        for name, plot in self.plots.items():
            (_, _), (_, top) = plot.getPlotItem().getViewBox().viewRange()
            self.labels[name].setPos(0.06, top)
        (_, _), (_, top) = self.ecg_plot.getPlotItem().getViewBox().viewRange()
        self.frozen_label.setPos(self.duration - 0.06, top)

    # -- display controls -----------------------------------------------------
    def set_gain(self, gain: float) -> None:
        """ECG gain: x2 halves the displayed millivolt span."""
        self.gain = float(gain)
        low, high = ECG_RANGE
        self.ecg_plot.getPlotItem().setYRange(low / self.gain, high / self.gain, padding=0.0)
        self.labels["ecg"].setHtml(self._label_html("II", ECG_COLOR, "ecg"))
        self._place_labels()

    def set_abp_scale_for(self, systolic: float) -> None:
        """Pick the smallest standard pressure scale that clears the target."""
        if getattr(self, "_replay_mode", False):
            return
        scale = next((s for s in ABP_SCALES if s >= systolic + 20), ABP_SCALES[-1])
        if scale != self.abp_scale:
            self.abp_scale = scale
            self.abp_plot.getPlotItem().setYRange(0.0, float(scale), padding=0.0)
            self.labels["abp"].setHtml(self._label_html("ART", ABP_COLOR, "abp"))
            self._place_labels()

    def set_grid(self, visible: bool) -> None:
        for item in self._grid:
            item.setVisible(bool(visible))

    @property
    def grid_visible(self) -> bool:
        return self._grid[0].isVisible()

    def set_frozen(self, frozen: bool) -> None:
        """Hold the picture; the simulation keeps running underneath."""
        self.frozen = bool(frozen)
        self.frozen_label.setVisible(self.frozen)

    # -- numerics -------------------------------------------------------------
    BADGES: ClassVar[dict[str, str]] = {
        "hr_high": "▲ HIGH", "hr_low": "▼ LOW", "asystole": "ASYSTOLE",
        "hr_unreadable": "NOT COUNTABLE", "spo2_low": "▼ LOW",
        "spo2_no_signal": "NO SIGNAL", "no_pulse": "NO PULSE", "map_low": "▼ LOW",
        "apnoea": "APNOEA", "rr_low": "▼ LOW", "rr_high": "▲ HIGH",
    }

    def show_vitals(self, v: Vitals, worst: dict[str, Alarm],
                    limits: AlarmLimits | None = None) -> None:
        """Write one set of numerics and their alarm states into the tiles."""
        if limits is None:
            limits = AlarmLimits()
        dash = "---"
        self.hr_tile.set_value(str(v.hr) if v.hr is not None else dash)
        self.hr_tile.set_sub(f"{limits.hr_low}–{limits.hr_high}")
        self.spo2_tile.set_value(str(v.spo2) if v.spo2 is not None else dash)
        self.spo2_tile.set_sub(f"≥ {limits.spo2_low}")
        if v.systolic is not None:
            self.abp_tile.set_value(f"{v.systolic}/{v.diastolic}")
        else:
            self.abp_tile.set_value(f"{dash}/{dash}")
        self.abp_tile.set_sub(f"({v.mean_pressure})" if v.mean_pressure is not None else "")
        self.rr_tile.set_value(str(v.resp_rate) if v.resp_rate is not None else dash)
        self.rr_tile.set_sub(f"{limits.rr_low}–{limits.rr_high}")

        for parameter, tile in (("hr", self.hr_tile), ("spo2", self.spo2_tile),
                                ("abp", self.abp_tile), ("resp", self.rr_tile)):
            alarm = worst.get(parameter)
            if alarm is None:
                tile.set_alarm(None)
            else:
                tile.set_alarm(alarm.message, alarm.is_high,
                               self.BADGES.get(alarm.code, alarm.message))

    def clear_vitals(self) -> None:
        for tile in self.tiles.values():
            tile.set_value("---/---" if tile is self.abp_tile else "---")
            tile.set_alarm(None)

    def flash_tiles(self, steady: bool = False) -> None:
        for tile in self.tiles.values():
            tile.flash(steady)

    # -- sweep ----------------------------------------------------------------
    def attach(self, buffer: RingBuffer) -> None:
        """Point the monitor at the buffer the producer is filling."""
        if buffer.capacity != self.n_samples:
            raise ValueError(
                f"buffer holds {buffer.capacity} samples, display expects {self.n_samples}"
            )
        self._buffer = buffer
        self._indices = {"ecg": buffer.channel_index("ecg"),
                         "resp": buffer.channel_index("resp")}
        for optional in ("pleth", "abp", "co2", "awp"):
            if optional in buffer.channels:
                self._indices[optional] = buffer.channel_index(optional)

    def start(self, fps: int = SWEEP_FPS) -> None:
        if self._buffer is None:
            raise RuntimeError("attach() a buffer before starting the sweep")
        self._timer.start(max(1, round(1000 / fps)))

    def stop(self) -> None:
        self._timer.stop()

    @property
    def is_running(self) -> bool:
        return self._timer.isActive()

    def refresh(self) -> None:
        """One frame: read the buffer, cut the erase bar, redraw."""
        if self._buffer is None or self.frozen:
            return
        data, write_index = self._buffer.snapshot()     # already a private copy
        apply_sweep_gap(data, write_index, self.gap_samples)
        self.update_traces(**{name: data[i] for name, i in self._indices.items()})

    # -- rendering ------------------------------------------------------------
    def update_traces(self, ecg: np.ndarray, resp: np.ndarray,
                      pleth: np.ndarray | None = None,
                      abp: np.ndarray | None = None,
                      co2: np.ndarray | None = None,
                      awp: np.ndarray | None = None) -> None:
        """Redraw traces, gapping replay outliers outside case-specific display limits."""
        for name, y in (("ecg", ecg), ("resp", resp), ("pleth", pleth),
                        ("abp", abp), ("co2", co2), ("awp", awp)):
            if y is None:
                continue
            values = y
            limits = getattr(self, "_replay_display_ranges", {}).get(name)
            if getattr(self, "_replay_mode", False) and limits is not None:
                values = np.asarray(y, dtype=np.float64).copy()
                low, high = limits
                valid = np.isfinite(values) & (values >= low) & (values <= high)
                if name == "abp":
                    valid &= values > 0.0
                values[~valid] = np.nan
            k = self.decimation[name]
            self.curves[name].setData(self._x[::k], values[::k])

    def set_replay_mode(self, enabled: bool) -> None:
        """Show replay-only rows and apply or restore the appropriate graph scales."""
        self._replay_mode = bool(enabled)
        self._replay_display_ranges = {}
        defaults = {
            "ecg": (ECG_RANGE[0] / self.gain, ECG_RANGE[1] / self.gain),
            "resp": RESP_RANGE,
            "pleth": PLETH_RANGE,
            "abp": (0.0, float(self.abp_scale)),
            "co2": (0.0, 100.0),
            "awp": (-20.0, 120.0),
        }
        for name, (low, high) in defaults.items():
            self.plots[name].getPlotItem().setYRange(float(low), float(high), padding=0.0)
        self.labels["abp"].setHtml(self._label_html("ART", ABP_COLOR, "abp"))
        for name in ("co2", "awp"):
            self.row_widgets[name].setVisible(enabled)
        for tile in self.tiles.values():
            tile.set_alarm(None)


    def set_replay_display_ranges(self, timeline) -> None:
        """Scale replay plots per case and retain robust bounds for outlier gaps."""
        if not getattr(self, "_replay_mode", False):
            return

        source_names = {
            "ecg": "ecg",
            "resp": "resp",
            "pleth": "ppg",
            "abp": "art",
            "co2": "capno",
            "awp": "awp",
        }
        ranges: dict[str, tuple[float, float]] = {}
        for name, source_name in source_names.items():
            signal = timeline.waveforms.get(source_name)
            if signal is None:
                continue
            values = np.asarray(signal.values, dtype=np.float64).reshape(-1)
            if values.size > 100_000:
                stride = max(1, (values.size + 99_999) // 100_000)
                values = values[::stride]
            values = values[np.isfinite(values)]
            if name in {"abp", "co2"}:
                values = values[values >= 0.0]
            if values.size < 2:
                continue

            low, high = np.percentile(values, (1.0, 99.0))
            span = float(high - low)
            if span <= 1e-6:
                padding = max(abs(float((high + low) * 0.5)) * 0.1, 1.0)
            else:
                padding = max(span * 0.05, 0.05)
            low = float(low) - padding
            high = float(high) + padding
            if name in {"abp", "co2"}:
                low = max(0.0, low)
            if high <= low:
                high = low + 1.0
            ranges[name] = (low, high)

        self._replay_display_ranges = ranges
        for name, (low, high) in ranges.items():
            self.plots[name].getPlotItem().setYRange(low, high, padding=0.0)
            if name == "abp":
                self.labels[name].setHtml(self._label_html("ART", ABP_COLOR, "abp"))


    def show_recorded_vitals(self, values: dict[str, float | None]) -> None:
        """Display recorded values and explain when no fresh reading is available."""
        def display(value: float | None) -> str:
            if value is None:
                return "---"
            try:
                number = float(value)
            except (TypeError, ValueError):
                return "---"
            return f"{number:.0f}" if np.isfinite(number) else "---"

        def has_value(value: float | None) -> bool:
            try:
                return value is not None and bool(np.isfinite(float(value)))
            except (TypeError, ValueError):
                return False

        hr = values.get("hr")
        spo2 = values.get("spo2")
        systolic, diastolic, mean = (
            values.get("sbp"), values.get("dbp"), values.get("map")
        )
        rr = values.get("rr")
        etco2 = values.get("etco2")
        awp = values.get("awp")

        self.hr_tile.set_value(display(hr))
        self.hr_tile.set_sub("Recorded" if has_value(hr) else "No recent sample")
        self.spo2_tile.set_value(display(spo2))
        self.spo2_tile.set_sub("Recorded" if has_value(spo2) else "No recent sample")

        self.abp_tile.set_value(f"{display(systolic)}/{display(diastolic)}")
        pressure_values = (systolic, diastolic, mean)
        if not any(has_value(value) for value in pressure_values):
            self.abp_tile.set_sub("No valid reading")
        elif all(has_value(value) for value in pressure_values):
            self.abp_tile.set_sub(f"MAP {display(mean)}")
        else:
            self.abp_tile.set_sub(f"MAP {display(mean)} · Partial reading")

        self.rr_tile.set_value(display(rr))
        self.rr_tile.set_sub("Recorded" if has_value(rr) else "No recent sample")
        self.co2_tile.set_value(display(etco2))
        self.co2_tile.set_sub("Recorded" if has_value(etco2) else "No recent sample")
        self.awp_tile.set_value(display(awp))
        self.awp_tile.set_sub("Recorded" if has_value(awp) else "No recent sample")

        for tile in self.tiles.values():
            tile.set_alarm(None)

    def clear(self) -> None:
        blank = np.full(self.n_samples, np.nan)
        self.update_traces(blank, blank, blank, blank)
