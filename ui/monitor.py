"""The monitor: four sweeping waveforms, each paired with its numeric.

The display does not scroll.  The x-axis is nailed to 0..10 s and the trace
stays where it was drawn; a block of ``NaN`` samples is blanked just ahead of
the buffer's write cursor, so what moves across the screen is the *gap*.  New
samples appear at its trailing edge and ten-second-old samples vanish at its
leading edge - the erase bar of a hospital monitor.

Each row pairs a waveform with its numeric tile, the layout every bedside
monitor uses: the number you read sits beside the trace it came from.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

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

        colours = {"ecg": ECG_COLOR, "pleth": PLETH_COLOR, "abp": ABP_COLOR,
                   "resp": RESP_COLOR}
        ranges = {"ecg": ECG_RANGE, "pleth": PLETH_RANGE,
                  "abp": (0.0, float(self.abp_scale)), "resp": RESP_RANGE}
        titles = {"ecg": "II", "pleth": "Pleth", "abp": "ART", "resp": "RESP"}

        self.hr_tile = VitalTile("HR", "bpm", ECG_COLOR, "heart", value_px=72)
        self.spo2_tile = VitalTile("SpO₂", "%", PLETH_COLOR, "droplet", value_px=60)
        self.abp_tile = VitalTile("ART", "mmHg", ABP_COLOR, "gauge", value_px=42)
        self.rr_tile = VitalTile("RR", "rpm", RESP_COLOR, "lungs", value_px=60)
        self.tiles = {"ecg": self.hr_tile, "pleth": self.spo2_tile,
                      "abp": self.abp_tile, "resp": self.rr_tile}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for name, stretch, decimate in ROWS:
            plot = self._make_plot(ranges[name])
            curve = plot.plot(self._x[::decimate], np.full(self._x[::decimate].size, np.nan),
                              pen=trace_pen(colours[name]))
            label = pg.TextItem(anchor=(0, 0))
            label.setHtml(self._label_html(titles[name], colours[name], name))
            plot.addItem(label, ignoreBounds=True)
            self.plots[name], self.curves[name], self.labels[name] = plot, curve, label
            self.decimation[name] = decimate

            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(plot, stretch=1)
            row.addWidget(self.tiles[name])
            layout.addLayout(row, stretch=stretch)

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

    def _label_html(self, title: str, colour: str, name: str) -> str:
        extra = ""
        if name == "ecg":
            extra = f"&nbsp;&nbsp;<span style='color:{TEXT_FAINT}'>{self._gain_text()}</span>"
        elif name == "abp":
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

    TICK_STEPS = {"ecg": 1.0, "pleth": 0.5, "abp": 50.0, "resp": 1.0}

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
    BADGES = {
        "hr_high": "▲ HIGH", "hr_low": "▼ LOW", "asystole": "ASYSTOLE",
        "hr_unreadable": "NOT COUNTABLE", "spo2_low": "▼ LOW",
        "spo2_no_signal": "NO SIGNAL", "no_pulse": "NO PULSE", "map_low": "▼ LOW",
        "apnoea": "APNOEA", "rr_low": "▼ LOW", "rr_high": "▲ HIGH",
    }

    def show_vitals(self, v: Vitals, worst: dict[str, Alarm],
                    limits: AlarmLimits = AlarmLimits()) -> None:
        """Write one set of numerics and their alarm states into the tiles."""
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
        for optional in ("pleth", "abp"):
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
                      abp: np.ndarray | None = None) -> None:
        """Redraw the traces.  ``NaN`` samples render as gaps, not zeros."""
        for name, y in (("ecg", ecg), ("resp", resp), ("pleth", pleth), ("abp", abp)):
            if y is None:
                continue
            k = self.decimation[name]
            self.curves[name].setData(self._x[::k], y[::k])

    def clear(self) -> None:
        blank = np.full(self.n_samples, np.nan)
        self.update_traces(blank, blank, blank, blank)
