"""Reusable widgets: sliders, switches, segmented controls, cards, vital tiles,
the alarm banner and the event log."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .theme import (
    ACCENT,
    BORDER,
    BORDER_STRONG,
    DANGER,
    INFO,
    TEXT,
    TEXT_FAINT,
    TEXT_MUTED,
    TILE_BG,
    WARNING,
    numeric_font,
)

URGENCY_LETHAL = "lethal"          # mirrors core.pathology; kept local to avoid a UI->core cycle


class LabeledSlider(QWidget):
    """A slider that reports a float in engineering units, with a live readout.

    Qt sliders are integer-only, so each control carries its own mapping from
    step index to physical value rather than scattering conversions at the
    call sites.
    """

    changed = pyqtSignal(str, float)

    def __init__(self, key: str, label: str, limits: tuple[float, float], value: float,
                 steps: int, unit: str = "", decimals: int = 0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.low, self.high = limits
        self.steps = int(steps)
        self.unit = unit
        self.decimals = int(decimals)

        self.name_label = QLabel(label, objectName="controlName")
        self.value_label = QLabel(objectName="controlValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.slider.setValue(self._to_step(value))
        self.slider.setAccessibleName(label)
        self.slider.valueChanged.connect(self._on_slider)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.name_label)
        header.addStretch(1)
        header.addWidget(self.value_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)
        layout.addLayout(header)
        layout.addWidget(self.slider)

        self._refresh_label(value)

    def _to_step(self, value: float) -> int:
        span = self.high - self.low
        frac = 0.0 if span == 0 else (value - self.low) / span
        return int(round(min(1.0, max(0.0, frac)) * self.steps))

    def _to_value(self, step: int) -> float:
        return self.low + (self.high - self.low) * step / self.steps

    def value(self) -> float:
        return self._to_value(self.slider.value())

    def set_value(self, value: float) -> None:
        """Move the slider without re-emitting - used to sync from state."""
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(self._to_step(value))
        self.slider.blockSignals(blocked)
        self._refresh_label(self.value())

    def _refresh_label(self, value: float) -> None:
        text = f"{value:.{self.decimals}f}"
        self.value_label.setText(f"{text} {self.unit}".strip())

    def _on_slider(self, step: int) -> None:
        value = self._to_value(step)
        self._refresh_label(value)
        self.changed.emit(self.key, value)


class ToggleSwitch(QCheckBox):
    """A pill switch.  Still a QCheckBox underneath, so it keeps that whole API."""

    TRACK_W, TRACK_H = 34, 18

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(26)

    def sizeHint(self) -> QSize:
        width = self.TRACK_W + 10 + self.fontMetrics().horizontalAdvance(self.text())
        return QSize(width + 4, max(26, self.TRACK_H + 8))

    def hitButton(self, pos) -> bool:              # the whole row is clickable
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = (self.height() - self.TRACK_H) / 2
        track = QRectF(1, y, self.TRACK_W, self.TRACK_H)
        on = self.isChecked()
        painter.setPen(QColor(ACCENT if on else BORDER_STRONG))
        painter.setBrush(QColor("#14532D" if on else "#0B1222"))
        painter.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        knob = self.TRACK_H - 6
        x = track.right() - knob - 3 if on else track.left() + 3
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT if on else TEXT_MUTED))
        painter.drawEllipse(QRectF(x, y + 3, knob, knob))
        if self.hasFocus():
            painter.setPen(QColor(TEXT))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track.adjusted(-2, -2, 2, 2), self.TRACK_H / 2 + 2,
                                    self.TRACK_H / 2 + 2)
        painter.setPen(QColor(TEXT if self.isEnabled() else TEXT_FAINT))
        painter.drawText(QRectF(self.TRACK_W + 10, 0, self.width(), self.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.text())
        painter.end()


class SegmentedControl(QWidget):
    """A row of mutually exclusive buttons; emits the chosen option's label."""

    changed = pyqtSignal(str)

    def __init__(self, options: list[str], current: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for option in options:
            button = QPushButton(option, objectName="segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(button)
            layout.addWidget(button)
            self.buttons[option] = button
            button.toggled.connect(lambda on, o=option: on and self.changed.emit(o))
        self.set_value(current or options[0], emit=False)

    def value(self) -> str:
        for option, button in self.buttons.items():
            if button.isChecked():
                return option
        return ""

    def set_value(self, option: str, emit: bool = True) -> None:
        button = self.buttons[option]
        blocked = self.blockSignals(not emit)
        button.setChecked(True)
        self.blockSignals(blocked)


class Card(QFrame):
    """A titled, rounded panel section."""

    def __init__(self, title: str, icon_name: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(7)
        if icon_name:
            glyph = QLabel()
            glyph.setPixmap(icons.render(icon_name, TEXT_MUTED, 14))
            header.addWidget(glyph)
        self.title = QLabel(title.upper(), objectName="cardTitle")
        header.addWidget(self.title)
        header.addStretch(1)
        self.header = header
        outer.addLayout(header)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget


class VitalTile(QFrame):
    """One large numeric, bedside-monitor style.

    Alarm state is shown three ways at once - tinted background, coloured
    border, and a text badge (``▲ HIGH``, ``NO PULSE``) - because colour alone
    is not enough to carry an alarm.
    """

    def __init__(self, title: str, unit: str, color: str, icon_name: str,
                 value_px: int = 50, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tile")
        self.color = color
        self.setMinimumWidth(190)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(214)
        self._alarm_message: str | None = None
        self._alarm_high = False
        self._bright = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(0)

        top = QHBoxLayout()
        top.setSpacing(6)
        glyph = QLabel()
        glyph.setPixmap(icons.render(icon_name, color, 15))
        top.addWidget(glyph)
        self.title = QLabel(title)
        self.title.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;"
                                 "background: transparent;")
        top.addWidget(self.title)
        top.addStretch(1)
        self.badge = QLabel()
        self.badge.setVisible(False)
        top.addWidget(self.badge)
        layout.addLayout(top)

        layout.addStretch(1)
        self.value_label = QLabel("---")
        self.value_label.setFont(numeric_font(value_px))
        self.value_label.setStyleSheet(f"color: {color}; background: transparent;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.value_label)
        layout.addStretch(1)

        bottom = QHBoxLayout()
        self.unit_label = QLabel(unit)
        self.unit_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 11px; background: transparent;")
        bottom.addWidget(self.unit_label)
        bottom.addStretch(1)
        self.sub_label = QLabel("")
        self.sub_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; background: transparent;")
        bottom.addWidget(self.sub_label)
        layout.addLayout(bottom)
        self._restyle()

    @property
    def value_text(self) -> str:
        return self.value_label.text()

    @property
    def alarm_message(self) -> str | None:
        return self._alarm_message

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)

    def set_sub(self, text: str) -> None:
        self.sub_label.setText(text)

    def set_alarm(self, message: str | None, high: bool = False, badge: str = "") -> None:
        changed = (message, high) != (self._alarm_message, self._alarm_high)
        self._alarm_message, self._alarm_high = message, high
        if message:
            colour = DANGER if high else WARNING
            self.badge.setText(badge or message)
            self.badge.setStyleSheet(
                f"color: {colour}; border: 1px solid {colour}; border-radius: 4px;"
                "padding: 0 5px; font-size: 10px; font-weight: 700; background: transparent;")
            self.badge.setVisible(True)
        else:
            self.badge.setVisible(False)
        if changed:
            self._bright = False
            self._restyle()

    def flash(self, steady: bool = False) -> None:
        """Advance the flash phase.  Only high-priority alarms flash."""
        if self._alarm_message and self._alarm_high and not steady:
            self._bright = not self._bright
        else:
            self._bright = False
        self._restyle()

    def _restyle(self) -> None:
        if self._alarm_message:
            accent = DANGER if self._alarm_high else WARNING
            fill = ("#3B0A0F" if self._bright else "#1F0709") if self._alarm_high else "#1F1406"
        else:
            accent, fill = self.color, TILE_BG
        self.setStyleSheet(
            f"QFrame#tile {{ background: {fill}; border: 1px solid {BORDER};"
            f" border-left: 3px solid {accent}; border-radius: 8px; }}")


class AlarmBanner(QLabel):
    """Alarm strip above the monitor.

    Lives above the waveforms rather than in the control panel: an alarm that
    disappears when the panel is collapsed is worse than no alarm at all.
    Lethal alarms flash; urgent ones are steady, so the two are distinguishable
    at a glance without reading the text.  The warning glyph is drawn, not an
    emoji, so it looks the same on every platform.
    """

    LETHAL = ("#4A0F16", "#7A1A25", "#FFB4BB")     # dim bg, bright bg, text
    URGENT = ("#3A2A0D", "#3A2A0D", "#FCD34D")     # steady

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("alarm")
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.setIndent(34)
        self._palette = self.URGENT
        self._bright = False
        self._active = False
        self._headline = ""
        self.clear_alarm()

    @property
    def is_active(self) -> bool:
        """Whether an alarm is raised.

        Not the same as ``isVisible()``: a widget in a window that has not been
        shown reports False regardless, which would make the banner rebuild
        itself every tick instead of flashing.
        """
        return self._active

    @property
    def headline(self) -> str:
        return self._headline

    def clear_alarm(self) -> None:
        self._active = False
        self._headline = ""
        self.setText("")
        self.setVisible(False)

    def show_alarm(self, headline: str, advice: str, urgency: str) -> None:
        self._palette = self.LETHAL if urgency == URGENCY_LETHAL else self.URGENT
        self._active = True
        self._headline = headline
        self.setText(f"{headline}  —  {advice}")
        self.setVisible(True)
        self._repaint()

    def flash(self) -> None:
        """Advance the flash phase; a no-op for the steady (urgent) palette."""
        self._bright = not self._bright
        self._repaint()

    def _repaint(self) -> None:
        dim, bright, text = self._palette
        background = bright if self._bright else dim
        self.setStyleSheet(f"QLabel#alarm {{ background: {background}; color: {text}; }}")

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._active:
            painter = QPainter(self)
            size = 18
            painter.drawPixmap(12, (self.height() - size) // 2,
                               icons.render("alert", self._palette[2], size))
            painter.end()


class EventLog(QListWidget):
    """Timestamped event history, newest first."""

    LIMIT = 500
    COLOURS = {"info": INFO, "ok": ACCENT, "warn": WARNING, "alarm": DANGER,
               "neutral": TEXT_MUTED}
    GLYPHS = {"info": "activity", "ok": "check", "warn": "alert", "alarm": "alert",
              "neutral": "timeline"}

    def add(self, stamp: str, message: str, severity: str = "neutral") -> QListWidgetItem:
        colour = self.COLOURS.get(severity, TEXT_MUTED)
        item = QListWidgetItem(icons.icon(self.GLYPHS.get(severity, "timeline"), colour, 14),
                               f"{stamp}   {message}")
        item.setForeground(QColor(TEXT if severity in ("alarm", "warn") else TEXT_MUTED))
        item.setData(Qt.ItemDataRole.UserRole, severity)
        self.insertItem(0, item)
        while self.count() > self.LIMIT:
            self.takeItem(self.count() - 1)
        return item

    def messages(self) -> list[str]:
        """Plain texts, newest first - for tests and export."""
        return [self.item(i).text() for i in range(self.count())]
