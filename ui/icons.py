"""A small vector icon set, drawn in code.

Emoji and font glyphs render differently on every platform - and some turn up
in colour - so each icon here is a path on a 24-unit grid, stroked or filled
in whatever colour the caller asks for and rendered at the screen's pixel ratio.
"""

from __future__ import annotations

import math
from functools import lru_cache

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from .theme import TEXT_MUTED


def _poly(*points: tuple[float, float], close: bool = False) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(*points[0])
    for p in points[1:]:
        path.lineTo(*p)
    if close:
        path.closeSubpath()
    return path


def _heart() -> tuple[QPainterPath, bool]:
    left, right = QPainterPath(), QPainterPath()
    left.addEllipse(QPointF(8.6, 9.3), 4.4, 4.4)
    right.addEllipse(QPointF(15.4, 9.3), 4.4, 4.4)
    point = QPainterPath()
    point.addPolygon(QPolygonF([QPointF(4.4, 10.8), QPointF(12.0, 20.0),
                                QPointF(19.6, 10.8), QPointF(12.0, 8.0)]))
    return left.united(right).united(point), True


def _droplet() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.moveTo(12, 3)
    p.cubicTo(12, 3, 18.5, 10, 18.5, 14.5)
    p.cubicTo(18.5, 18.1, 15.6, 21, 12, 21)
    p.cubicTo(8.4, 21, 5.5, 18.1, 5.5, 14.5)
    p.cubicTo(5.5, 10, 12, 3, 12, 3)
    return p, False


def _lungs() -> tuple[QPainterPath, bool]:
    p = _poly((12, 3), (12, 11))
    p.moveTo(10.5, 8.5)
    p.cubicTo(7, 7.5, 4, 11, 4, 16)
    p.cubicTo(4, 19, 6, 20.5, 8.5, 19.2)
    p.cubicTo(10.2, 18.3, 10.5, 16, 10.5, 13)
    p.closeSubpath()
    p.moveTo(13.5, 8.5)
    p.cubicTo(17, 7.5, 20, 11, 20, 16)
    p.cubicTo(20, 19, 18, 20.5, 15.5, 19.2)
    p.cubicTo(13.8, 18.3, 13.5, 16, 13.5, 13)
    p.closeSubpath()
    return p, False


def _gauge() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.arcMoveTo(QRectF(3, 4, 18, 18), 200)
    p.arcTo(QRectF(3, 4, 18, 18), 200, -220)
    p.moveTo(12, 13)
    p.lineTo(16.5, 8.5)
    return p, False


def _bolt() -> tuple[QPainterPath, bool]:
    return _poly((13, 2), (4, 14), (11, 14), (10, 22), (20, 10), (13, 10), (13, 2),
                 close=True), True


def _sync() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.addEllipse(QPointF(12, 12), 9, 9)
    p.addPath(_poly((13, 6), (8.5, 13), (12, 13), (11, 18), (15.5, 11), (12, 11),
                    (13, 6), close=True))
    return p, False


def _compress() -> tuple[QPainterPath, bool]:
    """Two chevrons driving down onto a chest line - not an arrow into a tray,
    which reads as 'download'."""
    p = _poly((6, 3.5), (12, 8.5), (18, 3.5))
    p.addPath(_poly((6, 9.5), (12, 14.5), (18, 9.5)))
    p.addPath(_poly((3.5, 19.5), (20.5, 19.5)))
    p.addPath(_poly((8, 17), (8, 19.5)))
    p.addPath(_poly((16, 17), (16, 19.5)))
    return p, False


def _activity() -> tuple[QPainterPath, bool]:
    return _poly((2, 12), (6, 12), (9, 5), (13.5, 19), (16.5, 12), (22, 12)), False


def _play() -> tuple[QPainterPath, bool]:
    return _poly((7, 4.5), (19.5, 12), (7, 19.5), close=True), True


def _stop() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.addRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    return p, True


def _snowflake() -> tuple[QPainterPath, bool]:
    """Six spokes, each ending in a V that opens outward."""
    p = QPainterPath()

    def at(radius: float, degrees: float) -> tuple[float, float]:
        rad = math.radians(degrees)
        return 12 + radius * math.cos(rad), 12 - radius * math.sin(rad)

    for spoke in range(0, 360, 60):
        p.moveTo(12, 12)
        p.lineTo(*at(9.0, spoke))
        for side in (-24, 24):
            p.moveTo(*at(6.0, spoke))
            p.lineTo(*at(8.6, spoke + side))
    return p, False


def _grid() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.addRoundedRect(QRectF(3.5, 3.5, 17, 17), 2, 2)
    for x in (9.2, 14.8):
        p.moveTo(x, 3.5)
        p.lineTo(x, 20.5)
        p.moveTo(3.5, x)
        p.lineTo(20.5, x)
    return p, False


def _download() -> tuple[QPainterPath, bool]:
    p = _poly((12, 3.5), (12, 15))
    p.addPath(_poly((7, 10), (12, 15), (17, 10)))
    p.addPath(_poly((4, 15), (4, 20), (20, 20), (20, 15)))
    return p, False


def _bell() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.moveTo(6, 16)
    p.lineTo(6, 11)
    p.cubicTo(6, 7.2, 8.6, 4.5, 12, 4.5)
    p.cubicTo(15.4, 4.5, 18, 7.2, 18, 11)
    p.lineTo(18, 16)
    p.lineTo(20, 18)
    p.lineTo(4, 18)
    p.closeSubpath()
    p.moveTo(10, 20.5)
    p.lineTo(14, 20.5)
    return p, False


def _alert() -> tuple[QPainterPath, bool]:
    p = _poly((12, 3), (22, 20.5), (2, 20.5), close=True)
    p.addPath(_poly((12, 9.5), (12, 14.5)))
    p.addEllipse(QPointF(12, 17.4), 0.4, 0.4)
    return p, False


def _list() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    for y in (6, 12, 18):
        p.moveTo(9, y)
        p.lineTo(20, y)
        p.addEllipse(QPointF(4.8, y), 0.6, 0.6)
    return p, False


def _user() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    p.addEllipse(QPointF(12, 8), 4, 4)
    p.moveTo(4, 21)
    p.cubicTo(4, 16, 7.6, 13.8, 12, 13.8)
    p.cubicTo(16.4, 13.8, 20, 16, 20, 21)
    return p, False


def _sliders() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    for y, knob in ((6, 15), (12, 8), (18, 13)):
        p.moveTo(3, y)
        p.lineTo(21, y)
        p.addEllipse(QPointF(knob, y), 2.2, 2.2)
    return p, False


def _timeline() -> tuple[QPainterPath, bool]:
    p = QPainterPath()
    for x in (5, 12, 19):
        p.addEllipse(QPointF(x, 12), 2.4, 2.4)
    p.moveTo(7.4, 12)
    p.lineTo(9.6, 12)
    p.moveTo(14.4, 12)
    p.lineTo(16.6, 12)
    return p, False


def _check() -> tuple[QPainterPath, bool]:
    return _poly((4.5, 12.5), (9.5, 17.5), (19.5, 6.5)), False


_SHAPES = {
    "heart": _heart, "droplet": _droplet, "lungs": _lungs, "gauge": _gauge,
    "bolt": _bolt, "sync": _sync, "compress": _compress, "activity": _activity,
    "play": _play, "stop": _stop, "freeze": _snowflake, "grid": _grid,
    "download": _download, "bell": _bell, "alert": _alert, "list": _list,
    "user": _user, "sliders": _sliders, "timeline": _timeline, "check": _check,
}

NAMES = tuple(_SHAPES)


def render(name: str, color: str = TEXT_MUTED, size: int = 18,
           ratio: float = 2.0) -> QPixmap:
    """The named icon as a pixmap ``size`` logical pixels square."""
    path, filled = _SHAPES[name]()
    pixmap = QPixmap(round(size * ratio), round(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 24.0, size / 24.0)
    colour = QColor(color)
    if filled:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
    else:
        pen = QPen(colour, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.end()
    return pixmap


@lru_cache(maxsize=256)
def icon(name: str, color: str = TEXT_MUTED, size: int = 18) -> QIcon:
    return QIcon(render(name, color, size))
