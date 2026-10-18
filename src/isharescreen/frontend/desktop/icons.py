"""Symbol icons drawn in code, so toolbars can use symbols instead of words.

Each glyph is painted on a unit square with the palette's current colours when
it is drawn, so one icon follows light/dark themes, the disabled state and the
checked (accent) state without any image files, at any DPI.
"""
from __future__ import annotations
from typing import Callable

from PySide6.QtCore import Qt, QPointF, QRect, QRectF, QPoint, QSize
from PySide6.QtGui import (QColor, QFont, QIcon, QIconEngine, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import QApplication

Draw = Callable[[QPainter, QColor, QColor], None]
_GLYPHS: dict[str, Draw] = {}


def glyph_drawer(name: str):
    def register(function: Draw) -> Draw:
        _GLYPHS[name] = function
        return function
    return register


def _stroke(painter, color, width=0.085):
    painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)


def _fill(painter, color):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)


def _line(painter, x1, y1, x2, y2):
    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def _dot(painter, x, y, r):
    painter.drawEllipse(QPointF(x, y), r, r)


def _text(painter, color, text, size=0.78, y=0.5):
    """A character drawn as an outline path (painter units are 0..1)."""
    font = QFont('Segoe UI', 100)
    font.setPixelSize(100)
    path = QPainterPath()
    path.addText(0, 0, font, text)
    box = path.boundingRect()
    scale = size / max(box.width(), box.height())
    painter.save()
    painter.translate(0.5 - box.center().x() * scale, y - box.center().y() * scale)
    painter.scale(scale, scale)
    _fill(painter, color)
    painter.drawPath(path)
    painter.restore()


def _magnifier(painter, color):
    _stroke(painter, color)
    painter.drawEllipse(QPointF(0.44, 0.44), 0.26, 0.26)
    _line(painter, 0.63, 0.63, 0.84, 0.84)


@glyph_drawer('pointer')
def _pointer(p, c, accent):
    _fill(p, c)
    p.drawPolygon(QPolygonF([QPointF(.30, .14), QPointF(.30, .80), QPointF(.46, .65), QPointF(.58, .88),
                             QPointF(.68, .83), QPointF(.56, .61), QPointF(.77, .61)]))


@glyph_drawer('clipboard')
def _clipboard(p, c, accent):
    _stroke(p, c)
    p.drawRoundedRect(QRectF(.22, .18, .56, .68), .08, .08)
    _fill(p, c)
    p.drawRoundedRect(QRectF(.38, .10, .24, .14), .04, .04)
    _stroke(p, c)
    _line(p, .34, .46, .66, .46); _line(p, .34, .62, .58, .62)


@glyph_drawer('keyboard')
def _keyboard(p, c, accent):
    _stroke(p, c)
    p.drawRoundedRect(QRectF(.08, .26, .84, .48), .08, .08)
    _fill(p, c)
    for x in (.24, .5, .76):
        _dot(p, x, .40, .035)
    _stroke(p, c, .07)
    _line(p, .30, .60, .70, .60)


@glyph_drawer('launchpad')
def _launchpad(p, c, accent):
    _fill(p, c)
    for row in range(3):
        for col in range(3):
            p.drawRoundedRect(QRectF(.16 + col * .26, .16 + row * .26, .17, .17), .04, .04)


@glyph_drawer('mission')
def _mission(p, c, accent):
    _stroke(p, c)
    p.drawRoundedRect(QRectF(.10, .20, .80, .60), .07, .07)
    _line(p, .42, .20, .42, .80)
    _line(p, .42, .50, .90, .50)


@glyph_drawer('desktop')
def _desktop(p, c, accent):
    _stroke(p, c)
    p.drawRoundedRect(QRectF(.10, .20, .80, .60), .07, .07)
    _fill(p, c)
    p.drawRoundedRect(QRectF(.24, .64, .52, .07), .03, .03)


@glyph_drawer('zoom-out')
def _zoom_out(p, c, accent):
    _magnifier(p, c); _stroke(p, c); _line(p, .34, .44, .54, .44)


@glyph_drawer('zoom-in')
def _zoom_in(p, c, accent):
    _magnifier(p, c); _stroke(p, c); _line(p, .34, .44, .54, .44); _line(p, .44, .34, .44, .54)


@glyph_drawer('zoom-actual')
def _zoom_actual(p, c, accent):
    _magnifier(p, c); _stroke(p, c, .07); p.drawEllipse(QPointF(.44, .44), .08, .08)


@glyph_drawer('zoom-fit')
def _zoom_fit(p, c, accent):
    _magnifier(p, c); _stroke(p, c, .07); p.drawRoundedRect(QRectF(.35, .35, .18, .18), .02, .02)


@glyph_drawer('search')
def _search(p, c, accent):
    _magnifier(p, c)


@glyph_drawer('more')
def _more(p, c, accent):
    _fill(p, c)
    for x in (.22, .5, .78):
        _dot(p, x, .5, .065)


@glyph_drawer('plus')
def _plus(p, c, accent):
    _stroke(p, c, .095)
    _line(p, .5, .22, .5, .78); _line(p, .22, .5, .78, .5)


@glyph_drawer('sidebar')
def _sidebar(p, c, accent):
    _stroke(p, c)
    p.drawRoundedRect(QRectF(.10, .20, .80, .60), .08, .08)
    _line(p, .38, .20, .38, .80)


@glyph_drawer('grid')
def _grid(p, c, accent):
    _fill(p, c)
    for row in range(2):
        for col in range(2):
            p.drawRoundedRect(QRectF(.18 + col * .34, .18 + row * .34, .26, .26), .05, .05)


@glyph_drawer('list')
def _list(p, c, accent):
    _stroke(p, c, .08)
    for y in (.30, .50, .70):
        _line(p, .38, y, .84, y)
    _fill(p, c)
    for y in (.30, .50, .70):
        _dot(p, .20, y, .04)


@glyph_drawer('monitor')
def _monitor(p, c, accent):
    tint = c if c.name() == QApplication.palette().color(QPalette.ColorRole.HighlightedText).name() else accent
    _fill(p, tint)
    p.drawRoundedRect(QRectF(.08, .14, .84, .55), .07, .07)
    _stroke(p, tint, .07)
    _line(p, .5, .70, .5, .82); _line(p, .30, .84, .70, .84)


@glyph_drawer('info')
def _info(p, c, accent):
    _stroke(p, c, .07)
    p.drawEllipse(QPointF(.5, .5), .36, .36)
    _fill(p, c)
    _dot(p, .5, .31, .045)
    _stroke(p, c, .085)
    _line(p, .5, .45, .5, .70)


@glyph_drawer('help')
def _help(p, c, accent):
    _stroke(p, c, .07)
    p.drawEllipse(QPointF(.5, .5), .36, .36)
    _text(p, c, '?', size=.40)


@glyph_drawer('eye')
def _eye(p, c, accent):
    _stroke(p, c, .07)
    path = QPainterPath()
    path.moveTo(.10, .5); path.quadTo(.5, .12, .90, .5); path.quadTo(.5, .88, .10, .5)
    p.drawPath(path)
    _fill(p, c)
    _dot(p, .5, .5, .10)


@glyph_drawer('eye-off')
def _eye_off(p, c, accent):
    _eye(p, c, accent)
    _stroke(p, c, .085)
    _line(p, .18, .82, .82, .18)


@glyph_drawer('fullscreen')
def _fullscreen(p, c, accent):
    _stroke(p, c, .08)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x, y = .5 + sx * .34, .5 + sy * .34
        _line(p, x, y, x - sx * .20, y); _line(p, x, y, x, y - sy * .20)


@glyph_drawer('refresh')
def _refresh(p, c, accent):
    _stroke(p, c, .08)
    path = QPainterPath()
    path.arcMoveTo(QRectF(.2, .2, .6, .6), 40)
    path.arcTo(QRectF(.2, .2, .6, .6), 40, 280)
    p.drawPath(path)
    _fill(p, c)
    p.drawPolygon(QPolygonF([QPointF(.72, .12), QPointF(.82, .38), QPointF(.56, .34)]))


@glyph_drawer('power')
def _power(p, c, accent):
    _stroke(p, c, .08)
    path = QPainterPath()
    path.arcMoveTo(QRectF(.2, .22, .6, .6), 125)
    path.arcTo(QRectF(.2, .22, .6, .6), 125, -290)
    p.drawPath(path)
    _line(p, .5, .14, .5, .46)


@glyph_drawer('link')
def _link(p, c, accent):
    _stroke(p, c, .08)
    p.save()
    p.translate(.5, .5); p.rotate(-45); p.translate(-.5, -.5)
    p.drawRoundedRect(QRectF(.08, .38, .46, .24), .12, .12)
    p.drawRoundedRect(QRectF(.46, .38, .46, .24), .12, .12)
    p.restore()


@glyph_drawer('swap')
def _swap(p, c, accent):
    _stroke(p, c, .08)
    _line(p, .18, .36, .78, .36); _line(p, .78, .36, .64, .22); _line(p, .78, .36, .64, .50)
    _line(p, .82, .66, .22, .66); _line(p, .22, .66, .36, .52); _line(p, .22, .66, .36, .80)


@glyph_drawer('command')
def _command(p, c, accent):
    _text(p, c, '⌘', size=.78)


@glyph_drawer('option')
def _option(p, c, accent):
    _text(p, c, '⌥', size=.74)


@glyph_drawer('control')
def _control(p, c, accent):
    _text(p, c, '⌃', size=.60, y=.42)


class GlyphEngine(QIconEngine):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def clone(self):
        return GlyphEngine(self.name)

    def paint(self, painter: QPainter, rect: QRect, mode, state):
        palette = QApplication.palette()
        accent = palette.color(QPalette.ColorRole.Highlight)
        if mode == QIcon.Mode.Disabled:
            color = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
        elif mode == QIcon.Mode.Selected or state == QIcon.State.On:
            accent = palette.color(QPalette.ColorRole.HighlightedText)       # sits on the selection colour
            color = accent
        elif state == QIcon.State.On:
            color = palette.color(QPalette.ColorRole.HighlightedText)    # checked buttons sit on the accent colour
        else:
            color = palette.color(QPalette.ColorRole.WindowText)
        size = min(rect.width(), rect.height())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(rect.center().x() - size / 2 + .5, rect.center().y() - size / 2 + .5)
        painter.scale(size, size)
        _GLYPHS[self.name](painter, color, accent)
        painter.restore()

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        painter.end()
        return pixmap


def glyph(name: str) -> QIcon:
    """A theme-aware icon for `name` (see the registered drawers above)."""
    if name not in _GLYPHS:
        raise KeyError(f'unknown glyph {name!r}')
    return QIcon(GlyphEngine(name))


def glyph_names() -> list[str]:
    return sorted(_GLYPHS)
