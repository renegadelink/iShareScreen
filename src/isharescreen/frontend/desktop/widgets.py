"""Painted pieces of the Liquid Glass look, and factories that fall back to the plain
widgets under the macOS look.

Qt cannot blur what is behind a widget, so glass is done the way the kit's own no-blur
variant does it: the window paints the soft green *field*, and each glass surface is a
translucent white gradient over it, with a bright top edge and a faint green shadow.
Data (lists, forms, readings) sits on opaque *wells*. Every effect can fall back to
opaque panels (see `theme.effects_reduced`). Buttons give hover and press feedback by
changing colour only, never size or position, and the reveal animation fades opacity
only, so reduced motion costs nothing but speed.
"""
from __future__ import annotations

from PySide6.QtCore import (Qt, QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRectF, QSize, QTimer,
                            QVariantAnimation)
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QIcon, QLinearGradient, QPainter, QPen,
                           QRadialGradient)
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton, QToolButton,
                               QVBoxLayout, QWidget)

from .theme import effects_reduced, is_glass, motion_ms, tokens


def _rgba(values, scale=1.0) -> QColor:
    r, g, b, a = values
    return QColor(r, g, b, max(0, min(255, int(a * scale))))


# ── the field and the shadows ────────────────────────────────────────

def paint_field(painter: QPainter, rect) -> None:
    """Soft brand-green light on a pale ground: the thing the glass appears to refract."""
    t = tokens()
    painter.save()
    painter.setClipRect(rect)
    painter.fillRect(rect, QColor(t['window']))
    painter.setPen(Qt.PenStyle.NoPen)
    lights = ((-.10, -.10, 1100, 700, t['field_a']), (1.10, 0.0, 900, 640, t['field_b']),
              (.60, 1.20, 1000, 800, t['field_c']), (.30, .60, 700, 500, (255, 255, 255, 140)))
    for fx, fy, rx, ry, colour in lights:
        painter.save()
        painter.translate(rect.x() + fx * rect.width(), rect.y() + fy * rect.height())
        painter.scale(rx, ry)
        gradient = QRadialGradient(0, 0, 1.0)
        centre = _rgba(colour)
        clear = QColor(centre.red(), centre.green(), centre.blue(), 0)
        gradient.setColorAt(0, centre); gradient.setColorAt(.6, clear); gradient.setColorAt(1, clear)
        painter.setBrush(QBrush(gradient))
        painter.drawRect(QRectF(-1.0, -1.0, 2.0, 2.0))
        painter.restore()
    painter.restore()


def paint_shadow(painter: QPainter, rect: QRectF, radius: float) -> None:
    """A soft green drop shadow, layered (a blur effect would forbid fading the panel)."""
    t = tokens()
    r, g, b = t['shadow']
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    base = rect.translated(0, 7)
    for i in range(16, 0, -1):
        alpha = int(7 * (1 - i / 17) ** 2)
        if alpha <= 0:
            continue
        painter.setBrush(QColor(r, g, b, alpha))
        painter.drawRoundedRect(base.adjusted(-i, -i, i, i), radius + i, radius + i)
    painter.restore()


def paint_backdrop(widget: QWidget) -> None:
    """Paint a window's background: the field and shadows under its glass, or the plain colour."""
    painter = QPainter(widget)
    if is_glass():
        paint_field(painter, widget.rect())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for surface in widget.findChildren(Surface):
            if surface.shadow and surface.isVisibleTo(widget) and not effects_reduced():
                top_left = surface.mapTo(widget, QPoint(0, 0))
                paint_shadow(painter, QRectF(top_left, QSize(surface.width(), surface.height())), surface.radius)
    else:
        painter.fillRect(widget.rect(), QColor(tokens()['window']))
    painter.end()


class FieldWidget(QWidget):
    """A plain container that paints the window background (the field, under Liquid Glass)."""
    def paintEvent(self, event):
        paint_backdrop(self)


# ── surfaces ─────────────────────────────────────────────────────────

class Surface(QFrame):
    """A panel. Under macOS it is a normal styled frame (by object name); under Liquid Glass it is
    painted here: `glass`/`rail`/`strong`/`flat` are frosted, `well` is the opaque surface for
    data and forms, `segment` is the pill behind grouped controls."""
    RADII = {'glass': 18, 'rail': 20, 'strong': 22, 'flat': 18, 'well': 14, 'segment': 999}

    def __init__(self, kind: str = 'glass', name: str = '', radius=None, shadow: bool = False, parent=None):
        super().__init__(parent)
        if name:
            self.setObjectName(name)
        self.kind = kind
        self.radius = radius if radius is not None else self.RADII.get(kind, 18)
        self.shadow = shadow

    def paintEvent(self, event):
        if not is_glass():
            super().paintEvent(event)
            return
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        radius = min(float(self.radius), rect.height() / 2)
        hairline, hairline2 = QColor(16, 36, 26, 23), QColor(16, 36, 26, 41)
        frosted = self.kind in ('glass', 'rail', 'strong', 'flat')
        if frosted and not effects_reduced():
            if self.kind == 'flat':
                painter.setBrush(_rgba(t['flat']))
            else:
                top, bottom = ('strong_top', 'strong_bottom') if self.kind == 'strong' else ('glass_top', 'glass_bottom')
                gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
                gradient.setColorAt(0, _rgba(t[top])); gradient.setColorAt(1, _rgba(t[bottom]))
                painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(_rgba(t['edge']), 1))
            painter.drawRoundedRect(rect, radius, radius)
            if self.kind != 'flat':
                painter.setPen(QPen(QColor(255, 255, 255, 242), 1))        # the bright inset top edge
                painter.drawLine(QPointF(rect.left() + radius * .9, rect.top() + 1.2),
                                 QPointF(rect.right() - radius * .9, rect.top() + 1.2))
        elif frosted:                                                      # reduced transparency: opaque
            painter.setBrush(QColor('#ffffff' if self.kind == 'strong' else t['well']))
            painter.setPen(QPen(hairline2, 1))
            painter.drawRoundedRect(rect, radius, radius)
        elif self.kind == 'well':
            painter.setBrush(QColor(t['well'])); painter.setPen(QPen(hairline, 1))
            painter.drawRoundedRect(rect, radius, radius)
        else:                                                              # segment
            painter.setBrush(QColor(16, 36, 26, 15)); painter.setPen(QPen(hairline, 1))
            painter.drawRoundedRect(rect, radius, radius)
        painter.end()


# ── animated pills and tool buttons ──────────────────────────────────

class _Feedback:
    """Hover and press as 0..1 values that ease over 120 ms. Only colour changes with them."""
    def _init_feedback(self):
        self._hover = 0.0
        self._press = 0.0
        self._tweens = {}

    def _tween(self, name: str, target: float):
        old = self._tweens.pop(name, None)
        if old is not None:
            old.stop()
        tween = QVariantAnimation(self)
        tween.setStartValue(float(getattr(self, name)))
        tween.setEndValue(float(target))
        tween.setDuration(motion_ms(120))
        tween.setEasingCurve(QEasingCurve.Type.InOutCubic)
        tween.valueChanged.connect(lambda value, n=name: (setattr(self, n, float(value)), self.update()))
        self._tweens[name] = tween
        tween.start()

    def enterEvent(self, event):
        self._tween('_hover', 1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._tween('_hover', 0.0)
        self._tween('_press', 0.0)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._tween('_press', 1.0)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._tween('_press', 0.0)
        super().mouseReleaseEvent(event)

    def focusInEvent(self, event):
        self._focus_reason = event.reason()
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.update()

    def _keyboard_focus(self) -> bool:
        """The focus ring is for keyboard users: it is not drawn after a mouse click."""
        return self.hasFocus() and getattr(self, '_focus_reason', None) in (
            Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason, Qt.FocusReason.ShortcutFocusReason)


def _mix(a: QColor, b: QColor, amount: float) -> QColor:
    amount = max(0.0, min(1.0, amount))
    return QColor(round(a.red() + (b.red() - a.red()) * amount), round(a.green() + (b.green() - a.green()) * amount),
                  round(a.blue() + (b.blue() - a.blue()) * amount), round(a.alpha() + (b.alpha() - a.alpha()) * amount))


class GlassButton(_Feedback, QPushButton):
    """A pill. Variants: default, primary (the one green action in a view), link, round."""
    def __init__(self, text: str = '', variant: str = 'default', parent=None):
        super().__init__(text, parent)
        self._init_feedback()
        self.variant = variant
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight({'link': 30, 'round': 30}.get(variant, 44))
        if variant == 'round':
            self.setFixedSize(30, 30)
        font = QFont(self.font()); font.setPixelSize(14); font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)

    def sizeHint(self):
        metrics = QFontMetrics(self.font())
        icon = 22 if not self.icon().isNull() else 0
        if self.variant == 'round':
            return QSize(30, 30)
        width = metrics.horizontalAdvance(self.text()) + icon + (8 if self.variant == 'link' else 36)
        return QSize(max(44, width), self.minimumHeight())

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        radius = rect.height() / 2
        enabled = self.isEnabled()
        hover, press = (self._hover, self._press) if enabled else (0.0, 0.0)
        text_colour = QColor(t['text'])
        if self.variant == 'link':
            text_colour = _mix(QColor(t['accent']), QColor(t['brand_deep']), hover) if enabled else QColor(t['muted'])
        elif self.variant == 'primary' and enabled:
            deep, green = QColor(t['brand_deep']), QColor(t['brand_green'])
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, _mix(green.lighter(112), green.lighter(128), hover))
            gradient.setColorAt(1, _mix(green, green.darker(112), hover))
            painter.setBrush(QBrush(gradient)); painter.setPen(QPen(deep, 1))
            painter.drawRoundedRect(rect, radius, radius)
            painter.setPen(QPen(QColor(255, 255, 255, 71), 1))               # inset highlight
            painter.drawLine(QPointF(rect.left() + radius * .8, rect.top() + 1.5), QPointF(rect.right() - radius * .8, rect.top() + 1.5))
            if press:
                painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(0, 0, 0, int(34 * press)))
                painter.drawRoundedRect(rect, radius, radius)
            text_colour = QColor(t['on_accent'])
        elif self.variant != 'link':
            if enabled:
                fill = _mix(QColor(t['well']), QColor('#ffffff'), hover)
                painter.setBrush(fill); painter.setPen(QPen(_mix(QColor(16, 36, 26, 41), QColor(16, 36, 26, 77), hover), 1))
            else:
                painter.setBrush(QColor(16, 36, 26, 10)); painter.setPen(QPen(QColor(16, 36, 26, 23), 1))
                text_colour = QColor(t['muted'])
            painter.drawRoundedRect(rect, radius, radius)
            if enabled:
                painter.setPen(QPen(QColor(255, 255, 255, 235), 1))
                painter.drawLine(QPointF(rect.left() + radius * .8, rect.top() + 1.2), QPointF(rect.right() - radius * .8, rect.top() + 1.2))
                if press:
                    painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(16, 36, 26, int(26 * press)))
                    painter.drawRoundedRect(rect, radius, radius)
        # label (and icon)
        painter.setPen(text_colour)
        painter.setFont(self.font())
        label_rect = QRectF(self.rect())
        if not self.icon().isNull():
            size = 18
            metrics = QFontMetrics(self.font())
            total = size + (6 + metrics.horizontalAdvance(self.text()) if self.text() else 0)
            left = label_rect.center().x() - total / 2
            icon_mode = QIcon.Mode.Disabled if not enabled else QIcon.Mode.Active if self.variant == 'primary' else QIcon.Mode.Normal
            self.icon().paint(painter, int(left), int(label_rect.center().y() - size / 2), size, size,
                              Qt.AlignmentFlag.AlignCenter, icon_mode)
            label_rect = QRectF(left + size + 6, label_rect.top(), label_rect.width(), label_rect.height())
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        else:
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self.text())
        if self.variant == 'link' and hover > 0.05:
            metrics = QFontMetrics(self.font())
            width = metrics.horizontalAdvance(self.text())
            y = label_rect.center().y() + metrics.ascent() / 2 + 2
            painter.setPen(QPen(_rgba((text_colour.red(), text_colour.green(), text_colour.blue(), 255), hover), 1))
            painter.drawLine(QPointF(label_rect.center().x() - width / 2, y), QPointF(label_rect.center().x() + width / 2, y))
        if self._keyboard_focus() and enabled:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(t['accent']), 2))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), radius + .5, radius + .5)
        painter.end()


class GlassToolButton(_Feedback, QToolButton):
    """A symbol button for grouped controls. Checked = the raised white pill with an accent symbol."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_feedback()
        self.setMinimumSize(40, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self):
        return QSize(max(40, super().sizeHint().width()), 36)

    def paintEvent(self, event):
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        radius = rect.height() / 2
        enabled = self.isEnabled()
        hover, press = (self._hover, self._press) if enabled else (0.0, 0.0)
        painter.setPen(Qt.PenStyle.NoPen)
        if self.isChecked() and enabled:
            painter.setBrush(QColor('#ffffff')); painter.setPen(QPen(QColor(15, 79, 52, 30), 1))
            painter.drawRoundedRect(rect, radius, radius)
        if hover or press:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, int(178 * hover)) if not self.isChecked() else QColor(16, 36, 26, int(14 * hover)))
            painter.drawRoundedRect(rect, radius, radius)
            if press:
                painter.setBrush(QColor(16, 36, 26, int(22 * press)))
                painter.drawRoundedRect(rect, radius, radius)
        size = 20
        mode = QIcon.Mode.Normal if enabled else QIcon.Mode.Disabled
        state = QIcon.State.On if self.isChecked() else QIcon.State.Off
        self.icon().paint(painter, int(rect.center().x() - size / 2 + .5), int(rect.center().y() - size / 2 + .5), size, size,
                          Qt.AlignmentFlag.AlignCenter, mode, state)
        if self._keyboard_focus() and enabled:
            painter.setBrush(Qt.BrushStyle.NoBrush); painter.setPen(QPen(QColor(t['accent']), 2))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), radius, radius)
        painter.end()


# ── factories (plain widgets under the macOS look) ───────────────────

def push_button(text: str = '', variant: str = 'default', tip: str = '') -> QPushButton:
    if is_glass():
        button = GlassButton(text, variant)
    else:
        button = QPushButton(text)
        if variant in ('primary', 'link', 'round'):
            button.setObjectName(variant)
    if tip:
        button.setToolTip(tip)
    return button


def tool_button() -> QToolButton:
    return GlassToolButton() if is_glass() else QToolButton()


def surface(kind: str = 'glass', name: str = '', radius=None, shadow: bool = False) -> Surface:
    return Surface(kind, name, radius, shadow)


# ── page head ────────────────────────────────────────────────────────

class PageHead(QWidget):
    """Eyebrow, title and one sentence, with the page's actions on the right (at most one primary)."""
    def __init__(self, eyebrow: str, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(12)
        text = QVBoxLayout(); text.setSpacing(2)
        self.eyebrow = QLabel(eyebrow.upper()); self.eyebrow.setObjectName('eyebrow')
        font = QFont(self.eyebrow.font()); font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.1)
        self.eyebrow.setFont(font)
        self.title = QLabel(title); self.title.setObjectName('pageTitle')
        title_font = QFont(self.title.font()); title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -.4)
        self.title.setFont(title_font)
        self.subtitle = QLabel(subtitle); self.subtitle.setObjectName('pageSub'); self.subtitle.setWordWrap(True)
        for label in (self.eyebrow, self.title, self.subtitle):
            text.addWidget(label)
        row.addLayout(text, 1)
        self.actions = QHBoxLayout(); self.actions.setSpacing(8)
        self.actions.setAlignment(Qt.AlignmentFlag.AlignTop)       # level with the title, not stretched to the head
        self.actions.setContentsMargins(0, 10, 0, 0)
        row.addLayout(self.actions)

    def set_text(self, eyebrow: str, title: str, subtitle: str):
        self.eyebrow.setText(eyebrow.upper()); self.title.setText(title); self.subtitle.setText(subtitle)


# ── motion ───────────────────────────────────────────────────────────

REVEAL_MS = 240
MAX_STAGGERED = 6        # the kit never staggers more than six items
STAGGER_MS = 28
MAX_STAGGER_DELAY_MS = 150


def reveal(widgets, stagger: bool = True) -> list:
    """Fade widgets in (opacity only, ease-out), the first few slightly staggered. Reduced motion
    keeps a short fade and drops the stagger. Returns the animations."""
    animations = []
    for index, widget in enumerate(w for w in widgets if w is not None):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b'opacity', widget)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(motion_ms(REVEAL_MS))
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda w=widget: w.setGraphicsEffect(None))
        delay = 0
        if stagger and index < MAX_STAGGERED and motion_ms(REVEAL_MS) == REVEAL_MS:
            delay = min(index * STAGGER_MS, MAX_STAGGER_DELAY_MS)
        widget._reveal = animation
        if delay:
            QTimer.singleShot(delay, animation.start)
        else:
            animation.start()
        animations.append(animation)
    return animations
