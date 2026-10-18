"""Primary window: the library of saved computers, laid out like macOS Screen
Sharing's (sidebar, symbol toolbar, striped rows with an ⓘ button, grid view)."""
from __future__ import annotations
import time
from typing import Iterable

from PySide6.QtCore import Qt, Signal, QSize, QRect, QRectF, QPoint, QPointF, QEvent
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QFont, QIcon, QKeySequence, QLinearGradient, QPainter,
                           QPainterPath, QPen, QShortcut)
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame,
    QStackedWidget, QListWidget, QListWidgetItem, QStyledItemDelegate, QStyle, QMenu, QAbstractItemView, QToolButton)

from .icons import glyph
from .library import SavedSystem, describe_last_connected
from .theme import tokens, fade_in, is_glass, label
from .widgets import FieldWidget, PageHead, push_button, reveal, surface, tool_button

ID_ROLE = Qt.ItemDataRole.UserRole
ACTIVE_ROLE = Qt.ItemDataRole.UserRole + 1
SUBTITLE_ROLE = Qt.ItemDataRole.UserRole + 2
WHEN_ROLE = Qt.ItemDataRole.UserRole + 3
ROW_HEIGHT = 54
INFO_SIZE = QSize(30, 22)


def short_when(when) -> str:
    return 'Not connected yet' if not when else describe_last_connected(when).removeprefix('Last connected ')


class RowDelegate(QStyledItemDelegate):
    """Draws one saved computer, as a list row or a grid tile."""
    info_clicked = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.grid = False

    def sizeHint(self, option, index):
        return QSize(150, 132) if self.grid else QSize(option.rect.width(), ROW_HEIGHT)

    @staticmethod
    def info_rect(row: QRect) -> QRect:
        return QRect(row.right() - INFO_SIZE.width() - 14, row.center().y() - INFO_SIZE.height() // 2,
                     INFO_SIZE.width(), INFO_SIZE.height())

    def paint(self, painter, option, index):
        t = tokens()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        name, subtitle = index.data(Qt.ItemDataRole.DisplayRole), index.data(SUBTITLE_ROLE)
        if self.grid:
            self._paint_tile(painter, option, name, subtitle, selected, t)
        else:
            self._paint_row(painter, option, index, name, subtitle, selected, t)
        painter.restore()

    def _paint_row_glass(self, painter, option, index, name, subtitle, selected, t):
        """Liquid Glass: an opaque well row, a hairline between rows, and for the selected row a soft
        green tint with the kit's lime-to-green indicator bar. Status is a pill with a word."""
        row = option.rect
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.fillRect(row, QColor(t['well']))
        if selected:
            painter.fillRect(row, QColor(18, 121, 66, 26))
            bar = QLinearGradient(QPointF(row.left(), row.top() + 9), QPointF(row.left(), row.bottom() - 9))
            bar.setColorAt(0, QColor(t['brand_lime'])); bar.setColorAt(1, QColor(t['accent']))
            path = QPainterPath(); path.addRoundedRect(QRectF(row.left(), row.top() + 9, 4, row.height() - 18), 2, 2)
            painter.fillPath(path, QBrush(bar))
        elif hover:
            painter.fillRect(row, QColor(255, 255, 255, 190))
        painter.setPen(QPen(QColor(16, 36, 26, 23), 1))
        painter.drawLine(row.left() + 16, row.bottom(), row.right() - 16, row.bottom())
        glyph('monitor').paint(painter, QRect(row.left() + 16, row.center().y() - 17, 34, 34))
        info = self.info_rect(row)
        active = index.data(ACTIVE_ROLE) or 0
        small = QFont(option.font); small.setPixelSize(12)
        painter.setFont(small)
        status = 'Connected' if active == 1 else f'{active} sessions open' if active else ''
        when = index.data(WHEN_ROLE)
        right_edge = info.left() - 10
        if status:
            pill_w = painter.fontMetrics().horizontalAdvance(status) + 30
            pill = QRectF(right_edge - pill_w, row.center().y() - 11, pill_w, 22)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(24, 138, 60, 31))
            painter.drawRoundedRect(pill, 11, 11)
            painter.setBrush(QColor(24, 138, 60)); painter.drawEllipse(QPointF(pill.left() + 12, pill.center().y()), 3.5, 3.5)
            painter.setPen(QColor(t['ok']))
            painter.drawText(QRectF(pill.left() + 21, pill.top(), pill.width() - 24, pill.height()), Qt.AlignmentFlag.AlignVCenter, status)
            right_edge = int(pill.left()) - 10
        else:
            width = painter.fontMetrics().horizontalAdvance(when) + 4
            painter.setPen(QColor(t['muted']))
            painter.drawText(QRect(right_edge - width, row.top(), width, row.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, when)
            right_edge -= width + 10
        left = row.left() + 64
        bold = QFont(option.font); bold.setPixelSize(14); bold.setWeight(QFont.Weight.DemiBold)
        painter.setFont(bold); painter.setPen(QColor(t['brand_deep'] if selected else t['text']))
        painter.drawText(QRect(left, row.top() + 8, max(40, right_edge - left), 20), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, max(40, right_edge - left)))
        painter.setFont(small); painter.setPen(QColor(t['muted']))
        painter.drawText(QRect(left, row.top() + 28, max(40, right_edge - left), 18), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(subtitle, Qt.TextElideMode.ElideRight, max(40, right_edge - left)))
        painter.setPen(QPen(QColor(16, 36, 26, 41), 1)); painter.setBrush(QColor('#ffffff'))
        painter.drawRoundedRect(QRectF(info).adjusted(.5, .5, -.5, -.5), 11, 11)
        glyph('info').paint(painter, QRect(info.center().x() - 8, info.center().y() - 8, 16, 16))

    def _paint_row(self, painter, option, index, name, subtitle, selected, t):
        if is_glass():
            self._paint_row_glass(painter, option, index, name, subtitle, selected, t)
            return
        row = option.rect
        painter.fillRect(row, QColor(t['accent'] if selected else t['stripe'] if index.row() % 2 == 0 else t['window']))
        main = QColor(t['selected_text'] if selected else t['text'])
        soft = QColor(t['selected_text'] if selected else t['muted'])
        glyph('monitor').paint(painter, QRect(row.left() + 14, row.center().y() - 17, 34, 34),
                               mode=QIcon.Mode.Selected if selected else QIcon.Mode.Normal)
        left = row.left() + 60
        info = self.info_rect(row)
        active = index.data(ACTIVE_ROLE) or 0
        when = '● Connected' if active == 1 else f'● {active} sessions open' if active else index.data(WHEN_ROLE)
        font = QFont(option.font); font.setPixelSize(12)
        painter.setFont(font)
        when_width = painter.fontMetrics().horizontalAdvance(when) + 16
        when_rect = QRect(info.left() - when_width - 4, row.top(), when_width, row.height())
        right_edge = when_rect.left() - 8
        bold = QFont(option.font); bold.setPixelSize(13); bold.setWeight(QFont.Weight.DemiBold)
        painter.setFont(bold); painter.setPen(main)
        painter.drawText(QRect(left, row.top() + 8, max(40, right_edge - left), 20), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, max(40, right_edge - left)))
        small = QFont(option.font); small.setPixelSize(11)
        painter.setFont(small); painter.setPen(soft)
        painter.drawText(QRect(left, row.top() + 28, max(40, right_edge - left), 18), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(subtitle, Qt.TextElideMode.ElideRight, max(40, right_edge - left)))
        painter.setFont(font)
        painter.setPen(QColor(t['selected_text'] if selected else t['ok']) if active else soft)
        painter.drawText(when_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, when)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 40) if selected else QColor(t['group']))
        painter.drawRoundedRect(info, 6, 6)
        glyph('info').paint(painter, QRect(info.center().x() - 8, info.center().y() - 8, 16, 16))

    def _paint_tile(self, painter, option, name, subtitle, selected, t):
        tile = option.rect.adjusted(6, 6, -6, -6)
        glass = is_glass()
        if selected:
            if glass:                                           # a soft green tint with a green edge
                painter.setPen(QPen(QColor(18, 121, 66, 71), 1)); painter.setBrush(QColor(18, 121, 66, 26))
            else:
                painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(t['accent']))
            painter.drawRoundedRect(QRectF(tile).adjusted(.5, .5, -.5, -.5), 14 if glass else 10, 14 if glass else 10)
        main = QColor(t['brand_deep'] if glass and selected else t['selected_text'] if selected else t['text'])
        soft = QColor(t['muted'] if glass else t['selected_text'] if selected else t['muted'])
        glyph('monitor').paint(painter, QRect(tile.center().x() - 30, tile.top() + 10, 60, 60),
                               mode=QIcon.Mode.Selected if selected else QIcon.Mode.Normal)
        bold = QFont(option.font); bold.setPixelSize(13); bold.setWeight(QFont.Weight.DemiBold)
        painter.setFont(bold); painter.setPen(main)
        painter.drawText(QRect(tile.left() + 4, tile.top() + 76, tile.width() - 8, 20), Qt.AlignmentFlag.AlignCenter,
                         painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, tile.width() - 12))
        small = QFont(option.font); small.setPixelSize(11)
        painter.setFont(small); painter.setPen(soft)
        painter.drawText(QRect(tile.left() + 4, tile.top() + 96, tile.width() - 8, 18), Qt.AlignmentFlag.AlignCenter,
                         painter.fontMetrics().elidedText(subtitle, Qt.TextElideMode.ElideRight, tile.width() - 12))

    def editorEvent(self, event, model, option, index):
        if not self.grid and event.type() == QEvent.Type.MouseButtonRelease \
                and self.info_rect(option.rect).contains(event.position().toPoint()):
            self.info_clicked.emit(index.data(ID_ROLE))
            return True
        return super().editorEvent(event, model, option, index)


class NavDelegate(QStyledItemDelegate):
    """The glass rail's navigation items: a raised white pill for the active one, with the
    lime-to-green indicator bar on the rail's edge; a soft white wash on hover."""
    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 46)

    def paint(self, painter, option, index):
        t = tokens()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        box = QRectF(option.rect).adjusted(2, 2, -2, -2)
        if selected:
            gradient = QLinearGradient(box.topLeft(), box.bottomLeft())
            gradient.setColorAt(0, QColor(255, 255, 255, 242)); gradient.setColorAt(1, QColor(255, 255, 255, 199))
            painter.setBrush(QBrush(gradient)); painter.setPen(QPen(QColor(15, 79, 52, 28), 1))
            painter.drawRoundedRect(box, 12, 12)
            painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
            painter.drawLine(QPointF(box.left() + 10, box.top() + 1.2), QPointF(box.right() - 10, box.top() + 1.2))
            bar = QLinearGradient(QPointF(0, box.top() + 8), QPointF(0, box.bottom() - 8))
            bar.setColorAt(0, QColor(t['brand_lime'])); bar.setColorAt(1, QColor(t['accent']))
            path = QPainterPath(); path.addRoundedRect(QRectF(option.rect.left() - 8, box.top() + 8, 3, box.height() - 16), 1.5, 1.5)
            painter.fillPath(path, QBrush(bar))
        elif hover:
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(255, 255, 255, 184))
            painter.drawRoundedRect(box, 12, 12)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon):
            icon.paint(painter, QRect(int(box.left()) + 12, int(box.center().y()) - 9, 18, 18), Qt.AlignmentFlag.AlignCenter,
                       QIcon.Mode.Selected if selected else QIcon.Mode.Normal)
        font = QFont(option.font); font.setPixelSize(14); font.setWeight(QFont.Weight.Bold if selected else QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor(t['brand_deep'] if selected else t['ink2']))
        painter.drawText(QRectF(box.left() + 42, box.top(), box.width() - 48, box.height()), Qt.AlignmentFlag.AlignVCenter,
                         index.data(Qt.ItemDataRole.DisplayRole))
        painter.restore()


class LibraryList(QListWidget):
    """A list that keeps the row stripes going below the last item, like Finder
    (under Liquid Glass it sits on an opaque well instead, with no stripes)."""
    def paintEvent(self, event):
        if is_glass():
            painter = QPainter(self.viewport())
            painter.fillRect(self.viewport().rect(), QColor(tokens()['well']))
            painter.end()
        elif self.viewMode() == QListWidget.ViewMode.ListMode:
            painter = QPainter(self.viewport())
            t = tokens()
            row, y = 0, 0
            while y < self.viewport().height():
                painter.fillRect(0, y, self.viewport().width(), ROW_HEIGHT, QColor(t['stripe'] if row % 2 == 0 else t['window']))
                row, y = row + 1, y + ROW_HEIGHT
            painter.end()
        super().paintEvent(event)


class LibraryWindow(QMainWindow):
    new_session = Signal()
    open_system = Signal(str)
    edit_system = Signal(str)
    rename_system = Signal(str)
    forget_password = Signal(str)
    remove_system = Signal(str)
    hp_toggled = Signal(bool)
    quit_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('iShareScreen')
        self.resize(900, 580)
        self.setMinimumSize(620, 420)
        self.systems: list[SavedSystem] = []
        self.active: dict[str, int] = {}
        self.sort_key = 'last'
        self.only_connected = False
        glass = is_glass()
        root = FieldWidget(); root.setObjectName('root'); self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(*(12, 12, 12, 12) if glass else (0, 0, 0, 0)); outer.setSpacing(12 if glass else 0)

        # ── sidebar (under Liquid Glass: a floating glass rail with the brand on top) ──
        self.sidebar = surface('rail', 'sidebar', shadow=True); self.sidebar.setFixedWidth(224 if glass else 200)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(*(10, 12, 10, 12) if glass else (10, 14, 10, 10)); side.setSpacing(4)
        self.sidebar_button = self._tool('sidebar', 'Show or hide the sidebar', checkable=True)
        self.sidebar_button.setChecked(True); self.sidebar_button.toggled.connect(self.sidebar.setVisible)
        if glass:
            brand = QHBoxLayout(); brand.setContentsMargins(8, 4, 0, 8); brand.setSpacing(10)
            mark = QLabel(); mark.setPixmap(glyph('monitor').pixmap(30, 30)); mark.setFixedSize(30, 30)
            names = QVBoxLayout(); names.setSpacing(0)
            product = QLabel('iShareScreen'); product.setObjectName('heading')
            tag = QLabel('SCREEN SHARING'); tag.setObjectName('eyebrow')
            tag_font = QFont(tag.font()); tag_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0); tag.setFont(tag_font)
            names.addWidget(product); names.addWidget(tag)
            brand.addWidget(mark); brand.addLayout(names, 1)
            side.addLayout(brand)
        self.side_list = QListWidget(); self.side_list.setObjectName('library'); self.side_list.setAccessibleName('Library sections')
        self.side_list.setFrameShape(QFrame.Shape.NoFrame)
        if glass:
            self.side_list.setItemDelegate(NavDelegate(self.side_list)); self.side_list.setMouseTracking(True)
        for text, icon in ((label('All Connections', 'All connections'), 'monitor'), ('Connected', 'link')):
            item = QListWidgetItem(glyph(icon), text); item.setSizeHint(QSize(160, 44 if glass else 34)); self.side_list.addItem(item)
        self.side_list.setCurrentRow(0)
        self.side_list.currentRowChanged.connect(self._section_changed)
        side.addWidget(self.side_list, 1)
        foot = QHBoxLayout(); foot.setSpacing(2)
        self.side_add = self._tool('plus', 'New connection (Ctrl+N)'); self.side_add.clicked.connect(self.new_session)
        foot.addWidget(self.side_add); foot.addStretch(1)
        if glass:
            foot.addWidget(self.sidebar_button)
        side.addLayout(foot)
        outer.addWidget(self.sidebar)

        # ── content ──
        content = QWidget(); column = QVBoxLayout(content); column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(16 if glass else 0)
        self.search = QLineEdit(); self.search.setPlaceholderText('Search'); self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName('Search computers'); self.search.setMaximumWidth(220); self.search.hide()
        self.search.textChanged.connect(self.refresh)
        self.grid_button = self._tool('grid', label('As Icons', 'As icons'), checkable=True)
        self.list_button = self._tool('list', label('As List', 'As list'), checkable=True); self.list_button.setChecked(True)
        views = surface('segment', 'segment'); row = QHBoxLayout(views)
        row.setContentsMargins(3, 2, 3, 2); row.setSpacing(1); row.addWidget(self.grid_button); row.addWidget(self.list_button)
        if glass:
            views.setFixedHeight(44)
        self.grid_button.clicked.connect(lambda: self.set_view('grid')); self.list_button.clicked.connect(lambda: self.set_view('list'))
        self.more = tool_button(); self.more.setIcon(glyph('more')); self.more.setToolTip('More'); self.more.setAccessibleName('More')
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more)
        sorts = QActionGroup(menu)
        self.sort_name = QAction(label('Sort by Name', 'Sort by name'), menu, checkable=True)
        self.sort_last = QAction(label('Sort by Last Connected', 'Sort by last connected'), menu, checkable=True)
        for action, key in ((self.sort_last, 'last'), (self.sort_name, 'name')):
            sorts.addAction(action); menu.addAction(action); action.triggered.connect(lambda _=False, k=key: self.set_sort(k))
        self.sort_last.setChecked(True)
        menu.addSeparator()
        self.hp_action = QAction('High Performance (Experimental)', menu, checkable=True)
        self.hp_action.setToolTip('Offer High Performance when connecting. It did not work with every Mac in testing.')
        self.hp_action.triggered.connect(self.hp_toggled)
        menu.addAction(self.hp_action)
        self.more.setMenu(menu)
        if glass:                                   # one primary action per view: a green pill with the ＋ symbol
            self.new_button = push_button('', 'primary', tip='New connection (Ctrl+N)')
            self.new_button.setIcon(glyph('plus')); self.new_button.setAccessibleName('New connection')
            self.new_button.clicked.connect(self.new_session)
        else:
            self.new_button = self._tool('plus', 'New session (Ctrl+N)'); self.new_button.clicked.connect(self.new_session)
        self.search_button = self._tool('search', 'Search (Ctrl+F)'); self.search_button.clicked.connect(self.toggle_search)
        if glass:
            self.head = PageHead('Library', 'All connections', 'Your Macs and other VNC computers. Double-click one to connect.')
            self.title = self.head.title
            for widget in (self.search, views, self.more, self.search_button, self.new_button):
                self.head.actions.addWidget(widget)
            column.addWidget(self.head)
        else:
            header = surface('well', 'header', radius=0); bar = QHBoxLayout(header)
            bar.setContentsMargins(12, 8, 12, 8); bar.setSpacing(8)
            self.title = QLabel('All Connections'); self.title.setObjectName('title')
            for widget in (self.sidebar_button, self.title):
                bar.addWidget(widget)
            bar.addStretch(1)
            for widget in (self.search, views, self.more, self.new_button, self.search_button):
                bar.addWidget(widget)
            column.addWidget(header)
        self.stack = QStackedWidget()
        if glass:                                   # a frosted card holding an opaque well for the list
            self.card = surface('glass', 'card', shadow=True)
            card_layout = QVBoxLayout(self.card); card_layout.setContentsMargins(14, 14, 14, 14)
            self.well = surface('well', 'well')
            well_layout = QVBoxLayout(self.well); well_layout.setContentsMargins(6, 6, 6, 6)
            well_layout.addWidget(self.stack)
            card_layout.addWidget(self.well)
            column.addWidget(self.card, 1)
        else:
            column.addWidget(self.stack, 1)
        outer.addWidget(content, 1)

        empty = QWidget(); box = QVBoxLayout(empty); box.setAlignment(Qt.AlignmentFlag.AlignCenter); box.setSpacing(12)
        title = QLabel('No saved computers yet'); title.setObjectName('emptyTitle'); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body = QLabel('Connect to a Mac that has Screen Sharing turned on. Once you connect successfully, '
                      'it is saved here so you can reopen it in two clicks. A password is saved only if you ask for it.')
        body.setObjectName('emptyBody'); body.setWordWrap(True); body.setAlignment(Qt.AlignmentFlag.AlignCenter); body.setFixedWidth(430)
        start = push_button(label('New session', 'New connection'), 'primary'); start.clicked.connect(self.new_session)
        for widget in (title, body):
            box.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        box.addSpacing(6); box.addWidget(start, 0, Qt.AlignmentFlag.AlignHCenter)
        self.stack.addWidget(empty)

        listing = QWidget(); inner = QVBoxLayout(listing); inner.setContentsMargins(0, 0, 0, 0)
        self.list = LibraryList(); self.list.setObjectName('library')
        self.delegate = RowDelegate(self.list); self.delegate.info_clicked.connect(self.edit_system)
        self.list.setItemDelegate(self.delegate)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setMouseTracking(True); self.list.setAccessibleName('Saved computers'); self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.itemActivated.connect(lambda item: self.open_system.emit(item.data(ID_ROLE)))
        inner.addWidget(self.list)
        self.no_match = QLabel(); self.no_match.setObjectName('emptyBody'); self.no_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self.no_match)
        self.stack.addWidget(listing)
        self.set_view('list')

        QShortcut(QKeySequence('Ctrl+N'), self).activated.connect(self.new_session)
        QShortcut(QKeySequence('Ctrl+F'), self).activated.connect(self.toggle_search)
        QShortcut(QKeySequence('Ctrl+Q'), self).activated.connect(self.quit_requested)
        for keys, signal in (('F2', self.rename_system), ('Ctrl+E', self.edit_system), ('Delete', self.remove_system)):
            shortcut = QShortcut(QKeySequence(keys), self.list)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)   # never steal Delete from the search box
            shortcut.activated.connect(lambda signal=signal: self._with_selected(signal))
        self.refresh()

    @staticmethod
    def _tool(icon, tip, checkable=False):
        button = tool_button(); button.setIcon(glyph(icon)); button.setToolTip(tip); button.setAccessibleName(tip)
        button.setCheckable(checkable)
        return button

    # ── content ──────────────────────────────────────────────────────
    def set_systems(self, systems: Iterable[SavedSystem]):
        self.systems = list(systems)
        self.refresh()

    def set_active(self, active: dict):
        self.active = dict(active)
        self.refresh()

    def set_view(self, view: str):
        grid = view == 'grid'
        self.delegate.grid = grid
        self.grid_button.setChecked(grid); self.list_button.setChecked(not grid)
        self.list.setViewMode(QListWidget.ViewMode.IconMode if grid else QListWidget.ViewMode.ListMode)
        if grid:
            self.list.setResizeMode(QListWidget.ResizeMode.Adjust); self.list.setMovement(QListWidget.Movement.Static)
            self.list.setWrapping(True); self.list.setSpacing(4); self.list.setGridSize(QSize(150, 132))
        else:
            self.list.setWrapping(False); self.list.setSpacing(0); self.list.setGridSize(QSize())
        self.refresh()                      # rebuild the items so they take the new size
        self.list.doItemsLayout(); self.list.viewport().update()

    def set_hp_checked(self, on: bool):
        """Show the Experimental switch's state without announcing it as a user choice."""
        self.hp_action.blockSignals(True)
        self.hp_action.setChecked(on)
        self.hp_action.blockSignals(False)

    def set_sort(self, key: str):
        self.sort_key = key
        self.sort_name.setChecked(key == 'name'); self.sort_last.setChecked(key == 'last')
        self.refresh()

    def _section_changed(self, row: int):
        self.only_connected = row == 1
        if is_glass():
            self.head.set_text('Library', 'Connected' if self.only_connected else 'All connections',
                               'Sessions that are open right now.' if self.only_connected
                               else 'Your Macs and other VNC computers. Double-click one to connect.')
        else:
            self.title.setText('Connected' if self.only_connected else 'All Connections')
        self.refresh()
        if is_glass() and self.isVisible():
            reveal([self.well], stagger=False)                  # a short fade between views

    def toggle_search(self):
        """The ⌕ button: show the search field; with text in it, refocus; empty, hide it again."""
        if not self.search.isVisible():
            self.search.show(); self.search.setFocus()
        elif self.search.text():
            self.search.setFocus(); self.search.selectAll()
        else:
            self.search.hide()

    def refresh(self, *_):
        query = self.search.text().strip().lower()
        selected = self._selected_id()
        self.search_button.setEnabled(bool(self.systems))
        self.stack.setCurrentIndex(1 if self.systems else 0)
        self.list.clear()
        order = (lambda s: s.title.lower()) if self.sort_key == 'name' else (lambda s: (-(s.last_connected or 0), s.title.lower()))
        shown = [s for s in sorted(self.systems, key=order)
                 if (not query or query in f'{s.title} {s.host} {s.user}'.lower())
                 and (not self.only_connected or self.active.get(s.id))]
        for system in shown:
            item = QListWidgetItem(system.title)
            subtitle = system.address + (f' · {system.user}' if system.user else '')
            if system.mode == 'high_performance':
                subtitle += ' · High Performance'
            if system.password_saved:
                subtitle += ' · password saved'
            when = short_when(system.last_connected)
            active = self.active.get(system.id, 0)
            item.setData(ID_ROLE, system.id); item.setData(ACTIVE_ROLE, active)
            item.setData(SUBTITLE_ROLE, subtitle); item.setData(WHEN_ROLE, when)
            item.setData(Qt.ItemDataRole.AccessibleTextRole,
                         f'{system.title}, {subtitle}, {"connected, " if active else ""}{when}')
            item.setToolTip(f'{system.title}\n{subtitle}')
            item.setSizeHint(QSize(150, 132) if self.delegate.grid else QSize(100, ROW_HEIGHT))
            self.list.addItem(item)
            if system.id == selected:
                self.list.setCurrentItem(item)
        if shown and self.list.currentRow() < 0:
            self.list.setCurrentRow(0)
        empty_note = ''
        if self.systems and not shown:
            empty_note = (f'No computers match “{self.search.text().strip()}”.' if query
                          else 'No sessions are open.' if self.only_connected else '')
        self.no_match.setVisible(bool(empty_note)); self.no_match.setText(empty_note)

    def _selected_id(self):
        item = self.list.currentItem() if hasattr(self, 'list') else None
        return item.data(ID_ROLE) if item else None

    def _with_selected(self, signal):
        selected = self._selected_id()
        if selected:
            signal.emit(selected)

    def _menu(self, point: QPoint):
        item = self.list.itemAt(point)
        if item is None:
            return
        self.list.setCurrentItem(item)
        self.build_menu(item.data(ID_ROLE)).exec(self.list.viewport().mapToGlobal(point))

    def build_menu(self, system_id: str) -> QMenu:
        """The context menu for one saved computer (separate so it can be inspected)."""
        system = next((s for s in self.systems if s.id == system_id), None)
        menu = QMenu(self)
        menu.addAction('Connect', lambda: self.open_system.emit(system_id))
        menu.addAction('Edit…', lambda: self.edit_system.emit(system_id)).setShortcut(QKeySequence('Ctrl+E'))
        menu.addAction('Rename…', lambda: self.rename_system.emit(system_id))
        if system and system.password_saved:
            menu.addAction('Forget saved password', lambda: self.forget_password.emit(system_id))
        menu.addSeparator()
        menu.addAction('Remove from library…', lambda: self.remove_system.emit(system_id))
        return menu

    # ── window events ────────────────────────────────────────────────
    def showEvent(self, event):
        first = not getattr(self, '_shown', False)
        self._shown = True
        super().showEvent(event)
        if first:
            fade_in(self)
            if is_glass():                  # the rail, the page head and the card arrive one after another
                reveal([self.sidebar, self.head, self.card])

    def closeEvent(self, event):
        event.ignore()                  # the app decides, since sessions may be open
        self.quit_requested.emit()
