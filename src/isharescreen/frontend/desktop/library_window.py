"""Primary window: the library of saved computers, laid out like macOS Screen
Sharing's (sidebar, symbol toolbar, striped rows with an ⓘ button, grid view)."""
from __future__ import annotations
import time
from typing import Iterable

from PySide6.QtCore import Qt, Signal, QSize, QRect, QRectF, QPoint, QEvent
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame,
    QStackedWidget, QListWidget, QListWidgetItem, QStyledItemDelegate, QStyle, QMenu, QAbstractItemView, QToolButton)

from .icons import glyph
from .library import SavedSystem, describe_last_connected
from .theme import tokens, fade_in

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

    def _paint_row(self, painter, option, index, name, subtitle, selected, t):
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
        if selected:
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(t['accent']))
            painter.drawRoundedRect(tile, 10, 10)
        main = QColor(t['selected_text'] if selected else t['text'])
        soft = QColor(t['selected_text'] if selected else t['muted'])
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


class LibraryList(QListWidget):
    """A list that keeps the row stripes going below the last item, like Finder."""
    def paintEvent(self, event):
        if self.viewMode() == QListWidget.ViewMode.ListMode:
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
        root = QWidget(); root.setObjectName('root'); self.setCentralWidget(root)
        outer = QHBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # ── sidebar ──
        self.sidebar = QFrame(); self.sidebar.setObjectName('sidebar'); self.sidebar.setFixedWidth(200)
        side = QVBoxLayout(self.sidebar); side.setContentsMargins(10, 14, 10, 10); side.setSpacing(4)
        self.side_list = QListWidget(); self.side_list.setObjectName('library'); self.side_list.setAccessibleName('Library sections')
        self.side_list.setFrameShape(QFrame.Shape.NoFrame)
        for text, icon in (('All Connections', 'monitor'), ('Connected', 'link')):
            item = QListWidgetItem(glyph(icon), text); item.setSizeHint(QSize(160, 34)); self.side_list.addItem(item)
        self.side_list.setCurrentRow(0)
        self.side_list.currentRowChanged.connect(self._section_changed)
        side.addWidget(self.side_list, 1)
        foot = QHBoxLayout(); foot.setSpacing(2)
        self.side_add = self._tool('plus', 'New session (Ctrl+N)'); self.side_add.clicked.connect(self.new_session)
        foot.addWidget(self.side_add); foot.addStretch(1)
        side.addLayout(foot)
        outer.addWidget(self.sidebar)

        # ── content ──
        content = QWidget(); column = QVBoxLayout(content); column.setContentsMargins(0, 0, 0, 0); column.setSpacing(0)
        header = QFrame(); header.setObjectName('header'); bar = QHBoxLayout(header)
        bar.setContentsMargins(12, 8, 12, 8); bar.setSpacing(8)
        self.sidebar_button = self._tool('sidebar', 'Show or hide the sidebar', checkable=True)
        self.sidebar_button.setChecked(True); self.sidebar_button.toggled.connect(self.sidebar.setVisible)
        bar.addWidget(self.sidebar_button)
        self.title = QLabel('All Connections'); self.title.setObjectName('title'); bar.addWidget(self.title)
        bar.addStretch(1)
        self.search = QLineEdit(); self.search.setPlaceholderText('Search'); self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName('Search computers'); self.search.setMaximumWidth(220); self.search.hide()
        self.search.textChanged.connect(self.refresh); bar.addWidget(self.search)
        self.grid_button = self._tool('grid', 'As Icons', checkable=True)
        self.list_button = self._tool('list', 'As List', checkable=True); self.list_button.setChecked(True)
        views = QFrame(); views.setObjectName('segment'); row = QHBoxLayout(views)
        row.setContentsMargins(3, 2, 3, 2); row.setSpacing(1); row.addWidget(self.grid_button); row.addWidget(self.list_button)
        bar.addWidget(views)
        self.grid_button.clicked.connect(lambda: self.set_view('grid')); self.list_button.clicked.connect(lambda: self.set_view('list'))
        self.more = QToolButton(); self.more.setIcon(glyph('more')); self.more.setToolTip('More'); self.more.setAccessibleName('More')
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more)
        sorts = QActionGroup(menu)
        self.sort_name = QAction('Sort by Name', menu, checkable=True); self.sort_last = QAction('Sort by Last Connected', menu, checkable=True)
        for action, key in ((self.sort_last, 'last'), (self.sort_name, 'name')):
            sorts.addAction(action); menu.addAction(action); action.triggered.connect(lambda _=False, k=key: self.set_sort(k))
        self.sort_last.setChecked(True)
        menu.addSeparator()
        self.hp_action = QAction('High Performance (Experimental)', menu, checkable=True)
        self.hp_action.setToolTip('Offer High Performance when connecting. It did not work with every Mac in testing.')
        self.hp_action.triggered.connect(self.hp_toggled)
        menu.addAction(self.hp_action)
        self.more.setMenu(menu); bar.addWidget(self.more)
        self.new_button = self._tool('plus', 'New session (Ctrl+N)'); self.new_button.clicked.connect(self.new_session)
        bar.addWidget(self.new_button)
        self.search_button = self._tool('search', 'Search (Ctrl+F)'); self.search_button.clicked.connect(self.toggle_search)
        bar.addWidget(self.search_button)
        column.addWidget(header)
        self.stack = QStackedWidget(); column.addWidget(self.stack, 1)
        outer.addWidget(content, 1)

        empty = QWidget(); box = QVBoxLayout(empty); box.setAlignment(Qt.AlignmentFlag.AlignCenter); box.setSpacing(12)
        title = QLabel('No saved computers yet'); title.setObjectName('emptyTitle'); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body = QLabel('Connect to a Mac that has Screen Sharing turned on. Once you connect successfully, '
                      'it is saved here so you can reopen it in two clicks. A password is saved only if you ask for it.')
        body.setObjectName('emptyBody'); body.setWordWrap(True); body.setAlignment(Qt.AlignmentFlag.AlignCenter); body.setFixedWidth(430)
        start = QPushButton('New session'); start.setObjectName('primary'); start.clicked.connect(self.new_session)
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
        button = QToolButton(); button.setIcon(glyph(icon)); button.setToolTip(tip); button.setAccessibleName(tip)
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
        self.title.setText('Connected' if self.only_connected else 'All Connections')
        self.refresh()

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

    def closeEvent(self, event):
        event.ignore()                  # the app decides, since sessions may be open
        self.quit_requested.emit()
