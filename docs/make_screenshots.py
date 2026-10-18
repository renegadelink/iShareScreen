"""Regenerate the README screenshots from the real windows, with made-up data.

    python docs/make_screenshots.py

Uses a local fake server and a drawn desktop picture, so no real computer, account or
desktop ever appears. Set QT_QPA_FONTDIR to your fonts folder if text renders as boxes
(for example C:\\Windows\\Fonts on Windows).
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['ISS_REDUCE_MOTION'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from PySide6.QtCore import Qt, QRectF            # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication       # noqa: E402

from fake_vnc import FakeVNC                     # noqa: E402
from isharescreen.frontend.desktop import theme  # noqa: E402
from isharescreen.frontend.desktop.credentials import MemoryCredentialStore  # noqa: E402
from isharescreen.frontend.desktop.edit_dialog import EditSystemDialog        # noqa: E402
from isharescreen.frontend.desktop.library import LibraryStore                # noqa: E402
from isharescreen.frontend.desktop.settings import Settings                   # noqa: E402
from isharescreen.frontend.desktop.standard_app import AppController          # noqa: E402

OUT = ROOT / 'docs' / 'screenshots'


def desktop_picture(width=1440, height=810) -> QImage:
    """A drawn stand-in for a Mac desktop: wallpaper, menu bar, two windows, a dock."""
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    sky = QLinearGradient(0, 0, width, height)
    sky.setColorAt(0, QColor('#3b4a8c')); sky.setColorAt(.55, QColor('#8a5fa8')); sky.setColorAt(1, QColor('#e58c7c'))
    p.fillRect(0, 0, width, height, sky)
    p.fillRect(0, 0, width, 26, QColor(20, 20, 24, 170))
    p.setPen(QColor('#f4f4f6')); p.setFont(QFont('Segoe UI', 11, QFont.Weight.DemiBold))
    p.drawText(14, 18, '  Finder     File     Edit     View     Go     Window')
    p.drawText(width - 150, 18, 'Mon 9:41 AM')

    def window(x, y, w, h, title, lines):
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 60)); p.drawRoundedRect(QRectF(x + 4, y + 8, w, h), 12, 12)
        p.setBrush(QColor('#fbfbfc')); p.drawRoundedRect(QRectF(x, y, w, h), 12, 12)
        p.setBrush(QColor('#ececef')); p.drawRoundedRect(QRectF(x, y, w, 34), 12, 12); p.drawRect(QRectF(x, y + 20, w, 14))
        for i, colour in enumerate(('#ff5f57', '#febc2e', '#28c840')):
            p.setBrush(QColor(colour)); p.drawEllipse(QRectF(x + 14 + i * 22, y + 11, 12, 12))
        p.setPen(QColor('#555')); p.setFont(QFont('Segoe UI', 10, QFont.Weight.DemiBold))
        p.drawText(QRectF(x, y, w, 34), Qt.AlignmentFlag.AlignCenter, title)
        p.setPen(Qt.PenStyle.NoPen)
        for i, ratio in enumerate(lines):
            p.setBrush(QColor('#d9dbe3' if i % 3 else '#bcc2d6'))
            p.drawRoundedRect(QRectF(x + 24, y + 58 + i * 26, (w - 48) * ratio, 11), 5, 5)
    window(150, 90, 620, 360, 'Project Notes', (.9, .72, .85, .4, .8, .66, .78, .3, .55, .7))
    window(520, 250, 640, 400, 'Documents', (.5, .8, .65, .9, .35, .75, .6, .82, .45, .7, .3, .55))
    p.setBrush(QColor(255, 255, 255, 70)); p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(width / 2 - 260, height - 70, 520, 56), 16, 16)
    for i, colour in enumerate(('#4aa3ff', '#ffb340', '#34c759', '#ff6961', '#af82ff', '#64d2ff', '#ffd60a', '#8e8e93')):
        p.setBrush(QColor(colour)); p.drawRoundedRect(QRectF(width / 2 - 244 + i * 63, height - 62, 44, 44), 11, 11)
    p.end()
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    theme.apply_theme(app, theme.DARK)
    work = Path(tempfile.mkdtemp())
    controller = AppController(LibraryStore(work / 'library.json'), MemoryCredentialStore(), Settings(work / 's.json'))
    controller.notify = lambda text: None
    server = FakeVNC(colour=(70, 120, 200), size=(64, 36))

    def pump(condition, timeout=8):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if condition():
                return True
            time.sleep(0.01)
        return False

    def shot(widget, name):
        app.processEvents()
        widget.grab().save(str(OUT / name))
        print('wrote', OUT / name)

    store = controller.store
    now = time.time()
    store.record_success('studio-mac.local', 5900, 'alex', name='Studio Mac', when=now - 3600 * 3, password_saved=True)
    store.record_success('192.0.2.42', 5900, 'sam', name='Build Mini', when=now - 86400 * 2)
    store.record_success('conference-pc.example.net', 5900, '', name='Conference PC', when=now - 86400 * 40)
    store.record_success('192.0.2.77', 5900, 'alex', name='Design iMac', when=now - 86400 * 9)
    controller.refresh_library()
    library = controller.library
    library.resize(900, 520); library.show()
    library.list.setCurrentRow(0)
    shot(library, 'library-dark.png')
    theme.apply_theme(app, theme.LIGHT)
    library.set_view('grid'); library.list.setCurrentRow(1)
    shot(library, 'library-grid-light.png')
    library.set_view('list')

    theme.apply_theme(app, theme.DARK)
    wizard = controller.open_wizard(); wizard.show()
    shot(wizard, 'connect-to.png')
    wizard.address.setText(f'127.0.0.1:{server.port}'); wizard.check_address()
    pump(lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS)
    wizard.message.setText('Screen Sharing requires a user name and password to sign in to “studio-mac.local”.')
    wizard.username.setText('alex'); wizard.remember.setChecked(True)
    wizard._fit()
    shot(wizard, 'sign-in.png')
    wizard.username.setText('fixture-user'); wizard.password.setText('fixture-password'); wizard.connect_session()
    pump(lambda: controller.sessions)
    window = controller.sessions[0]
    pump(lambda: window.is_live)
    window.template.host = 'studio-mac.local'
    window.host = 'studio-mac.local'
    window.resize(1180, 720); window.show()
    window.screen_widget.set_image(desktop_picture())
    window.info_label.setText('studio-mac.local · 1440 × 810 · Standard VNC · 312 updates')
    window.statusBar().clearMessage()
    shot(window, 'session.png')
    window.close()

    controller.store.items[0].password_saved = True
    dialog = EditSystemDialog(None, controller.store.items[0])
    dialog.show()
    shot(dialog, 'info-sheet.png')
    dialog.close()
    controller.quit_now()
    server.close()


if __name__ == '__main__':
    main()
