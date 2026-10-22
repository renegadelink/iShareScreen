"""RVDP adapter for the existing Qt session window. Socket I/O stays off the GUI."""
from __future__ import annotations
import asyncio
import struct
import time
from PySide6.QtCore import Signal, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QToolButton
import aiohttp

from ...protocols.rvdp import wire as W
from ...protocols.rvdp.client import Framebuffer, endpoint
from .session import SessionController, SessionWindow, classify_error


class RVDPController(SessionController):
    mode_label = 'routeOS RVDP'
    pointer_only = True
    control_changed = Signal(bool)

    def __init__(self, url, port=None, user='', password='', auth='none', *, request_control=False):
        url, port = endpoint(url)
        super().__init__(url, port, '', '', auth='none')
        self.request_control = request_control
        self.control_granted = False
        self.keyboard_available = False
        self.loop = None
        self.outgoing = None
        self.ws = None
        self.task = None

    def clone(self, secret=''):
        return RVDPController(self.host, request_control=self.request_control)

    def stop(self):
        self.stopping = True
        if self.loop and self.task and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.task.cancel)

    def send(self, method, *args):
        if self.stopping or not self.loop or self.loop.is_closed():
            return
        packet = None
        if method == 'pointer' and self.control_granted:
            x, y, buttons = args
            if self.width and self.height:
                x = max(0, min(self.width - 1, int(x)))
                y = max(0, min(self.height - 1, int(y)))
                packet = (W.Type.POINTER, W.POINTER.pack(x, y, bool(buttons & 1), 0 if buttons & 1 else 2))
        elif method == 'request':
            packet = (W.Type.RESYNC, b'\x02')
        elif method == 'key' and self.control_granted and self.keyboard_available:
            key, down = args
            if W.valid_key(key):
                packet = (W.Type.KEY, W.KEY.pack(key, bool(down)))
        if packet:
            self.loop.call_soon_threadsafe(self._enqueue, packet)

    def _enqueue(self, packet):
        if self.stopping or self.outgoing is None:
            return
        if self.outgoing.full():
            # Do not lose a release and leave a held pointer on the device.
            self.failed.emit('network', 'Input queue filled; reconnect to continue.')
            self.task.cancel()
        else:
            self.outgoing.put_nowait(packet)

    async def _session(self):
        self.outgoing = asyncio.Queue(maxsize=32)
        self.width = self.height = 0
        framebuffer = Framebuffer()
        sequence = 0
        self.phase.emit('Connecting to routeOS…')
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(self.host, max_msg_size=W.HEADER.size + W.MAX_PAYLOAD,
                                          compress=0, autoping=True) as ws:
                self.ws = ws
                await ws.send_bytes(W.pack(W.Type.HELLO, sequence, W.hello(self.request_control, keyboard=True)))
                sequence += 1
                self.phase.emit('Waiting for a complete routeOS screen…')
                deadline = time.monotonic() + 10
                last_packet = time.monotonic()
                last_ping = time.monotonic()
                received_frame = False
                while not self.stopping:
                    if not received_frame and time.monotonic() > deadline:
                        raise TimeoutError('No complete routeOS screen arrived')
                    while not self.outgoing.empty():
                        kind, payload = self.outgoing.get_nowait()
                        if kind == W.Type.RESYNC:
                            framebuffer.resync()
                        await asyncio.wait_for(ws.send_bytes(W.pack(kind, sequence, payload, minor=framebuffer.minor)), 2)
                        sequence = (sequence + 1) & 0xffffffff
                    try:
                        message = await ws.receive(timeout=0.02)
                    except asyncio.TimeoutError:
                        if not self.is_established and time.monotonic() > deadline:
                            raise TimeoutError('No complete routeOS screen arrived')
                        if time.monotonic() - last_packet > 15:
                            raise TimeoutError('routeOS heartbeat timed out')
                        if time.monotonic() - last_ping > 5:
                            self._enqueue((W.Type.PING, struct.pack('>I', int(time.monotonic()) & 0xffffffff)))
                            last_ping = time.monotonic()
                        continue
                    if message.type != aiohttp.WSMsgType.BINARY:
                        raise ConnectionError('routeOS closed the display connection or sent a nonbinary message')
                    last_packet = time.monotonic()
                    event, value = framebuffer.receive(message.data)
                    if event == 'display':
                        self.width, self.height = framebuffer.width, framebuffer.height
                        self.keyboard_available = framebuffer.keyboard
                    elif event == 'control':
                        self.control_granted = value and self.request_control
                        self.control_changed.emit(self.control_granted)
                    elif event == 'resync':
                        self._enqueue((W.Type.RESYNC, b'\x01'))
                    elif event == 'ping':
                        self._enqueue((W.Type.PONG, value))
                    elif event == 'frame':
                        received_frame = True
                        image = QImage(value.data, self.width, self.height, self.width * 4,
                                       QImage.Format.Format_RGBA8888).copy()
                        self._raw_frame.emit(image)

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.task = self.loop.create_task(self._session())
        try:
            if not self.stopping:
                self.loop.run_until_complete(self.task)
            else:
                self.task.cancel()
                self.loop.run_until_complete(asyncio.gather(self.task, return_exceptions=True))
        except asyncio.CancelledError:
            pass
        except Exception as error:
            if not self.stopping:
                self.failed.emit(*classify_error(error))
        finally:
            self.control_granted = False
            self.control_changed.emit(False)
            self.ws = None
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()


class RVDPSessionWindow(SessionWindow):
    """Reuse display/scaling/lifecycle without exposing unsupported VNC actions."""

    def __init__(self, *args, **kwargs):
        self._retry_count = 0
        self._retry = None
        super().__init__(*args, **kwargs)
        self._retry = QTimer(self)
        self._retry.setSingleShot(True)
        self._retry.timeout.connect(self.reconnect)

    def attach(self, controller):
        super().attach(controller)
        controller.control_changed.connect(self._permission)
        self._permission(controller.control_granted)
        self.clip_action.setChecked(False)
        unsupported = (self.clip_action, self.type_action, self.launchpad_action,
                       self.mission_action, self.desktop_action, self.ctrl_action, self.swap_action)
        for action in unsupported:
            action.setEnabled(False)
            action.setVisible(False)
        for button in self.toolbar.findChildren(QToolButton):
            if button.defaultAction() in unsupported:
                button.hide()

    def _permission(self, granted):
        self.control_action.setEnabled(bool(granted))
        self.control_action.setChecked(bool(granted))
        self.set_control(bool(granted))

    def send(self, method, *args):
        if method in ('pointer', 'request', 'key'):
            super().send(method, *args)

    def on_failed(self, kind, message):
        super().on_failed(kind, message)
        if self._retry is not None and self._retry_count < 3 and kind in ('network', 'disconnect', 'timeout', 'refused'):
            self._retry.start(500 * (2 ** self._retry_count))
            self._retry_count += 1

    def on_frame(self, image):
        self._retry_count = 0
        if self._retry:
            self._retry.stop()
        super().on_frame(image)
        self.info_label.setText(f'{self.host} · {image.width()} × {image.height()} · routeOS RVDP')

    def closeEvent(self, event):
        if self._retry:
            self._retry.stop()
        super().closeEvent(event)
