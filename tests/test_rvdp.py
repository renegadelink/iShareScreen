"""RVDP boundaries, transactional display and native adapter integration."""
import asyncio
import struct
import threading
import pytest
from aiohttp import web
from desktop_helpers import pump
from isharescreen.protocols.rvdp import wire as W
from isharescreen.protocols.rvdp.client import Framebuffer, endpoint


GOLDEN = {
    'hello': '5256445001000100000000000000000d01000100000000010000400000',
    'display': '5256445001000200000000000000000e0003000101000000000100000000',
    'rect': '525644500100030000000001000000170000000100000000000300010100000006f80007e0001f',
    'end': '5256445001000400000000020000000b0000000100010100000000',
    'pointer': '52564450010006000000000100000006007800440101',
}


def test_golden_wire_bytes():
    assert W.pack(W.Type.HELLO, 0, W.hello()).hex() == GOLDEN['hello']
    assert W.pack(W.Type.DISPLAY_INFO, 0, W.display_info(3, 1)).hex() == GOLDEN['display']
    assert W.pack(W.Type.RECT, 1, W.rectangle(1, 0, 0, 3, 1, bytes.fromhex('f80007e0001f'))).hex() == GOLDEN['rect']
    assert W.pack(W.Type.FRAME_END, 2, W.END.pack(1, 1, 1, 0)).hex() == GOLDEN['end']


def test_primary_colours_and_transaction_boundary():
    f = Framebuffer()
    f.receive(bytes.fromhex(GOLDEN['display']))
    before = f.pixels.copy()
    f.receive(bytes.fromhex(GOLDEN['rect']))
    assert (f.pixels == before).all()
    event, pixels = f.receive(bytes.fromhex(GOLDEN['end']))
    assert event == 'frame'
    assert pixels.tolist() == [[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]]


@pytest.mark.parametrize('change', [
    lambda b: b[:8], lambda b: b[:-1], lambda b: b+b'\0',
    lambda b: b'FAIL'+b[4:], lambda b: b[:4]+b'\2'+b[5:],
    lambda b: b[:7]+b'\1'+b[8:], lambda b: b[:6]+b'\7'+b[7:],
    lambda b: b[:12]+b'\xff'*4+b[16:], lambda b: b'\0'*20000,
])
def test_malformed_envelopes_rejected(change):
    with pytest.raises(ValueError):
        W.unpack(change(bytes.fromhex(GOLDEN['rect'])))


@pytest.mark.parametrize('w,h', [(0, 1), (321, 1), (240, 241), (65535, 65535)])
def test_bad_display_dimensions(w, h):
    with pytest.raises(ValueError):
        W.read_display(W.display_info(w, h))


def test_incomplete_snapshot_forces_resync_without_presenting():
    f = Framebuffer()
    f.receive(W.pack(2, 0, W.display_info(3, 2)))
    f.receive(W.pack(3, 1, W.rectangle(1, 0, 0, 3, 1, b'\xff'*6)))
    assert f.receive(W.pack(4, 2, W.END.pack(1, 1, 1, 0)))[0] == 'resync'
    assert f.needs_full and f.last_frame is None


def test_bad_rectangle_and_sequence():
    f = Framebuffer()
    f.receive(bytes.fromhex(GOLDEN['display']))
    with pytest.raises(ValueError, match='sequence'):
        f.receive(bytes.fromhex(GOLDEN['display']))
    with pytest.raises(ValueError, match='bounds'):
        W.read_rectangle(W.rectangle(1, 1, 0, 3, 1, b'\0'*6), 3, 1)
    p = W.rectangle(1, 0, 0, 3, 1, b'\0'*6)
    with pytest.raises(ValueError, match='length'):
        W.read_rectangle(p[:-1], 3, 1)


def test_wraparound_sequence_and_fresh_session():
    s = W.Sequence()
    s.accept(0xffffffff); s.accept(0)
    # The receiver enforces the first sequence separately.
    assert Framebuffer().sequence.last is None


@pytest.mark.parametrize('url', ['http://localhost/rvdp', 'ws://u:p@localhost/rvdp',
                                'ws://localhost/other', 'ws://localhost/rvdp?token=secret', 'ws://localhost:99999/rvdp'])
def test_reject_invalid_or_secret_bearing_urls(url):
    with pytest.raises(ValueError):
        endpoint(url)


class FakeRVDP:
    def __init__(self, grant=False, keyboard=False):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.pointer = []
        self.grant = grant
        self.clients = []
        self.keyboard = keyboard
        self.keys = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        assert self.ready.wait(5)

    def run(self):
        asyncio.set_event_loop(self.loop)
        async def start():
            app = web.Application()
            app.router.add_get('/rvdp', self.socket)
            self.runner = web.AppRunner(app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '127.0.0.1', 0)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]
        self.loop.run_until_complete(start())
        self.ready.set()
        self.loop.run_forever()
        self.loop.run_until_complete(self.runner.cleanup())
        self.loop.close()

    async def socket(self, req):
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self.clients.append(ws)
        msg = await ws.receive()
        _, requested = W.read_hello(W.unpack(msg.data).payload)
        minor = min(W.unpack(msg.data).payload[3], int(self.keyboard))
        sequence = 0
        async def send(kind, payload):
            nonlocal sequence
            await ws.send_bytes(W.pack(kind, sequence, payload, minor=minor)); sequence += 1
        await send(2, W.display_info(3, 1, control=self.grant, keyboard=bool(minor)))
        await send(11, bytes([self.grant and requested]))
        await send(3, W.rectangle(1, 0, 0, 3, 1, bytes.fromhex('f80007e0001f')))
        await send(4, W.END.pack(1, 1, 1, 0))
        async for message in ws:
            if message.type == web.WSMsgType.BINARY:
                p = W.unpack(message.data)
                if p.kind == W.Type.POINTER:
                    self.pointer.append(W.read_pointer(p.payload, 3, 1))
                elif p.kind == W.Type.KEY:
                    self.keys.append(W.read_key(p.payload))
        return ws

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


@pytest.mark.parametrize('grant,requested', [(False, False), (True, False), (True, True)])
def test_native_window_and_pointer_authorization(qapp, app, grant, requested):
    fake = FakeRVDP(grant)
    try:
        wizard = app.open_wizard()
        wizard.address.setText(f'ws://127.0.0.1:{fake.port}/rvdp')
        wizard.check_address()
        assert wizard.stack.currentIndex() == wizard.CREDENTIALS
        wizard.rvdp_control.setChecked(requested)
        wizard.connect_session()
        assert pump(qapp, lambda: bool(app.sessions)), wizard.cred_status.text()
        window = app.sessions[0]
        assert window.is_live and window.screen_widget.image.width() == 3
        assert window.control_action.isEnabled() == (grant and requested)
        assert not window.clip_action.isEnabled()
        assert app.store.items[0].mode == 'rvdp'
        window.controller.send('pointer', 1, 0, 1)
        window.controller.send('pointer', 1, 0, 0)
        if grant and requested:
            assert pump(qapp, lambda: len(fake.pointer) == 2)
        else:
            assert not fake.pointer
        window.close()
        assert pump(qapp, lambda: not app.sessions)
    finally:
        fake.close()


@pytest.mark.parametrize('keyboard,granted', [(True, True), (True, False), (False, True)])
def test_keyboard_needs_negotiation_and_control(qapp, keyboard, granted):
    from isharescreen.frontend.desktop.rvdp import RVDPController
    from isharescreen.frontend.desktop.session import shutdown_all
    fake = FakeRVDP(granted, keyboard)
    controller = RVDPController(f'ws://127.0.0.1:{fake.port}/rvdp', request_control=granted)
    try:
        controller.start()
        assert pump(qapp, lambda: controller.is_established)
        controller.send('key', ord('7'), True)
        controller.send('key', ord('7'), False)
        controller.send('key', 0xff0d, True)
        controller.send('key', 0xffe3, True)  # unsupported modifier never transmitted
        if keyboard and granted:
            assert pump(qapp, lambda: len(fake.keys) == 3)
            assert fake.keys == [(ord('7'), True), (ord('7'), False), (0xff0d, True)]
        else:
            for _ in range(5):qapp.processEvents()
            assert not fake.keys
    finally:
        controller.stop();shutdown_all();fake.close()


@pytest.mark.parametrize('key,down', [(0, 1), (0xffffffff, 1), (ord('7'), 2), (0xffe3, 1)])
def test_invalid_keyboard_payload(key, down):
    with pytest.raises(ValueError):
        W.read_key(W.KEY.pack(key, down))


def test_keyboard_capability_requires_minor_one():
    f = Framebuffer()
    with pytest.raises(ValueError, match='1.1'):
        f.receive(W.pack(2, 0, W.display_info(3, 1, control=True, keyboard=True)))
    with pytest.raises(ValueError, match='1.1'):
        W.unpack(W.pack(7, 0, W.KEY.pack(ord('2'), True)))


def test_native_reconnect_and_saved_rvdp_roundtrip(qapp, app):
    from isharescreen.frontend.desktop.library import LibraryStore
    fake = FakeRVDP()
    try:
        wizard = app.open_wizard()
        url = f'ws://127.0.0.1:{fake.port}/rvdp'
        wizard.address.setText(url)
        wizard.check_address(); wizard.connect_session()
        assert pump(qapp, lambda: bool(app.sessions))
        window = app.sessions[0]
        original = window.controller
        asyncio.run_coroutine_threadsafe(fake.clients[-1].close(), fake.loop).result(5)
        assert pump(qapp, lambda: window.controller is not original and window.is_live)
        assert window.controller.host == url
        assert not window.control_action.isEnabled()
        loaded = LibraryStore(app.store.path)
        assert loaded.items[0].mode == 'rvdp' and loaded.items[0].address == url
        window.close()
        assert pump(qapp, lambda: not app.sessions)
    finally:
        fake.close()
