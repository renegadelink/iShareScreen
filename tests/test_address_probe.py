"""Address parsing and the VNC availability probe."""
import socket
import threading
import time

import pytest

from fake_vnc import FakeVNC
from isharescreen.proxy.address import ProbeError, format_address, parse_address, probe_server


@pytest.mark.parametrize('text,expected', [
    ('192.168.1.20', ('192.168.1.20', 5900)),
    ('  studio-mac.local  ', ('studio-mac.local', 5900)),
    ('studio-mac.local:5901', ('studio-mac.local', 5901)),
    ('vnc://10.0.0.5:5902/', ('10.0.0.5', 5902)),
    ('[fe80::1]:5903', ('fe80::1', 5903)),
    ('[fe80::1]', ('fe80::1', 5900)),
    ('fe80::1', ('fe80::1', 5900)),
    ('mac.example.com.', ('mac.example.com.', 5900)),
    ('my_mac', ('my_mac', 5900)),
])
def test_parse_valid(text, expected):
    assert parse_address(text) == expected


@pytest.mark.parametrize('text', [
    '', '   ', 'two words', '999.1.1.1', '1.2.3', 'host:0', 'host:65536', 'host:abc', ':5900',
    '[fe80::1', '[fe80::1]x', '[nope]:5900', 'a..b', '-bad.example', 'bad-.example', 'host:5900:1',
    'exa$mple.com', 'x' * 64 + '.example',
])
def test_parse_invalid(text):
    with pytest.raises(ValueError):
        parse_address(text)


def test_default_port_sentinel_reports_whether_a_port_was_given():
    assert parse_address('host', default_port=0) == ('host', 0)
    assert parse_address('host:5901', default_port=0) == ('host', 5901)


def test_format_address_round_trips():
    for host, port in [('studio.local', 5900), ('studio.local', 5901), ('fe80::1', 5902)]:
        assert parse_address(format_address(host, port)) == (host, port)


def test_probe_reads_banner_and_leaves_no_server_session():
    server = FakeVNC()
    try:
        result = probe_server('127.0.0.1', server.port, timeout=3)
        assert (result.host, result.port, result.version) == ('127.0.0.1', server.port, '003.889')
        deadline = time.time() + 3
        while server.disconnects < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert server.connections == 1 and server.disconnects == 1
        assert server.authenticated == 0 and server.rejected == 0   # the probe never signed in
    finally:
        server.close()


def test_probe_rejects_non_vnc_service():
    server = FakeVNC(mode='http')
    try:
        with pytest.raises(ProbeError) as caught:
            probe_server('127.0.0.1', server.port, timeout=3)
        assert caught.value.kind == 'wrong_service'
    finally:
        server.close()


def test_probe_silent_service_is_wrong_service_not_a_hang():
    server = FakeVNC(mode='silent')
    try:
        started = time.monotonic()
        with pytest.raises(ProbeError) as caught:
            probe_server('127.0.0.1', server.port, timeout=0.5)
        assert caught.value.kind == 'wrong_service'
        assert time.monotonic() - started < 4
    finally:
        server.close()


def test_probe_reports_closed_port_and_unknown_name():
    listener = socket.socket(); listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]; listener.close()
    with pytest.raises(ProbeError) as refused:
        probe_server('127.0.0.1', port, timeout=3)
    assert refused.value.kind == 'refused'
    with pytest.raises(ProbeError) as dns:
        probe_server('no-such-host.invalid', 5900, timeout=3)
    assert dns.value.kind == 'dns'


def test_probe_can_be_cancelled_by_closing_its_socket():
    server = FakeVNC(mode='silent')
    holder = {}
    outcome = []

    def run():
        try:
            probe_server('127.0.0.1', server.port, timeout=20, on_socket=lambda s: holder.setdefault('sock', s))
        except ProbeError as error:
            outcome.append(error.kind)
    try:
        worker = threading.Thread(target=run)
        worker.start()
        deadline = time.time() + 3
        while server.connections < 1 and time.time() < deadline:
            time.sleep(0.02)
        started = time.monotonic()
        holder['sock'].close()
        worker.join(5)
        assert not worker.is_alive() and time.monotonic() - started < 3 and outcome
    finally:
        server.close()
