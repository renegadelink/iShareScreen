"""Address parsing and VNC availability probing for the connection wizard.

A probe resolves the name, opens a TCP connection, reads the RFB version banner,
answers with the client version and reads the list of security types the server
offers, so the wizard can ask for the right credentials. It then closes the
socket without choosing a security type or authenticating, so the server has
created no session when the probe ends. Closing is the explicit choice (handing
the socket to authentication would tie the credentials stage to a server-side
handshake timeout).
"""
from __future__ import annotations
import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .standard import (ServerRefused, UnsupportedSecurity, choose_security, describe_security,
                       read_security_types)

DEFAULT_PORT = 5900

_LABEL = re.compile(r'^[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?$')
_BANNER = re.compile(rb'^RFB (\d{3})\.(\d{3})\n$')


def parse_address(text: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """Split `host[:port]` into a validated `(host, port)`.

    Accepts a hostname, IPv4, IPv6 (bare or `[v6]:port`) and an optional
    `vnc://` prefix. Raises `ValueError` with a user-presentable message.
    """
    value = (text or '').strip()
    if value.lower().startswith('vnc://'):
        value = value[6:]
    value = value.rstrip('/')
    if not value:
        raise ValueError('Enter an IP address or hostname.')
    if re.search(r'\s', value):
        raise ValueError('The address cannot contain spaces.')
    port = default_port
    if value.startswith('['):
        end = value.find(']')
        if end < 0:
            raise ValueError('The IPv6 address is missing its closing bracket.')
        host, rest = value[1:end], value[end+1:]
        if rest:
            if not rest.startswith(':'):
                raise ValueError('Unexpected text after the IPv6 address.')
            port = _port(rest[1:])
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            raise ValueError('That is not a valid IPv6 address.') from None
        return host, port
    if value.count(':') > 1:
        try:
            ipaddress.IPv6Address(value)
        except ValueError:
            raise ValueError('That is not a valid IPv6 address. Use [address]:port to give a port.') from None
        return value, port
    host = value
    if ':' in value:
        host, _, port_text = value.partition(':')
        port = _port(port_text)
    if not host:
        raise ValueError('Enter an IP address or hostname before the port.')
    if re.fullmatch(r'[\d.]+', host):
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            raise ValueError('That is not a valid IPv4 address.') from None
        return host, port
    name = host[:-1] if host.endswith('.') else host
    if len(name) > 253 or not all(_LABEL.match(label) for label in name.split('.')):
        raise ValueError('That is not a valid hostname or IP address.')
    return host, port


def _port(text: str) -> int:
    if not re.fullmatch(r'\d{1,5}', text or '') or not 1 <= int(text) <= 65535:
        raise ValueError('The port must be a number from 1 to 65535.')
    return int(text)


def format_address(host: str, port: int) -> str:
    shown = f'[{host}]' if ':' in host else host
    return shown if port == DEFAULT_PORT else f'{shown}:{port}'


class ProbeError(Exception):
    """`kind` is one of dns, unreachable, timeout, refused, wrong_service, unsupported."""
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class ProbeResult:
    host: str
    port: int
    version: str      # e.g. "003.889"
    address: str      # the IP address that answered
    security: tuple = ()   # security types the server offered
    auth: str = ''         # what to ask for: 'account', 'password' or 'none'


def probe_server(host: str, port: int = DEFAULT_PORT, timeout: float = 5.0,
                 on_socket: Optional[Callable[[socket.socket], None]] = None) -> ProbeResult:
    """Verify that `host:port` is reachable and speaks RFB.

    `on_socket` receives the socket before it connects so another thread can
    close it to cancel a blocked probe.
    """
    deadline = time.monotonic() + timeout
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ProbeError('dns', f'Could not find “{host}”. Check the spelling, or use its IP address.') from None
    except UnicodeError:
        raise ProbeError('dns', f'“{host}” is not a usable name.') from None
    failure: Optional[ProbeError] = None
    for family, kind, proto, _, sockaddr in infos[:4]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sock = socket.socket(family, kind, proto)
        try:
            if on_socket:
                on_socket(sock)
            sock.settimeout(remaining)
            try:
                sock.connect(sockaddr)
            except ConnectionRefusedError:
                failure = ProbeError('refused', f'{host} answered, but nothing is listening on port {port}. '
                                                'On a Mac, turn on Screen Sharing in System Settings → General → Sharing.')
                continue
            except (socket.timeout, TimeoutError):
                failure = ProbeError('timeout', f'No answer from {host}. It may be off, asleep, or blocked by a firewall or VPN.')
                continue
            except OSError as error:
                failure = ProbeError('unreachable', f'Could not reach {host}: {error.strerror or error}.')
                continue
            return _read_banner(sock, host, port, sockaddr[0], max(1.0, deadline - time.monotonic()))
        finally:
            sock.close()
    raise failure or ProbeError('timeout', f'No answer from {host}.')


def _read_banner(sock: socket.socket, host: str, port: int, address: str, timeout: float) -> ProbeResult:
    data = bytearray()
    try:
        sock.settimeout(timeout)
        while len(data) < 12:
            chunk = sock.recv(12 - len(data))
            if not chunk:
                break
            data += chunk
    except (socket.timeout, TimeoutError):
        raise ProbeError('wrong_service', f'{host}:{port} accepted the connection but did not identify itself as a '
                                          'VNC server.') from None
    except OSError as error:
        raise ProbeError('unreachable', f'The connection to {host} was interrupted: {error.strerror or error}.') from None
    match = _BANNER.match(bytes(data))
    if not match:
        raise ProbeError('wrong_service', f'{host}:{port} is running a service, but it is not a VNC server.')
    version = f'{match.group(1).decode()}.{match.group(2).decode()}'
    try:
        _, types = read_security_types(sock, bytes(data))
    except ServerRefused as error:
        raise ProbeError('refused', str(error)) from None
    except UnsupportedSecurity as error:
        raise ProbeError('unsupported', str(error)) from None
    except (OSError, ValueError):   # timeout, closed mid-handshake, or garbage
        raise ProbeError('wrong_service', f'{host}:{port} sent a VNC banner but did not complete the VNC handshake.') from None
    choice = choose_security(types)
    if choice is None:
        raise ProbeError('unsupported', f'{host} is a VNC server, but it offers only {describe_security(types)} '
                                        'authentication. This viewer supports Apple Remote Desktop, VNC password and none.')
    return ProbeResult(host, port, version, address, tuple(types), choice[1])
