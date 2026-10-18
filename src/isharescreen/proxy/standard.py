"""Standard RFB framebuffer transport. TCP only; no HP/media negotiation."""
from __future__ import annotations
import socket
import struct
import zlib
import threading
import hashlib
import secrets
import os
from Crypto.Cipher import AES, DES
import numpy as np

from .protocol.rfb import recv_exact, build_set_encodings, build_key_event, build_pointer_event


def _connect(host, port, on_socket):
    """Like `socket.create_connection`, but hands every attempt's socket to
    `on_socket` before it connects so that a cancel can close it."""
    error = None
    for family,kind,proto,_,address in socket.getaddrinfo(host,port,type=socket.SOCK_STREAM):
        sock = socket.socket(family,kind,proto)
        try:
            if on_socket:
                on_socket(sock)
            sock.settimeout(15)
            sock.connect(address)
            return sock
        except OSError as failure:
            sock.close()
            error = failure
    raise error or OSError(f'Could not resolve {host}')


class AuthRejected(PermissionError):
    """The server refused the supplied credentials."""


class UnsupportedSecurity(ValueError):
    """The server offers no authentication type this viewer implements."""


class UnsupportedEncoding(ValueError):
    """The server sent a rectangle encoding that was not requested."""


class DesktopUnavailable(ConnectionError):
    """The server authenticated the user but did not provide a desktop."""


class ServerRefused(ConnectionError):
    """The server declined the connection during the handshake, with a reason."""


SECURITY_ARD, SECURITY_VNC, SECURITY_NONE = 30, 2, 1
# Preference order: Apple's type 30 first, so Macs always take the tested path.
SUPPORTED_SECURITY = ((SECURITY_ARD,'account'), (SECURITY_VNC,'password'), (SECURITY_NONE,'none'))
_SECURITY_NAMES = {1:'None', 2:'VNC password', 5:'RA2', 6:'RA2ne', 16:'Tight', 18:'TLS', 19:'VeNCrypt',
                   30:'Apple Remote Desktop', 33:'Apple SRP', 35:'Apple Remote Desktop (legacy)'}


def choose_security(types):
    """Return `(security_type, auth_kind)` for the best type offered, else None.

    auth_kind is 'account' (username and password), 'password' (password only)
    or 'none' (no credentials)."""
    for number,kind in SUPPORTED_SECURITY:
        if number in types:
            return number,kind
    return None


def describe_security(types):
    return ', '.join(_SECURITY_NAMES.get(t,f'type {t}') for t in types) or 'nothing'


def _read_reason(sock):
    """Best-effort read of a failure reason; never waits longer than a moment,
    since some servers close or stay silent instead of sending one."""
    try:
        sock.settimeout(1.5)
        length=struct.unpack('>I',recv_exact(sock,4))[0]
        if length>1024:
            return ''
        text=recv_exact(sock,length).decode('utf-8','replace')
        return ''.join(c for c in text if c.isprintable())[:300]
    except OSError:     # includes timeouts and ConnectionError
        return ''


def read_security_types(sock, banner):
    """Reply to the server's version banner and read its security-type list.

    Returns `(minor, types)` where minor is the negotiated RFB 3.x minor version
    (7 or 8). Servers newer than 3.8, including Apple's 3.889, get 3.8."""
    if len(banner)!=12 or not banner.startswith(b'RFB ') or not banner.endswith(b'\n'):
        raise ValueError('The endpoint is not a VNC server')
    try:
        major,minor=int(banner[4:7]),int(banner[8:11])
    except ValueError:
        raise ValueError('The endpoint is not a VNC server') from None
    if (major,minor)<(3,7):
        raise UnsupportedSecurity(f'The server speaks RFB {major}.{minor}, which is older than this viewer supports (3.7).')
    chosen=8 if (major,minor)>=(3,8) else 7
    sock.sendall(b'RFB 003.00%d\n'%chosen)
    count=recv_exact(sock,1)[0]
    if count==0:
        reason=_read_reason(sock)
        raise ServerRefused('The server refused the connection'+(f': {reason}' if reason else '.'))
    return chosen,recv_exact(sock,count)


def _vnc_password_response(challenge, password):
    """RFC 6143 §7.2.2: DES-encrypt the 16-byte challenge with the password
    (first 8 bytes, zero padded) as the key, each key byte bit-reversed."""
    key=password.encode('latin-1','replace')[:8].ljust(8,b'\0')
    key=bytes(int(f'{byte:08b}'[::-1],2) for byte in key)
    return DES.new(key,DES.MODE_ECB).encrypt(challenge)


def authenticate_standard(host, port, user, password, on_socket=None, on_phase=None):
    """Authenticate and return the connected socket.

    Apple Remote Desktop (type 30) is preferred, as implemented by LibVNCClient,
    so Macs use the ordinary desktop path rather than the Apple 3.889 SRP path
    whose session setup expects additional ARD messages. Other servers may use
    VNC password (type 2, `user` ignored) or None (type 1, nothing used).
    `on_socket` receives the socket as soon as it exists, so another thread can
    close it to cancel a connect or handshake that is blocked on the network.
    """
    phase = on_phase or (lambda name: None)
    phase('connecting')
    sock = _connect(host,port,on_socket)
    try:
        minor,types=read_security_types(sock,recv_exact(sock,12))
        choice=choose_security(types)
        if choice is None:
            raise UnsupportedSecurity(f'The server offers only {describe_security(types)} authentication. '
                                      'This viewer supports Apple Remote Desktop, VNC password and none.')
        security,_=choice
        phase('authenticating')
        sock.sendall(bytes([security]))
        if security==SECURITY_ARD:
            generator,length=struct.unpack('>HH',recv_exact(sock,4))
            if not 64<=length<=512:
                raise ValueError('Invalid ARD key length')
            modulus=int.from_bytes(recv_exact(sock,length),'big')
            peer=int.from_bytes(recv_exact(sock,length),'big')
            if not 1<generator<modulus or not 1<peer<modulus:
                raise ValueError('Invalid ARD key exchange')
            private=secrets.randbelow(modulus-3)+2
            public=pow(generator,private,modulus).to_bytes(length,'big')
            shared=pow(peer,private,modulus).to_bytes(length,'big')
            key=hashlib.md5(shared).digest()
            def field(text):
                value=text.encode('utf-8')
                if len(value)>63:
                    raise ValueError('ARD usernames/passwords must be at most 63 UTF-8 bytes')
                return value+b'\0'+os.urandom(63-len(value))
            encrypted=AES.new(key,AES.MODE_ECB).encrypt(field(user)+field(password))
            sock.sendall(encrypted+public)
        elif security==SECURITY_VNC:
            sock.sendall(_vnc_password_response(recv_exact(sock,16),password))
        if security==SECURITY_NONE and minor<8:
            return sock     # RFB 3.7 sends no SecurityResult for None
        result=struct.unpack('>I',recv_exact(sock,4))[0]
        if result:
            reason=_read_reason(sock) if minor>=8 else ''
            subject={SECURITY_ARD:'The Mac rejected the username or password',
                     SECURITY_VNC:'The server rejected the password'}.get(security,'The server refused the connection')
            raise AuthRejected(subject+(f' ({reason})' if reason else ''))
        return sock
    except BaseException:
        sock.close()
        raise


class StandardClient:
    def __init__(self, host: str, user: str, password: str, port: int = 5900, *,
                 on_socket=None, on_phase=None):
        """`on_phase(name)` reports 'connecting', 'authenticating' and
        'requesting' as each really begins: the TCP connect and RFB banner
        exchange, the ARD key exchange, then ClientInit/ServerInit."""
        phase = on_phase or (lambda name: None)
        self.send_lock = threading.Lock()
        self.sock = authenticate_standard(host,port,user,password,on_socket,phase)
        try:
            self.sock.settimeout(20)
            phase('requesting')
            self._send(b'\x01')  # Shared, ordinary framebuffer session.
            try:
                header = recv_exact(self.sock, 24)
            except ConnectionError:
                raise DesktopUnavailable('The Mac accepted the login but closed the connection before sharing a desktop. '
                                         'Check that Screen Sharing allows this user and that the Mac is not at a locked login window.') from None
            self.width, self.height = struct.unpack('>HH', header[:4])
            name_len = struct.unpack('>I', header[20:])[0]
            if name_len > 1024 * 1024:
                raise ValueError('Invalid desktop name length')
            self.name = recv_exact(self.sock, name_len).decode('utf-8', 'replace')
            self._resize(self.width, self.height)
            self.zlib = zlib.decompressobj()
            # RGBA byte order, 24-bit true colour in a 32-bit pixel.
            pixel_format = struct.pack('>BBBBHHHBBBxxx',32,24,0,1,255,255,255,0,8,16)
            self._send(b'\x00\x00\x00\x00' + pixel_format)
            self._send(build_set_encodings((6, 0, -223)))
            self.sock.settimeout(None)
            self.request(full=True)
        except BaseException:
            self.sock.close()
            raise

    def _send(self, data):
        with self.send_lock:
            self.sock.sendall(data)

    def _resize(self, width, height):
        if not (0 < width <= 16384 and 0 < height <= 16384 and width*height <= 64_000_000):
            raise ValueError(f'Invalid framebuffer dimensions: {width}x{height}')
        self.width, self.height = width, height
        self.pixels = np.zeros((height,width,4), dtype=np.uint8)
        self.pixels[:,:,3] = 255

    def request(self, full=False):
        self._send(struct.pack('>BBHHHH',3,not full,0,0,self.width,self.height))

    def read_update(self):
        kind = recv_exact(self.sock,1)[0]
        if kind == 0:
            _, count = struct.unpack('>BH',recv_exact(self.sock,3))
            changed = False
            for _ in range(count):
                x,y,w,h,encoding = struct.unpack('>HHHHi',recv_exact(self.sock,12))
                if encoding == -223:
                    self._resize(w,h)
                    changed = True
                    continue
                if x+w > self.width or y+h > self.height:
                    raise ValueError('Rectangle exceeds framebuffer')
                size = w*h*4
                if encoding == 0:
                    data = recv_exact(self.sock,size)
                elif encoding == 6:
                    length = struct.unpack('>I',recv_exact(self.sock,4))[0]
                    if length > 256_000_000:
                        raise ValueError('Oversized compressed rectangle')
                    data = self.zlib.decompress(recv_exact(self.sock,length), size+1)
                else:
                    raise UnsupportedEncoding(f'Unsupported standard RFB encoding: {encoding}')
                if len(data) != size:
                    raise ValueError('Invalid rectangle pixel length')
                self.pixels[y:y+h,x:x+w] = np.frombuffer(data,dtype=np.uint8).reshape(h,w,4)
                self.pixels[y:y+h,x:x+w,3] = 255
                changed = changed or bool(size)
            self.request()
            return 'frame' if changed else None
        if kind == 2:
            return 'bell'
        if kind == 3:
            header = recv_exact(self.sock,7)
            length = struct.unpack('>I',header[3:])[0]
            if length > 16_000_000:
                raise ValueError('Oversized clipboard')
            self.clipboard = recv_exact(self.sock,length).decode('latin-1')
            return 'clipboard'
        if kind == 1:
            header = recv_exact(self.sock,5)
            recv_exact(self.sock,struct.unpack('>H',header[3:])[0]*6)
            return None
        raise ValueError(f'Unexpected RFB server message: {kind}')

    def pointer(self, x, y, buttons):
        self._send(build_pointer_event(x=x,y=y,buttons=buttons))

    def key(self, keysym, down):
        self._send(build_key_event(keysym=keysym,down=down))

    MAX_CLIPBOARD = 1_000_000

    def clipboard_text(self, text):
        """Encode `text` for ClientCutText. RFB's basic clipboard is Latin-1, so
        other characters become '?'. Returns `(bytes, lossless)`."""
        data = text.encode('latin-1','replace')[:self.MAX_CLIPBOARD]
        return data, len(data)==len(text) and data.decode('latin-1')==text

    def send_clipboard(self, text):
        """Send local clipboard text to the server (RFB ClientCutText)."""
        data,_ = self.clipboard_text(text)
        self._send(struct.pack('>B3xI',6,len(data))+data)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
