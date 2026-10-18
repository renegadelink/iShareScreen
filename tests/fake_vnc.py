"""A deterministic local RFB server speaking ordinary ARD (security type 30).

Modes: 'vnc' (normal), 'http' (answers like a web server), 'silent' (accepts
and says nothing), 'stall' (sends the RFB banner, then never answers).
Test data only; the credentials below are fixtures, not real accounts.
"""
import hashlib
import socket
import struct
import threading

from Crypto.Cipher import AES, DES

from isharescreen.proxy.protocol.rfb import recv_exact

PRIME = (1 << 521) - 1
KEY_LENGTH = 66


def _des_response(challenge, password):
    """Independent restatement of the RFC 6143 VNC authentication response."""
    key = bytearray(password.encode('latin-1')[:8].ljust(8, b'\0'))
    for index, byte in enumerate(key):
        key[index] = int('{:08b}'.format(byte)[::-1], 2)
    return DES.new(bytes(key), DES.MODE_ECB).encrypt(challenge)


class FakeVNC:
    def __init__(self, users=None, colour=(225, 37, 61), size=(2, 1), mode='vnc', name='Fixture',
                 security='ard', banner=b'RFB 003.889\n', reason='', refuse=None):
        """security: 'ard' (Apple type 30), 'vnc' (password, checked against any
        value in `users`), 'none', or 'unsupported' (offers only TLS).
        `reason` follows a failed login on 3.8; `refuse` makes the server send
        a zero-length security list with that reason."""
        self.users = users or {'fixture-user': 'fixture-password'}
        self.colour, self.size, self.mode, self.name = colour, size, mode, name
        self.security, self.banner, self.reason, self.refuse = security, banner, reason, refuse
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen()
        self.port = self.listener.getsockname()[1]
        self.lock = threading.Lock()
        self.clients = []          # live connection sockets
        self.connections = 0       # TCP connections accepted
        self.authenticated = 0     # successful logins
        self.rejected = 0          # failed logins
        self.disconnects = 0       # connections that ended
        self.events = []           # ('key', keysym, down) / ('pointer', x, y, buttons)
        self.requests = 0
        self.errors = []
        threading.Thread(target=self._accept, daemon=True).start()

    # ── lifecycle ────────────────────────────────────────────────────
    def close(self):
        try:
            self.listener.close()
        except OSError:
            pass
        self.drop_all()

    def drop_all(self):
        with self.lock:
            clients, self.clients = self.clients, []
        for sock in clients:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _accept(self):
        while True:
            try:
                sock, _ = self.listener.accept()
            except OSError:
                return
            with self.lock:
                self.connections += 1
                self.clients.append(sock)
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    # ── protocol ─────────────────────────────────────────────────────
    def _serve(self, sock):
        try:
            sock.settimeout(30)
            if self.mode == 'silent':
                sock.recv(1)
                return
            if self.mode == 'http':
                sock.sendall(b'HTTP/1.1 400 Bad Request\r\n\r\n')
                return
            sock.sendall(self.banner)
            expected = b'RFB 003.008\n' if self.banner >= b'RFB 003.008' else b'RFB 003.007\n'
            if recv_exact(sock, 12) != expected:
                return
            minor8 = expected == b'RFB 003.008\n'
            if self.refuse is not None:
                reason = self.refuse.encode()
                sock.sendall(b'\0' + struct.pack('>I', len(reason)) + reason)
                return
            offered = {'ard': b'\x02\x1e\x21', 'vnc': b'\x01\x02', 'none': b'\x01\x01',
                       'unsupported': b'\x01\x12'}[self.security]
            sock.sendall(offered)
            choice = recv_exact(sock, 1)
            if self.mode == 'stall':          # accept the security choice, then never answer
                while sock.recv(1):
                    pass
                return
            ok = self._authenticate(sock, choice)
            if choice == b'\x01' and not minor8:
                pass                           # RFB 3.7: no SecurityResult for None
            elif ok:
                sock.sendall(b'\0\0\0\0')
            else:
                with self.lock:
                    self.rejected += 1
                reason = self.reason.encode()
                sock.sendall(b'\0\0\0\1' + (struct.pack('>I', len(reason)) + reason if minor8 and reason else b''))
                return
            with self.lock:
                self.authenticated += 1
            recv_exact(sock, 1)  # ClientInit
            width, height = self.size
            pixel_format = struct.pack('>BBBBHHHBBBxxx', 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
            name = self.name.encode()
            sock.sendall(struct.pack('>HH', width, height) + pixel_format + struct.pack('>I', len(name)) + name)
            self._messages(sock)
        except (OSError, ConnectionError):
            pass
        except BaseException as error:  # surfaced by the test via .errors
            self.errors.append(error)
        finally:
            with self.lock:
                self.disconnects += 1
                if sock in self.clients:
                    self.clients.remove(sock)
            try:
                sock.close()
            except OSError:
                pass

    def _authenticate(self, sock, choice):
        """Run the chosen security type; True when the client may proceed."""
        if choice == b'\x1e':                 # Apple Remote Desktop
            private = 11
            sock.sendall(struct.pack('>HH', 5, KEY_LENGTH) + PRIME.to_bytes(KEY_LENGTH, 'big')
                         + pow(5, private, PRIME).to_bytes(KEY_LENGTH, 'big'))
            credentials = recv_exact(sock, 128)
            public = int.from_bytes(recv_exact(sock, KEY_LENGTH), 'big')
            secret = pow(public, private, PRIME).to_bytes(KEY_LENGTH, 'big')
            plain = AES.new(hashlib.md5(secret).digest(), AES.MODE_ECB).decrypt(credentials)
            user = plain[:64].split(b'\0')[0].decode()
            password = plain[64:].split(b'\0')[0].decode()
            return self.users.get(user) == password
        if choice == b'\x02':                 # VNC password (RFC 6143 7.2.2)
            challenge = bytes(range(16, 32))
            sock.sendall(challenge)
            response = recv_exact(sock, 16)
            return any(response == _des_response(challenge, password) for password in self.users.values())
        return choice == b'\x01'

    def _messages(self, sock):
        answered = False   # one initial frame per connection; later frames come from push_frame()
        while True:
            kind = recv_exact(sock, 1)[0]
            if kind == 0:
                recv_exact(sock, 19)
            elif kind == 2:
                count = struct.unpack('>xH', recv_exact(sock, 3))[0]
                recv_exact(sock, count * 4)
            elif kind == 3:
                recv_exact(sock, 9)
                with self.lock:
                    self.requests += 1
                if not answered:
                    answered = True
                    self.push_frame(sock=sock)
            elif kind == 4:
                down, _, keysym = struct.unpack('>BHI', recv_exact(sock, 7))
                self.events.append(('key', keysym, bool(down)))
            elif kind == 5:
                buttons, x, y = struct.unpack('>BHH', recv_exact(sock, 5))
                self.events.append(('pointer', x, y, buttons))
            elif kind == 6:
                recv_exact(sock, 3)
                text = recv_exact(sock, struct.unpack('>I', recv_exact(sock, 4))[0])
                self.events.append(('clipboard', text.decode('latin-1')))
            else:
                return

    def push_clipboard(self, text):
        """Tell every client the server's clipboard changed (ServerCutText)."""
        data = text.encode('latin-1')
        with self.lock:
            targets = list(self.clients)
        for target in targets:
            try:
                target.sendall(b'\x03\0\0\0' + struct.pack('>I', len(data)) + data)
            except OSError:
                pass

    def push_frame(self, colour=None, sock=None):
        """Send a raw full-screen update of one colour to one or all clients."""
        colour = colour or self.colour
        width, height = self.size
        pixels = bytes([*colour, 0]) * (width * height)
        message = b'\0\0\0\1' + struct.pack('>HHHHi', 0, 0, width, height, 0) + pixels
        with self.lock:
            targets = [sock] if sock else list(self.clients)
        for target in targets:
            try:
                target.sendall(message)
            except OSError:
                pass
