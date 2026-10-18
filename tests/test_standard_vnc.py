"""Standard ARD auth and visible pixels, with a deterministic local RFB server."""
import hashlib
import socket
import struct
import threading
import zlib

import numpy as np
from Crypto.Cipher import AES

from isharescreen.proxy.standard import StandardClient
from isharescreen.proxy.protocol.rfb import recv_exact


def test_standard_authentication_and_coloured_framebuffer():
    listener=socket.socket()
    listener.bind(('127.0.0.1',0))
    listener.listen()
    errors=[]

    def server():
        try:
            sock,_=listener.accept()
            with sock:
                sock.settimeout(5)
                sock.sendall(b'RFB 003.889\n')
                assert recv_exact(sock,12)==b'RFB 003.008\n'
                sock.sendall(b'\x02\x1e\x21')
                assert recv_exact(sock,1)==b'\x1e'  # ARD 30, never extended SRP 33.
                prime=(1<<521)-1
                length=66
                sock.sendall(struct.pack('>HH',5,length)+prime.to_bytes(length,'big')+pow(5,11,prime).to_bytes(length,'big'))
                credentials=recv_exact(sock,128)
                public=int.from_bytes(recv_exact(sock,length),'big')
                secret=pow(public,11,prime).to_bytes(length,'big')
                plaintext=AES.new(hashlib.md5(secret).digest(),AES.MODE_ECB).decrypt(credentials)
                assert plaintext[:64].split(b'\0')[0]==b'fixture-user'
                assert plaintext[64:].split(b'\0')[0]==b'fixture-password'
                sock.sendall(b'\0\0\0\0')
                assert recv_exact(sock,1)==b'\x01'
                pf=struct.pack('>BBBBHHHBBBxxx',32,24,0,1,255,255,255,16,8,0)
                sock.sendall(struct.pack('>HH',2,1)+pf+struct.pack('>I',7)+b'Fixture')
                assert recv_exact(sock,20)[:4]==b'\0\0\0\0'
                _,_,count=struct.unpack('>BBH',recv_exact(sock,4))
                encodings=struct.unpack('>'+'i'*count,recv_exact(sock,count*4))
                assert encodings==(6,0,-223)
                request=recv_exact(sock,10)
                assert request[:2]==b'\x03\x00'
                pixels=bytes([255,0,0,0,0,255,0,0])
                sock.sendall(b'\0\0\0\1'+struct.pack('>HHHHi',0,0,2,1,0)+pixels)
                recv_exact(sock,10)
        except BaseException as error:
            errors.append(error)

    worker=threading.Thread(target=server,daemon=True)
    worker.start()
    client=StandardClient('127.0.0.1','fixture-user','fixture-password',listener.getsockname()[1])
    try:
        assert client.read_update()=='frame'
        assert client.pixels.tolist()==[[[255,0,0,255],[0,255,0,255]]]
    finally:
        client.close()
        worker.join(5)
        listener.close()
    assert not errors, errors


def test_continuous_zlib_and_desktop_resize():
    client=StandardClient.__new__(StandardClient)
    left,right=socket.socketpair()
    client.sock=left
    client.send_lock=threading.Lock()
    client.zlib=zlib.decompressobj()
    client._resize(1,1)
    compressor=zlib.compressobj()
    try:
        for colour in ([12,34,56,0],[71,81,91,0]):
            compressed=compressor.compress(bytes(colour))+compressor.flush(zlib.Z_SYNC_FLUSH)
            right.sendall(b'\0\0\0\1'+struct.pack('>HHHHi',0,0,1,1,6)+struct.pack('>I',len(compressed))+compressed)
            assert client.read_update()=='frame'
            assert client.pixels[0,0].tolist()==colour[:3]+[255]
            recv_exact(right,10)
        right.sendall(b'\0\0\0\1'+struct.pack('>HHHHi',0,0,2,3,-223))
        assert client.read_update()=='frame'
        assert client.pixels.shape==(3,2,4)
    finally:
        left.close();right.close()


def test_native_screen_paints_received_pixels(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImage
    from isharescreen.frontend.desktop.standard_app import Screen
    app=QApplication.instance() or QApplication([])
    class Receiver:
        def send(self,*args): pass
    screen=Screen(Receiver())
    pixels=np.array([[[225,37,61,255],[31,175,72,255]]],dtype=np.uint8)
    screen.image=QImage(pixels.data,2,1,8,QImage.Format.Format_RGBA8888).copy()
    screen.resize(320,240)
    result=screen.grab().toImage()
    assert result.pixelColor(40,120).red()>200
    assert result.pixelColor(280,120).green()>150
    screen.close()
