"""Authenticate and verify framebuffer pixels without storing credentials/images."""
import asyncio
import getpass
import sys
import time
from isharescreen.proxy.standard import StandardClient


def probe(host, user, password):
    client = StandardClient(host,user,password)
    try:
        print(f'Authenticated. Standard VNC desktop: {client.width}x{client.height}', flush=True)
        client.sock.settimeout(12)
        deadline=time.monotonic()+15
        updates=0
        while time.monotonic()<deadline:
            if client.read_update() != 'frame':
                continue
            updates+=1
            frame=client.pixels
            print(f'FRAME {updates}: {frame.shape[1]}x{frame.shape[0]}, range {frame[:,:,:3].min()}..{frame[:,:,:3].max()}, std {frame[:,:,:3].std():.1f}', flush=True)
            if frame[:,:,:3].std()>1:
                if '--capture' in sys.argv:
                    from PIL import Image
                    Image.fromarray(frame).save('build/live-screen-check.png')
                break
    finally:
        client.close()


if __name__ == '__main__':
    probe(sys.argv[1], sys.argv[2], getpass.getpass('Mac password: '))
