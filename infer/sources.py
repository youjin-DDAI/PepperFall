"""Frame sources for infer/run.py. Every source returns BGR frames at 10 fps.

  video file : resampled to 10 fps (nearest frame, as in build_clips), processed as fast as possible
  camera     : laptop webcam, latest frame every 0.1 s (older frames are dropped so latency doesn't grow)
  pepper     : TCP client of infer/pepper_bridge.py running on Pepper; latest frame only. send() makes Pepper speak.
"""
import json
import socket
import struct
import threading
import time

import cv2
import numpy as np

FPS = 10
HEADER = struct.Struct("!cIIId")  # format (b"J" jpeg / b"R" raw BGR), width, height, payload bytes, timestamp


class VideoFile:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.i, self.k = -1, 0  # source frame index, next 10 fps frame

    def read(self):
        want = int(round(self.k / FPS * self.src_fps))
        frame = None
        while self.i < want:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.i += 1
        self.k += 1
        return frame

    def close(self):
        self.cap.release()


class _Latest:
    """Background reader that keeps only the newest frame; read() paces output to 10 fps."""

    def __init__(self):
        self.frame, self.new, self.alive = None, threading.Event(), True
        self.last = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _put(self, frame):
        self.frame = frame
        self.new.set()

    def read(self):
        time.sleep(max(0.0, self.last + 1 / FPS - time.time()))
        self.last = time.time()
        while self.alive and not self.new.wait(1.0):
            pass
        self.new.clear()
        return self.frame if self.alive else None


class Camera(_Latest):
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        super().__init__()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                self.alive = False
                break
            self._put(f)

    def close(self):
        self.alive = False
        self.cap.release()


class Pepper(_Latest):
    def __init__(self, host, port=5600):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(None)
        super().__init__()

    def _recv(self, n):
        b = bytearray()
        while len(b) < n:
            chunk = self.sock.recv(n - len(b))
            if not chunk:
                raise ConnectionError("Pepper bridge closed the connection")
            b += chunk
        return bytes(b)

    def _loop(self):
        try:
            while self.alive:
                fmt, w, h, n, _ = HEADER.unpack(self._recv(HEADER.size))
                data = self._recv(n)
                if fmt == b"J":
                    f = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    f = np.frombuffer(data, np.uint8).reshape(h, w, 3)
                self._put(f)
        except (ConnectionError, OSError) as e:
            print(f"pepper stream ended: {e}")
            self.alive = False
            self.new.set()

    def send(self, msg):
        """msg dict, e.g. {"say": "Are you okay?"} -> one JSON line to the bridge."""
        self.sock.sendall((json.dumps(msg) + "\n").encode())

    def close(self):
        self.alive = False
        self.sock.close()


def open_source(spec):
    """'pepper:HOST[:PORT]' | 'camera[:INDEX]' | video file path."""
    if spec.startswith("pepper:"):
        host, _, port = spec[7:].partition(":")
        return Pepper(host, int(port or 5600))
    if spec.startswith("camera"):
        return Camera(int(spec.partition(":")[2] or 0))
    return VideoFile(spec)
