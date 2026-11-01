# -*- coding: utf-8 -*-
"""Runs ON Pepper (Python 2.7 + NAOqi, also works with Python 3 + naoqi bindings): streams the head camera to the laptop
and speaks the laptop's messages. The laptop connects with `infer/run.py --source pepper:PEPPER_IP`.

  scp infer/pepper_bridge.py nao@PEPPER_IP:~/
  ssh nao@PEPPER_IP
  python pepper_bridge.py --port 5600 --fps 10

Protocol (same as infer/sources.py): Pepper -> laptop, per frame: header "!cIIId" (b"J" jpeg or b"R" raw BGR, width,
height, payload bytes, timestamp) + payload. Laptop -> Pepper: JSON lines, e.g. {"say": "Are you okay?"}.
Frames are JPEG-encoded if OpenCV is available on the robot, raw BGR otherwise (640x480 raw ~ 9 MB/s at 10 fps).
"""
import argparse
import json
import socket
import struct
import threading
import time

from naoqi import ALProxy

HEADER = struct.Struct("!cIIId")
K_TOP_CAMERA, K_VGA, K_BGR = 0, 2, 13  # ALVideoDevice: top camera, 640x480, BGR

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


def listen_commands(conn, tts):
    f = conn.makefile("r")
    for line in f:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("say"):
            tts.post.say(str(msg["say"]))  # non-blocking so streaming continues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--naoqi", default="127.0.0.1")
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    video = ALProxy("ALVideoDevice", args.naoqi, 9559)
    tts = ALProxy("ALTextToSpeech", args.naoqi, 9559)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print("waiting for laptop on port %d (jpeg=%s)" % (args.port, cv2 is not None))
    while True:
        conn, addr = srv.accept()
        print("laptop connected: %s" % (addr,))
        t = threading.Thread(target=listen_commands, args=(conn, tts))
        t.daemon = True
        t.start()
        handle = video.subscribeCamera("fall_stream", K_TOP_CAMERA, K_VGA, K_BGR, args.fps)
        try:
            nxt = time.time()
            while True:
                img = video.getImageRemote(handle)
                if img is None:
                    continue
                w, h, data = img[0], img[1], img[6]
                if cv2 is not None:
                    frame = np.frombuffer(bytearray(data), np.uint8).reshape(h, w, 3)
                    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                    fmt, payload = b"J", enc.tobytes()
                else:
                    fmt, payload = b"R", bytes(data)
                conn.sendall(HEADER.pack(fmt, w, h, len(payload), time.time()) + payload)
                nxt += 1.0 / args.fps
                time.sleep(max(0.0, nxt - time.time()))
        except (socket.error, IOError) as e:
            print("laptop disconnected: %s" % e)
        finally:
            video.unsubscribe(handle)
            conn.close()


if __name__ == "__main__":
    main()
