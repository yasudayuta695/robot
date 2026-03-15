import logging
import threading
import time
from typing import Optional

import numpy as np
import cv2
import zmq


class CameraReceiver:
    def __init__(self, pi_ip: str, camera_port: int, logger: logging.Logger) -> None:
        self.logger = logger
        self.pi_ip = pi_ip
        self.camera_port = camera_port
        self._lock = threading.Lock()
        self._image = np.zeros((480, 640, 3), dtype=np.uint8)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.logger.info("カメラ受信スレッド起動...")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest_frame(self) -> np.ndarray:
        with self._lock:
            return self._image.copy()

    def _run(self) -> None:
        ctx = zmq.Context()
        cam_socket = ctx.socket(zmq.PULL)
        cam_socket.setsockopt(zmq.CONFLATE, 1)
        cam_socket.connect(f"tcp://{self.pi_ip}:{self.camera_port}")

        try:
            while self._running:
                try:
                    data = cam_socket.recv(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    time.sleep(0.01)
                    continue

                try:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((240, 320, 3))
                    img = cv2.resize(img, (640, 480))
                    with self._lock:
                        self._image = img
                except Exception as exc:
                    self.logger.debug("Frame decode failed: %s", exc)
        finally:
            cam_socket.close()
            ctx.term()
