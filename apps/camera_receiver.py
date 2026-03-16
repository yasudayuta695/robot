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

    def find_line(self, img: np.ndarray) -> Optional[np.ndarray]:
        vis = img.copy()
        h, w = vis.shape[:2]

        # 下側だけを見る（床の黒ライン検出を安定化）
        roi_top = int(h * 0.55)
        roi = vis[roi_top:, :]

        # 黒を抽出（反転2値化）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 画面中央の基準線（青）
        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (255, 0, 0), 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            target = max(contours, key=cv2.contourArea)
            if cv2.contourArea(target) > 300:  # 小ノイズ除外
                m = cv2.moments(target)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])   # ROI内のx
                    cy = int(m["m01"] / m["m00"])   # ROI内のy
                    cy_global = cy + roi_top
                    # 輪郭表示（黄）
                    target_shifted = target + np.array([[[0, roi_top]]], dtype=target.dtype)
                    cv2.drawContours(vis, [target_shifted], -1, (0, 255, 255), 2)

                    # 黒ライン中心線（緑）
                    cv2.line(vis, (cx, 0), (cx, h - 1), (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy_global), 5, (0, 255, 0), -1)
                    cv2.putText(
                        vis,
                        f"line_x={cx}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
        return vis
        
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
