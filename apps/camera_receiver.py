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

        # 遠/中/近も取りたいので、ROIはやや広めに設定
        roi_top = int(h * 0.20)
        roi = vis[roi_top:, :]

        # 黒抽出
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3分割ガイド（遠/中/近）
        y1 = h // 3
        y2 = (2 * h) // 3
        cv2.line(vis, (0, y1), (w - 1, y1), (120, 120, 120), 1)
        cv2.line(vis, (0, y2), (w - 1, y2), (120, 120, 120), 1)
        cv2.putText(vis, "Far", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "Mid", (8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "Near", (8, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return vis

        target = max(contours, key=cv2.contourArea)
        if cv2.contourArea(target) <= 300:
            return vis

        # 描画用にグローバル座標へ戻す
        target_shifted = target + np.array([[[0, roi_top]]], dtype=target.dtype)
        cv2.drawContours(vis, [target_shifted], -1, (0, 255, 255), 2)

        # 最大輪郭のみ塗りつぶしマスク化（幅計測/3点抽出を安定化）
        target_mask = np.zeros_like(mask)
        cv2.drawContours(target_mask, [target], -1, 255, thickness=cv2.FILLED)

        # ライン方向に沿った緑線（fitLine）
        vx, vy, x0, y0 = cv2.fitLine(target_shifted, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        L = max(w, h)
        p1 = (int(x0 - vx * L), int(y0 - vy * L))
        p2 = (int(x0 + vx * L), int(y0 + vy * L))
        cv2.line(vis, p1, p2, (0, 255, 0), 2)

        # 全体幅（各行の x幅 の中央値）
        ys_all, xs_all = np.where(target_mask > 0)
        row_widths_all = []
        for r in np.unique(ys_all):
            row_xs = xs_all[ys_all == r]
            if row_xs.size >= 2:
                row_widths_all.append(int(row_xs.max() - row_xs.min()))
        if row_widths_all:
            width_px = int(np.median(row_widths_all))
            cv2.putText(
                vis,
                f"width={width_px}px",
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 遠/中/近 3点（各帯域でライン中心を算出）
        zones = [("Far", 0, y1), ("Mid", y1, y2), ("Near", y2, h)]
        for name, z0, z1 in zones:
            ry0 = max(0, z0 - roi_top)
            ry1 = min(target_mask.shape[0], z1 - roi_top)
            if ry1 <= ry0:
                continue

            strip = target_mask[ry0:ry1, :]
            ys, xs = np.where(strip > 0)
            if xs.size < 10:
                continue

            centers = []
            widths = []
            for rr in np.unique(ys):
                row_xs = xs[ys == rr]
                if row_xs.size >= 2:
                    x_min = int(row_xs.min())
                    x_max = int(row_xs.max())
                    centers.append((x_min + x_max) / 2.0)
                    widths.append(x_max - x_min)

            if not centers:
                continue

            px = int(np.median(centers))
            py = int(roi_top + ry0 + np.median(np.unique(ys)))
            wz = int(np.median(widths)) if widths else 0

            cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"{name}:{wz}px",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
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
