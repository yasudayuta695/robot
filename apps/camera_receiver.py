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

        # 奥側も取りたい場合はこの値を小さくする（0.20 -> 0.10）
        roi_top = int(h * 0.10)
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

        # 黒ライン中心を各y行で求め、緑の折れ線として描画（曲線にも追従しやすい）
        ys_all, xs_all = np.where(target_mask > 0)
        row_indices = np.unique(ys_all)
        row_centers = []
        row_widths = []
        for r in row_indices:
            row_xs = xs_all[ys_all == r]
            if row_xs.size >= 2:
                x_min = int(row_xs.min())
                x_max = int(row_xs.max())
                row_centers.append((x_min + x_max) / 2.0)
                row_widths.append(x_max - x_min)
            else:
                row_centers.append(np.nan)
                row_widths.append(0)

        row_centers_np = np.array(row_centers, dtype=np.float32)
        valid_mask = ~np.isnan(row_centers_np)
        if np.count_nonzero(valid_mask) < 8:
            return vis

        valid_rows = row_indices[valid_mask]
        valid_centers = row_centers_np[valid_mask]
        valid_widths = np.array(row_widths, dtype=np.float32)[valid_mask]

        # 近傍平均でx方向のガタつきを抑える
        if valid_centers.size >= 7:
            kernel_smooth = np.ones(7, dtype=np.float32) / 7.0
            smooth_centers = np.convolve(valid_centers, kernel_smooth, mode="same")
        else:
            smooth_centers = valid_centers

        centerline_points = np.stack(
            [
                np.clip(np.round(smooth_centers), 0, w - 1).astype(np.int32),
                (valid_rows + roi_top).astype(np.int32),
            ],
            axis=1,
        )
        if centerline_points.shape[0] >= 2:
            pts = centerline_points[::2].reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        # 全体幅（各行の幅の中央値）
        if valid_widths.size > 0:
            width_px = int(np.median(valid_widths))
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

        # 遠/中/近 3点（固定深度位置で算出）
        zones = [
            ("Far", 0, y1, y1 // 2),
            ("Mid", y1, y2, (y1 + y2) // 2),
            ("Near", y2, h, (y2 + h) // 2),
        ]
        for name, z0, z1, ay_global in zones:
            ry0 = max(0, z0 - roi_top)
            ry1 = min(target_mask.shape[0], z1 - roi_top)
            if ry1 <= ry0:
                continue

            # 帯域中央の固定y（ROI座標）
            ay = int(np.clip(ay_global - roi_top, ry0, ry1 - 1))

            # 固定y付近の細い帯で、中心線xと幅を計算
            band_top = max(ry0, ay - 6)
            band_bot = min(ry1, ay + 7)
            band_mask = (valid_rows >= band_top) & (valid_rows < band_bot)
            if np.count_nonzero(band_mask) < 2:
                continue

            px = int(np.median(smooth_centers[band_mask]))
            wz = int(np.median(valid_widths[band_mask]))

            py = ay_global  # 表示位置は固定深度
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
