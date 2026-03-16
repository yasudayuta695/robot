import logging
import threading
import time
from typing import Dict, Optional, Tuple

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
        self._latest_line_features = self._default_line_features()
        self._far_threshold = 100
        self._near_threshold = 70
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _default_line_features(self) -> Dict[str, float]:
        return {
            "line_detect_top": 0.0,
            "line_detect_mid": 0.0,
            "line_detect_bottom": 0.0,
            "line_offset_top": 0.0,
            "line_offset_mid": 0.0,
            "line_offset_bottom": 0.0,
        }

    def _set_latest_line_features(self, features: Dict[str, float]) -> None:
        with self._lock:
            self._latest_line_features = features.copy()

    def get_latest_line_features(self) -> Dict[str, float]:
        with self._lock:
            return self._latest_line_features.copy()

    def get_thresholds(self) -> Tuple[int, int]:
        with self._lock:
            return self._far_threshold, self._near_threshold

    def set_thresholds(self, far_threshold: Optional[int] = None, near_threshold: Optional[int] = None) -> Tuple[int, int]:
        with self._lock:
            if far_threshold is not None:
                self._far_threshold = int(np.clip(int(far_threshold), 0, 255))
            if near_threshold is not None:
                self._near_threshold = int(np.clip(int(near_threshold), 0, 255))
            return self._far_threshold, self._near_threshold

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
        features = self._default_line_features()

        # 奥側を強めたいので、ROI上端を少し上げて取得範囲を広げる
        roi_top = int(h * 0.03)
        roi = vis[roi_top:, :]

        # 黒抽出（遠方は細く低コントラストになりやすいので閾値を分ける）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        roi_h = blur.shape[0]
        far_split = int(roi_h * 0.45)

        far_threshold, near_threshold = self.get_thresholds()

        _, mask_far = cv2.threshold(blur[:far_split, :], far_threshold, 255, cv2.THRESH_BINARY_INV)
        _, mask_near = cv2.threshold(blur[far_split:, :], near_threshold, 255, cv2.THRESH_BINARY_INV)

        mask = np.zeros_like(blur, dtype=np.uint8)
        mask[:far_split, :] = mask_far
        mask[far_split:, :] = mask_near

        # 遠方は細線を残すため弱め、手前はノイズ除去を強めに処理
        kernel_far = np.ones((3, 3), np.uint8)
        kernel_near = np.ones((5, 5), np.uint8)
        far_region = cv2.morphologyEx(mask[:far_split, :], cv2.MORPH_OPEN, kernel_far)
        far_region = cv2.morphologyEx(far_region, cv2.MORPH_CLOSE, kernel_far)
        near_region = cv2.morphologyEx(mask[far_split:, :], cv2.MORPH_OPEN, kernel_near)
        near_region = cv2.morphologyEx(near_region, cv2.MORPH_CLOSE, kernel_near)
        mask[:far_split, :] = far_region
        mask[far_split:, :] = near_region

        # 上下の切れ目をつなげる
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))

        # 3分割ガイド（遠/中/近）
        y1 = h // 3
        y2 = (2 * h) // 3
        cv2.line(vis, (0, y1), (w - 1, y1), (120, 120, 120), 1)
        cv2.line(vis, (0, y2), (w - 1, y2), (120, 120, 120), 1)
        cv2.putText(vis, "Far", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "Mid", (8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "Near", (8, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"th_far={far_threshold} th_near={near_threshold}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._set_latest_line_features(features)
            return vis

        def contour_score(cnt: np.ndarray) -> float:
            area = float(cv2.contourArea(cnt))
            if area <= 0.0:
                return -1.0
            _, y, _, _ = cv2.boundingRect(cnt)
            # ROI上側まで伸びる輪郭をやや優先する
            far_bonus = 1.0 + 0.45 * (1.0 - (float(y) / max(1.0, float(roi_h))))
            return area * far_bonus

        target = max(contours, key=contour_score)
        if cv2.contourArea(target) <= 220:
            self._set_latest_line_features(features)
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
            self._set_latest_line_features(features)
            return vis

        valid_rows = row_indices[valid_mask].astype(np.int32)
        valid_centers = row_centers_np[valid_mask].astype(np.float32)
        valid_widths = np.array(row_widths, dtype=np.float32)[valid_mask]

        # 端で細くなる部分は中心が暴れやすいので、極端に細い行を除外
        median_width = float(np.median(valid_widths))
        min_stable_width = max(2.0, median_width * 0.25)
        stable_mask = valid_widths >= min_stable_width
        if np.count_nonzero(stable_mask) >= 8:
            valid_rows = valid_rows[stable_mask]
            valid_centers = valid_centers[stable_mask]
            valid_widths = valid_widths[stable_mask]

        # 検出できた範囲をROI全体へ補間して、途中で線が切れないようにする
        roi_h = target_mask.shape[0]
        dense_rows = np.arange(roi_h, dtype=np.int32)
        interp_centers = np.interp(
            dense_rows,
            valid_rows,
            valid_centers,
            left=float(valid_centers[0]),
            right=float(valid_centers[-1]),
        ).astype(np.float32)
        interp_widths = np.interp(
            dense_rows,
            valid_rows,
            valid_widths,
            left=float(valid_widths[0]),
            right=float(valid_widths[-1]),
        ).astype(np.float32)

        # edge padして平滑化し、奥/手前端のくねりを抑える
        smooth_window = 15 if roi_h >= 15 else max(3, (roi_h // 2) * 2 + 1)
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel_smooth = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        pad = smooth_window // 2
        padded = np.pad(interp_centers, (pad, pad), mode="edge")
        smooth_centers = np.convolve(padded, kernel_smooth, mode="valid")

        centerline_points = np.stack(
            [
                np.clip(np.round(smooth_centers), 0, w - 1).astype(np.int32),
                (dense_rows + roi_top).astype(np.int32),
            ],
            axis=1,
        )
        if centerline_points.shape[0] >= 2:
            pts_xy = np.ascontiguousarray(centerline_points[::2], dtype=np.int32)
            if pts_xy.shape[0] >= 2:
                try:
                    cv2.polylines(
                        vis,
                        [pts_xy.reshape(-1, 1, 2)],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                except cv2.error:
                    # OpenCV build差異でpolylinesが失敗する場合のフォールバック
                    for i in range(1, pts_xy.shape[0]):
                        p_prev = tuple(pts_xy[i - 1])
                        p_curr = tuple(pts_xy[i])
                        cv2.line(vis, p_prev, p_curr, (0, 255, 0), 2)

        # 全体幅（各行の幅の中央値）
        if interp_widths.size > 0:
            width_px = int(np.median(interp_widths))
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
            ("Far", "top", 0, y1, y1 // 2),
            ("Mid", "mid", y1, y2, (y1 + y2) // 2),
            ("Near", "bottom", y2, h, (y2 + h) // 2),
        ]
        for name, zone_key, z0, z1, ay_global in zones:
            ry0 = max(0, z0 - roi_top)
            ry1 = min(target_mask.shape[0], z1 - roi_top)
            if ry1 <= ry0:
                continue

            # 帯域中央の固定y（ROI座標）
            ay = int(np.clip(ay_global - roi_top, ry0, ry1 - 1))

            # 3点は描画中の中心線そのものから取得する
            px = int(np.clip(np.round(smooth_centers[ay]), 0, w - 1))
            wz = int(max(0.0, interp_widths[ay]))

            # guide_learn用の保存特徴量（0/1フラグ + 中心からの正規化距離）
            offset_norm = float((px - (w / 2.0)) / max(w / 2.0, 1.0))
            features[f"line_detect_{zone_key}"] = 1.0
            features[f"line_offset_{zone_key}"] = float(np.clip(offset_norm, -1.0, 1.0))

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

        self._set_latest_line_features(features)
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
