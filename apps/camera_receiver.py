import logging
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import zmq

from vision.line_detection_config import LineDetectionConfig
from vision.line_detection_config import (
    DEFAULT_LINE_DETECTION_PROFILE,
    build_line_detection_profile,
)
from vision.line_detection_rules import (
    calc_blob_penalty,
    calc_edge_penalty,
    evaluate_contour_reject_reason,
    evaluate_post_track_reject_reason,
    is_soft_recover_candidate,
)


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
        self._line_color_space = "lab"
        self._auto_threshold_enabled = False
        self._debug_overlay_enabled = True
        self._prev_center_x: Optional[float] = None
        self._consecutive_misses = 0
        self._lost_feature_hold_frames = 8
        self._line_detection_profile_name = DEFAULT_LINE_DETECTION_PROFILE
        self._line_detection_config = LineDetectionConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _draw_debug_overlay(
        self,
        vis: np.ndarray,
        roi_top: int,
        entries: list[Dict[str, float | int | str]],
        best_idx: Optional[int],
    ) -> None:
        if not entries:
            cv2.putText(
                vis,
                "dbg: no contours",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            return

        sorted_entries = sorted(entries, key=lambda e: float(e["score"]), reverse=True)
        top_entries = sorted_entries[:8]

        for rank, entry in enumerate(top_entries):
            idx = int(entry["idx"])
            x = int(entry["x"])
            y = int(entry["y"]) + roi_top
            bw = int(entry["bw"])
            bh = int(entry["bh"])
            score = float(entry["score"])
            reason = str(entry["reason"])

            if best_idx is not None and idx == best_idx and score > 0.0:
                color = (0, 220, 0)
            elif score > 0.0:
                color = (0, 170, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 1)

            reason_text = reason if reason else "ok"
            label = f"c{idx} s={score:.0f} {reason_text}"
            text_y = max(14, y - 4)
            cv2.putText(
                vis,
                label,
                (max(2, x), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

        best_score = float(sorted_entries[0]["score"])
        cv2.putText(
            vis,
            f"dbg contours={len(entries)} best={best_score:.0f}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (40, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _on_detection_lost(self) -> None:
        self._consecutive_misses += 1
        if self._consecutive_misses > 8:
            self._prev_center_x = None

    def _on_detection_success(self, center_x: float) -> None:
        if self._prev_center_x is None:
            self._prev_center_x = float(center_x)
        else:
            # 急なジャンプを抑えるため、検出中心を緩やかに追従
            self._prev_center_x = 0.7 * float(self._prev_center_x) + 0.3 * float(center_x)
        self._consecutive_misses = 0

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
            incoming = features.copy()

            incoming_detect_sum = (
                float(incoming.get("line_detect_top", 0.0))
                + float(incoming.get("line_detect_mid", 0.0))
                + float(incoming.get("line_detect_bottom", 0.0))
            )

            # Short-term hold: if detection is lost briefly, keep the previous
            # feature trend with decay to avoid abrupt 0,0 control on curves.
            if incoming_detect_sum <= 0.0 and (0 < self._consecutive_misses <= self._lost_feature_hold_frames):
                prev = self._latest_line_features.copy()
                prev_detect_sum = (
                    float(prev.get("line_detect_top", 0.0))
                    + float(prev.get("line_detect_mid", 0.0))
                    + float(prev.get("line_detect_bottom", 0.0))
                )
                if prev_detect_sum > 0.0:
                    decay = 1.0 - (float(self._consecutive_misses) / float(self._lost_feature_hold_frames + 1))
                    decay = float(np.clip(decay, 0.15, 1.0))
                    hold_conf = float(np.clip(decay, 0.20, 1.0))

                    incoming = {
                        "line_detect_top": hold_conf,
                        "line_detect_mid": hold_conf,
                        "line_detect_bottom": hold_conf,
                        "line_offset_top": float(np.clip(float(prev.get("line_offset_top", 0.0)) * decay, -1.0, 1.0)),
                        "line_offset_mid": float(np.clip(float(prev.get("line_offset_mid", 0.0)) * decay, -1.0, 1.0)),
                        "line_offset_bottom": float(np.clip(float(prev.get("line_offset_bottom", 0.0)) * decay, -1.0, 1.0)),
                    }

            self._latest_line_features = incoming

    def get_latest_line_features(self) -> Dict[str, float]:
        with self._lock:
            return self._latest_line_features.copy()

    def get_thresholds(self) -> Tuple[int, int]:
        with self._lock:
            return self._far_threshold, self._near_threshold

    def is_auto_threshold_enabled(self) -> bool:
        with self._lock:
            return bool(self._auto_threshold_enabled)

    def set_auto_threshold_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._auto_threshold_enabled = bool(enabled)
            return self._auto_threshold_enabled

    def is_debug_overlay_enabled(self) -> bool:
        with self._lock:
            return bool(self._debug_overlay_enabled)

    def set_debug_overlay_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._debug_overlay_enabled = bool(enabled)
            return self._debug_overlay_enabled

    def set_thresholds(self, far_threshold: Optional[int] = None, near_threshold: Optional[int] = None) -> Tuple[int, int]:
        with self._lock:
            if far_threshold is not None:
                self._far_threshold = int(np.clip(int(far_threshold), 0, 255))
            if near_threshold is not None:
                self._near_threshold = int(np.clip(int(near_threshold), 0, 255))
            return self._far_threshold, self._near_threshold

    def get_line_color_space(self) -> str:
        with self._lock:
            return str(self._line_color_space)

    def set_line_color_space(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"lab", "hsv"}:
            normalized = "lab"
        with self._lock:
            self._line_color_space = normalized
            return str(self._line_color_space)

    def get_line_detection_config(self) -> LineDetectionConfig:
        with self._lock:
            return self._line_detection_config

    def set_line_detection_config(self, config: LineDetectionConfig) -> LineDetectionConfig:
        with self._lock:
            self._line_detection_config = config
            return self._line_detection_config

    def get_line_detection_profile_name(self) -> str:
        with self._lock:
            return str(self._line_detection_profile_name)

    def set_line_detection_profile_name(self, profile_name: str) -> str:
        profile = str(profile_name).strip().lower() or DEFAULT_LINE_DETECTION_PROFILE
        cfg = build_line_detection_profile(profile)
        with self._lock:
            self._line_detection_profile_name = profile
            self._line_detection_config = cfg
            return str(self._line_detection_profile_name)

    def _prepare_intensity_channel(self, roi: np.ndarray) -> Tuple[np.ndarray, str]:
        mode = self.get_line_color_space()

        if mode == "hsv":
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v_chan = hsv[:, :, 2]
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            norm_v = clahe.apply(v_chan)
            blur = cv2.GaussianBlur(norm_v, (5, 5), 0)
            return blur, "HSV-V"

        # Default: LAB L-channel preprocessing.
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        norm_l = clahe.apply(l_chan)
        blur = cv2.GaussianBlur(norm_l, (5, 5), 0)
        return blur, "LAB-L"

    def _resolve_thresholds(self, blur: np.ndarray, far_split: int) -> Tuple[int, int]:
        with self._lock:
            far_threshold = int(self._far_threshold)
            near_threshold = int(self._near_threshold)
            auto_enabled = bool(self._auto_threshold_enabled)

        if not auto_enabled:
            return far_threshold, near_threshold

        far_region = blur[:far_split, :]
        near_region = blur[far_split:, :]

        if far_region.size > 0:
            far_median = float(np.median(far_region))
            far_p35 = float(np.percentile(far_region, 35))
            far_otsu, _ = cv2.threshold(far_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            far_target = max(
                round(far_median * 0.82),
                round(far_p35 * 0.95),
                round(float(far_otsu) * 0.90),
            )
            far_target = int(np.clip(far_target, 40, 185))
            far_threshold = int(round((0.8 * far_threshold) + (0.2 * far_target)))

        if near_region.size > 0:
            near_median = float(np.median(near_region))
            near_p35 = float(np.percentile(near_region, 35))
            near_otsu, _ = cv2.threshold(near_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            near_target = max(
                round(near_median * 0.76),
                round(near_p35 * 0.95),
                round(float(near_otsu) * 0.88),
            )
            near_target = int(np.clip(near_target, 30, 170))
            near_threshold = int(round((0.8 * near_threshold) + (0.2 * near_target)))

        with self._lock:
            self._far_threshold = int(np.clip(far_threshold, 0, 255))
            self._near_threshold = int(np.clip(near_threshold, 0, 255))
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

    def _build_line_mask(
        self,
        vis: np.ndarray,
        roi_top: int,
        frame_width: int,
    ) -> tuple[np.ndarray, int, int, int, str]:
        roi = vis[roi_top:, :]
        blur, intensity_mode = self._prepare_intensity_channel(roi)
        roi_h = blur.shape[0]
        far_split = int(roi_h * 0.45)

        far_threshold, near_threshold = self._resolve_thresholds(blur=blur, far_split=far_split)
        _, mask_far = cv2.threshold(blur[:far_split, :], far_threshold, 255, cv2.THRESH_BINARY_INV)
        _, mask_near = cv2.threshold(blur[far_split:, :], near_threshold, 255, cv2.THRESH_BINARY_INV)

        global_mask = np.zeros_like(blur, dtype=np.uint8)
        global_mask[:far_split, :] = mask_far
        global_mask[far_split:, :] = mask_near
        mask = global_mask.copy()

        if self._prev_center_x is not None and self._consecutive_misses < 3:
            center_hint = float(np.clip(self._prev_center_x, 0.0, float(frame_width - 1)))
            relax_ratio = min(0.30, 0.08 * float(self._consecutive_misses))
            corridor_mask = np.zeros_like(mask, dtype=np.uint8)
            upper_safe_limit = int(roi_h * 0.78)
            edge_lock_risk = abs(center_hint - (frame_width * 0.5)) > (0.28 * float(frame_width))

            for r in range(roi_h):
                t = float(r) / max(1.0, float(roi_h - 1))
                main_half_ratio = (0.56 * (1.0 - t)) + (0.24 * t) + relax_ratio
                main_half_ratio = float(np.clip(main_half_ratio, 0.18, 0.78))
                main_half = int(main_half_ratio * float(frame_width))
                x1_main = max(0, int(center_hint) - main_half)
                x2_main = min(frame_width, int(center_hint) + main_half)
                if x2_main > x1_main:
                    corridor_mask[r, x1_main:x2_main] = 255

                if r < upper_safe_limit:
                    safe_half_ratio = (0.62 * (1.0 - t)) + (0.32 * t)
                    safe_half_ratio = float(np.clip(safe_half_ratio, 0.28, 0.74))
                    safe_half = int(safe_half_ratio * float(frame_width))
                    x1_safe = max(0, (frame_width // 2) - safe_half)
                    x2_safe = min(frame_width, (frame_width // 2) + safe_half)
                    if x2_safe > x1_safe:
                        corridor_mask[r, x1_safe:x2_safe] = 255

                if edge_lock_risk:
                    rescue_half_ratio = (0.26 * (1.0 - t)) + (0.18 * t)
                    rescue_half_ratio = float(np.clip(rescue_half_ratio, 0.16, 0.30))
                    rescue_half = int(rescue_half_ratio * float(frame_width))
                    x1_rescue = max(0, (frame_width // 2) - rescue_half)
                    x2_rescue = min(frame_width, (frame_width // 2) + rescue_half)
                    if x2_rescue > x1_rescue:
                        corridor_mask[r, x1_rescue:x2_rescue] = 255

            mask = cv2.bitwise_and(mask, corridor_mask)

        kernel_far = np.ones((3, 3), np.uint8)
        kernel_near = np.ones((5, 5), np.uint8)
        far_region = cv2.morphologyEx(mask[:far_split, :], cv2.MORPH_OPEN, kernel_far)
        far_region = cv2.morphologyEx(far_region, cv2.MORPH_CLOSE, kernel_far)
        near_region = cv2.morphologyEx(mask[far_split:, :], cv2.MORPH_OPEN, kernel_near)
        near_region = cv2.morphologyEx(near_region, cv2.MORPH_CLOSE, kernel_near)
        mask[:far_split, :] = far_region
        mask[far_split:, :] = near_region

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
        return mask, roi_h, far_threshold, near_threshold, intensity_mode

    def _find_best_contour(
        self,
        contours: list[np.ndarray],
        roi_h: int,
        frame_width: int,
    ) -> tuple[float, Optional[int], Optional[np.ndarray], list[Dict[str, float | int | str]]]:
        cfg = self.get_line_detection_config()

        def max_segment_width_px(row_index: int) -> int:
            t = float(row_index) / max(1.0, float(roi_h - 1))
            base_ratio = 0.18 + (0.22 * t)
            relax_ratio = min(0.10, 0.025 * float(self._consecutive_misses))
            ratio = float(np.clip(base_ratio + relax_ratio, 0.16, 0.52))
            return int(np.clip(int(ratio * float(frame_width)), 8, 260))

        contour_entries: list[Dict[str, float | int | str]] = []
        scored_contours: list[tuple[float, int, np.ndarray]] = []

        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            x, y, bw, bh = cv2.boundingRect(cnt)
            span_ratio = float(bh) / max(1.0, float(roi_h))
            fill_ratio = area / max(1.0, float(bw * bh)) if bw > 0 and bh > 0 else 0.0
            bw_ratio = float(bw) / max(1.0, float(frame_width))
            avg_row_width = area / max(1.0, float(bh))
            mid_allowed_width = float(max_segment_width_px(int(y + (bh * 0.5))))
            aspect = float(bh) / max(1.0, float(bw))

            touch_left = x <= 2
            touch_right = (x + bw) >= (frame_width - 2)
            touch_top = y <= 2
            touch_bottom = (y + bh) >= (roi_h - 2)

            reason = evaluate_contour_reject_reason(
                area=area,
                span_ratio=span_ratio,
                fill_ratio=fill_ratio,
                bw_ratio=bw_ratio,
                avg_row_width=avg_row_width,
                mid_allowed_width=mid_allowed_width,
                aspect=aspect,
                touch_left=touch_left,
                touch_right=touch_right,
                touch_top=touch_top,
                touch_bottom=touch_bottom,
                cfg=cfg.contour,
            )

            score = -1.0
            if reason is None:
                bottom_reach = float(y + bh) / max(1.0, float(roi_h))
                cx = float(x) + float(bw) * 0.5

                if self._prev_center_x is not None and self._consecutive_misses < 2:
                    center_ref = float(self._prev_center_x)
                    center_weight = 0.55
                    center_floor = 0.45
                else:
                    center_ref = float(frame_width * 0.5)
                    center_weight = 0.28
                    center_floor = 0.62

                dist_center = abs(cx - center_ref) / max(1.0, (frame_width * 0.5))
                center_bonus = max(center_floor, 1.0 - center_weight * dist_center)
                span_bonus = 0.7 + 1.2 * min(span_ratio, 1.0)
                bottom_bonus = 0.7 + 0.6 * min(bottom_reach, 1.0)
                shape_bonus = 0.75 + 0.35 * min(aspect / 3.0, 1.0)
                blob_penalty = calc_blob_penalty(width_ratio=bw_ratio, span_ratio=span_ratio, cfg=cfg.contour)
                edge_penalty = calc_edge_penalty(
                    touch_left=touch_left,
                    touch_right=touch_right,
                    prev_center_x=self._prev_center_x,
                    frame_width=frame_width,
                    cfg=cfg.contour,
                )
                score = area * span_bonus * bottom_bonus * shape_bonus * center_bonus * blob_penalty * edge_penalty

            contour_entries.append(
                {
                    "idx": idx,
                    "score": float(score),
                    "reason": reason if reason else "",
                    "x": float(x),
                    "y": float(y),
                    "bw": float(bw),
                    "bh": float(bh),
                }
            )
            scored_contours.append((float(score), idx, cnt))

        best_score, best_idx, target = max(scored_contours, key=lambda item: item[0])
        if best_score > 0.0:
            return best_score, best_idx, target, contour_entries

        soft_best_score = -1.0
        soft_best_idx = None
        soft_best_cnt = None
        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            x, y, bw, bh = cv2.boundingRect(cnt)
            span_ratio = float(bh) / max(1.0, float(roi_h))
            bw_ratio = float(bw) / max(1.0, float(frame_width))
            fill_ratio = area / max(1.0, float(bw * bh)) if bw > 0 and bh > 0 else 0.0
            touch_left = x <= 2
            touch_right = (x + bw) >= (frame_width - 2)

            if not is_soft_recover_candidate(
                area=area,
                span_ratio=span_ratio,
                bw_ratio=bw_ratio,
                fill_ratio=fill_ratio,
                touch_left=touch_left,
                touch_right=touch_right,
                cfg=cfg.soft_recover,
            ):
                continue

            cx = float(x) + (0.5 * float(bw))
            bottom_reach = float(y + bh) / max(1.0, float(roi_h))
            aspect = float(bh) / max(1.0, float(bw))

            if self._prev_center_x is not None and self._consecutive_misses < 4:
                center_ref = float(self._prev_center_x)
            else:
                center_ref = float(frame_width * 0.5)
            center_dist = abs(cx - center_ref) / max(1.0, (frame_width * 0.5))
            center_bonus = max(0.35, 1.0 - (0.60 * center_dist))
            shape_bonus = 0.65 + (0.65 * min(span_ratio, 1.0)) + (0.20 * min(aspect / 3.0, 1.0))
            bottom_bonus = 0.65 + (0.55 * min(bottom_reach, 1.0))
            score = area * shape_bonus * bottom_bonus * center_bonus
            if score > soft_best_score:
                soft_best_score = float(score)
                soft_best_idx = idx
                soft_best_cnt = cnt

        if soft_best_idx is None or soft_best_cnt is None:
            return best_score, best_idx, target, contour_entries

        for entry in contour_entries:
            if int(entry["idx"]) == int(soft_best_idx):
                entry["reason"] = "soft_recover"
                entry["score"] = float(soft_best_score)
                break
        return soft_best_score, int(soft_best_idx), soft_best_cnt, contour_entries

    def find_line(self, img: np.ndarray) -> Optional[np.ndarray]:
        vis = img.copy()
        h, w = vis.shape[:2]
        features = self._default_line_features()
        debug_overlay_enabled = self.is_debug_overlay_enabled()

        # 奥側を強めたいので、ROI上端を少し上げて取得範囲を広げる
        roi_top = int(h * 0.03)
        mask, roi_h, far_threshold, near_threshold, intensity_mode = self._build_line_mask(
            vis=vis,
            roi_top=roi_top,
            frame_width=w,
        )

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
            f"{intensity_mode} th_far={far_threshold} th_near={near_threshold}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if debug_overlay_enabled:
                self._draw_debug_overlay(vis, roi_top, [], best_idx=None)
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis
        best_score, best_idx, target, contour_entries = self._find_best_contour(
            contours=contours,
            roi_h=roi_h,
            frame_width=w,
        )

        if debug_overlay_enabled:
            self._draw_debug_overlay(vis, roi_top, contour_entries, best_idx=best_idx)

        min_final_area = self.get_line_detection_config().post_track.final_contour_area_min
        if best_score <= 0.0 or cv2.contourArea(target) <= min_final_area:
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        # 描画用にグローバル座標へ戻す
        target_shifted = target + np.array([[[0, roi_top]]], dtype=target.dtype)
        cv2.drawContours(vis, [target_shifted], -1, (0, 255, 255), 2)

        # 最大輪郭のみ塗りつぶしマスク化（幅計測/3点抽出を安定化）
        target_mask = np.zeros_like(mask)
        cv2.drawContours(target_mask, [target], -1, 255, thickness=cv2.FILLED)

        # 行ごとに連続区間を追跡し、分岐/ノイズ混在でも一本の中心線を選ぶ
        # 下側ノイズの影響を減らすため、中段のシードから上下に追跡する
        roi_h = target_mask.shape[0]
        seed_x = float(self._prev_center_x) if self._prev_center_x is not None else (w * 0.5)
        seed_x = float(np.clip(seed_x, 0.0, float(w - 1)))

        def max_segment_width_px(row_index: int) -> int:
            t = float(row_index) / max(1.0, float(roi_h - 1))
            base_ratio = 0.18 + (0.22 * t)
            relax_ratio = min(0.10, 0.025 * float(self._consecutive_misses))
            ratio = float(np.clip(base_ratio + relax_ratio, 0.16, 0.52))
            return int(np.clip(int(ratio * float(w)), 8, 260))

        def row_segments(row_index: int, row: np.ndarray) -> list[tuple[float, float, int, int]]:
            xs = np.flatnonzero(row > 0)
            if xs.size < 2:
                return []
            split_idx = np.where(np.diff(xs) > 1)[0] + 1
            segments = np.split(xs, split_idx)
            result: list[tuple[float, float, int, int]] = []
            max_seg_w = max_segment_width_px(row_index)
            for seg in segments:
                if seg.size < 2:
                    continue
                x_min = int(seg[0])
                x_max = int(seg[-1])
                seg_w = x_max - x_min
                if seg_w <= 1:
                    continue
                if seg_w > max_seg_w:
                    continue
                cx = 0.5 * (x_min + x_max)
                result.append((float(cx), float(seg_w), x_min, x_max))
            return result

        seed_center_row = int(np.clip(roi_h * 0.52, 0, roi_h - 1))
        seed_r0 = max(0, seed_center_row - 30)
        seed_r1 = min(roi_h, seed_center_row + 31)

        best_seed = None
        best_seed_cost = float("inf")
        for r in range(seed_r0, seed_r1):
            segs = row_segments(r, target_mask[r, :])
            for cx, seg_w, x_min, x_max in segs:
                width_cost = max(0.0, seg_w - (0.45 * float(w))) * 0.20
                border_penalty = 20.0 if (x_min <= 1 or x_max >= (w - 2)) else 0.0
                cost = abs(cx - seed_x) + width_cost + border_penalty
                if cost < best_seed_cost:
                    best_seed_cost = cost
                    best_seed = (r, cx, seg_w)

        if best_seed is None:
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        seed_row, seed_cx, seed_w = best_seed

        def track_direction(
            start_row: int,
            step: int,
            start_x: float,
            start_w: float,
        ) -> tuple[list[int], list[float], list[float]]:
            rows: list[int] = []
            centers: list[float] = []
            widths: list[float] = []
            track_x = float(start_x)
            track_w = max(8.0, float(start_w))
            prev_dx = 0.0
            misses = 0
            max_misses = 24

            end = roi_h if step > 0 else -1
            for r in range(start_row, end, step):
                segs = row_segments(r, target_mask[r, :])
                if not segs:
                    misses += 1
                    if misses > max_misses:
                        break
                    continue

                best = None
                best_cost = float("inf")
                for cx, seg_w, x_min, x_max in segs:
                    pred_x = track_x + (0.6 * prev_dx)
                    jump_cost = abs(cx - pred_x)
                    width_consistency_cost = abs(float(seg_w) - track_w) * 0.18
                    width_cost = max(0.0, seg_w - (0.48 * float(w))) * 0.15
                    border_penalty = 12.0 if (x_min <= 1 or x_max >= (w - 2)) else 0.0
                    cost = jump_cost + width_consistency_cost + width_cost + border_penalty
                    if cost < best_cost:
                        best_cost = cost
                        best = (cx, seg_w)

                if best is None:
                    misses += 1
                    if misses > max_misses:
                        break
                    continue

                cx, seg_w = best
                delta_x = float(cx - track_x)
                jump_limit = (0.24 * float(w)) + (0.95 * float(seg_w)) + (1.40 * abs(prev_dx))
                if abs(delta_x) > jump_limit:
                    misses += 1
                    if misses > max_misses:
                        break
                    continue

                rows.append(r)
                centers.append(cx)
                widths.append(seg_w)
                track_x = 0.68 * track_x + 0.32 * cx
                track_w = 0.75 * track_w + 0.25 * float(seg_w)
                prev_dx = 0.50 * prev_dx + 0.50 * delta_x
                misses = 0

            return rows, centers, widths

        up_rows, up_centers, up_widths = track_direction(seed_row - 1, -1, seed_cx, seed_w)
        down_rows, down_centers, down_widths = track_direction(seed_row + 1, 1, seed_cx, seed_w)

        tracked_rows = [seed_row] + up_rows + down_rows
        tracked_centers = [seed_cx] + up_centers + down_centers
        tracked_widths = [seed_w] + up_widths + down_widths

        if len(tracked_rows) < 8:
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        # 補間用に上→下へ並び替える
        order = np.argsort(np.array(tracked_rows, dtype=np.int32))
        tracked_rows_np = np.array(tracked_rows, dtype=np.int32)[order]
        tracked_centers_np = np.array(tracked_centers, dtype=np.float32)[order]
        tracked_widths_np = np.array(tracked_widths, dtype=np.float32)[order]

        # 重複rowがあれば最初の1件を採用
        valid_rows, unique_idx = np.unique(tracked_rows_np, return_index=True)
        valid_centers = tracked_centers_np[unique_idx]
        valid_widths = tracked_widths_np[unique_idx]

        if valid_rows.size < 8:
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        # 端で細くなる部分は中心が暴れやすいので、極端に細い行を除外
        median_width = float(np.median(valid_widths))
        min_stable_width = max(2.0, median_width * 0.18)
        stable_mask = valid_widths >= min_stable_width
        if np.count_nonzero(stable_mask) >= 8:
            valid_rows = valid_rows[stable_mask]
            valid_centers = valid_centers[stable_mask]
            valid_widths = valid_widths[stable_mask]

        if valid_rows.size < 8:
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        # 中心線追跡後の最終除外判定を行う
        cfg = self.get_line_detection_config()
        width_med = float(np.median(valid_widths))
        width_p90 = float(np.percentile(valid_widths, 90))
        near_anchor_global = int((2 * h) // 3)
        near_anchor_row = int(np.clip(near_anchor_global - roi_top, 0, roi_h - 1))
        has_near_support = bool(np.any(valid_rows >= max(0, near_anchor_row - 10)))
        bottom_reach_ratio = float(valid_rows.max()) / max(1.0, float(roi_h - 1))
        near_rows_mask = valid_rows >= near_anchor_row
        near_rows_count = int(np.count_nonzero(near_rows_mask))
        near_width_med = float(np.median(valid_widths[near_rows_mask])) if near_rows_count > 0 else 0.0
        span_rows = int(valid_rows.max() - valid_rows.min() + 1)
        span_ratio_rows = float(span_rows) / max(1.0, float(roi_h))

        post_reject_reason = evaluate_post_track_reject_reason(
            width_med=width_med,
            width_p90=width_p90,
            frame_width=w,
            has_near_support=has_near_support,
            bottom_reach_ratio=bottom_reach_ratio,
            near_rows_count=near_rows_count,
            near_width_med=near_width_med,
            span_ratio_rows=span_ratio_rows,
            cfg=cfg.post_track,
        )
        if post_reject_reason is not None:
            if debug_overlay_enabled:
                cv2.putText(
                    vis,
                    f"reject: {post_reject_reason}",
                    (10, 122),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

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

        valid_row_min = int(valid_rows.min())
        valid_row_max = int(valid_rows.max())
        valid_range_mask = (dense_rows >= valid_row_min) & (dense_rows <= valid_row_max)
        draw_rows = dense_rows[valid_range_mask]

        centerline_points = np.stack(
            [
                np.clip(np.round(smooth_centers[draw_rows]), 0, w - 1).astype(np.int32),
                (draw_rows + roi_top).astype(np.int32),
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
        if draw_rows.size > 0:
            width_px = int(np.median(interp_widths[draw_rows]))
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

            # 追跡中心を更新（下端ノイズに引っ張られすぎないよう少し上を採用）
            track_row = int(np.clip(roi_h * 0.65, valid_row_min, valid_row_max))
            self._on_detection_success(float(smooth_centers[track_row]))

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

            # valid_rows範囲外は見失い扱い（featuresは0.0のまま）
            if ay < valid_row_min or ay > valid_row_max:
                continue

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

        frame_sizes = {
            320 * 240 * 3: (240, 320),
            640 * 480 * 3: (480, 640),
        }

        try:
            while self._running:
                try:
                    data = cam_socket.recv(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    time.sleep(0.01)
                    continue
                try:
                    shape_hw = frame_sizes.get(len(data))
                    if shape_hw is None:
                        self.logger.debug("Unexpected frame size: %d", len(data))
                        continue
                    h, w = shape_hw
                    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if (w, h) != (640, 480):
                        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
                    with self._lock:
                        self._image = img
                except Exception as exc:
                    self.logger.debug("Frame decode failed: %s", exc)
        finally:
            cam_socket.close()
            ctx.term()
