import logging
import os
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import zmq

try:
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
except ModuleNotFoundError:
    # Support mixed launch layouts where modules may live under ./vision,
    # ./apps/vision, or as flat files next to this module.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    for candidate in (
        os.path.join(current_dir, "apps"),
        os.path.join(current_dir, "vision"),
        current_dir,
        os.path.join(parent_dir, "apps"),
        os.path.join(parent_dir, "vision"),
        parent_dir,
    ):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
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
    except ModuleNotFoundError:
        try:
            from apps.vision.line_detection_config import LineDetectionConfig
            from apps.vision.line_detection_config import (
                DEFAULT_LINE_DETECTION_PROFILE,
                build_line_detection_profile,
            )
            from apps.vision.line_detection_rules import (
                calc_blob_penalty,
                calc_edge_penalty,
                evaluate_contour_reject_reason,
                evaluate_post_track_reject_reason,
                is_soft_recover_candidate,
            )
        except ModuleNotFoundError:
            from line_detection_config import LineDetectionConfig
            from line_detection_config import (
                DEFAULT_LINE_DETECTION_PROFILE,
                build_line_detection_profile,
            )
            from line_detection_rules import (
                calc_blob_penalty,
                calc_edge_penalty,
                evaluate_contour_reject_reason,
                evaluate_post_track_reject_reason,
                is_soft_recover_candidate,
            )


class CameraReceiver:
    def __init__(
        self,
        pi_ip: str,
        camera_port: int,
        logger: logging.Logger,
        line_feature_port: Optional[int] = None,
    ) -> None:
        self.logger = logger
        self.pi_ip = pi_ip
        self.camera_port = camera_port
        self.line_feature_port = int(line_feature_port) if line_feature_port is not None else None
        self._lock = threading.Lock()
        self._image = np.zeros((480, 640, 3), dtype=np.uint8)
        self._latest_line_features = self._default_line_features()
        self._far_threshold = 100
        self._near_threshold = 70
        self._line_color_space = "lab"
        self._auto_threshold_enabled = False
        self._auto_threshold_update_interval_sec = 2.5
        self._last_auto_threshold_update_ts = 0.0
        self._last_remote_feature_ts = 0.0
        self._debug_overlay_enabled = True
        self._prev_center_x: Optional[float] = None
        self._consecutive_misses = 0
        self._lost_feature_hold_frames = 8
        self._offset_jump_threshold: float = 0.28
        self._feature_smoothing_alpha: float = 0.22
        self._offset_deadband: float = 0.02
        self._feature_history_len: int = 5
        self._offset_histories = {"top": deque(maxlen=self._feature_history_len), "mid": deque(maxlen=self._feature_history_len), "bottom": deque(maxlen=self._feature_history_len)}
        self._width_histories = {"top": deque(maxlen=self._feature_history_len), "mid": deque(maxlen=self._feature_history_len), "bottom": deque(maxlen=self._feature_history_len)}
        self._prev_accepted_offsets: Dict[str, Optional[float]] = {"top": None, "mid": None, "bottom": None}
        self._line_detection_profile_name = DEFAULT_LINE_DETECTION_PROFILE
        self._line_detection_config = build_line_detection_profile(DEFAULT_LINE_DETECTION_PROFILE)
        self._calibrated_width_norm: Optional[float] = None
        self._last_width_med_norm: float = 0.0
        self._show_binary_mask: bool = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_show_binary_mask(self, enabled: bool) -> None:
        with self._lock:
            self._show_binary_mask = bool(enabled)

    def is_show_binary_mask(self) -> bool:
        with self._lock:
            return self._show_binary_mask

    def is_remote_line_feature_mode(self) -> bool:
        return self.line_feature_port is not None and int(self.line_feature_port) > 0

    def get_remote_feature_age_sec(self) -> Optional[float]:
        if not self.is_remote_line_feature_mode():
            return None
        with self._lock:
            ts = float(self._last_remote_feature_ts)
        if ts <= 0.0:
            return None
        return max(0.0, float(time.perf_counter() - ts))

    def _normalize_line_features(self, incoming: Dict[str, float]) -> Dict[str, float]:
        normalized = self._default_line_features()
        for key in normalized.keys():
            try:
                normalized[key] = float(incoming.get(key, normalized[key]))
            except (TypeError, ValueError):
                continue
        return normalized

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
            for zone in ("top", "mid", "bottom"):
                self._prev_accepted_offsets[zone] = None
                self._offset_histories[zone].clear()
                self._width_histories[zone].clear()

    def _on_detection_success(self, center_x: float) -> None:
        if self._prev_center_x is None:
            self._prev_center_x = float(center_x)
        else:
            # 急なジャンプを抑えるため、検出中心を緩やかに追従
            self._prev_center_x = 0.82 * float(self._prev_center_x) + 0.18 * float(center_x)
        self._consecutive_misses = 0

    def calibrate_straight_width(self) -> Optional[float]:
        """直前に検出した幅中央値をキャリブレーション値として保存する。
        remote mode では最新フィーチャの各ゾーン幅をフォールバックとして使用する。
        """
        with self._lock:
            w = float(self._last_width_med_norm)
            if w <= 0.0:
                # remote mode フォールバック: 最新フィーチャから幅を推定
                feats = self._latest_line_features
                widths = [
                    float(feats.get(f"line_width_{z}", 0.0))
                    for z in ("top", "mid", "bottom")
                    if float(feats.get(f"line_detect_{z}", 0.0)) > 0.0
                ]
                if widths:
                    w = float(np.median(widths))
        if w <= 0.0:
            return None
        with self._lock:
            self._calibrated_width_norm = w
        return w

    def get_calibrated_width_norm(self) -> Optional[float]:
        with self._lock:
            return self._calibrated_width_norm

    def set_calibrated_width_norm(self, value: Optional[float]) -> None:
        """外部から直接キャリブレーション値を設定する（remote mode でPC→Pi 送信用）。"""
        with self._lock:
            self._calibrated_width_norm = float(value) if value is not None else None

    def clear_calibrated_width(self) -> None:
        with self._lock:
            self._calibrated_width_norm = None

    def _default_line_features(self) -> Dict[str, float]:
        return {
            "line_detect_top": 0.0,
            "line_detect_mid": 0.0,
            "line_detect_bottom": 0.0,
            "line_offset_top": 0.0,
            "line_offset_mid": 0.0,
            "line_offset_bottom": 0.0,
            "line_width_top": 0.0,
            "line_width_mid": 0.0,
            "line_width_bottom": 0.0,
        }

    def get_offset_jump_threshold(self) -> float:
        with self._lock:
            return float(self._offset_jump_threshold)

    def set_offset_jump_threshold(self, value: float) -> float:
        with self._lock:
            self._offset_jump_threshold = float(np.clip(value, 0.05, 1.0))
            return self._offset_jump_threshold

    def _set_latest_line_features(self, features: Dict[str, float]) -> None:
        with self._lock:
            incoming = features.copy()
            prev_features = self._latest_line_features.copy()

            # Per-zone outlier rejection: offset が前フレームから急変したゾーンを detect=0 に落とす
            threshold = self._offset_jump_threshold
            for zone in ("top", "mid", "bottom"):
                detect_key = f"line_detect_{zone}"
                offset_key = f"line_offset_{zone}"
                width_key = f"line_width_{zone}"
                if float(incoming.get(detect_key, 0.0)) > 0.0:
                    new_offset = float(incoming.get(offset_key, 0.0))
                    new_width = float(incoming.get(width_key, 0.0))
                    prev = self._prev_accepted_offsets[zone]
                    if prev is not None and abs(new_offset - prev) > threshold:
                        # 外れ値: このゾーンをロスト扱いにする
                        incoming[detect_key] = 0.0
                    else:
                        prev_detect = float(prev_features.get(detect_key, 0.0))
                        prev_offset = float(prev_features.get(offset_key, 0.0))
                        prev_width = float(prev_features.get(width_key, 0.0))

                        if prev_detect > 0.0:
                            alpha = float(np.clip(self._feature_smoothing_alpha, 0.05, 0.95))
                            smoothed_offset = (1.0 - alpha) * prev_offset + alpha * new_offset
                            smoothed_width = (1.0 - alpha) * prev_width + alpha * new_width
                        else:
                            smoothed_offset = new_offset
                            smoothed_width = new_width

                        if abs(smoothed_offset) < float(self._offset_deadband):
                            smoothed_offset = 0.0

                        smoothed_offset = float(np.clip(smoothed_offset, -1.0, 1.0))
                        smoothed_width = float(np.clip(smoothed_width, 0.0, 1.0))

                        # 単発ノイズ抑制: 直近履歴の中央値でロバスト化
                        self._offset_histories[zone].append(smoothed_offset)
                        self._width_histories[zone].append(smoothed_width)
                        med_offset = float(np.median(np.array(self._offset_histories[zone], dtype=np.float32)))
                        med_width = float(np.median(np.array(self._width_histories[zone], dtype=np.float32)))

                        stable_offset = 0.65 * med_offset + 0.35 * smoothed_offset
                        stable_width = 0.65 * med_width + 0.35 * smoothed_width

                        incoming[offset_key] = float(np.clip(stable_offset, -1.0, 1.0))
                        incoming[width_key] = float(np.clip(stable_width, 0.0, 1.0))
                        self._prev_accepted_offsets[zone] = float(incoming[offset_key])

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
                        "line_width_top": float(np.clip(float(prev.get("line_width_top", 0.0)) * decay, 0.0, 1.0)),
                        "line_width_mid": float(np.clip(float(prev.get("line_width_mid", 0.0)) * decay, 0.0, 1.0)),
                        "line_width_bottom": float(np.clip(float(prev.get("line_width_bottom", 0.0)) * decay, 0.0, 1.0)),
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
            if self._auto_threshold_enabled:
                # Enable時に初回更新を即実行できるようにする
                self._last_auto_threshold_update_ts = 0.0
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
            last_update_ts = float(self._last_auto_threshold_update_ts)
            auto_update_interval_sec = float(self._auto_threshold_update_interval_sec)

        if not auto_enabled:
            return far_threshold, near_threshold

        now_ts = time.perf_counter()
        if (now_ts - last_update_ts) < auto_update_interval_sec:
            # しきい値統計計算（percentile/otsu）を毎フレーム実行しない
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
            far_threshold = int(round((0.9 * far_threshold) + (0.1 * far_target)))

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
            near_threshold = int(round((0.9 * near_threshold) + (0.1 * near_target)))

        with self._lock:
            self._far_threshold = int(np.clip(far_threshold, 0, 255))
            self._near_threshold = int(np.clip(near_threshold, 0, 255))
            self._last_auto_threshold_update_ts = now_ts
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
        roi_full = vis[roi_top:, :]
        roi_h_full = roi_full.shape[0]
        color_mode = self.get_line_color_space()

        # 低解像度入力(320x240など)では情報欠落を防ぐため、縮小率を弱める。
        if frame_width >= 640:
            proc_scale = 0.5
        elif frame_width >= 480:
            proc_scale = 0.6
        else:
            proc_scale = 0.75
        proc_w = max(96, int(round(frame_width * proc_scale)))
        proc_h = max(72, int(round(roi_h_full * proc_scale)))
        roi = cv2.resize(roi_full, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        blur, intensity_mode = self._prepare_intensity_channel(roi)
        roi_h = blur.shape[0]
        far_split = int(roi_h * 0.45)

        far_threshold, near_threshold = self._resolve_thresholds(blur=blur, far_split=far_split)
        _, mask_far = cv2.threshold(blur[:far_split, :], far_threshold, 255, cv2.THRESH_BINARY_INV)
        _, mask_near = cv2.threshold(blur[far_split:, :], near_threshold, 255, cv2.THRESH_BINARY_INV)

        if color_mode == "hsv":
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            s_chan = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
            sat_threshold = int(np.clip(np.percentile(s_chan, 60), 55, 130))
            _, sat_low_mask = cv2.threshold(s_chan, sat_threshold, 255, cv2.THRESH_BINARY_INV)

            gated_far = cv2.bitwise_and(mask_far, sat_low_mask[:far_split, :])
            gated_near = cv2.bitwise_and(mask_near, sat_low_mask[far_split:, :])

            # 低彩度ゲートが強すぎて候補が消える場合は元マスクへ戻す。
            if cv2.countNonZero(gated_far) > int(mask_far.size * 0.012):
                mask_far = gated_far
            if cv2.countNonZero(gated_near) > int(mask_near.size * 0.012):
                mask_near = gated_near

        global_mask = np.zeros_like(blur, dtype=np.uint8)
        global_mask[:far_split, :] = mask_far
        global_mask[far_split:, :] = mask_near
        mask = global_mask.copy()

        # 固定閾値運用時に環境が変わるとマスクが全欠損/全飽和しやすい。
        # 占有率に応じて当該フレームのみ閾値を自己回復させる。
        mask_nonzero = int(cv2.countNonZero(mask))
        mask_size = int(mask.size)
        occ_ratio = float(mask_nonzero) / max(1.0, float(mask_size))

        if occ_ratio < 0.003:
            relaxed_far = int(np.clip(far_threshold + 16, 0, 255))
            relaxed_near = int(np.clip(near_threshold + 16, 0, 255))
            _, relaxed_far_mask = cv2.threshold(blur[:far_split, :], relaxed_far, 255, cv2.THRESH_BINARY_INV)
            _, relaxed_near_mask = cv2.threshold(blur[far_split:, :], relaxed_near, 255, cv2.THRESH_BINARY_INV)
            relaxed_mask = np.zeros_like(mask)
            relaxed_mask[:far_split, :] = relaxed_far_mask
            relaxed_mask[far_split:, :] = relaxed_near_mask
            if cv2.countNonZero(relaxed_mask) > mask_nonzero:
                mask = relaxed_mask
                global_mask = relaxed_mask.copy()
                far_threshold = relaxed_far
                near_threshold = relaxed_near
        elif occ_ratio > 0.52:
            tight_far = int(np.clip(far_threshold - 12, 0, 255))
            tight_near = int(np.clip(near_threshold - 12, 0, 255))
            _, tight_far_mask = cv2.threshold(blur[:far_split, :], tight_far, 255, cv2.THRESH_BINARY_INV)
            _, tight_near_mask = cv2.threshold(blur[far_split:, :], tight_near, 255, cv2.THRESH_BINARY_INV)
            tight_mask = np.zeros_like(mask)
            tight_mask[:far_split, :] = tight_far_mask
            tight_mask[far_split:, :] = tight_near_mask
            if cv2.countNonZero(tight_mask) < mask_nonzero:
                mask = tight_mask
                global_mask = tight_mask.copy()
                far_threshold = tight_far
                near_threshold = tight_near

        if self._prev_center_x is not None and self._consecutive_misses < 3:
            center_hint = float(np.clip(self._prev_center_x * 0.5, 0.0, float(proc_w - 1)))
            relax_ratio = min(0.30, 0.08 * float(self._consecutive_misses))
            upper_safe_limit = int(roi_h * 0.78)
            edge_lock_risk = abs(center_hint - (proc_w * 0.5)) > (0.28 * float(proc_w))

            # 全行の t を一括計算
            rows = np.arange(roi_h, dtype=np.float64)
            t = rows / max(1.0, float(roi_h - 1))
            cols = np.arange(proc_w, dtype=np.float64)

            # main帯: center_hint 中心で幅が行ごとに変わる
            main_half = np.clip((0.56 * (1.0 - t) + 0.24 * t + relax_ratio), 0.18, 0.78) * float(proc_w)
            x1_main = np.clip(center_hint - main_half, 0, proc_w).astype(np.int32)
            x2_main = np.clip(center_hint + main_half, 0, proc_w).astype(np.int32)
            # cols[None, :] >= x1[:, None] でブロードキャスト比較 → bool マスク
            corridor_mask = ((cols[None, :] >= x1_main[:, None]) & (cols[None, :] < x2_main[:, None]))

            # safe帯: 画面中心基準、upper_safe_limit行まで
            safe_half = np.clip((0.62 * (1.0 - t) + 0.32 * t), 0.28, 0.74) * float(proc_w)
            center_mid = float(proc_w // 2)
            x1_safe = np.clip(center_mid - safe_half, 0, proc_w).astype(np.int32)
            x2_safe = np.clip(center_mid + safe_half, 0, proc_w).astype(np.int32)
            safe_band = ((cols[None, :] >= x1_safe[:, None]) & (cols[None, :] < x2_safe[:, None]))
            safe_band[upper_safe_limit:, :] = False
            corridor_mask |= safe_band

            # rescue帯: edge_lock_risk時のみ、画面中心基準
            if edge_lock_risk:
                rescue_half = np.clip((0.26 * (1.0 - t) + 0.18 * t), 0.16, 0.30) * float(proc_w)
                x1_rescue = np.clip(center_mid - rescue_half, 0, proc_w).astype(np.int32)
                x2_rescue = np.clip(center_mid + rescue_half, 0, proc_w).astype(np.int32)
                corridor_mask |= ((cols[None, :] >= x1_rescue[:, None]) & (cols[None, :] < x2_rescue[:, None]))

            mask = mask & (corridor_mask.view(np.uint8) * 255)

            # corridor が厳しすぎて消失する場合は、当該フレームのみ corridor を外す。
            if cv2.countNonZero(mask) < int(max(24, mask.size * 0.004)):
                mask = global_mask.copy()

        kernel_far = np.ones((2, 2), np.uint8)
        kernel_near_open = np.ones((3, 3), np.uint8)
        kernel_near_close = np.ones((4, 4), np.uint8)
        far_region = cv2.morphologyEx(mask[:far_split, :], cv2.MORPH_OPEN, kernel_far)
        far_region = cv2.morphologyEx(far_region, cv2.MORPH_CLOSE, kernel_far)
        near_region = cv2.morphologyEx(mask[far_split:, :], cv2.MORPH_OPEN, kernel_near_open)
        near_region = cv2.morphologyEx(near_region, cv2.MORPH_CLOSE, kernel_near_close)
        mask[:far_split, :] = far_region
        mask[far_split:, :] = near_region

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))

        # 元解像度に戻す（輪郭検出・追跡・スコアリングは元解像度で動作）
        mask = cv2.resize(mask, (frame_width, roi_h_full), interpolation=cv2.INTER_NEAREST)
        return mask, roi_h_full, far_threshold, near_threshold, intensity_mode

    def _find_best_contour(
        self,
        contours: list[np.ndarray],
        roi_h: int,
        frame_width: int,
    ) -> tuple[float, Optional[int], Optional[np.ndarray], list[Dict[str, float | int | str]], bool]:
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
            return best_score, best_idx, target, contour_entries, False

        soft_best_score = -1.0
        soft_best_idx = None
        soft_best_cnt = None
        soft_near_anchor = int(np.clip(roi_h * 0.62, 0, roi_h - 1))
        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            x, y, bw, bh = cv2.boundingRect(cnt)
            span_ratio = float(bh) / max(1.0, float(roi_h))
            bw_ratio = float(bw) / max(1.0, float(frame_width))
            fill_ratio = area / max(1.0, float(bw * bh)) if bw > 0 and bh > 0 else 0.0
            avg_row_width = area / max(1.0, float(bh))
            mid_allowed_width = float(max_segment_width_px(int(y + (bh * 0.5))))
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

            # soft recover は誤検出を拾いやすいため、近距離支持と縦長性を最低条件にする
            has_soft_near_support = (y + bh) >= soft_near_anchor
            if not has_soft_near_support:
                continue
            if aspect < 0.95:
                continue
            if avg_row_width > (mid_allowed_width * 1.35):
                continue
            if bottom_reach < 0.56:
                continue

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
            return best_score, best_idx, target, contour_entries, False

        for entry in contour_entries:
            if int(entry["idx"]) == int(soft_best_idx):
                entry["reason"] = "soft_recover"
                entry["score"] = float(soft_best_score)
                break
        return soft_best_score, int(soft_best_idx), soft_best_cnt, contour_entries, True

    def _precompute_row_segments(
        self,
        target_mask: np.ndarray,
        frame_width: int,
    ) -> Dict[int, list[tuple[float, float, int, int]]]:
        """target_mask の全行セグメント (cx, width, xmin, xmax) を一括計算する。"""
        roi_h = target_mask.shape[0]
        ys, xs = np.nonzero(target_mask)
        if ys.size == 0:
            return {}

        # 行ごとの最大許容セグメント幅をベクトル化計算
        # カーブ時にコース幅が広がることを考慮し、Near側は最大75%まで許容
        all_rows = np.arange(roi_h, dtype=np.float64)
        t = all_rows / max(1.0, float(roi_h - 1))
        base_ratio = 0.18 + 0.57 * t
        relax = min(0.10, 0.025 * float(self._consecutive_misses))
        ratio = np.clip(base_ratio + relax, 0.16, 0.75)
        max_widths = np.clip((ratio * float(frame_width)).astype(np.int32), 8, int(frame_width * 0.90))

        unique_rows, starts, counts = np.unique(ys, return_index=True, return_counts=True)
        result: Dict[int, list[tuple[float, float, int, int]]] = {}
        for i in range(unique_rows.size):
            r = int(unique_rows[i])
            row_xs = xs[starts[i]:starts[i] + counts[i]]
            if row_xs.size < 2:
                continue

            max_w = int(max_widths[r])
            gaps = np.where(np.diff(row_xs) > 1)[0] + 1
            bounds = np.empty(gaps.size + 2, dtype=np.intp)
            bounds[0] = 0
            bounds[1:gaps.size + 1] = gaps
            bounds[-1] = row_xs.size

            segs: list[tuple[float, float, int, int]] = []
            for j in range(bounds.size - 1):
                seg_slice = row_xs[bounds[j]:bounds[j + 1]]
                if seg_slice.size < 2:
                    continue
                x_min = int(seg_slice[0])
                x_max = int(seg_slice[-1])
                seg_w = x_max - x_min
                if seg_w <= 1 or seg_w > max_w:
                    continue
                segs.append((0.5 * (x_min + x_max), float(seg_w), x_min, x_max))

            if segs:
                result[r] = segs
        return result

    def _apply_simple_line_fallback(
        self,
        vis: np.ndarray,
        roi_top: int,
        y1: int,
        y2: int,
        frame_h: int,
        frame_w: int,
        debug_overlay_enabled: bool,
    ) -> bool:
        roi = vis[roi_top:, :]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        roi_h = gray.shape[0]

        features = self._default_line_features()
        detected_centers: Dict[str, float] = {}

        zones = [
            ("Far", "top", (roi_top + y1) // 2),
            ("Mid", "mid", (y1 + y2) // 2),
            ("Near", "bottom", (y2 + frame_h) // 2),
        ]

        center_ref_default = float(self._prev_center_x) if self._prev_center_x is not None else (frame_w * 0.5)
        min_seg_w = max(4, int(frame_w * 0.015))
        max_seg_w = max(min_seg_w + 2, int(frame_w * 0.55))

        for name, zone_key, ay_global in zones:
            ay = int(np.clip(ay_global - roi_top, 0, roi_h - 1))
            by0 = max(0, ay - 6)
            by1 = min(roi_h, ay + 7)
            band = gray[by0:by1, :]
            if band.size == 0:
                continue

            profile = np.median(band, axis=0).astype(np.float32)
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (9, 1), 0).reshape(-1)

            p20 = float(np.percentile(profile, 20))
            p50 = float(np.percentile(profile, 50))
            dark_thr = min(p20 + 8.0, p50 - 4.0)
            dark_mask = profile <= dark_thr

            xs = np.flatnonzero(dark_mask)
            if xs.size < 2:
                continue

            split_idx = np.where(np.diff(xs) > 1)[0] + 1
            groups = np.split(xs, split_idx)

            best_cx = None
            best_w = None
            best_cost = float("inf")
            for group in groups:
                if group.size < 2:
                    continue
                x0 = int(group[0])
                x1_ = int(group[-1])
                seg_w = x1_ - x0
                if seg_w < min_seg_w or seg_w > max_seg_w:
                    continue
                cx = 0.5 * float(x0 + x1_)
                center_ref = float(detected_centers.get("bottom", detected_centers.get("mid", center_ref_default)))
                cost = abs(cx - center_ref) - (0.10 * float(seg_w))
                if cost < best_cost:
                    best_cost = cost
                    best_cx = cx
                    best_w = float(seg_w)

            if best_cx is None or best_w is None:
                continue

            px = int(np.clip(round(best_cx), 0, frame_w - 1))
            width_norm = float(np.clip(best_w / max(1.0, float(frame_w)), 0.0, 1.0))
            offset_norm = float(np.clip((float(px) - (frame_w * 0.5)) / max(frame_w * 0.5, 1.0), -1.0, 1.0))

            features[f"line_detect_{zone_key}"] = 1.0
            features[f"line_offset_{zone_key}"] = offset_norm
            features[f"line_width_{zone_key}"] = width_norm
            detected_centers[zone_key] = float(px)

            if debug_overlay_enabled:
                py = ay_global
                cv2.circle(vis, (px, py), 5, (0, 200, 255), -1)
                cv2.putText(
                    vis,
                    f"{name}:fb x={px} y={py}",
                    (px + 6, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

        if (
            features["line_detect_top"] <= 0.0
            and features["line_detect_mid"] <= 0.0
            and features["line_detect_bottom"] <= 0.0
        ):
            return False

        track_cx = detected_centers.get("bottom", detected_centers.get("mid", detected_centers.get("top")))
        if track_cx is not None:
            self._on_detection_success(float(track_cx))

        if debug_overlay_enabled:
            cv2.putText(
                vis,
                "fallback active",
                (10, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

        self._set_latest_line_features(features)
        return True

    def find_line(self, img: np.ndarray) -> Optional[np.ndarray]:
        debug_overlay_enabled = self.is_debug_overlay_enabled()
        show_binary = self.is_show_binary_mask()
        vis = img.copy() if (debug_overlay_enabled or show_binary) else img
        h, w = vis.shape[:2]
        features = self._default_line_features()

        # 上部25%をカットして処理負荷を削減
        roi_top = int(h * 0.25)
        mask, roi_h, far_threshold, near_threshold, intensity_mode = self._build_line_mask(
            vis=vis,
            roi_top=roi_top,
            frame_width=w,
        )

        if show_binary:
            # マスクを BGR に変換して vis を差し替え（検出オーバーレイはこの上に描く）
            mask_full = np.zeros((h, w), dtype=np.uint8)
            mask_full[roi_top:, :] = mask
            vis = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)

        # 3分割ガイド（遠/中/近）: ROI内を均等3分割
        _roi_available = h - roi_top
        y1 = roi_top + _roi_available // 3
        y2 = roi_top + 2 * _roi_available // 3
        if debug_overlay_enabled:
            cv2.line(vis, (0, y1), (w - 1, y1), (120, 120, 120), 1)
            cv2.line(vis, (0, y2), (w - 1, y2), (120, 120, 120), 1)
            cv2.putText(vis, "Far", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(vis, "Mid", (8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(vis, "Near", (8, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            with self._lock:
                calib_done = self._calibrated_width_norm is not None
            if not calib_done:
                cx_guide = w // 2
                cv2.line(vis, (cx_guide, roi_top), (cx_guide, h - 1), (0, 200, 255), 1)
                cv2.putText(vis, "CALIB?", (cx_guide + 4, roi_top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
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
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis
        best_score, best_idx, target, contour_entries, using_soft_recover = self._find_best_contour(
            contours=contours,
            roi_h=roi_h,
            frame_width=w,
        )

        if debug_overlay_enabled:
            self._draw_debug_overlay(vis, roi_top, contour_entries, best_idx=best_idx)

        min_final_area = self.get_line_detection_config().post_track.final_contour_area_min
        if best_score <= 0.0 or cv2.contourArea(target) <= min_final_area:
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        # 描画用にグローバル座標へ戻す
        if debug_overlay_enabled:
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

        # 全行のセグメント情報を一括計算（行ごとのNumPy呼び出しを排除）
        all_row_segs = self._precompute_row_segments(target_mask, w)

        # soft_recover のときは遠方ノイズを避けるため近距離側をシードにする
        seed_center_ratio = 0.68 if using_soft_recover else 0.52
        seed_center_row = int(np.clip(roi_h * seed_center_ratio, 0, roi_h - 1))
        seed_r0 = max(0, seed_center_row - 30)
        seed_r1 = min(roi_h, seed_center_row + 31)

        best_seed = None
        best_seed_cost = float("inf")
        for r in range(seed_r0, seed_r1):
            segs = all_row_segs.get(r, [])
            for cx, seg_w, x_min, x_max in segs:
                width_cost = max(0.0, seg_w - (0.45 * float(w))) * 0.20
                border_penalty = 20.0 if (x_min <= 1 or x_max >= (w - 2)) else 0.0
                cost = abs(cx - seed_x) + width_cost + border_penalty
                if cost < best_seed_cost:
                    best_seed_cost = cost
                    best_seed = (r, cx, seg_w)

        if best_seed is None:
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
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
            max_misses = 14 if using_soft_recover else 24

            end = roi_h if step > 0 else -1
            for r in range(start_row, end, step):
                segs = all_row_segs.get(r, [])
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
                if using_soft_recover:
                    jump_limit = (0.16 * float(w)) + (0.70 * float(seg_w)) + (0.90 * abs(prev_dx))
                else:
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
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
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
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
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
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
            self._on_detection_lost()
            self._set_latest_line_features(features)
            return vis

        # 中心線追跡後の最終除外判定を行う
        cfg = self.get_line_detection_config()
        width_med = float(np.median(valid_widths))
        width_p90 = float(np.percentile(valid_widths, 90))
        near_anchor_global = y2
        near_anchor_row = int(np.clip(near_anchor_global - roi_top, 0, roi_h - 1))
        has_near_support = bool(np.any(valid_rows >= max(0, near_anchor_row - 10)))
        bottom_reach_ratio = float(valid_rows.max()) / max(1.0, float(roi_h - 1))
        near_rows_mask = valid_rows >= near_anchor_row
        near_rows_count = int(np.count_nonzero(near_rows_mask))
        near_width_med = float(np.median(valid_widths[near_rows_mask])) if near_rows_count > 0 else 0.0
        span_rows = int(valid_rows.max() - valid_rows.min() + 1)
        span_ratio_rows = float(span_rows) / max(1.0, float(roi_h))
        center_step_p90 = 0.0
        if valid_centers.size >= 3:
            center_step_p90 = float(np.percentile(np.abs(np.diff(valid_centers)), 90))

        # キャリブレーション用に幅を保存（too_wide で棄却される前でも記録）
        with self._lock:
            self._last_width_med_norm = width_med / max(1.0, float(w))

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
        if post_reject_reason == "too_wide":
            with self._lock:
                calib_w_norm = self._calibrated_width_norm
            if calib_w_norm is not None and calib_w_norm > 0.0:
                # valid_centers は上→下でソート済み → 端点差分でカーブ度を推定
                curve_disp = abs(float(valid_centers[-1]) - float(valid_centers[0]))
                curve_ratio = curve_disp / max(1.0, float(w))
                width_scale = 1.0 + 2.5 * curve_ratio
                if width_med <= calib_w_norm * width_scale * float(w):
                    post_reject_reason = None
        if post_reject_reason is None and using_soft_recover:
            soft_jitter_limit = max(14.0, 0.09 * float(w))
            if center_step_p90 > soft_jitter_limit:
                post_reject_reason = "soft_jitter"
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
            if self._apply_simple_line_fallback(vis, roi_top, y1, y2, h, w, debug_overlay_enabled):
                return vis
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
        if debug_overlay_enabled and centerline_points.shape[0] >= 2:
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
            if debug_overlay_enabled:
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
            ("Far", "top", roi_top, y1, (roi_top + y1) // 2),
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

            # 単一行だと行ノイズで揺れやすいので近傍行の中央値を採用
            win_half = 5
            wy0 = int(max(valid_row_min, ay - win_half))
            wy1 = int(min(valid_row_max, ay + win_half))
            row_win = np.arange(wy0, wy1 + 1, dtype=np.int32)
            px_f = float(np.median(smooth_centers[row_win])) if row_win.size > 0 else float(smooth_centers[ay])
            wz_f = float(np.median(interp_widths[row_win])) if row_win.size > 0 else float(interp_widths[ay])
            px = int(np.clip(np.round(px_f), 0, w - 1))
            wz = int(max(0.0, wz_f))

            # guide_learn用の保存特徴量（0/1フラグ + 中心からの正規化距離 + 幅）
            offset_norm = float((px - (w / 2.0)) / max(w / 2.0, 1.0))
            width_norm = float(wz) / max(1.0, float(w))
            features[f"line_detect_{zone_key}"] = 1.0
            features[f"line_offset_{zone_key}"] = float(np.clip(offset_norm, -1.0, 1.0))
            features[f"line_width_{zone_key}"] = float(np.clip(width_norm, 0.0, 1.0))

            if debug_overlay_enabled:
                py = ay_global  # 表示位置は固定深度
                cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    f"{name} x={px} y={py} w={wz}px off={offset_norm:+.2f}",
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
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

        feature_socket = None
        if self.is_remote_line_feature_mode():
            try:
                feature_socket = ctx.socket(zmq.PULL)
                feature_socket.setsockopt(zmq.CONFLATE, 1)
                feature_socket.connect(f"tcp://{self.pi_ip}:{int(self.line_feature_port)}")
            except Exception as exc:
                self.logger.warning("Feature socket setup failed: %s", exc)
                feature_socket = None

        frame_sizes = {
            320 * 240 * 3: (240, 320),
            640 * 480 * 3: (480, 640),
        }

        try:
            while self._running:
                received_any = False
                try:
                    data = cam_socket.recv(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    data = None
                try:
                    if data is None:
                        raise ValueError("no_frame")
                    shape_hw = frame_sizes.get(len(data))
                    if shape_hw is None:
                        # JPEG経路（BGRで復元）
                        jpeg_buf = np.frombuffer(data, dtype=np.uint8)
                        img = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
                        if img is None:
                            self.logger.debug("Unexpected frame size: %d", len(data))
                        else:
                            with self._lock:
                                self._image = img
                            received_any = True
                    else:
                        # 後方互換: 旧raw経路（RGB -> BGR）
                        h, w = shape_hw
                        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        with self._lock:
                            self._image = img
                        received_any = True
                except ValueError:
                    pass
                except Exception as exc:
                    self.logger.debug("Frame decode failed: %s", exc)

                if feature_socket is not None:
                    try:
                        packet = feature_socket.recv_json(flags=zmq.NOBLOCK)
                        payload = packet.get("line_features", packet) if isinstance(packet, dict) else None
                        if isinstance(payload, dict):
                            normalized = self._normalize_line_features(payload)
                            with self._lock:
                                self._latest_line_features = normalized
                                self._last_remote_feature_ts = time.perf_counter()
                            received_any = True
                    except zmq.ZMQError:
                        pass
                    except Exception as exc:
                        self.logger.debug("Remote feature decode failed: %s", exc)

                if not received_any:
                    time.sleep(0.005)
        finally:
            cam_socket.close()
            if feature_socket is not None:
                feature_socket.close()
            ctx.term()
