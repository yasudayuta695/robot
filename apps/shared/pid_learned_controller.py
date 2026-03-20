from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


class LearnedPIDONNXController:
    FEATURE_KEYS = [
        "line_detect_top",
        "line_detect_mid",
        "line_detect_bottom",
        "line_offset_top",
        "line_offset_mid",
        "line_offset_bottom",
        "line_width_top",
        "line_width_mid",
        "line_width_bottom",
    ]

    def __init__(
        self,
        history: int = 10,
        smoothing_alpha: float = 0.35,
        max_motor_speed: int = 100,
        stop_on_no_line: bool = True,
        steer_limit: float = 1.0,
        no_line_hold_frames: int = 3,
        no_line_brake_frames: int = 8,
        gain_smoothing_alpha: float = 0.25,
        steer_rate_limit: float = 0.18,
    ) -> None:
        if history <= 0:
            raise ValueError("history must be >= 1")

        self.history = int(history)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_motor_speed = int(max(1, max_motor_speed))
        self.stop_on_no_line = bool(stop_on_no_line)
        self.steer_limit = float(max(1e-6, abs(steer_limit)))
        self.no_line_hold_frames = max(0, int(no_line_hold_frames))
        self.no_line_brake_frames = max(1, int(no_line_brake_frames))
        self.gain_smoothing_alpha = float(np.clip(gain_smoothing_alpha, 0.0, 1.0))
        self.steer_rate_limit = float(max(1e-4, abs(steer_rate_limit)))

        self._net: Optional[cv2.dnn_Net] = None
        self._model_path: str = ""
        self._feature_history: Deque[np.ndarray] = deque(maxlen=self.history)
        self._left_norm_prev: float = 0.0
        self._right_norm_prev: float = 0.0
        self._last_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._no_line_frames: int = 0
        self._last_steer: float = 0.0

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def last_gains(self) -> Tuple[float, float, float]:
        return self._last_gains

    def is_loaded(self) -> bool:
        return self._net is not None

    def load_model(self, onnx_path: str) -> None:
        net = cv2.dnn.readNetFromONNX(onnx_path)
        self._net = net
        self._model_path = onnx_path
        self.reset_state()

    def reset_state(self) -> None:
        self._feature_history.clear()
        self._left_norm_prev = 0.0
        self._right_norm_prev = 0.0
        self._last_gains = (0.0, 0.0, 0.0)
        self._no_line_frames = 0
        self._last_steer = 0.0

    def _norm_to_speed(self, left_norm: float, right_norm: float) -> Tuple[int, int]:
        left_speed = int(np.clip(np.round(left_norm * 100.0), -self.max_motor_speed, self.max_motor_speed))
        right_speed = int(np.clip(np.round(right_norm * 100.0), -self.max_motor_speed, self.max_motor_speed))
        return left_speed, right_speed

    def _make_frame_feature(
        self,
        line_features: Dict[str, float],
        current_left_speed: int,
        current_right_speed: int,
        base_speed: int,
    ) -> np.ndarray:
        values: List[float] = []

        for key in self.FEATURE_KEYS:
            raw = float(line_features.get(key, 0.0))
            if key.startswith("line_detect") or key.startswith("line_width"):
                values.append(float(np.clip(raw, 0.0, 1.0)))
            else:
                values.append(float(np.clip(raw, -1.0, 1.0)))

        current_left_norm = float(np.clip(float(current_left_speed) / 100.0, -1.0, 1.0))
        current_right_norm = float(np.clip(float(current_right_speed) / 100.0, -1.0, 1.0))
        base_speed_norm = float(np.clip(float(base_speed) / 100.0, 0.0, 1.0))

        values.append(current_left_norm)
        values.append(current_right_norm)
        values.append(base_speed_norm)

        return np.asarray(values, dtype=np.float32)

    def _build_window_input(self) -> np.ndarray:
        history_list = list(self._feature_history)
        if not history_list:
            raise RuntimeError("No feature history available for inference.")

        if len(history_list) < self.history:
            pad_count = self.history - len(history_list)
            history_list = [history_list[0]] * pad_count + history_list

        flat = np.concatenate(history_list, axis=0).astype(np.float32)
        frame_dim = len(self.FEATURE_KEYS) + 3  # +3: current_left, current_right, base_speed
        return flat.reshape(1, self.history * frame_dim)

    def _pid_terms_from_window(self, window_flat: np.ndarray) -> Tuple[float, float, float, float, float]:
        frame_dim = len(self.FEATURE_KEYS) + 3  # +3: current_left, current_right, base_speed
        seq = np.asarray(window_flat, dtype=np.float32).reshape(self.history, frame_dim)
        detect = np.clip(seq[:, 0:3], 0.0, 1.0)
        offset = np.clip(seq[:, 3:6], -1.0, 1.0)
        zone_weights = np.asarray([0.20, 0.35, 0.45], dtype=np.float32).reshape(1, 3)

        weighted_detect = detect * zone_weights
        weighted_sum = np.sum(weighted_detect, axis=1)
        weighted_error_num = np.sum(weighted_detect * offset, axis=1)

        effective_error = np.zeros_like(weighted_sum, dtype=np.float32)
        valid = weighted_sum > 0.01
        effective_error[valid] = weighted_error_num[valid] / np.maximum(weighted_sum[valid], 1e-6)

        error = float(effective_error[-1])
        if np.any(valid):
            integral = float(np.sum(effective_error * valid.astype(np.float32)) / max(1.0, float(np.sum(valid))))
        else:
            integral = 0.0
        derivative = float(effective_error[-1] - effective_error[-2]) if self.history >= 2 else 0.0
        base_speed_norm = float(np.clip(seq[-1, frame_dim - 1], 0.0, 1.0))
        detect_sum = float(np.sum(seq[-1, 0:3]))

        return error, integral, derivative, base_speed_norm, detect_sum

    def predict_motor_speed(
        self,
        line_features: Dict[str, float],
        current_left_speed: int,
        current_right_speed: int,
        base_speed: int,
    ) -> Tuple[int, int]:
        if self._net is None:
            raise RuntimeError("ONNX model is not loaded.")

        frame_feature = self._make_frame_feature(
            line_features=line_features,
            current_left_speed=current_left_speed,
            current_right_speed=current_right_speed,
            base_speed=base_speed,
        )
        self._feature_history.append(frame_feature)

        inp = self._build_window_input()
        error, integral, derivative, base_speed_norm, detect_sum = self._pid_terms_from_window(inp.reshape(-1))

        if self.stop_on_no_line and detect_sum <= 0.0:
            self._no_line_frames += 1
            if self._no_line_frames <= self.no_line_hold_frames:
                return self._norm_to_speed(self._left_norm_prev, self._right_norm_prev)

            brake_step = self._no_line_frames - self.no_line_hold_frames
            if brake_step <= self.no_line_brake_frames:
                decay = 1.0 - (float(brake_step) / float(self.no_line_brake_frames))
                decay = float(np.clip(decay, 0.0, 1.0))
                min_decay = 0.15
                scale = min_decay + ((1.0 - min_decay) * decay)
                left_norm = self._left_norm_prev * scale
                right_norm = self._right_norm_prev * scale
                return self._norm_to_speed(left_norm, right_norm)

            self._left_norm_prev = 0.0
            self._right_norm_prev = 0.0
            return 0, 0

        self._no_line_frames = 0

        self._net.setInput(inp)
        output = self._net.forward()
        out = np.asarray(output, dtype=np.float32).reshape(-1)
        if out.size < 3:
            raise RuntimeError(f"Unexpected ONNX output shape for PID gains: {output.shape}")

        kp_raw = float(np.clip(out[0], 0.0, 5.0))
        ki_raw = float(np.clip(out[1], 0.0, 5.0))
        kd_raw = float(np.clip(out[2], 0.0, 5.0))

        g = self.gain_smoothing_alpha
        kp_prev, ki_prev, kd_prev = self._last_gains
        kp = ((1.0 - g) * kp_prev) + (g * kp_raw)
        ki = ((1.0 - g) * ki_prev) + (g * ki_raw)
        kd = ((1.0 - g) * kd_prev) + (g * kd_raw)
        self._last_gains = (kp, ki, kd)

        steer_target = (kp * error) + (ki * integral) + (kd * derivative)
        steer_target = float(np.clip(steer_target, -self.steer_limit, self.steer_limit))

        delta = steer_target - self._last_steer
        max_delta = self.steer_rate_limit
        if delta > max_delta:
            steer = self._last_steer + max_delta
        elif delta < -max_delta:
            steer = self._last_steer - max_delta
        else:
            steer = steer_target
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        self._last_steer = steer

        left_norm = float(np.clip(base_speed_norm + steer, -1.0, 1.0))
        right_norm = float(np.clip(base_speed_norm - steer, -1.0, 1.0))

        a = self.smoothing_alpha
        left_norm_smooth = ((1.0 - a) * self._left_norm_prev) + (a * left_norm)
        right_norm_smooth = ((1.0 - a) * self._right_norm_prev) + (a * right_norm)
        self._left_norm_prev = left_norm_smooth
        self._right_norm_prev = right_norm_smooth

        return self._norm_to_speed(left_norm_smooth, right_norm_smooth)
