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
    ]

    def __init__(
        self,
        history: int = 10,
        smoothing_alpha: float = 0.35,
        max_motor_speed: int = 100,
        stop_on_no_line: bool = True,
        steer_limit: float = 1.0,
    ) -> None:
        if history <= 0:
            raise ValueError("history must be >= 1")

        self.history = int(history)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_motor_speed = int(max(1, max_motor_speed))
        self.stop_on_no_line = bool(stop_on_no_line)
        self.steer_limit = float(max(1e-6, abs(steer_limit)))

        self._net: Optional[cv2.dnn_Net] = None
        self._model_path: str = ""
        self._feature_history: Deque[np.ndarray] = deque(maxlen=self.history)
        self._left_norm_prev: float = 0.0
        self._right_norm_prev: float = 0.0
        self._last_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)

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
            if key.startswith("line_detect"):
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
        return flat.reshape(1, self.history * 9)

    def _pid_terms_from_window(self, window_flat: np.ndarray) -> Tuple[float, float, float, float, float]:
        seq = np.asarray(window_flat, dtype=np.float32).reshape(self.history, 9)
        detect_mid = np.clip(seq[:, 1], 0.0, 1.0)
        offset_mid = np.clip(seq[:, 4], -1.0, 1.0)
        effective_error = detect_mid * offset_mid

        error = float(effective_error[-1])
        integral = float(np.mean(effective_error))
        derivative = float(effective_error[-1] - effective_error[-2]) if self.history >= 2 else 0.0
        base_speed_norm = float(np.clip(seq[-1, 8], 0.0, 1.0))
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
            self._left_norm_prev = 0.0
            self._right_norm_prev = 0.0
            return 0, 0

        self._net.setInput(inp)
        output = self._net.forward()
        out = np.asarray(output, dtype=np.float32).reshape(-1)
        if out.size < 3:
            raise RuntimeError(f"Unexpected ONNX output shape for PID gains: {output.shape}")

        kp = float(np.clip(out[0], 0.0, 5.0))
        ki = float(np.clip(out[1], 0.0, 5.0))
        kd = float(np.clip(out[2], 0.0, 5.0))
        self._last_gains = (kp, ki, kd)

        steer = (kp * error) + (ki * integral) + (kd * derivative)
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        left_norm = float(np.clip(base_speed_norm + steer, -1.0, 1.0))
        right_norm = float(np.clip(base_speed_norm - steer, -1.0, 1.0))

        a = self.smoothing_alpha
        left_norm_smooth = ((1.0 - a) * self._left_norm_prev) + (a * left_norm)
        right_norm_smooth = ((1.0 - a) * self._right_norm_prev) + (a * right_norm)
        self._left_norm_prev = left_norm_smooth
        self._right_norm_prev = right_norm_smooth

        return self._norm_to_speed(left_norm_smooth, right_norm_smooth)
