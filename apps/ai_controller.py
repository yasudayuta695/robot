from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


class LineTraceONNXController:
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
        smoothing_alpha: float = 0.5,
        max_motor_speed: int = 100,
        stop_on_no_line: bool = True,
        curve_slowdown_sensitivity: float = 0.7,
        curve_slowdown_min_scale: float = 0.45,
        no_line_hold_frames: int = 10,
        no_line_brake_frames: int = 20,
    ) -> None:
        if history <= 0:
            raise ValueError("history must be >= 1")
        self.history = int(history)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_motor_speed = int(max(1, max_motor_speed))
        self.stop_on_no_line = bool(stop_on_no_line)
        self.curve_slowdown_sensitivity = float(np.clip(curve_slowdown_sensitivity, 0.0, 2.0))
        self.curve_slowdown_min_scale = float(np.clip(curve_slowdown_min_scale, 0.05, 1.0))
        self.no_line_hold_frames = max(0, int(no_line_hold_frames))
        self.no_line_brake_frames = max(1, int(no_line_brake_frames))

        self._net: Optional[cv2.dnn_Net] = None
        self._model_path: str = ""
        self._feature_history: Deque[np.ndarray] = deque(maxlen=self.history)
        self._left_norm_prev: float = 0.0
        self._right_norm_prev: float = 0.0
        self._no_line_frames: int = 0

    @property
    def model_path(self) -> str:
        return self._model_path

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
        self._no_line_frames = 0

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

        line_detect_sum = (
            float(line_features.get("line_detect_top", 0.0))
            + float(line_features.get("line_detect_mid", 0.0))
            + float(line_features.get("line_detect_bottom", 0.0))
        )
        if self.stop_on_no_line and line_detect_sum <= 0.0:
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

        inp = self._build_window_input()
        self._net.setInput(inp)
        output = self._net.forward()
        out = np.asarray(output, dtype=np.float32).reshape(-1)

        if out.size < 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {output.shape}")

        left_norm = float(np.clip(out[0], -1.0, 1.0))
        right_norm = float(np.clip(out[1], -1.0, 1.0))

        # Curve-aware slow-down: reduce speed when far/near offsets diverge.
        top_offset = float(np.clip(line_features.get("line_offset_top", 0.0), -1.0, 1.0))
        bottom_offset = float(np.clip(line_features.get("line_offset_bottom", 0.0), -1.0, 1.0))
        curvature = abs(top_offset - bottom_offset)
        slowdown = 1.0 - (self.curve_slowdown_sensitivity * curvature)
        slowdown = float(np.clip(slowdown, self.curve_slowdown_min_scale, 1.0))
        left_norm *= slowdown
        right_norm *= slowdown

        a = self.smoothing_alpha
        left_norm_smooth = (1.0 - a) * self._left_norm_prev + a * left_norm
        right_norm_smooth = (1.0 - a) * self._right_norm_prev + a * right_norm
        self._left_norm_prev = left_norm_smooth
        self._right_norm_prev = right_norm_smooth
        return self._norm_to_speed(left_norm_smooth, right_norm_smooth)
