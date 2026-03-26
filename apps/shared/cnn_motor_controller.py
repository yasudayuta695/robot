from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class CnnMotorONNXController:
    """
    End-to-End CNN モータ制御コントローラ。
    OpenCV DNN は複数入力 ONNX 非対応のため onnxruntime を使用。

    Inputs (ONNX):
        "image"  : (1, N, img_h, img_w)  float32 in [0, 1]  ← N フレームスタック（N = history）
        "scalars": (1, 3)                 float32            ← [left_norm, right_norm, base_speed_norm]
    Output:
        "motor_output": (1, 2)  float32 in [-1, 1]          ← [left_norm, right_norm]
    """

    def __init__(
        self,
        img_h: int = 60,
        img_w: int = 80,
        history: int = 1,
        smoothing_alpha: float = 0.35,
        steer_rate_limit: float = 0.15,
        max_motor_speed: int = 100,
        no_line_min_pixels: int = 10,
        no_line_hold_frames: int = 3,
        no_line_brake_frames: int = 8,
    ) -> None:
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.history = max(1, int(history))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.steer_rate_limit = float(max(1e-4, abs(steer_rate_limit)))
        self.max_motor_speed = int(max(1, max_motor_speed))
        self.no_line_min_pixels = int(max(1, no_line_min_pixels))
        self.no_line_hold_frames = max(0, int(no_line_hold_frames))
        self.no_line_brake_frames = max(1, int(no_line_brake_frames))

        self._session: Optional[ort.InferenceSession] = None
        self._model_path: str = ""
        self._left_norm_prev: float = 0.0
        self._right_norm_prev: float = 0.0
        self._no_line_frames: int = 0
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.history)

    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    def load_model(self, onnx_path: str) -> None:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # モデルの実際の入力チャンネル数（= history）を読み取って自動設定
        img_shape = sess.get_inputs()[0].shape  # e.g. [1, 5, 60, 80] or ['batch', 1, 60, 80]
        model_history = img_shape[1] if isinstance(img_shape[1], int) and img_shape[1] > 0 else self.history
        if model_history != self.history:
            import logging
            logging.getLogger(__name__).info(
                "CNN model history mismatch: controller=%d, model=%d → using model value",
                self.history, model_history,
            )
            self.history = model_history
            self._frame_buffer = deque(maxlen=self.history)
        self._session = sess
        self._model_path = onnx_path
        self.reset_state()

    def reset_state(self) -> None:
        self._left_norm_prev = 0.0
        self._right_norm_prev = 0.0
        self._no_line_frames = 0
        self._frame_buffer.clear()

    def _norm_to_speed(self, left_norm: float, right_norm: float) -> Tuple[int, int]:
        left = int(np.clip(round(left_norm * 100.0), -self.max_motor_speed, self.max_motor_speed))
        right = int(np.clip(round(right_norm * 100.0), -self.max_motor_speed, self.max_motor_speed))
        return left, right

    def predict_motor_speed(
        self,
        mask_gray: np.ndarray,
        current_left_speed: int,
        current_right_speed: int,
        base_speed: int,
    ) -> Tuple[int, int]:
        if self._session is None:
            raise RuntimeError("CNN model is not loaded.")

        # ライン検出判定（マスクの白画素数で判断）
        n_pixels = int(np.count_nonzero(mask_gray))
        no_line = n_pixels < self.no_line_min_pixels

        if no_line:
            self._no_line_frames += 1
            if self._no_line_frames <= self.no_line_hold_frames:
                return self._norm_to_speed(self._left_norm_prev, self._right_norm_prev)
            brake_step = self._no_line_frames - self.no_line_hold_frames
            if brake_step <= self.no_line_brake_frames:
                decay = 1.0 - float(brake_step) / float(self.no_line_brake_frames)
                scale = 0.15 + 0.85 * float(np.clip(decay, 0.0, 1.0))
                return self._norm_to_speed(
                    self._left_norm_prev * scale,
                    self._right_norm_prev * scale,
                )
            self._left_norm_prev = 0.0
            self._right_norm_prev = 0.0
            return 0, 0

        self._no_line_frames = 0

        # 前処理：現フレームをバッファに追加し、N フレームをスタック
        resized = cv2.resize(mask_gray, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        frame = resized.astype(np.float32) / 255.0  # (H, W)
        self._frame_buffer.append(frame)

        frames = list(self._frame_buffer)
        # バッファが history に満たない場合は先頭フレームで埋める
        while len(frames) < self.history:
            frames.insert(0, frames[0])
        img_blob = np.stack(frames, axis=0)[np.newaxis, :, :, :]  # (1, N, H, W)

        scalar_blob = np.array(
            [[
                float(np.clip(current_left_speed / 100.0, -1.0, 1.0)),
                float(np.clip(current_right_speed / 100.0, -1.0, 1.0)),
                float(np.clip(base_speed / 100.0, 0.0, 1.0)),
            ]],
            dtype=np.float32,
        )  # (1, 3)

        outputs = self._session.run(None, {"image": img_blob, "scalars": scalar_blob})
        out = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        left_norm = float(np.clip(out[0], -1.0, 1.0))
        right_norm = float(np.clip(out[1], -1.0, 1.0))

        # EMA 平滑化 + 出力変化率の制限
        prev_left = self._left_norm_prev
        prev_right = self._right_norm_prev
        a = self.smoothing_alpha
        left_smooth = (1.0 - a) * prev_left + a * left_norm
        right_smooth = (1.0 - a) * prev_right + a * right_norm

        max_delta = self.steer_rate_limit
        left_limited = prev_left + float(np.clip(left_smooth - prev_left, -max_delta, max_delta))
        right_limited = prev_right + float(np.clip(right_smooth - prev_right, -max_delta, max_delta))

        self._left_norm_prev = float(np.clip(left_limited, -1.0, 1.0))
        self._right_norm_prev = float(np.clip(right_limited, -1.0, 1.0))

        return self._norm_to_speed(self._left_norm_prev, self._right_norm_prev)
