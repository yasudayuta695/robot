from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class CnnMotorONNXController:
    """
    End-to-End CNN モータ制御コントローラ。
    OpenCV DNN は複数入力 ONNX 非対応のため onnxruntime を使用。

    Inputs (ONNX):
        "image"  : (1, 1, img_h, img_w)  float32 in [0, 1]  ← セグメンテーションマスク
        "scalars": (1, 3)                 float32            ← [left_norm, right_norm, base_speed_norm]
    Output:
        "motor_output": (1, 2)  float32 in [-1, 1]          ← [left_norm, right_norm]
    """

    def __init__(
        self,
        img_h: int = 60,
        img_w: int = 80,
        smoothing_alpha: float = 0.35,
        max_motor_speed: int = 100,
        no_line_min_pixels: int = 10,
        no_line_hold_frames: int = 3,
        no_line_brake_frames: int = 8,
    ) -> None:
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_motor_speed = int(max(1, max_motor_speed))
        self.no_line_min_pixels = int(max(1, no_line_min_pixels))
        self.no_line_hold_frames = max(0, int(no_line_hold_frames))
        self.no_line_brake_frames = max(1, int(no_line_brake_frames))

        self._session: Optional[ort.InferenceSession] = None
        self._model_path: str = ""
        self._left_norm_prev: float = 0.0
        self._right_norm_prev: float = 0.0
        self._no_line_frames: int = 0

    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    def load_model(self, onnx_path: str) -> None:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._session = sess
        self._model_path = onnx_path
        self.reset_state()

    def reset_state(self) -> None:
        self._left_norm_prev = 0.0
        self._right_norm_prev = 0.0
        self._no_line_frames = 0

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

        # 前処理
        resized = cv2.resize(mask_gray, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        img_blob = resized.astype(np.float32) / 255.0
        img_blob = img_blob[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

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

        # EMA 平滑化
        a = self.smoothing_alpha
        left_smooth = (1.0 - a) * self._left_norm_prev + a * left_norm
        right_smooth = (1.0 - a) * self._right_norm_prev + a * right_norm
        self._left_norm_prev = left_smooth
        self._right_norm_prev = right_smooth

        return self._norm_to_speed(left_smooth, right_smooth)
