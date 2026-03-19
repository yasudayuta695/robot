import logging
import os
import threading
import tkinter as tk
import time
import datetime
import cv2
import numpy as np
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageTk

from lstm_pid_controller import LSTMPIDONNXController
from pid_learned_controller import LearnedPIDONNXController
from camera_receiver import CameraReceiver
from config_loader import AppConfig, ensure_config_file, load_config, migrate_legacy_config_if_needed
from control_logic import DriveParams, compute_dpad_command, compute_joystick_command
from motor_client import MotorClient
from path_utils import sanitize_for_dirname, to_wsl_unc_path
from recorder import DataRecorder, RecorderState


PI_IP = "192.168.23.162"
CENTER = 150
RADIUS = 120
STICK_RADIUS = 30
DEADZONE = 35
CAMERA_PORT = 5556
LINE_FEATURE_PORT = 5557
MOTOR_PORT = 5555
LEFT_SIGN = 1
RIGHT_SIGN = 1
PIVOT_Y_THRESHOLD = 0.15
PIVOT_TURN_GAIN = 0.25
DRIVE_SPEED = 60
REVERSE_SPEED_SCALE = 0.8
REVERSE_STRAIGHT_X_THRESHOLD = 0.2
OUTER_SPEED_SCALE = 1.05
INNER_SPEED_SCALE = 0.9
DEFAULT_DPAD_DRIVE_SPEED_SCALE = 1.0
DEFAULT_DPAD_TURN_SPEED_SCALE = 30.0 / 60.0
DEFAULT_AI_MODEL_REL_PATH = os.path.join("robot_MLP", "model.onnx")
DATA_RECORD_INTERVAL_SEC = 0.1
EMERGENCY_STOP_KEYS = {"space", "Escape"}
COORD_STREAM_INTERVAL_SEC = 0.05
COORD_STREAM_MAX_LINES = 220


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class _LineDetectionWorker:
    """find_line() をバックグラウンドスレッドで実行し、UIスレッドのブロックを防ぐ。"""

    def __init__(self, camera_receiver: CameraReceiver, interval_sec: float) -> None:
        self._camera_receiver = camera_receiver
        self._interval = max(0.02, float(interval_sec))
        self._lock = threading.Lock()
        self._latest_vis: Optional[np.ndarray] = None
        self._latest_features: Dict[str, float] = camera_receiver.get_latest_line_features()
        self._remote_stale_logged = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _draw_remote_feature_overlay(frame: np.ndarray, features: Dict[str, float], calib_done: bool = True) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]
        roi_top = int(h * 0.25)
        roi_available = h - roi_top
        y1 = roi_top + (roi_available // 3)
        y2 = roi_top + ((2 * roi_available) // 3)
        if not calib_done:
            cx_guide = w // 2
            cv2.line(vis, (cx_guide, roi_top), (cx_guide, h - 1), (0, 200, 255), 1)
            cv2.putText(vis, "CALIB?", (cx_guide + 4, roi_top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
        zone_rows = {
            "top": (roi_top + y1) // 2,
            "mid": (y1 + y2) // 2,
            "bottom": (y2 + h) // 2,
        }
        zone_colors = {
            "top": (0, 220, 220),
            "mid": (0, 220, 120),
            "bottom": (0, 255, 0),
        }

        cv2.line(vis, (0, y1), (w - 1, y1), (90, 90, 90), 1)
        cv2.line(vis, (0, y2), (w - 1, y2), (90, 90, 90), 1)

        line_points: List[Tuple[int, int]] = []
        detect_count = 0
        for zone_key in ("top", "mid", "bottom"):
            y = int(zone_rows[zone_key])
            color = zone_colors[zone_key]
            detect = float(features.get(f"line_detect_{zone_key}", 0.0))
            if detect <= 0.0:
                continue

            detect_count += 1
            offset = float(np.clip(features.get(f"line_offset_{zone_key}", 0.0), -1.0, 1.0))
            width_norm = float(np.clip(features.get(f"line_width_{zone_key}", 0.0), 0.0, 1.0))
            x = int(np.clip((w * 0.5) + (offset * (w * 0.5)), 0, w - 1))
            half_w = int(max(3, round((width_norm * w) * 0.5)))

            line_points.append((x, y))
            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.line(vis, (max(0, x - half_w), y), (min(w - 1, x + half_w), y), color, 2)
            cv2.putText(
                vis,
                f"{zone_key} x={x} y={y} off={offset:+.2f} w={int(width_norm * w)}",
                (min(w - 120, x + 8), max(14, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        if len(line_points) >= 2:
            pts = np.array(line_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], False, (255, 255, 0), 2)

        status_text = "remote_line: ok" if detect_count > 0 else "remote_line: none"
        status_color = (0, 255, 255) if detect_count > 0 else (0, 120, 255)
        cv2.putText(
            vis,
            status_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2,
            cv2.LINE_AA,
        )
        return vis

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_tick = time.perf_counter()
        while self._running:
            frame = self._camera_receiver.get_latest_frame()
            if self._camera_receiver.is_remote_line_feature_mode():
                features = self._camera_receiver.get_latest_line_features()
                remote_age = self._camera_receiver.get_remote_feature_age_sec()
                has_recent_remote = remote_age is not None and remote_age <= max(1.0, self._interval * 4.0)
                if has_recent_remote:
                    calib_done = self._camera_receiver.get_calibrated_width_norm() is not None
                    vis = self._draw_remote_feature_overlay(frame, features, calib_done=calib_done)
                    self._remote_stale_logged = False
                else:
                    vis = self._camera_receiver.find_line(frame)
                    features = self._camera_receiver.get_latest_line_features()
                    if not self._remote_stale_logged:
                        logging.getLogger("pi_pc_app").warning(
                            "Remote line features stale/missing. Falling back to local find_line()."
                        )
                        self._remote_stale_logged = True
            else:
                vis = self._camera_receiver.find_line(frame)
                features = self._camera_receiver.get_latest_line_features()
            with self._lock:
                self._latest_vis = vis
                self._latest_features = features

            next_tick += self._interval
            sleep_sec = next_tick - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                # 処理が間隔を超過した場合はドリフトをリセット
                next_tick = time.perf_counter()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        with self._lock:
            vis = self._latest_vis.copy() if self._latest_vis is not None else None
            return vis, self._latest_features.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class UnifiedApp:
    def __init__(self, root: tk.Tk) -> None:
        self.logger = logging.getLogger("pi_pc_app")
        self.root = root
        self.root.title("FPV Controller & Data Collector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.wsl_distro_name = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-24.04")
        self.module_path = os.path.abspath(__file__)
        self.module_dir = os.path.dirname(self.module_path)
        self.project_dir = os.path.dirname(self.module_dir)

        self.config_path = os.path.join(self.project_dir, "comfig.txt")
        legacy_config_path = os.path.join(self.project_dir, "drive_types.txt")
        migrate_legacy_config_if_needed(self.config_path, legacy_config_path)
        ensure_config_file(self.config_path, self.project_dir)

        self.config: AppConfig = load_config(
            config_path=self.config_path,
            project_dir=self.project_dir,
            logger=self.logger,
            default_drive_speed=DRIVE_SPEED,
            default_dpad_drive_scale=DEFAULT_DPAD_DRIVE_SPEED_SCALE,
            default_dpad_turn_scale=DEFAULT_DPAD_TURN_SPEED_SCALE,
        )

        self.drive_params = DriveParams(
            left_sign=LEFT_SIGN,
            right_sign=RIGHT_SIGN,
            pivot_y_threshold=PIVOT_Y_THRESHOLD,
            pivot_turn_gain=PIVOT_TURN_GAIN,
            reverse_speed_scale=REVERSE_SPEED_SCALE,
            reverse_straight_x_threshold=REVERSE_STRAIGHT_X_THRESHOLD,
            outer_speed_scale=OUTER_SPEED_SCALE,
            inner_speed_scale=INNER_SPEED_SCALE,
        )

        self.camera_receiver = CameraReceiver(
            PI_IP,
            CAMERA_PORT,
            self.logger,
            line_feature_port=LINE_FEATURE_PORT,
        )
        self.camera_receiver.set_line_detection_profile_name(self.config.line_detection_profile)
        self.camera_receiver.set_line_color_space(self.config.line_color_space)
        self.camera_receiver.set_auto_threshold_enabled(bool(self.config.auto_threshold_enabled))
        self.camera_receiver.set_thresholds(
            far_threshold=int(self.config.far_threshold),
            near_threshold=int(self.config.near_threshold),
        )
        self.camera_receiver.start()

        self.motor_client = MotorClient(PI_IP, MOTOR_PORT, self.logger)
        self.recorder = DataRecorder(self.config.save_base_dir, self.logger)
        self.ai_controller = LearnedPIDONNXController(
            history=10,
            smoothing_alpha=self.config.ai_smoothing_alpha,
            max_motor_speed=100,
            stop_on_no_line=True,
            no_line_hold_frames=self.config.ai_no_line_hold_frames,
            no_line_brake_frames=self.config.ai_no_line_brake_frames,
            gain_smoothing_alpha=self.config.pid_gain_smoothing_alpha,
            steer_rate_limit=self.config.pid_steer_rate_limit,
        )
        self.line_process_interval_sec = max(0.02, float(self.config.line_process_interval_ms) / 1000.0)
        self.ai_control_interval_sec = max(0.02, float(self.config.ai_control_interval_ms) / 1000.0)
        self.ui_update_interval_ms = max(10, int(self.config.ui_update_interval_ms))
        self._last_ai_control_time = 0.0
        self._cached_line_vis = None
        self._cached_line_features = self.camera_receiver.get_latest_line_features()
        self._coord_stream_interval_sec = COORD_STREAM_INTERVAL_SEC
        self._coord_stream_last_ts = 0.0
        self._coord_stream_line_count = 0

        self._line_worker = _LineDetectionWorker(
            camera_receiver=self.camera_receiver,
            interval_sec=self.line_process_interval_sec,
        )
        self._line_worker.start()
        self.ai_model_default_path = os.path.join(self.project_dir, DEFAULT_AI_MODEL_REL_PATH)
        self._ai_error_logged = False

        self.current_l_speed = 0
        self.current_r_speed = 0
        self.emergency_stop_active = False
        self.preview_images: List[ImageTk.PhotoImage] = []
        self.active_dirs: Set[str] = set()
        self.key_to_dir: Dict[str, str] = {
            "Up": "up",
            "Down": "down",
            "Left": "left",
            "Right": "right",
            "w": "up",
            "W": "up",
            "s": "down",
            "S": "down",
            "a": "left",
            "A": "left",
            "d": "right",
            "D": "right",
        }

        self._build_ui()
        if os.path.isfile(self.ai_model_default_path):
            self.ai_model_path_var.set(self.ai_model_default_path)
            self.load_ai_model()
        self._log_paths()
        self.update_camera_frame()

    def _build_ui(self) -> None:
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        self.settings_frame = tk.Frame(self.main_frame)
        self.settings_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        tk.Label(self.settings_frame, text="カメラ:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            self.settings_frame,
            textvariable=self.camera_var,
            values=self.config.camera_ids,
            width=10,
            state="readonly",
        )
        if self.config.default_camera_id in self.config.camera_ids:
            self.camera_combo.current(self.config.camera_ids.index(self.config.default_camera_id))
        else:
            self.camera_combo.current(0)
        self.camera_combo.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(self.settings_frame, text="モデル:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.settings_frame,
            textvariable=self.model_var,
            values=["なし (生データ収集)", "YOLOv8", "Pose"],
            width=15,
        )
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(self.settings_frame, text="速度基準:").pack(side=tk.LEFT)
        self.drive_speed_var = tk.IntVar(value=self.config.drive_speed_default)
        self.drive_speed_scale = tk.Scale(
            self.settings_frame,
            from_=20,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.drive_speed_var,
            command=self.on_drive_speed_change,
            showvalue=False,
            length=140,
        )
        self.drive_speed_scale.pack(side=tk.LEFT)
        self.drive_speed_value_label = tk.Label(self.settings_frame, text=f"{self.get_drive_speed()}%", width=5)
        self.drive_speed_value_label.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(self.settings_frame, text="走行種別:").pack(side=tk.LEFT)
        self.drive_type_var = tk.StringVar(value=self.config.drive_types[0])
        self.drive_type_combo = ttk.Combobox(
            self.settings_frame,
            textvariable=self.drive_type_var,
            values=self.config.drive_types,
            width=14,
            state="readonly",
        )
        self.drive_type_combo.current(0)
        self.drive_type_combo.pack(side=tk.LEFT, padx=(0, 15))

        self.auto_stop_on_zero_var = tk.BooleanVar(value=False)
        self.auto_stop_check = tk.Checkbutton(
            self.settings_frame,
            text="停止(0,0)で自動終了",
            variable=self.auto_stop_on_zero_var,
        )
        self.auto_stop_check.pack(side=tk.LEFT, padx=(0, 15))

        self.record_btn = tk.Button(
            self.settings_frame,
            text="● 録画開始",
            bg="lightgreen",
            font=("", 10, "bold"),
            command=self.toggle_recording,
        )
        self.record_btn.pack(side=tk.LEFT)

        far_threshold, near_threshold = self.camera_receiver.get_thresholds()
        self.threshold_frame = tk.Frame(self.main_frame)
        self.threshold_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        tk.Label(self.threshold_frame, text="黒ライン閾値(遠):").pack(side=tk.LEFT)
        self.far_threshold_var = tk.IntVar(value=far_threshold)
        self.far_threshold_scale = tk.Scale(
            self.threshold_frame,
            from_=40,
            to=180,
            orient=tk.HORIZONTAL,
            variable=self.far_threshold_var,
            command=self.on_far_threshold_change,
            showvalue=False,
            length=140,
        )
        self.far_threshold_scale.pack(side=tk.LEFT)
        self.far_threshold_value_label = tk.Label(self.threshold_frame, text=str(far_threshold), width=4)
        self.far_threshold_value_label.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(self.threshold_frame, text="黒ライン閾値(近):").pack(side=tk.LEFT)
        self.near_threshold_var = tk.IntVar(value=near_threshold)
        self.near_threshold_scale = tk.Scale(
            self.threshold_frame,
            from_=20,
            to=160,
            orient=tk.HORIZONTAL,
            variable=self.near_threshold_var,
            command=self.on_near_threshold_change,
            showvalue=False,
            length=140,
        )
        self.near_threshold_scale.pack(side=tk.LEFT)
        self.near_threshold_value_label = tk.Label(self.threshold_frame, text=str(near_threshold), width=4)
        self.near_threshold_value_label.pack(side=tk.LEFT)

        self.auto_threshold_var = tk.BooleanVar(value=self.camera_receiver.is_auto_threshold_enabled())
        self.auto_threshold_check = tk.Checkbutton(
            self.threshold_frame,
            text="閾値自動補正",
            variable=self.auto_threshold_var,
            command=self.on_toggle_auto_threshold,
        )
        self.auto_threshold_check.pack(side=tk.LEFT, padx=(12, 0))

        self.debug_overlay_var = tk.BooleanVar(value=self.camera_receiver.is_debug_overlay_enabled())
        self.debug_overlay_check = tk.Checkbutton(
            self.threshold_frame,
            text="Debug Overlay",
            variable=self.debug_overlay_var,
            command=self.on_toggle_debug_overlay,
        )
        self.debug_overlay_check.pack(side=tk.LEFT, padx=(12, 0))

        self.offset_filter_frame = tk.Frame(self.main_frame)
        self.offset_filter_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        tk.Label(self.offset_filter_frame, text="Offsetジャンプ閾値(各ゾーン):").pack(side=tk.LEFT)
        self.offset_jump_threshold_var = tk.DoubleVar(value=self.camera_receiver.get_offset_jump_threshold())
        self.offset_jump_threshold_scale = tk.Scale(
            self.offset_filter_frame,
            from_=0.10,
            to=1.00,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.offset_jump_threshold_var,
            command=self.on_offset_jump_threshold_change,
            showvalue=False,
            length=160,
        )
        self.offset_jump_threshold_scale.pack(side=tk.LEFT)
        self.offset_jump_threshold_label = tk.Label(
            self.offset_filter_frame,
            text=f"{self.camera_receiver.get_offset_jump_threshold():.2f}",
            width=5,
        )
        self.offset_jump_threshold_label.pack(side=tk.LEFT)
        tk.Label(self.offset_filter_frame, text="(1.00=無効)", fg="gray50").pack(side=tk.LEFT, padx=(4, 0))

        self.calib_width_frame = tk.Frame(self.main_frame)
        self.calib_width_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        tk.Label(self.calib_width_frame, text="幅キャリブ:").pack(side=tk.LEFT)
        self.calib_width_btn = tk.Button(
            self.calib_width_frame,
            text="直線幅をキャリブ",
            command=self.on_calibrate_width,
        )
        self.calib_width_btn.pack(side=tk.LEFT, padx=(4, 8))
        self.calib_width_label = tk.Label(
            self.calib_width_frame,
            text="未設定",
            width=14,
            fg="gray50",
        )
        self.calib_width_label.pack(side=tk.LEFT)
        self.calib_width_clear_btn = tk.Button(
            self.calib_width_frame,
            text="クリア",
            command=self.on_clear_calibrated_width,
        )
        self.calib_width_clear_btn.pack(side=tk.LEFT, padx=(4, 0))

        self.dpad_sensitivity_frame = tk.Frame(self.main_frame)
        self.dpad_sensitivity_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        tk.Label(self.dpad_sensitivity_frame, text="十字キー感度(前進):").pack(side=tk.LEFT)
        self.dpad_drive_scale_var = tk.DoubleVar(value=float(self.config.dpad_drive_speed_scale))
        self.dpad_drive_scale_slider = tk.Scale(
            self.dpad_sensitivity_frame,
            from_=0.20,
            to=2.00,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.dpad_drive_scale_var,
            command=self.on_dpad_drive_sensitivity_change,
            showvalue=False,
            length=140,
        )
        self.dpad_drive_scale_slider.pack(side=tk.LEFT)
        self.dpad_drive_scale_value_label = tk.Label(
            self.dpad_sensitivity_frame,
            text=f"{self.get_dpad_drive_scale():.2f}",
            width=5,
        )
        self.dpad_drive_scale_value_label.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(self.dpad_sensitivity_frame, text="十字キー感度(旋回):").pack(side=tk.LEFT)
        self.dpad_turn_scale_var = tk.DoubleVar(value=float(self.config.dpad_turn_speed_scale))
        self.dpad_turn_scale_slider = tk.Scale(
            self.dpad_sensitivity_frame,
            from_=0.10,
            to=2.00,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.dpad_turn_scale_var,
            command=self.on_dpad_turn_sensitivity_change,
            showvalue=False,
            length=140,
        )
        self.dpad_turn_scale_slider.pack(side=tk.LEFT)
        self.dpad_turn_scale_value_label = tk.Label(
            self.dpad_sensitivity_frame,
            text=f"{self.get_dpad_turn_scale():.2f}",
            width=5,
        )
        self.dpad_turn_scale_value_label.pack(side=tk.LEFT)

        self.path_info_label = tk.Label(
            self.main_frame,
            text=(
                f"実行ファイル: {to_wsl_unc_path(self.module_path, self.wsl_distro_name)}\n"
                f"設定ファイル: {to_wsl_unc_path(self.config_path, self.wsl_distro_name)}\n"
                f"データ保存ルート: {to_wsl_unc_path(self.config.save_base_dir, self.wsl_distro_name)}"
            ),
            justify="left",
            anchor="w",
            fg="gray30",
        )
        self.path_info_label.pack(fill=tk.X, pady=(0, 10))

        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.TOP)

        self.camera_label = tk.Label(self.bottom_frame, bg="black", width=640, height=480)
        self.camera_label.pack(side=tk.LEFT, padx=10)

        self.coord_panel = tk.LabelFrame(self.bottom_frame, text="座標ストリーム")
        self.coord_panel.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        self.coord_text = tk.Text(
            self.coord_panel,
            width=56,
            height=29,
            font=("Courier", 9),
            wrap="none",
        )
        self.coord_text_scroll = tk.Scrollbar(self.coord_panel, orient=tk.VERTICAL, command=self.coord_text.yview)
        self.coord_text.configure(yscrollcommand=self.coord_text_scroll.set)
        self.coord_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        self.coord_text_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

        self.joy_frame = tk.Frame(self.bottom_frame)
        self.joy_frame.pack(side=tk.LEFT, padx=10)

        self.control_mode = tk.StringVar(value="joystick")
        mode_frame = tk.LabelFrame(self.joy_frame, text="操作モード")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Radiobutton(
            mode_frame,
            text="ジョイスティック",
            variable=self.control_mode,
            value="joystick",
            command=self.on_mode_change,
        ).pack(anchor="w")
        tk.Radiobutton(
            mode_frame,
            text="十字キー",
            variable=self.control_mode,
            value="dpad",
            command=self.on_mode_change,
        ).pack(anchor="w")
        tk.Radiobutton(
            mode_frame,
            text="AI (ONNX)",
            variable=self.control_mode,
            value="ai",
            command=self.on_mode_change,
        ).pack(anchor="w")
        self.emergency_status_var = tk.StringVar(value="Emergency stop standby (hold Space/Esc)")
        self.emergency_status_label = tk.Label(
            mode_frame,
            textvariable=self.emergency_status_var,
            fg="firebrick",
            anchor="w",
        )
        self.emergency_status_label.pack(fill=tk.X, pady=(4, 0))

        self.ai_frame = tk.LabelFrame(self.joy_frame, text="AI Controller")
        self.ai_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.ai_frame, text="ONNX:").grid(row=0, column=0, padx=(6, 4), pady=6, sticky="w")
        self.ai_model_path_var = tk.StringVar(value=self.ai_model_default_path)
        self.ai_model_path_entry = tk.Entry(self.ai_frame, textvariable=self.ai_model_path_var, width=36)
        self.ai_model_path_entry.grid(row=0, column=1, padx=(0, 6), pady=6, sticky="we")

        self.ai_load_btn = tk.Button(self.ai_frame, text="Load", command=self.load_ai_model)
        self.ai_load_btn.grid(row=0, column=2, padx=(0, 6), pady=6)

        self.ai_reset_btn = tk.Button(self.ai_frame, text="Reset", command=self.reset_ai_state)
        self.ai_reset_btn.grid(row=0, column=3, padx=(0, 6), pady=6)

        self.ai_status_var = tk.StringVar(value="AI model: not loaded")
        self.ai_status_label = tk.Label(self.ai_frame, textvariable=self.ai_status_var, fg="gray30", anchor="w")
        self.ai_status_label.grid(row=1, column=0, columnspan=4, padx=6, pady=(0, 6), sticky="we")
        self.ai_frame.grid_columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(self.joy_frame, width=CENTER * 2, height=CENTER * 2, bg="white")
        self.canvas.pack(pady=20)
        self.canvas.create_oval(CENTER - RADIUS, CENTER - RADIUS, CENTER + RADIUS, CENTER + RADIUS, outline="gray")
        self.canvas.create_oval(
            CENTER - DEADZONE,
            CENTER - DEADZONE,
            CENTER + DEADZONE,
            CENTER + DEADZONE,
            outline="lightgray",
            dash=(4, 4),
        )
        self.stick = self.canvas.create_oval(
            CENTER - STICK_RADIUS,
            CENTER - STICK_RADIUS,
            CENTER + STICK_RADIUS,
            CENTER + STICK_RADIUS,
            fill="blue",
        )

        self.dpad_frame = tk.Frame(self.joy_frame)
        self.build_dpad_ui()

        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)

    def _log_paths(self) -> None:
        self.logger.info("実行ファイル: %s", to_wsl_unc_path(self.module_path, self.wsl_distro_name))
        self.logger.info("設定ファイル: %s", to_wsl_unc_path(self.config_path, self.wsl_distro_name))
        self.logger.info("データ保存ルート: %s", to_wsl_unc_path(self.config.save_base_dir, self.wsl_distro_name))

    def _zone_rows_for_frame(self, frame_h: int) -> Dict[str, int]:
        roi_top = int(frame_h * 0.25)
        roi_available = frame_h - roi_top
        y1 = roi_top + (roi_available // 3)
        y2 = roi_top + ((2 * roi_available) // 3)
        return {
            "top": (roi_top + y1) // 2,
            "mid": (y1 + y2) // 2,
            "bottom": (y2 + frame_h) // 2,
        }

    def _extract_zone_coordinate(
        self,
        zone_key: str,
        line_features: Dict[str, float],
        frame_w: int,
        zone_rows: Dict[str, int],
    ) -> Tuple[int, int, float, int, float]:
        detect = float(line_features.get(f"line_detect_{zone_key}", 0.0))
        y = int(zone_rows[zone_key])
        if detect <= 0.0:
            return -1, y, 0.0, 0, 0.0

        offset = float(np.clip(line_features.get(f"line_offset_{zone_key}", 0.0), -1.0, 1.0))
        width_norm = float(np.clip(line_features.get(f"line_width_{zone_key}", 0.0), 0.0, 1.0))
        x = int(np.clip((frame_w * 0.5) + (offset * (frame_w * 0.5)), 0, frame_w - 1))
        width_px = int(max(0, round(width_norm * frame_w)))
        return x, y, offset, width_px, detect

    def _append_coordinate_stream(
        self,
        line_features: Dict[str, float],
        frame_w: int,
        frame_h: int,
    ) -> None:
        if not hasattr(self, "coord_text"):
            return

        now = time.perf_counter()
        if (now - self._coord_stream_last_ts) < self._coord_stream_interval_sec:
            return
        self._coord_stream_last_ts = now

        zone_rows = self._zone_rows_for_frame(frame_h)
        top = self._extract_zone_coordinate("top", line_features, frame_w, zone_rows)
        mid = self._extract_zone_coordinate("mid", line_features, frame_w, zone_rows)
        bottom = self._extract_zone_coordinate("bottom", line_features, frame_w, zone_rows)

        tstr = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = (
            f"{tstr} | "
            f"top(x={top[0]:4d},y={top[1]:3d},off={top[2]:+5.2f},w={top[3]:3d},d={top[4]:.1f}) "
            f"mid(x={mid[0]:4d},y={mid[1]:3d},off={mid[2]:+5.2f},w={mid[3]:3d},d={mid[4]:.1f}) "
            f"btm(x={bottom[0]:4d},y={bottom[1]:3d},off={bottom[2]:+5.2f},w={bottom[3]:3d},d={bottom[4]:.1f})"
        )
        self.coord_text.insert("end", line + "\n")
        self._coord_stream_line_count += 1
        if self._coord_stream_line_count > COORD_STREAM_MAX_LINES:
            self.coord_text.delete("1.0", "2.0")
            self._coord_stream_line_count -= 1
        self.coord_text.see("end")

    def on_drive_speed_change(self, _value: str) -> None:
        self.drive_speed_value_label.config(text=f"{self.get_drive_speed()}%")

    def on_offset_jump_threshold_change(self, _value: str) -> None:
        val = self.camera_receiver.set_offset_jump_threshold(float(self.offset_jump_threshold_var.get()))
        self.offset_jump_threshold_label.config(text=f"{val:.2f}")

    def on_far_threshold_change(self, _value: str) -> None:
        far, _ = self.camera_receiver.set_thresholds(far_threshold=int(self.far_threshold_var.get()))
        self.far_threshold_value_label.config(text=str(far))

    def on_near_threshold_change(self, _value: str) -> None:
        _, near = self.camera_receiver.set_thresholds(near_threshold=int(self.near_threshold_var.get()))
        self.near_threshold_value_label.config(text=str(near))

    def on_toggle_auto_threshold(self) -> None:
        enabled = self.camera_receiver.set_auto_threshold_enabled(bool(self.auto_threshold_var.get()))
        self.auto_threshold_var.set(enabled)
        if enabled:
            self.logger.info("黒ライン閾値の自動補正: ON")
        else:
            self.logger.info("黒ライン閾値の自動補正: OFF")

    def on_toggle_debug_overlay(self) -> None:
        enabled = self.camera_receiver.set_debug_overlay_enabled(bool(self.debug_overlay_var.get()))
        self.debug_overlay_var.set(enabled)
        if enabled:
            self.logger.info("Debug Overlay: ON")
        else:
            self.logger.info("Debug Overlay: OFF")

    def on_calibrate_width(self) -> None:
        w_norm = self.camera_receiver.calibrate_straight_width()
        if w_norm is None:
            self.calib_width_label.config(text="検出なし", fg="red")
            self.logger.warning("幅キャリブレーション失敗: 直線検出データなし")
        else:
            self.calib_width_label.config(text=f"{w_norm:.3f} (norm)", fg="green")
            self.logger.info("幅キャリブレーション設定: %.3f (norm)", w_norm)

    def on_clear_calibrated_width(self) -> None:
        self.camera_receiver.clear_calibrated_width()
        self.calib_width_label.config(text="未設定", fg="gray50")
        self.logger.info("幅キャリブレーションをクリア")

    def on_dpad_drive_sensitivity_change(self, _value: str) -> None:
        self.dpad_drive_scale_value_label.config(text=f"{self.get_dpad_drive_scale():.2f}")
        if self.control_mode.get() == "dpad" and self.active_dirs:
            self.apply_dpad_keys()

    def on_dpad_turn_sensitivity_change(self, _value: str) -> None:
        self.dpad_turn_scale_value_label.config(text=f"{self.get_dpad_turn_scale():.2f}")
        if self.control_mode.get() == "dpad" and self.active_dirs:
            self.apply_dpad_keys()

    def get_drive_speed(self) -> int:
        return max(0, min(100, int(self.drive_speed_var.get())))

    def get_dpad_drive_scale(self) -> float:
        return max(0.05, float(self.dpad_drive_scale_var.get()))

    def get_dpad_turn_scale(self) -> float:
        return max(0.05, float(self.dpad_turn_scale_var.get()))

    def get_dpad_drive_speed(self) -> int:
        return int(self.get_drive_speed() * self.get_dpad_drive_scale())

    def get_dpad_turn_speed(self) -> int:
        return int(self.get_drive_speed() * self.get_dpad_turn_scale())

    def toggle_recording(self) -> None:
        if self.recorder.state == RecorderState.IDLE:
            try:
                self.recorder.arm()
            except Exception as exc:
                save_root = to_wsl_unc_path(self.config.save_base_dir, self.wsl_distro_name)
                self.logger.exception("録画待機への切替に失敗しました。save_base_dir=%s", save_root)
                messagebox.showerror(
                    "録画開始エラー",
                    "録画開始の準備に失敗しました。\n"
                    f"保存先: {save_root}\n"
                    f"詳細: {exc}",
                )
                return
            self.record_btn.config(text="■ 録画待機中", bg="khaki")
            self.logger.info("録画待機中: 操作入力を待っています。")
            return

        self.stop_recording_flow("")

    def stop_recording_flow(self, reason: str) -> None:
        self.recorder.stop()
        self.record_btn.config(text="● 録画開始", bg="lightgreen")
        if reason:
            self.logger.info(reason)
        if self.recorder.has_temp_data():
            self.open_preview_dialog()

    def open_preview_dialog(self) -> None:
        image_files = self.recorder.list_temp_images()

        preview = tk.Toplevel(self.root)
        preview.title("保存前プレビュー")
        preview.transient(self.root)
        preview.grab_set()
        preview.geometry("900x520")

        info_text = (
            f"走行種別: {self.drive_type_var.get()}\n"
            f"撮影枚数: {len(image_files)}\n"
            f"保存先: {to_wsl_unc_path(self.get_final_base_dir(), self.wsl_distro_name)}"
        )
        tk.Label(preview, text=info_text, justify="left", anchor="w").pack(fill=tk.X, padx=12, pady=(12, 8))

        thumbs_frame = tk.Frame(preview)
        thumbs_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.preview_images = []
        if image_files:
            candidate_indices = sorted({0, len(image_files) // 2, len(image_files) - 1})
            for idx in candidate_indices:
                img_path = os.path.join(self.recorder.temp_img_dir, image_files[idx])
                try:
                    pil = Image.open(img_path)
                    pil.thumbnail((260, 180))
                    thumb = ImageTk.PhotoImage(pil)
                    self.preview_images.append(thumb)
                    card = tk.Frame(thumbs_frame, bd=1, relief="solid")
                    card.pack(side=tk.LEFT, padx=8, pady=8)
                    tk.Label(card, image=thumb).pack(padx=6, pady=6)
                    tk.Label(card, text=image_files[idx], wraplength=250).pack(padx=6, pady=(0, 6))
                except Exception as exc:
                    self.logger.debug("Preview image load failed: %s", exc)
        else:
            tk.Label(thumbs_frame, text="プレビュー可能な画像がありません。", fg="gray40").pack(anchor="w", padx=8, pady=8)

        button_row = tk.Frame(preview)
        button_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        def save_and_close() -> None:
            preview.destroy()
            self.commit_data()

        def discard_and_close() -> None:
            preview.destroy()
            self.discard_data()

        tk.Button(button_row, text="保存する", bg="lightgreen", command=save_and_close).pack(side=tk.LEFT)
        tk.Button(button_row, text="破棄する", bg="tomato", command=discard_and_close).pack(side=tk.LEFT, padx=8)
        tk.Button(button_row, text="キャンセル", command=preview.destroy).pack(side=tk.LEFT)

    def get_final_base_dir(self) -> str:
        folder_name = self.recorder.get_output_folder_name()
        camera_id = sanitize_for_dirname(self.camera_var.get())
        drive_type = sanitize_for_dirname(self.drive_type_var.get())
        return os.path.join(self.config.save_base_dir, "dataset", camera_id, drive_type, folder_name)

    def commit_data(self) -> None:
        base_dir = self.get_final_base_dir()
        self.recorder.commit_data(
            base_dir=base_dir,
            drive_type=self.drive_type_var.get(),
            drive_speed_base=self.get_drive_speed(),
            control_mode=self.control_mode.get(),
            camera_id=self.camera_var.get(),
            recording_duration_sec=self.recorder.get_recording_duration_sec(),
        )
        self.logger.info("データを %s に保存しました。", to_wsl_unc_path(base_dir, self.wsl_distro_name))

    def discard_data(self) -> None:
        self.recorder.discard_data()
        self.logger.info("走行データを破棄しました。")

    def build_dpad_ui(self) -> None:
        up_btn = tk.Button(self.dpad_frame, text="↑", width=6, height=2)
        left_btn = tk.Button(self.dpad_frame, text="←", width=6, height=2)
        stop_btn = tk.Button(self.dpad_frame, text="■", width=6, height=2)
        right_btn = tk.Button(self.dpad_frame, text="→", width=6, height=2)
        down_btn = tk.Button(self.dpad_frame, text="↓", width=6, height=2)

        up_btn.grid(row=0, column=1, padx=4, pady=4)
        left_btn.grid(row=1, column=0, padx=4, pady=4)
        stop_btn.grid(row=1, column=1, padx=4, pady=4)
        right_btn.grid(row=1, column=2, padx=4, pady=4)
        down_btn.grid(row=2, column=1, padx=4, pady=4)

        up_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(self.get_dpad_drive_speed(), self.get_dpad_drive_speed()))
        up_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        down_btn.bind(
            "<ButtonPress-1>",
            lambda _e: self.send_command(
                -int(self.get_dpad_drive_speed() * REVERSE_SPEED_SCALE),
                -int(self.get_dpad_drive_speed() * REVERSE_SPEED_SCALE),
            ),
        )
        down_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        left_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(-self.get_dpad_turn_speed(), self.get_dpad_turn_speed()))
        left_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        right_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(self.get_dpad_turn_speed(), -self.get_dpad_turn_speed()))
        right_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        stop_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(0, 0))

    def on_mode_change(self) -> None:
        self.active_dirs.clear()
        self.send_command(0, 0)
        self.reset_ai_state()
        self._last_ai_control_time = 0.0
        self.canvas.coords(
            self.stick,
            CENTER - STICK_RADIUS,
            CENTER - STICK_RADIUS,
            CENTER + STICK_RADIUS,
            CENTER + STICK_RADIUS,
        )
        if self.control_mode.get() == "joystick":
            self.dpad_frame.pack_forget()
            self.canvas.pack(pady=20)
        elif self.control_mode.get() == "dpad":
            self.canvas.pack_forget()
            self.dpad_frame.pack(pady=20)
        else:
            self.canvas.pack_forget()
            self.dpad_frame.pack_forget()
            if not self.ai_controller.is_loaded():
                self.load_ai_model()

    def on_key_press(self, event: tk.Event) -> None:
        if event.keysym in EMERGENCY_STOP_KEYS:
            self.set_emergency_stop(True)
            self.send_command(0, 0)
            return

        if self.control_mode.get() != "dpad":
            return
        direction = self.key_to_dir.get(event.keysym)
        if direction is None:
            return
        self.active_dirs.add(direction)
        self.apply_dpad_keys()

    def on_key_release(self, event: tk.Event) -> None:
        if event.keysym in EMERGENCY_STOP_KEYS:
            self.set_emergency_stop(False)
            return

        direction = self.key_to_dir.get(event.keysym)
        if direction is None:
            return
        self.active_dirs.discard(direction)
        if self.control_mode.get() == "dpad":
            self.apply_dpad_keys()

    def set_emergency_stop(self, active: bool) -> None:
        if self.emergency_stop_active == active:
            return

        self.emergency_stop_active = active
        if active:
            self.emergency_status_var.set("EMERGENCY STOP ACTIVE (release Space/Esc to resume)")
            self.logger.warning("Emergency stop activated")
        else:
            self.emergency_status_var.set("Emergency stop standby (hold Space/Esc)")
            self.logger.info("Emergency stop released")

    def apply_dpad_keys(self) -> None:
        left_speed, right_speed = compute_dpad_command(
            active_dirs=self.active_dirs,
            dpad_drive_speed=self.get_dpad_drive_speed(),
            dpad_turn_speed=self.get_dpad_turn_speed(),
            reverse_speed_scale=REVERSE_SPEED_SCALE,
        )
        self.send_command(left_speed, right_speed)

    def _make_ai_controller(self, model_path: str):
        """ファイル名に 'lstm' が含まれていれば LSTM コントローラーを返す。"""
        use_lstm = "lstm" in os.path.basename(model_path).lower()
        cls = LSTMPIDONNXController if use_lstm else LearnedPIDONNXController
        return cls(
            history=10,
            smoothing_alpha=self.config.ai_smoothing_alpha,
            max_motor_speed=100,
            stop_on_no_line=True,
            no_line_hold_frames=self.config.ai_no_line_hold_frames,
            no_line_brake_frames=self.config.ai_no_line_brake_frames,
            gain_smoothing_alpha=self.config.pid_gain_smoothing_alpha,
            steer_rate_limit=self.config.pid_steer_rate_limit,
        )

    def load_ai_model(self) -> None:
        model_path = self.ai_model_path_var.get().strip()
        if not model_path:
            self.ai_status_var.set("AI model: path is empty")
            return

        if not os.path.isfile(model_path):
            self.ai_status_var.set(f"AI model not found: {model_path}")
            self.logger.warning("AI model not found: %s", model_path)
            return

        try:
            self.ai_controller = self._make_ai_controller(model_path)
            self.ai_controller.load_model(model_path)
            controller_type = type(self.ai_controller).__name__
            self.ai_status_var.set(f"[{controller_type}] AI model loaded: {model_path}")
            self._ai_error_logged = False
            self.logger.info("AI ONNX model loaded (%s): %s", controller_type, to_wsl_unc_path(model_path, self.wsl_distro_name))
        except Exception as exc:
            self.ai_status_var.set(f"AI load failed: {exc}")
            self.logger.warning("AI model load failed: %s", exc)

    def reset_ai_state(self) -> None:
        self.ai_controller.reset_state()
        self._ai_error_logged = False
        self._last_ai_control_time = 0.0

    def run_ai_control(self, line_features: Dict[str, float]) -> None:
        if not self.ai_controller.is_loaded():
            self.send_command(0, 0)
            return

        try:
            left_speed, right_speed = self.ai_controller.predict_motor_speed(
                line_features=line_features,
                current_left_speed=self.current_l_speed,
                current_right_speed=self.current_r_speed,
                base_speed=self.get_drive_speed(),
            )
            self._ai_error_logged = False
            self.send_command(left_speed, right_speed)
        except Exception as exc:
            if not self._ai_error_logged:
                self.logger.warning("AI inference failed: %s", exc)
                self._ai_error_logged = True
            self.send_command(0, 0)

    def update_camera_frame(self) -> None:
        frame = self.camera_receiver.get_latest_frame()
        now = time.perf_counter()

        # バックグラウンドスレッドから最新のライン検出結果を取得
        line_vis, line_features_from_worker = self._line_worker.get_latest()
        if line_vis is not None:
            self._cached_line_vis = line_vis
            self._cached_line_features = line_features_from_worker

        display_src = self._cached_line_vis if self._cached_line_vis is not None else frame
        display_frame = cv2.cvtColor(display_src, cv2.COLOR_BGR2RGB)
        line_features = self._cached_line_features.copy()
        far_now, near_now = self.camera_receiver.get_thresholds()
        if int(self.far_threshold_var.get()) != far_now:
            self.far_threshold_var.set(far_now)
            self.far_threshold_value_label.config(text=str(far_now))
        if int(self.near_threshold_var.get()) != near_now:
            self.near_threshold_var.set(near_now)
            self.near_threshold_value_label.config(text=str(near_now))

        if self.control_mode.get() == "ai":
            if (now - self._last_ai_control_time) >= self.ai_control_interval_sec:
                self.run_ai_control(line_features)
                self._last_ai_control_time = now
        
        pil_image = Image.fromarray(display_frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        self.camera_label.config(image=tk_image)
        self.camera_label.image = tk_image

        if self.recorder.state == RecorderState.RECORDING:
            self._append_coordinate_stream(
                line_features=line_features,
                frame_w=display_frame.shape[1],
                frame_h=display_frame.shape[0],
            )

        self.recorder.record_frame(
            image_rgb=frame,
            left_speed=self.current_l_speed,
            right_speed=self.current_r_speed,
            line_features=line_features,
            drive_type=self.drive_type_var.get(),
            drive_speed_base=self.get_drive_speed(),
            control_mode=self.control_mode.get(),
            interval_sec=DATA_RECORD_INTERVAL_SEC,
        )

        self.root.after(self.ui_update_interval_ms, self.update_camera_frame)

    def send_command(self, left_speed: int, right_speed: int) -> None:
        if self.emergency_stop_active:
            left_speed = 0
            right_speed = 0

        self.current_l_speed = left_speed
        self.current_r_speed = right_speed

        started = self.recorder.start_recording_if_needed(left_speed, right_speed)
        if started:
            self.record_btn.config(text="■ 録画停止", bg="pink")
            self.logger.info("録画開始: 操作入力を検知しました。")

        if (
            self.recorder.state == RecorderState.RECORDING
            and self.auto_stop_on_zero_var.get()
            and not self.emergency_stop_active
            and left_speed == 0
            and right_speed == 0
        ):
            self.stop_recording_flow("録画自動終了: 停止(0,0)を検知しました。")

        self.motor_client.send(left_speed, right_speed)

    def drag(self, event: tk.Event) -> None:
        if self.control_mode.get() != "joystick":
            return

        dx = event.x - CENTER
        dy = event.y - CENTER

        distance = (dx * dx + dy * dy) ** 0.5
        if distance > RADIUS:
            dx = dx * RADIUS / distance
            dy = dy * RADIUS / distance

        self.canvas.coords(
            self.stick,
            CENTER + dx - STICK_RADIUS,
            CENTER + dy - STICK_RADIUS,
            CENTER + dx + STICK_RADIUS,
            CENTER + dy + STICK_RADIUS,
        )

        left_speed, right_speed = compute_joystick_command(
            dx=dx,
            dy=dy,
            radius=RADIUS,
            deadzone=DEADZONE,
            drive_speed=self.get_drive_speed(),
            params=self.drive_params,
        )
        self.send_command(left_speed, right_speed)

    def release(self, _event: tk.Event) -> None:
        if self.control_mode.get() != "joystick":
            return
        self.canvas.coords(
            self.stick,
            CENTER - STICK_RADIUS,
            CENTER - STICK_RADIUS,
            CENTER + STICK_RADIUS,
            CENTER + STICK_RADIUS,
        )
        self.send_command(0, 0)

    def on_closing(self) -> None:
        self._line_worker.stop()
        self.recorder.close_and_discard_on_exit()
        self.camera_receiver.stop()
        self.motor_client.close()
        self.root.unbind_all("<KeyPress>")
        self.root.unbind_all("<KeyRelease>")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()