import logging
import os
import tkinter as tk
import cv2
from tkinter import ttk
from typing import Dict, List, Set, Tuple

from PIL import Image, ImageTk

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


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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

        self.camera_receiver = CameraReceiver(PI_IP, CAMERA_PORT, self.logger)
        self.camera_receiver.start()

        self.motor_client = MotorClient(PI_IP, MOTOR_PORT, self.logger)
        self.recorder = DataRecorder(self.config.save_base_dir, self.logger)

        self.current_l_speed = 0
        self.current_r_speed = 0
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

    def on_drive_speed_change(self, _value: str) -> None:
        self.drive_speed_value_label.config(text=f"{self.get_drive_speed()}%")

    def get_drive_speed(self) -> int:
        return max(0, min(100, int(self.drive_speed_var.get())))

    def get_dpad_drive_speed(self) -> int:
        return int(self.get_drive_speed() * self.config.dpad_drive_speed_scale)

    def get_dpad_turn_speed(self) -> int:
        return int(self.get_drive_speed() * self.config.dpad_turn_speed_scale)

    def toggle_recording(self) -> None:
        if self.recorder.state == RecorderState.IDLE:
            self.recorder.arm()
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
        else:
            self.canvas.pack_forget()
            self.dpad_frame.pack(pady=20)

    def on_key_press(self, event: tk.Event) -> None:
        if self.control_mode.get() != "dpad":
            return
        direction = self.key_to_dir.get(event.keysym)
        if direction is None:
            return
        self.active_dirs.add(direction)
        self.apply_dpad_keys()

    def on_key_release(self, event: tk.Event) -> None:
        direction = self.key_to_dir.get(event.keysym)
        if direction is None:
            return
        self.active_dirs.discard(direction)
        if self.control_mode.get() == "dpad":
            self.apply_dpad_keys()

    def apply_dpad_keys(self) -> None:
        left_speed, right_speed = compute_dpad_command(
            active_dirs=self.active_dirs,
            dpad_drive_speed=self.get_dpad_drive_speed(),
            dpad_turn_speed=self.get_dpad_turn_speed(),
            reverse_speed_scale=REVERSE_SPEED_SCALE,
        )
        self.send_command(left_speed, right_speed)

    def update_camera_frame(self) -> None:
        frame = self.camera_receiver.get_latest_frame()

        lines = self.camera_receiver.find_line(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (51, 255, 105), 2)

        pil_image = Image.fromarray(frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        self.camera_label.config(image=tk_image)
        self.camera_label.image = tk_image

        self.recorder.record_frame(
            image_rgb=frame,
            left_speed=self.current_l_speed,
            right_speed=self.current_r_speed,
            drive_type=self.drive_type_var.get(),
            drive_speed_base=self.get_drive_speed(),
            control_mode=self.control_mode.get(),
            interval_sec=0.1,
        )

        self.root.after(30, self.update_camera_frame)

    def send_command(self, left_speed: int, right_speed: int) -> None:
        self.current_l_speed = left_speed
        self.current_r_speed = right_speed

        started = self.recorder.start_recording_if_needed(left_speed, right_speed)
        if started:
            self.record_btn.config(text="■ 録画停止", bg="pink")
            self.logger.info("録画開始: 操作入力を検知しました。")

        if (
            self.recorder.state == RecorderState.RECORDING
            and self.auto_stop_on_zero_var.get()
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