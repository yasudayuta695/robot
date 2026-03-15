import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
import zmq
import math
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
import os
import csv
import datetime
import shutil


def to_wsl_unc_path(path, distro_name="Ubuntu-24.04"):
    abs_path = os.path.abspath(path)
    linux_like = abs_path.replace("\\", "/")
    if linux_like.startswith("/"):
        return "\\\\wsl.localhost\\{}{}".format(distro_name, linux_like.replace("/", "\\"))
    return abs_path


def normalize_config_path(path, distro_name="Ubuntu-24.04"):
    candidate = path.strip().strip('"').strip("'")
    if not candidate:
        return candidate

    is_windows = os.name == "nt"

    # UNC path (\\wsl.localhost\Distro\path\to\dir) can be edited from Windows.
    if candidate.startswith("\\\\wsl.localhost\\"):
        if is_windows:
            return os.path.normpath(candidate)
        parts = candidate.split("\\")
        if len(parts) >= 5:
            return "/" + "/".join(parts[4:])

    # Windows path (C:\Users\...) to WSL path (/mnt/c/Users/...)
    if len(candidate) >= 2 and candidate[1] == ":":
        if is_windows:
            return os.path.abspath(os.path.expanduser(candidate))
        drive_letter = candidate[0].lower()
        rest = candidate[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive_letter}{rest}"

    return os.path.abspath(os.path.expanduser(candidate))


def sanitize_for_dirname(name):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe or "default"

# --- 設定 ---
PI_IP = "192.168.23.162"  # Raspberry PiのIPアドレス
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

image_data = np.zeros((480, 640, 3), dtype=np.uint8)
running = True

# モーター受信用ZeroMQ
motor_context = zmq.Context()
motor_socket = motor_context.socket(zmq.PUSH)
motor_socket.connect(f"tcp://{PI_IP}:{MOTOR_PORT}")

# --- カメラ受信スレッド ---
def receiver_thread():
    global image_data, running
    
    ctx = zmq.Context()
    cam_socket = ctx.socket(zmq.PULL)
    cam_socket.setsockopt(zmq.CONFLATE, 1)
    cam_socket.connect(f"tcp://{PI_IP}:{CAMERA_PORT}")
    print("カメラ受信スレッド起動...")

    while running:
        try:
            data = cam_socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            time.sleep(0.01)
            continue
            
        try:
            img = np.frombuffer(data, dtype=np.uint8).reshape((240, 320, 3))
            img = cv2.resize(img, (640, 480))
            image_data = img
        except Exception as e:
            pass

# --- GUIアプリケーション ---
class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FPV Controller & Data Collector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- データ保存用の状態変数 ---
        self.is_recording = False
        self.is_record_armed = False
        self.csv_file = None
        self.csv_writer = None
        self.preview_images = []
        self.wsl_distro_name = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-24.04")
        self.module_path = os.path.abspath(__file__)
        self.module_dir = os.path.dirname(self.module_path)
        self.project_dir = os.path.dirname(self.module_dir)
        launcher_path = os.path.join(self.project_dir, "pi_pc_app.py")
        self.script_path = launcher_path if os.path.exists(launcher_path) else self.module_path
        self.script_path_display = to_wsl_unc_path(self.script_path, self.wsl_distro_name)
        
        # 設定ファイルはプロジェクト配下の comfig.txt を使用
        self.drive_type_file_path = os.path.join(self.project_dir, "comfig.txt")
        legacy_config_path = os.path.join(self.project_dir, "drive_types.txt")
        if (not os.path.exists(self.drive_type_file_path)) and os.path.exists(legacy_config_path):
            shutil.copyfile(legacy_config_path, self.drive_type_file_path)
        self.drive_type_file_display = to_wsl_unc_path(self.drive_type_file_path, self.wsl_distro_name)
        self.ensure_drive_type_file()

        self.save_base_dir = self.project_dir
        (self.save_base_dir,
         self.drive_types,
         self.initial_drive_speed,
         self.dpad_drive_speed_scale,
         self.dpad_turn_speed_scale,
         self.camera_ids,
         self.default_camera_id) = self.load_drive_config()
        self.save_base_dir_display = to_wsl_unc_path(self.save_base_dir, self.wsl_distro_name)
        self.current_drive_type = self.drive_types[0]
        
        # 一時フォルダのパス
        self.temp_dir = os.path.join(self.save_base_dir, "temp_record")
        self.temp_img_dir = os.path.join(self.temp_dir, "images")
        self.temp_csv_path = os.path.join(self.temp_dir, "driving_log.csv")
        
        self.last_save_time = 0
        self.current_l_speed = 0
        self.current_r_speed = 0
        self.auto_stop_on_zero_var = tk.BooleanVar(value=False)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        # ====== 設定・録画エリア ======
        self.settings_frame = tk.Frame(self.main_frame)
        self.settings_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        tk.Label(self.settings_frame, text="カメラ:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(self.settings_frame, textvariable=self.camera_var, values=self.camera_ids, width=10, state="readonly")
        if self.default_camera_id in self.camera_ids:
            self.camera_combo.current(self.camera_ids.index(self.default_camera_id))
        else:
            self.camera_combo.current(0)
        self.camera_combo.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(self.settings_frame, text="モデル:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.settings_frame, textvariable=self.model_var, values=["なし (生データ収集)", "YOLOv8", "Pose"], width=15)
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(self.settings_frame, text="速度基準:").pack(side=tk.LEFT)
        self.drive_speed_var = tk.IntVar(value=self.initial_drive_speed)
        self.drive_speed_scale = tk.Scale(
            self.settings_frame,
            from_=20,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.drive_speed_var,
            command=self.on_drive_speed_change,
            showvalue=False,
            length=140
        )
        self.drive_speed_scale.pack(side=tk.LEFT)
        self.drive_speed_value_label = tk.Label(self.settings_frame, text=f"{self.get_drive_speed()}%", width=5)
        self.drive_speed_value_label.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(self.settings_frame, text="走行種別:").pack(side=tk.LEFT)
        self.drive_type_var = tk.StringVar(value=self.current_drive_type)
        self.drive_type_combo = ttk.Combobox(self.settings_frame, textvariable=self.drive_type_var, values=self.drive_types, width=14, state="readonly")
        self.drive_type_combo.current(0)
        self.drive_type_combo.pack(side=tk.LEFT, padx=(0, 15))

        self.auto_stop_check = tk.Checkbutton(
            self.settings_frame,
            text="停止(0,0)で自動終了",
            variable=self.auto_stop_on_zero_var,
        )
        self.auto_stop_check.pack(side=tk.LEFT, padx=(0, 15))

        self.record_btn = tk.Button(self.settings_frame, text="● 録画開始", bg="lightgreen", font=("", 10, "bold"), command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT)

        self.path_info_label = tk.Label(
            self.main_frame,
            text=(
                f"実行ファイル: {self.script_path_display}\n"
                f"設定ファイル: {self.drive_type_file_display}\n"
                f"データ保存ルート: {self.save_base_dir_display}"
            ),
            justify="left",
            anchor="w",
            fg="gray30"
        )
        self.path_info_label.pack(fill=tk.X, pady=(0, 10))

        print(f"実行ファイル: {self.script_path_display}")
        print(f"設定ファイル: {self.drive_type_file_display}")
        print(f"データ保存ルート: {self.save_base_dir_display}")

        # UIの下部（カメラ映像とコントローラー）
        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.TOP)

        self.camera_label = tk.Label(self.bottom_frame, bg="black", width=640, height=480)
        self.camera_label.pack(side=tk.LEFT, padx=10)

        self.joy_frame = tk.Frame(self.bottom_frame)
        self.joy_frame.pack(side=tk.LEFT, padx=10)

        self.control_mode = tk.StringVar(value="joystick")
        mode_frame = tk.LabelFrame(self.joy_frame, text="操作モード")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Radiobutton(mode_frame, text="ジョイスティック", variable=self.control_mode,
                   value="joystick", command=self.on_mode_change).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="十字キー", variable=self.control_mode,
                   value="dpad", command=self.on_mode_change).pack(anchor="w")
        
        self.canvas = tk.Canvas(self.joy_frame, width=CENTER*2, height=CENTER*2, bg="white")
        self.canvas.pack(pady=20)

        self.canvas.create_oval(CENTER-RADIUS, CENTER-RADIUS, CENTER+RADIUS, CENTER+RADIUS, outline="gray")
        self.canvas.create_oval(CENTER-DEADZONE, CENTER-DEADZONE, CENTER+DEADZONE, CENTER+DEADZONE, outline="lightgray", dash=(4, 4))
        self.stick = self.canvas.create_oval(CENTER-STICK_RADIUS, CENTER-STICK_RADIUS, 
                                             CENTER+STICK_RADIUS, CENTER+STICK_RADIUS, fill="blue")

        self.dpad_frame = tk.Frame(self.joy_frame)
        self.build_dpad_ui()

        self.current_cmd = "0,0"
        self.active_dirs = set()
        self.key_to_dir = {"Up": "up", "Down": "down", "Left": "left", "Right": "right",
                           "w": "up", "W": "up", "s": "down", "S": "down",
                           "a": "left", "A": "left", "d": "right", "D": "right"}
        
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)

        self.update_camera_frame()

    def on_drive_speed_change(self, _value):
        self.drive_speed_value_label.config(text=f"{self.get_drive_speed()}%")

    def get_drive_speed(self):
        return max(0, min(100, int(self.drive_speed_var.get())))

    def get_dpad_drive_speed(self):
        return int(self.get_drive_speed() * self.dpad_drive_speed_scale)

    def get_dpad_turn_speed(self):
        return int(self.get_drive_speed() * self.dpad_turn_speed_scale)

    def ensure_drive_type_file(self):
        if os.path.exists(self.drive_type_file_path):
            return
        with open(self.drive_type_file_path, mode="w", encoding="utf-8") as f:
            f.write("# Config file for data collection\n")
            f.write("# 各PCで、このプロジェクトフォルダのパスを save_base_dir に設定してください\n")
            f.write("# Linux例: save_base_dir=/home/ryuryu/lab/Robot_car\n")
            f.write("# UNC例: save_base_dir=\\\\wsl.localhost\\Ubuntu-24.04\\home\\ryuryu\\lab\\Robot_car\n")
            f.write("save_base_dir=/home/ryuryu/lab/Robot_car\n")
            f.write("drive_speed_default=60\n")
            f.write("dpad_drive_speed_scale=1.0\n")
            f.write("dpad_turn_speed_scale=0.5\n")
            f.write("camera_id=camera_1\n")
            f.write("camera_id=camera_2\n")
            f.write("default_camera_id=camera_1\n")
            f.write("\n")
            f.write("# 走行種別は drive_type= で複数行書けます\n")
            f.write("drive_type=straight\n")
            f.write("drive_type=left_curve\n")
            f.write("drive_type=right_curve\n")
            f.write("drive_type=stop_and_go\n")

    def load_drive_config(self):
        configured_save_base_dir = self.project_dir
        configured_drive_speed = DRIVE_SPEED
        configured_dpad_drive_scale = DEFAULT_DPAD_DRIVE_SPEED_SCALE
        configured_dpad_turn_scale = DEFAULT_DPAD_TURN_SPEED_SCALE
        configured_camera_ids = []
        configured_default_camera_id = ""
        drive_types = []
        with open(self.drive_type_file_path, mode="r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "save_base_dir" and value:
                        configured_save_base_dir = normalize_config_path(value, self.wsl_distro_name)
                    elif key == "drive_speed_default" and value:
                        try:
                            configured_drive_speed = int(float(value))
                        except ValueError:
                            pass
                    elif key == "dpad_drive_speed_scale" and value:
                        try:
                            configured_dpad_drive_scale = float(value)
                        except ValueError:
                            pass
                    elif key == "dpad_turn_speed_scale" and value:
                        try:
                            configured_dpad_turn_scale = float(value)
                        except ValueError:
                            pass
                    elif key == "camera_id" and value:
                        configured_camera_ids.append(value)
                    elif key == "default_camera_id" and value:
                        configured_default_camera_id = value
                    elif key == "drive_type" and value:
                        drive_types.append(value)
                    continue

                # 後方互換: 旧形式（1行1走行種別）も受け付ける
                drive_types.append(line)

        if not drive_types:
            drive_types = ["straight"]
        if not configured_save_base_dir:
            configured_save_base_dir = self.project_dir
        configured_drive_speed = max(20, min(100, int(configured_drive_speed)))
        if configured_dpad_drive_scale <= 0:
            configured_dpad_drive_scale = DEFAULT_DPAD_DRIVE_SPEED_SCALE
        if configured_dpad_turn_scale <= 0:
            configured_dpad_turn_scale = DEFAULT_DPAD_TURN_SPEED_SCALE
        if not configured_camera_ids:
            configured_camera_ids = ["camera_1", "camera_2"]
        if not configured_default_camera_id:
            configured_default_camera_id = configured_camera_ids[0]
        return (
            configured_save_base_dir,
            drive_types,
            configured_drive_speed,
            configured_dpad_drive_scale,
            configured_dpad_turn_scale,
            configured_camera_ids,
            configured_default_camera_id,
        )

    def list_temp_images(self):
        if not os.path.exists(self.temp_img_dir):
            return []
        images = [x for x in os.listdir(self.temp_img_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))]
        images.sort()
        return images

    # --- 録画切り替え処理 ---
    def toggle_recording(self):
        if (not self.is_recording) and (not self.is_record_armed):
            # 録画待機開始: 一旦 temp_record フォルダに保存準備だけ行う
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir) # 前回のゴミがあれば消す
            os.makedirs(self.temp_img_dir, exist_ok=True)
            
            self.csv_file = open(self.temp_csv_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["image_path", "left_speed", "right_speed", "drive_type", "drive_speed_base", "control_mode"])

            self.is_record_armed = True
            self.last_save_time = 0
            self.record_btn.config(text="■ 録画待機中", bg="khaki")
            print("録画待機中: ジョイスティック操作 or 十字キー入力で録画を開始します...")
        else:
            self.stop_recording_flow()

    def stop_recording_flow(self, reason=""):
        self.is_recording = False
        self.is_record_armed = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.record_btn.config(text="● 録画開始", bg="lightgreen")
        if reason:
            print(reason)
        # 停止直後に確認ポップアップを出す
        self.ask_save_data()

    def ask_save_data(self):
        if not os.path.exists(self.temp_csv_path):
            return

        self.open_preview_dialog()

    def open_preview_dialog(self):
        image_files = self.list_temp_images()

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
                img_path = os.path.join(self.temp_img_dir, image_files[idx])
                try:
                    pil = Image.open(img_path)
                    pil.thumbnail((260, 180))
                    thumb = ImageTk.PhotoImage(pil)
                    self.preview_images.append(thumb)
                    card = tk.Frame(thumbs_frame, bd=1, relief="solid")
                    card.pack(side=tk.LEFT, padx=8, pady=8)
                    tk.Label(card, image=thumb).pack(padx=6, pady=6)
                    tk.Label(card, text=image_files[idx], wraplength=250).pack(padx=6, pady=(0, 6))
                except Exception:
                    continue
        else:
            tk.Label(thumbs_frame, text="プレビュー可能な画像がありません。", fg="gray40").pack(anchor="w", padx=8, pady=8)

        button_row = tk.Frame(preview)
        button_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        def save_and_close():
            preview.destroy()
            self.commit_data()

        def discard_and_close():
            preview.destroy()
            self.discard_data()

        tk.Button(button_row, text="保存する", bg="lightgreen", command=save_and_close).pack(side=tk.LEFT)
        tk.Button(button_row, text="破棄する", bg="tomato", command=discard_and_close).pack(side=tk.LEFT, padx=8)
        tk.Button(button_row, text="キャンセル", command=preview.destroy).pack(side=tk.LEFT)

    def get_final_base_dir(self):
        cam_id = self.camera_var.get()
        drive_type = sanitize_for_dirname(self.drive_type_var.get())
        return os.path.join(self.save_base_dir, "datasets", "1_raw_data", cam_id, drive_type)

    def commit_data(self):
        # 一時フォルダから本番フォルダへデータを移す
        base_dir = self.get_final_base_dir()
        final_img_dir = os.path.join(base_dir, "images")
        final_csv_path = os.path.join(base_dir, "driving_log.csv")
        
        os.makedirs(final_img_dir, exist_ok=True)
        
        # 1. 画像ファイルを移動
        for img_file in os.listdir(self.temp_img_dir):
            src = os.path.join(self.temp_img_dir, img_file)
            dst = os.path.join(final_img_dir, img_file)
            shutil.move(src, dst)
            
        # 2. CSVデータを本番のCSVに追記
        file_exists = os.path.isfile(final_csv_path)
        with open(final_csv_path, mode='a', newline='') as f_out:
            writer = csv.writer(f_out)
            # 本番ファイルが新規ならヘッダーを書く
            if not file_exists:
                writer.writerow(["image_path", "left_speed", "right_speed", "drive_type", "drive_speed_base", "control_mode"])
            
            # 一時CSVを読み込んで追記
            with open(self.temp_csv_path, mode='r', newline='') as f_in:
                reader = csv.reader(f_in)
                next(reader)  # ヘッダーをスキップ
                for row in reader:
                    if len(row) == 3:
                        row.append(self.drive_type_var.get())
                        row.append(self.get_drive_speed())
                        row.append(self.control_mode.get())
                    elif len(row) == 4:
                        row.append(self.get_drive_speed())
                        row.append(self.control_mode.get())
                    elif len(row) == 5:
                        row.append(self.control_mode.get())
                    writer.writerow(row)
                    
        # 3. 一時フォルダをお掃除
        shutil.rmtree(self.temp_dir)
        print(f"データを {to_wsl_unc_path(base_dir, self.wsl_distro_name)} に保存しました！")

    def discard_data(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("走行データを破棄しました。")

    def build_dpad_ui(self):
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
        down_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(-int(self.get_dpad_drive_speed() * REVERSE_SPEED_SCALE), -int(self.get_dpad_drive_speed() * REVERSE_SPEED_SCALE)))
        down_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        left_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(-self.get_dpad_turn_speed(), self.get_dpad_turn_speed()))
        left_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        right_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(self.get_dpad_turn_speed(), -self.get_dpad_turn_speed()))
        right_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        stop_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(0, 0))

    def on_mode_change(self):
        self.active_dirs.clear()
        self.send_command(0, 0)
        self.canvas.coords(self.stick, CENTER-STICK_RADIUS, CENTER-STICK_RADIUS,
                           CENTER+STICK_RADIUS, CENTER+STICK_RADIUS)
        if self.control_mode.get() == "joystick":
            self.dpad_frame.pack_forget()
            self.canvas.pack(pady=20)
        else:
            self.canvas.pack_forget()
            self.dpad_frame.pack(pady=20)

    def on_key_press(self, event):
        if self.control_mode.get() != "dpad":
            return
        direction = self.key_to_dir.get(event.keysym)
        if direction is None:
            return
        self.active_dirs.add(direction)
        self.apply_dpad_keys()

    def on_key_release(self, event):
        direction = self.key_to_dir.get(event.keysym)
        if direction is None:
            return
        self.active_dirs.discard(direction)
        if self.control_mode.get() == "dpad":
            self.apply_dpad_keys()

    def apply_dpad_keys(self):
        dpad_drive_speed = self.get_dpad_drive_speed()
        dpad_turn_speed = self.get_dpad_turn_speed()

        up = "up" in self.active_dirs
        down = "down" in self.active_dirs
        left = "left" in self.active_dirs
        right = "right" in self.active_dirs

        if up and not down:
            base = dpad_drive_speed
        elif down and not up:
            base = -int(dpad_drive_speed * REVERSE_SPEED_SCALE)
        else:
            base = 0

        if right and not left:
            turn = dpad_turn_speed
        elif left and not right:
            turn = -dpad_turn_speed
        else:
            turn = 0

        if base == 0 and turn == 0:
            l_speed, r_speed = 0, 0
        elif base == 0:
            l_speed, r_speed = turn, -turn
        else:
            l_speed = base + turn
            r_speed = base - turn

        l_speed = max(-100, min(100, l_speed))
        r_speed = max(-100, min(100, r_speed))
        self.send_command(l_speed, r_speed)

    def update_camera_frame(self):
        if running:
            pil_image = Image.fromarray(image_data)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.camera_label.config(image=tk_image)
            self.camera_label.image = tk_image

            # 録画処理: 一時フォルダ(temp_img_dir)へ書き込む
            if self.is_recording:
                current_time = time.time()
                if current_time - self.last_save_time >= 0.1:  # 10FPS
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    filename = f"{timestamp}.jpg"
                    filepath = os.path.join(self.temp_img_dir, filename)

                    bgr_img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, bgr_img)

                    # CSVへのパス記録
                    self.csv_writer.writerow([
                        f"images/{filename}",
                        self.current_l_speed,
                        self.current_r_speed,
                        self.drive_type_var.get(),
                        self.get_drive_speed(),
                        self.control_mode.get(),
                    ])
                    self.last_save_time = current_time

            self.root.after(30, self.update_camera_frame)

    def send_command(self, l_speed, r_speed):
        self.current_l_speed = l_speed
        self.current_r_speed = r_speed

        if self.is_record_armed and (l_speed != 0 or r_speed != 0):
            self.is_recording = True
            self.is_record_armed = False
            self.last_save_time = 0
            self.record_btn.config(text="■ 録画停止", bg="pink")
            print("録画開始: 操作入力を検知したため記録を開始します...")

        if self.is_recording and self.auto_stop_on_zero_var.get() and l_speed == 0 and r_speed == 0:
            self.stop_recording_flow("録画自動終了: 停止(0,0)を検知しました。")
        
        cmd = f"{l_speed},{r_speed}"
        if cmd != self.current_cmd:
            motor_socket.send_string(cmd)
            self.current_cmd = cmd

    def drag(self, event):
        if self.control_mode.get() != "joystick":
            return

        dx = event.x - CENTER
        dy = event.y - CENTER
        distance = math.sqrt(dx**2 + dy**2)

        if distance > RADIUS:
            dx = dx * RADIUS / distance
            dy = dy * RADIUS / distance
        
        self.canvas.coords(self.stick, CENTER+dx-STICK_RADIUS, CENTER+dy-STICK_RADIUS, 
                                     CENTER+dx+STICK_RADIUS, CENTER+dy+STICK_RADIUS)

        if distance < DEADZONE:
            self.send_command(0, 0)
            return

        dir_mag = math.sqrt(dx * dx + dy * dy)
        dir_x = dx / max(dir_mag, 1e-6)
        dir_y = dy / max(dir_mag, 1e-6)
        drive_speed = self.get_drive_speed()

        base_speed = int((-dir_y) * drive_speed)
        if base_speed < 0 and abs(dir_x) < REVERSE_STRAIGHT_X_THRESHOLD:
            base_speed = int(base_speed * REVERSE_SPEED_SCALE)

        if abs(dir_y) < PIVOT_Y_THRESHOLD:
            turn = int((dir_x * abs(dir_x)) * drive_speed * PIVOT_TURN_GAIN)
            l_speed = turn
            r_speed = -turn
        else:
            angle_scale = abs(dir_y)
            outer = int(base_speed * OUTER_SPEED_SCALE)
            inner = int(base_speed * angle_scale * INNER_SPEED_SCALE)

            if dir_x > 0:  
                l_speed = outer
                r_speed = inner
            else:           
                l_speed = inner
                r_speed = outer

        l_speed *= LEFT_SIGN
        r_speed *= RIGHT_SIGN
        l_speed = max(-100, min(100, l_speed))
        r_speed = max(-100, min(100, r_speed))

        self.send_command(l_speed, r_speed)

    def release(self, event):
        if self.control_mode.get() != "joystick":
            return
        self.canvas.coords(self.stick, CENTER-STICK_RADIUS, CENTER-STICK_RADIUS, 
                                     CENTER+STICK_RADIUS, CENTER+STICK_RADIUS)
        self.send_command(0, 0)

    def on_closing(self):
        global running
        running = False
        if (self.is_recording or self.is_record_armed) and self.csv_file:
            self.csv_file.close()
            # 閉じる際に強制終了された場合は一時データを破棄
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
        self.root.unbind_all("<KeyPress>")
        self.root.unbind_all("<KeyRelease>")
        self.root.destroy()

if __name__ == "__main__":
    thread1 = threading.Thread(target=receiver_thread)
    thread1.start()

    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()
    
    thread1.join()