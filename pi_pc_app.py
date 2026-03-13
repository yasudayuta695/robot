import tkinter as tk
import zmq
import math
import cv2
import numpy as np
import threading
import struct
import time
from PIL import Image, ImageTk

# --- 設定 ---
PI_IP = "192.168.23.162"  # ★Raspberry PiのIPアドレス（確認してください）
CENTER = 150
RADIUS = 120
STICK_RADIUS = 30
DEADZONE = 35
CAMERA_PORT = 5556        # カメラ用ポート
MOTOR_PORT = 5555         # モーター用ポート
LEFT_SIGN = 1             # 左モーター配線に応じて 1 / -1 を切替
RIGHT_SIGN = 1            # 右モーター配線に応じて 1 / -1 を切替
PIVOT_Y_THRESHOLD = 0.15  # この値より水平に近いときだけその場旋回を許可
PIVOT_TURN_GAIN = 0.25   # その場旋回時の旋回量
DRIVE_SPEED = 60         # 走行時の基準速度（倒し量ではなく方向だけを使う）
REVERSE_SPEED_SCALE = 0.8 # 後退時のみ速度を抑える係数
REVERSE_STRAIGHT_X_THRESHOLD = 0.2  # |x|が小さい真後ろ寄りのときだけ後退減速
OUTER_SPEED_SCALE = 1.05  # 斜め走行時の外輪倍率
INNER_SPEED_SCALE = 0.9   # 斜め走行時の内輪倍率
DPAD_DRIVE_SPEED = 50     # 十字キーモードの前進/後退速度
DPAD_TURN_SPEED = 30      # 十字キーモードのその場旋回速度

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
            # ★修正: recv_multipart ではなく、通常の recv で1つの塊を受け取る！
            data = cam_socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            time.sleep(0.01)
            continue
            
        try:
            # ★修正: 届いたデータをそのまま 320x240 の画像に復元する！
            # (縦240, 横320, 色3チャンネル)
            img = np.frombuffer(data, dtype=np.uint8).reshape((240, 320, 3))
            
            # 画面用に 640x480 に引き伸ばす
            img = cv2.resize(img, (640, 480))
            
            image_data = img
        except Exception as e:
            # 万が一通信のゴミが混ざって変換失敗した時は無視して次へ
            pass

# --- GUIアプリケーション ---
class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FPV Controller")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        self.camera_label = tk.Label(self.main_frame, bg="black", width=640, height=480)
        self.camera_label.pack(side=tk.LEFT, padx=10)

        self.joy_frame = tk.Frame(self.main_frame)
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
        self.key_to_dir = {
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
        
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)

        self.update_camera_frame()

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

        up_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(DPAD_DRIVE_SPEED, DPAD_DRIVE_SPEED))
        up_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        down_speed = int(DPAD_DRIVE_SPEED * REVERSE_SPEED_SCALE)
        down_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(-down_speed, -down_speed))
        down_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        left_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(-DPAD_TURN_SPEED, DPAD_TURN_SPEED))
        left_btn.bind("<ButtonRelease-1>", lambda _e: self.send_command(0, 0))
        right_btn.bind("<ButtonPress-1>", lambda _e: self.send_command(DPAD_TURN_SPEED, -DPAD_TURN_SPEED))
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
        up = "up" in self.active_dirs
        down = "down" in self.active_dirs
        left = "left" in self.active_dirs
        right = "right" in self.active_dirs

        if up and not down:
            base = DPAD_DRIVE_SPEED
        elif down and not up:
            base = -int(DPAD_DRIVE_SPEED * REVERSE_SPEED_SCALE)
        else:
            base = 0

        if right and not left:
            turn = DPAD_TURN_SPEED
        elif left and not right:
            turn = -DPAD_TURN_SPEED
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
            self.root.after(30, self.update_camera_frame)

    def send_command(self, l_speed, r_speed):
        cmd = f"{l_speed},{r_speed}"
        if cmd != self.current_cmd:
            motor_socket.send_string(cmd)
            # print(f"送信: 左 {l_speed}%, 右 {r_speed}%") # ログが多すぎる場合はコメントアウト
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

        # 方向のみを使うため、単位ベクトルに変換する
        dir_mag = math.sqrt(dx * dx + dy * dy)
        dir_x = dx / max(dir_mag, 1e-6)
        dir_y = dy / max(dir_mag, 1e-6)

        # 前後の基準速度（前:正, 後:負）
        # 2乗カーブではなく一次にして、斜め入力時の失速を抑える
        base_speed = int((-dir_y) * DRIVE_SPEED)
        if base_speed < 0 and abs(dir_x) < REVERSE_STRAIGHT_X_THRESHOLD:
            base_speed = int(base_speed * REVERSE_SPEED_SCALE)

        # 真横付近はその場旋回
        if abs(dir_y) < PIVOT_Y_THRESHOLD:
            turn = int((dir_x * abs(dir_x)) * DRIVE_SPEED * PIVOT_TURN_GAIN)
            l_speed = turn
            r_speed = -turn
        else:
            # 走行中は外輪を基準速度のまま、内輪のみ0へ近づける
            # angle_scale = cos(theta) = |y| / sqrt(x^2 + y^2)
            angle_scale = abs(dir_y)

            outer = int(base_speed * OUTER_SPEED_SCALE)
            inner = int(base_speed * angle_scale * INNER_SPEED_SCALE)

            if dir_x > 0:  # 右へ曲がる: 右輪を内輪にする
                l_speed = outer
                r_speed = inner
            else:           # 左へ曲がる: 左輪を内輪にする
                l_speed = inner
                r_speed = outer

        # モーター配線や取付方向の差をここで吸収
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