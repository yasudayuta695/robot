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
DEADZONE = 25
CAMERA_PORT = 5556        # カメラ用ポート
MOTOR_PORT = 5555         # モーター用ポート
LEFT_SIGN = 1             # 左モーター配線に応じて 1 / -1 を切替
RIGHT_SIGN = 1            # 右モーター配線に応じて 1 / -1 を切替
PIVOT_Y_THRESHOLD = 0.05  # この値より水平に近いときだけその場旋回を許可
PIVOT_TURN_GAIN = 1.0     # その場旋回時の旋回量

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
        
        self.canvas = tk.Canvas(self.joy_frame, width=CENTER*2, height=CENTER*2, bg="white")
        self.canvas.pack(pady=20)

        self.canvas.create_oval(CENTER-RADIUS, CENTER-RADIUS, CENTER+RADIUS, CENTER+RADIUS, outline="gray")
        self.canvas.create_oval(CENTER-DEADZONE, CENTER-DEADZONE, CENTER+DEADZONE, CENTER+DEADZONE, outline="lightgray", dash=(4, 4))
        self.stick = self.canvas.create_oval(CENTER-STICK_RADIUS, CENTER-STICK_RADIUS, 
                                             CENTER+STICK_RADIUS, CENTER+STICK_RADIUS, fill="blue")

        self.current_cmd = "0,0"
        
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)

        self.update_camera_frame()

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

        norm_x = dx / RADIUS
        norm_y = dy / RADIUS

        # 前後の基準速度（前:正, 後:負）
        base_speed = int((-norm_y * abs(norm_y)) * 100)

        # 真横付近はその場旋回
        if abs(norm_y) < PIVOT_Y_THRESHOLD:
            turn = int((norm_x * abs(norm_x)) * 100 * PIVOT_TURN_GAIN)
            l_speed = turn
            r_speed = -turn
        else:
            # 走行中は外輪を基準速度のまま、内輪のみ0へ近づける
            # angle_scale = cos(theta) = |y| / sqrt(x^2 + y^2)
            mag = math.sqrt(norm_x * norm_x + norm_y * norm_y)
            angle_scale = abs(norm_y) / max(mag, 1e-6)

            outer = base_speed
            inner = int(base_speed * angle_scale)

            if norm_x > 0:  # 右へ曲がる: 右輪を内輪にする
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
        self.canvas.coords(self.stick, CENTER-STICK_RADIUS, CENTER-STICK_RADIUS, 
                                     CENTER+STICK_RADIUS, CENTER+STICK_RADIUS)
        self.send_command(0, 0)

    def on_closing(self):
        global running
        running = False
        self.root.destroy()

if __name__ == "__main__":
    thread1 = threading.Thread(target=receiver_thread)
    thread1.start()

    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()
    
    thread1.join()