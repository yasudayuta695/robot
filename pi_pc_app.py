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
    cam_socket.setsockopt(zmq.CONFLATE, 1) # ★追加：渋滞している古い映像を読み捨てる！
    cam_socket.connect(f"tcp://{PI_IP}:{CAMERA_PORT}")
    print("カメラ受信スレッド起動...")

    while running:
        try:
            byte_rows, byte_cols, byte_mat_type, data = cam_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            time.sleep(0.01)
            continue
            
        row = struct.unpack("q", byte_rows)[0]
        cols = struct.unpack("q", byte_cols)
        mat_type = struct.unpack("q", byte_mat_type)
        
        if mat_type[0] == 0:
            img = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0]))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0], 3))
            # ★削除：色が変な原因だった「cvtColor」の行を丸ごと消しました！
            
        # ★追加：ラズパイから来た軽い映像(320x240)を、画面サイズ(640x480)に引き伸ばす
        img = cv2.resize(img, (640, 480))
        image_data = img

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

        steering = int((norm_x * abs(norm_x)) * 100)
        throttle = -int((norm_y * abs(norm_y)) * 100)
        steering = int(steering * 0.7)

        l_speed = -throttle - steering
        r_speed = throttle - steering

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