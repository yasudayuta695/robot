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
PI_IP = "192.168.23.162"  # Raspberry PiのIPアドレス
CENTER = 150              # ジョイスティックの中心
RADIUS = 120              # 動かせる範囲
STICK_RADIUS = 30         # スティックの大きさ
DEADZONE = 25             # 遊び
CAMERA_PORT = 5555        # カメラ受信用ポート

# グローバル変数
image_data = np.zeros((480, 640, 3), dtype=np.uint8)
running = True

# --- ZeroMQの設定 (モーター送信用) ---
motor_context = zmq.Context()
motor_socket = motor_context.socket(zmq.PUSH)
motor_socket.connect(f"tcp://{PI_IP}:5555")


# --- カメラ受信スレッド ---
def receiver_thread():
    global image_data, running
    
    ctx = zmq.Context()
    cam_socket = ctx.socket(zmq.REP)
    cam_socket.bind(f"tcp://*:{CAMERA_PORT}")
    print("カメラ受信スレッド起動...")

    count = 0
    while running:
        try:
            # NOBLOCKで受信を試みる
            byte_rows, byte_cols, byte_mat_type, data = cam_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            time.sleep(0.01)
            continue
            
        count += 1
        cam_socket.send_string(f"ok {count}")

        # データの解凍
        row = struct.unpack("q", byte_rows)[0]
        cols = struct.unpack("q", byte_cols)
        mat_type = struct.unpack("q", byte_mat_type)
        
        # NumPy配列に変換
        if mat_type[0] == 0:  # Gray Scale
            img = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0]))
            # Tkinterで表示するためにRGB形式に変換
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # BGR Color
            img = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0], 3))
            # OpenCVのBGR形式から、Tkinter用のRGB形式に変換
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        image_data = img


# --- GUIアプリケーション ---
class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FPV Controller")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 画面レイアウト: 左側にカメラ、右側にジョイスティックを配置
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        # 【左側】カメラ映像表示用ラベル
        self.camera_label = tk.Label(self.main_frame, bg="black", width=640, height=480)
        self.camera_label.pack(side=tk.LEFT, padx=10)

        # 【右側】ジョイスティック用フレーム
        self.joy_frame = tk.Frame(self.main_frame)
        self.joy_frame.pack(side=tk.LEFT, padx=10)
        
        self.canvas = tk.Canvas(self.joy_frame, width=CENTER*2, height=CENTER*2, bg="white")
        self.canvas.pack(pady=20)

        # 背景の円とデッドゾーン
        self.canvas.create_oval(CENTER-RADIUS, CENTER-RADIUS, CENTER+RADIUS, CENTER+RADIUS, outline="gray")
        self.canvas.create_oval(CENTER-DEADZONE, CENTER-DEADZONE, CENTER+DEADZONE, CENTER+DEADZONE, outline="lightgray", dash=(4, 4))
        # スティック本体
        self.stick = self.canvas.create_oval(CENTER-STICK_RADIUS, CENTER-STICK_RADIUS, 
                                             CENTER+STICK_RADIUS, CENTER+STICK_RADIUS, fill="blue")

        self.current_cmd = "0,0"
        
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)

        # 映像の定期更新処理を開始
        self.update_camera_frame()

    def update_camera_frame(self):
        """別スレッドで更新されている image_data を Tkinter 画面に反映する"""
        if running:
            # OpenCVの配列をPIL画像に変換し、さらにTkinter画像に変換
            pil_image = Image.fromarray(image_data)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # ラベルの画像を差し替え
            self.camera_label.config(image=tk_image)
            self.camera_label.image = tk_image # ガベージコレクション対策で参照を残す
            
            # 30ミリ秒後に自分自身をもう一度呼び出す（ループ更新）
            self.root.after(30, self.update_camera_frame)

    # --- 以下、ジョイスティックの制御ロジック（前回と同じ） ---
    def send_command(self, l_speed, r_speed):
        cmd = f"{l_speed},{r_speed}"
        if cmd != self.current_cmd:
            motor_socket.send_string(cmd)
            print(f"送信: 左 {l_speed}%, 右 {r_speed}%")
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
        """ウィンドウを閉じたときの終了処理"""
        global running
        running = False
        self.root.destroy()

if __name__ == "__main__":
    # カメラ受信スレッドの開始
    thread1 = threading.Thread(target=receiver_thread)
    thread1.start()

    # Tkinterメインループの開始
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()
    
    # ウィンドウが閉じたらスレッドの終了を待つ
    thread1.join()