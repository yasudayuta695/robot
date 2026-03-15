import tkinter as tk
import zmq
import math

# --- 設定 ---
PI_IP = "192.168.23.162"  # Raspberry PiのIPアドレス
CENTER = 150              # 画面サイズを少し大きくして操作しやすく
RADIUS = 120              # 動かせる範囲を拡大
STICK_RADIUS = 30         # スティックも少し大きく
DEADZONE = 25             # 遊び（この範囲の動きは無視する）
LEFT_SIGN = 1             # 左モーター配線に応じて 1 / -1 を切替
RIGHT_SIGN = 1            # 右モーター配線に応じて 1 / -1 を切替
PIVOT_Y_THRESHOLD = 0.05  # この値より水平に近いときだけその場旋回を許可
PIVOT_TURN_GAIN = 1.0     # その場旋回時の旋回量

# ZeroMQ の設定
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://{PI_IP}:5555")

class JoystickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analog Controller (Mild)")
        
        self.canvas = tk.Canvas(root, width=CENTER*2, height=CENTER*2, bg="white")
        self.canvas.pack(pady=20)

        # 背景の円
        self.canvas.create_oval(CENTER-RADIUS, CENTER-RADIUS, CENTER+RADIUS, CENTER+RADIUS, outline="gray")
        # 遊び（デッドゾーン）の範囲を薄い点線の円で表示
        self.canvas.create_oval(CENTER-DEADZONE, CENTER-DEADZONE, CENTER+DEADZONE, CENTER+DEADZONE, outline="lightgray", dash=(4, 4))
        # スティック本体
        self.stick = self.canvas.create_oval(CENTER-STICK_RADIUS, CENTER-STICK_RADIUS, 
                                             CENTER+STICK_RADIUS, CENTER+STICK_RADIUS, fill="blue")

        self.current_cmd = "0,0"
        
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)

    def send_command(self, l_speed, r_speed):
        cmd = f"{l_speed},{r_speed}"
        if cmd != self.current_cmd:
            socket.send_string(cmd)
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

        # --- 判定を緩くするための工夫 ---
        
        # 1. 遊び（デッドゾーン）の判定
        if distance < DEADZONE:
            self.send_command(0, 0)
            return

        # 2. -1.0 〜 1.0 の比率に変換
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

if __name__ == "__main__":
    root = tk.Tk()
    app = JoystickApp(root)
    root.mainloop()