import RPi.GPIO as GPIO
import time
import zmq
import threading
import numpy as np
from picamera2 import Picamera2

# --- ネットワーク設定 ---
PC_IP = "192.168.23.204"  # 画像を受信するPCのIPアドレス（★要確認）
CAMERA_PORT = 5556        # カメラ用のポート（PC側と合わせる）
MOTOR_PORT = 5555         # モーター用のポート

# --- モーター（GPIO）の設定 ---
GPIO.setmode(GPIO.BOARD)
MA_IN1, MA_IN2, MA_PWM = 19, 21, 23
MB_IN1, MB_IN2, MB_PWM = 15, 13, 11

for pin in [MA_IN1, MA_IN2, MA_PWM, MB_IN1, MB_IN2, MB_PWM]:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

pwm_a = GPIO.PWM(MA_PWM, 100)
pwm_b = GPIO.PWM(MB_PWM, 100)
pwm_a.start(0)
pwm_b.start(0)

# 正しい動きをするように修正したモーター制御関数
def set_left_motor(speed):
    if speed > 0:
        GPIO.output(MB_IN1, GPIO.HIGH)
        GPIO.output(MB_IN2, GPIO.LOW)
    elif speed < 0:
        GPIO.output(MB_IN1, GPIO.LOW)
        GPIO.output(MB_IN2, GPIO.HIGH)
    else:
        GPIO.output(MB_IN1, GPIO.LOW)
        GPIO.output(MB_IN2, GPIO.LOW)
    pwm_b.ChangeDutyCycle(min(abs(speed), 100))

def set_right_motor(speed):
    if speed > 0:
        GPIO.output(MA_IN1, GPIO.LOW)
        GPIO.output(MA_IN2, GPIO.HIGH)
    elif speed < 0:
        GPIO.output(MA_IN1, GPIO.HIGH)
        GPIO.output(MA_IN2, GPIO.LOW)
    else:
        GPIO.output(MA_IN1, GPIO.LOW)
        GPIO.output(MA_IN2, GPIO.LOW)
    pwm_a.ChangeDutyCycle(min(abs(speed), 100))

# --- プログラム全体の実行状態 ---
running = True

# --- カメラ配信用の別スレッド ---
def camera_thread():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{PC_IP}:{CAMERA_PORT}")
    
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    print("カメラ配信スレッド起動...")
    try:
        while running:
            img = picam2.capture_array()
            height, width = img.shape[:2]
            ndim = img.ndim

            # 画像データを送信
            data = [np.array([height]), np.array([width]), np.array([ndim]), img.data]
            sock.send_multipart(data)
            
            # PCからの受信確認（OK）を待つ
            sock.recv_string()
    except Exception as e:
        print(f"カメラ処理でエラー: {e}")
    finally:
        picam2.stop()
        sock.close()

# --- メイン処理（モーター受信） ---
if __name__ == "__main__":
    # カメラ処理を裏側（別スレッド）でスタートさせる
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    # モーター受信用のZeroMQ準備
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{MOTOR_PORT}")
    print("モーター待機サーバー起動...")

    try:
        while running:
            # PCからの操縦コマンドを待つ
            cmd = socket.recv_string()
            try:
                l_str, r_str = cmd.split(',')
                set_left_motor(int(l_str))
                set_right_motor(int(r_str))
            except ValueError:
                set_left_motor(0)
                set_right_motor(0)
                
    except KeyboardInterrupt:
        print("終了します...")
        running = False
        
    finally:
        # モーターを安全に停止
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()