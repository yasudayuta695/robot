import RPi.GPIO as GPIO
import time
import zmq
import threading
import numpy as np
from picamera2 import Picamera2

# --- ネットワーク設定 ---
CAMERA_PORT = 5556        # カメラ用のポート
MOTOR_PORT = 5555         # モーター用のポート
# （PCのIPアドレスはもう不要になりました！）
CMD_LEFT_SIGN = -1         # 受信した左速度の符号補正: 1 または -1
CMD_RIGHT_SIGN = 1       # 受信した右速度の符号補正: 1 または -1

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

running = True

# --- カメラ配信スレッド ---
def camera_thread():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://*:{CAMERA_PORT}")
    
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({"AeEnable":False,
                        "AwbEnable":False,
                        "ExposureTime":3000,  # 露光時間（マイクロ秒）
                        "AnalogueGain": 1.0})   # アナログゲイン（例: 4倍）
    print("カメラ配信スレッド起動（PCからの接続待機中...）")
    try:
        while running:
            img = picam2.capture_array()
            
            # ★修正: 分割送信をやめ、画像データ(bytes)だけを1つの塊として送る！
            sock.send(img.tobytes())
            
            time.sleep(0.03) 
            
    except Exception as e:
        print(f"カメラ処理でエラー: {e}")
    finally:
        picam2.stop()
        sock.close()

# --- メイン処理（モーター受信） ---
if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{MOTOR_PORT}")
    print("モーター待機サーバー起動...")

    try:
        while running:
            cmd = socket.recv_string()
            try:
                l_str, r_str = cmd.split(',')
                l_speed = int(l_str) * CMD_LEFT_SIGN
                r_speed = int(r_str) * CMD_RIGHT_SIGN
                set_left_motor(l_speed)
                set_right_motor(r_speed)
            except ValueError:
                set_left_motor(0)
                set_right_motor(0)
                
    except KeyboardInterrupt:
        print("終了します...")
        running = False
        
    finally:
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()