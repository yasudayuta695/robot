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
    sock.setsockopt(zmq.CONFLATE, 1)  # ★追加：古い映像を捨てて最新1枚だけを送る！
    sock.bind(f"tcp://*:{CAMERA_PORT}")
    
    picam2 = Picamera2()
    # ★変更：サイズを 320x240 に下げて通信量を劇的に軽くする！
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    print("カメラ配信スレッド起動（PCからの接続待機中...）")
    try:
        while running:
            img = picam2.capture_array()
            height, width = img.shape[:2]
            ndim = img.ndim

            data = [np.array([height]), np.array([width]), np.array([ndim]), img.data]
            sock.send_multipart(data)
            
            # ★追加：少しだけお休みを入れてWi-Fiのパンクを防ぐ (約30FPSに制限)
            time.sleep(0.03) 
            
    except Exception as e:
        print(f"カメラ処理でエラー: {e}")
    finally:
        picam2.stop()
        sock.close()
        
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
                set_left_motor(int(l_str))
                set_right_motor(int(r_str))
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