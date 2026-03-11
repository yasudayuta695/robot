import RPi.GPIO as GPIO
import time
import zmq

# ピン設定
GPIO.setmode(GPIO.BOARD)
MA_IN1, MA_IN2, MA_PWM = 19, 21, 23
MB_IN1, MB_IN2, MB_PWM = 15, 13, 11

for pin in [MA_IN1, MA_IN2, MA_PWM, MB_IN1, MB_IN2, MB_PWM]:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

pwm_a = GPIO.PWM(MA_PWM, 100)
pwm_b = GPIO.PWM(MB_PWM, 100)
pwm_a.start(0)
pwm_b.start(0)

def set_motor(in1, in2, pwm, speed):
    """速度(-100〜100)に応じてモーターの正転・逆転・停止を制御"""
    if speed > 0:    # 正転
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
    elif speed < 0:  # 逆転
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    else:            # 停止
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)

    # 速度を0〜100の範囲にしてPWMに適用
    pwm.ChangeDutyCycle(min(abs(speed), 100))

# --- ZeroMQの設定 ---
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")
print("アナログ操作サーバー起動、待機中...")

try:
    while True:
        cmd = socket.recv_string()
        # "左の速度,右の速度" の形で送られてくるのを分割する (例: "80,40")
        try:
            l_str, r_str = cmd.split(',')
            l_speed = int(l_str)
            r_speed = int(r_str)

            # モーターに速度を反映 (MA=左, MB=右 と仮定しています)
            set_motor(MA_IN1, MA_IN2, pwm_a, l_speed)
            set_motor(MB_IN1, MB_IN2, pwm_b, r_speed)
        except ValueError:
            # 万が一変なデータが来たときは停止
            set_motor(MA_IN1, MA_IN2, pwm_a, 0)
            set_motor(MB_IN1, MB_IN2, pwm_b, 0)

except KeyboardInterrupt:
    print("終了します・・・")