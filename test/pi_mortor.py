import RPi.GPIO as GPIO
import zmq

# ピン設定
GPIO.setmode(GPIO.BOARD)
MA_IN1, MA_IN2, MA_PWM = 19, 21, 23
MB_IN1, MB_IN2, MB_PWM = 15, 13, 11
CMD_LEFT_SIGN = 1   # 左コマンドの符号補正: 1 または -1
CMD_RIGHT_SIGN = -1  # 右コマンドの符号補正: 1 または -1

for pin in [MA_IN1, MA_IN2, MA_PWM, MB_IN1, MB_IN2, MB_PWM]:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

pwm_a = GPIO.PWM(MA_PWM, 100)
pwm_b = GPIO.PWM(MB_PWM, 100)
pwm_a.start(0)
pwm_b.start(0)

def set_left_motor(speed):
    """左モーター速度(-100〜100)を設定"""
    if speed > 0:    # 正転
        GPIO.output(MB_IN1, GPIO.HIGH)
        GPIO.output(MB_IN2, GPIO.LOW)
    elif speed < 0:  # 逆転
        GPIO.output(MB_IN1, GPIO.LOW)
        GPIO.output(MB_IN2, GPIO.HIGH)
    else:            # 停止
        GPIO.output(MB_IN1, GPIO.LOW)
        GPIO.output(MB_IN2, GPIO.LOW)

    pwm_b.ChangeDutyCycle(min(abs(speed), 100))

def set_right_motor(speed):
    """右モーター速度(-100〜100)を設定"""
    if speed > 0:    # 正転
        GPIO.output(MA_IN1, GPIO.LOW)
        GPIO.output(MA_IN2, GPIO.HIGH)
    elif speed < 0:  # 逆転
        GPIO.output(MA_IN1, GPIO.HIGH)
        GPIO.output(MA_IN2, GPIO.LOW)
    else:            # 停止
        GPIO.output(MA_IN1, GPIO.LOW)
        GPIO.output(MA_IN2, GPIO.LOW)

    pwm_a.ChangeDutyCycle(min(abs(speed), 100))

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
            l_speed = int(l_str) * CMD_LEFT_SIGN
            r_speed = int(r_str) * CMD_RIGHT_SIGN

            set_left_motor(r_speed)
            set_right_motor(l_speed)
        except ValueError:
            # 万が一変なデータが来たときは停止
            set_left_motor(0)
            set_right_motor(0)

except KeyboardInterrupt:
    print("終了します・・・")
finally:
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()