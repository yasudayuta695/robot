import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import RPi.GPIO as GPIO
import zmq
from picamera2 import Picamera2

from ai_controller import LineTraceONNXController
from camera_receiver import CameraReceiver


# Network ports used by legacy remote mode.
CAMERA_PORT = 5556
MOTOR_PORT = 5555

# Command sign correction in remote mode.
CMD_LEFT_SIGN = -1
CMD_RIGHT_SIGN = 1

# Motor output sign correction in local AI mode.
MOTOR_LEFT_SIGN = -1
MOTOR_RIGHT_SIGN = 1

# GPIO pin mapping.
MA_IN1, MA_IN2, MA_PWM = 19, 21, 23
MB_IN1, MB_IN2, MB_PWM = 15, 13, 11


def setup_gpio() -> tuple[GPIO.PWM, GPIO.PWM]:
    GPIO.setmode(GPIO.BOARD)
    for pin in [MA_IN1, MA_IN2, MA_PWM, MB_IN1, MB_IN2, MB_PWM]:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    pwm_a = GPIO.PWM(MA_PWM, 100)
    pwm_b = GPIO.PWM(MB_PWM, 100)
    pwm_a.start(0)
    pwm_b.start(0)
    return pwm_a, pwm_b


def set_left_motor(speed: int, pwm_b: GPIO.PWM) -> None:
    speed = int(np.clip(speed, -100, 100))
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


def set_right_motor(speed: int, pwm_a: GPIO.PWM) -> None:
    speed = int(np.clip(speed, -100, 100))
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


def stop_all_motors(pwm_a: GPIO.PWM, pwm_b: GPIO.PWM) -> None:
    set_left_motor(0, pwm_b)
    set_right_motor(0, pwm_a)


@dataclass
class PiRuntimeConfig:
    curve_slowdown_sensitivity: float = 0.70
    ai_smoothing_alpha: float = 0.35
    ai_no_line_hold_frames: int = 3
    ai_no_line_brake_frames: int = 8
    line_process_interval_ms: int = 70
    ai_control_interval_ms: int = 100
    far_threshold: int = 100
    near_threshold: int = 70


def load_runtime_config(config_path: str) -> PiRuntimeConfig:
    cfg = PiRuntimeConfig()
    if not config_path or not os.path.isfile(config_path):
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith("#") or ("=" not in line):
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            try:
                if key == "curve_slowdown_sensitivity":
                    cfg.curve_slowdown_sensitivity = float(value)
                elif key == "ai_smoothing_alpha":
                    cfg.ai_smoothing_alpha = float(value)
                elif key == "ai_no_line_hold_frames":
                    cfg.ai_no_line_hold_frames = int(float(value))
                elif key == "ai_no_line_brake_frames":
                    cfg.ai_no_line_brake_frames = int(float(value))
                elif key == "line_process_interval_ms":
                    cfg.line_process_interval_ms = int(float(value))
                elif key == "ai_control_interval_ms":
                    cfg.ai_control_interval_ms = int(float(value))
                elif key == "far_threshold":
                    cfg.far_threshold = int(float(value))
                elif key == "near_threshold":
                    cfg.near_threshold = int(float(value))
            except ValueError:
                continue

    cfg.curve_slowdown_sensitivity = float(np.clip(cfg.curve_slowdown_sensitivity, 0.0, 2.0))
    cfg.ai_smoothing_alpha = float(np.clip(cfg.ai_smoothing_alpha, 0.0, 1.0))
    cfg.ai_no_line_hold_frames = int(np.clip(cfg.ai_no_line_hold_frames, 0, 60))
    cfg.ai_no_line_brake_frames = int(np.clip(cfg.ai_no_line_brake_frames, 1, 120))
    cfg.line_process_interval_ms = int(np.clip(cfg.line_process_interval_ms, 20, 300))
    cfg.ai_control_interval_ms = int(np.clip(cfg.ai_control_interval_ms, 20, 300))
    cfg.far_threshold = int(np.clip(cfg.far_threshold, 0, 255))
    cfg.near_threshold = int(np.clip(cfg.near_threshold, 0, 255))
    return cfg


def run_remote_mode(pwm_a: GPIO.PWM, pwm_b: GPIO.PWM, camera_fps: float) -> None:
    running = True

    def camera_thread() -> None:
        nonlocal running
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.bind(f"tcp://*:{CAMERA_PORT}")

        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": (320, 240)})
        picam2.configure(config)
        picam2.start()
        picam2.set_controls(
            {
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": 5000,
                "AnalogueGain": 1.0,
            }
        )

        sleep_sec = max(0.0, 1.0 / max(1e-6, camera_fps))
        print("Camera stream thread started in remote mode...")
        try:
            while running:
                img = picam2.capture_array()
                sock.send(img.tobytes())
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
        except Exception as exc:
            print(f"Camera thread error: {exc}")
        finally:
            picam2.stop()
            sock.close()
            ctx.term()

    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{MOTOR_PORT}")
    print("Remote motor command server started...")

    try:
        while running:
            cmd = socket.recv_string()
            try:
                l_str, r_str = cmd.split(",")
                l_speed = int(l_str) * CMD_LEFT_SIGN
                r_speed = int(r_str) * CMD_RIGHT_SIGN
                set_left_motor(l_speed, pwm_b)
                set_right_motor(r_speed, pwm_a)
            except ValueError:
                stop_all_motors(pwm_a, pwm_b)
    except KeyboardInterrupt:
        print("Stopping remote mode...")
        running = False
    finally:
        stop_all_motors(pwm_a, pwm_b)
        socket.close()
        context.term()


def run_local_ai_mode(args: argparse.Namespace, pwm_a: GPIO.PWM, pwm_b: GPIO.PWM) -> None:
    logger = logging.getLogger("pi_local_ai")
    logger.setLevel(logging.INFO)

    runtime_cfg = load_runtime_config(args.config_path)

    onnx_path = os.path.abspath(os.path.expanduser(args.onnx_path))
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    line_detector = CameraReceiver("127.0.0.1", 0, logger)
    line_detector.set_debug_overlay_enabled(False)
    line_detector.set_auto_threshold_enabled(False)
    line_detector.set_thresholds(
        far_threshold=int(runtime_cfg.far_threshold),
        near_threshold=int(runtime_cfg.near_threshold),
    )

    ai_controller = LineTraceONNXController(
        history=int(args.history),
        smoothing_alpha=float(runtime_cfg.ai_smoothing_alpha),
        max_motor_speed=int(args.max_motor_speed),
        stop_on_no_line=bool(args.stop_on_no_line),
        curve_slowdown_sensitivity=float(runtime_cfg.curve_slowdown_sensitivity),
        no_line_hold_frames=int(runtime_cfg.ai_no_line_hold_frames),
        no_line_brake_frames=int(runtime_cfg.ai_no_line_brake_frames),
    )
    ai_controller.load_model(onnx_path)

    control_interval_sec = max(0.02, float(runtime_cfg.ai_control_interval_ms) / 1000.0)
    line_interval_sec = max(0.02, float(runtime_cfg.line_process_interval_ms) / 1000.0)
    camera_sleep_sec = max(0.0, 1.0 / max(1e-6, float(args.camera_fps)))

    current_left_speed = 0
    current_right_speed = 0
    last_control_ts = 0.0
    last_line_ts = 0.0
    cached_features = line_detector.get_latest_line_features()

    picam2 = Picamera2()
    cam_conf = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(cam_conf)
    picam2.start()
    picam2.set_controls(
        {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(args.exposure_time),
            "AnalogueGain": float(args.analogue_gain),
        }
    )

    print("Local AI mode started on Pi.")
    print(f"ONNX={onnx_path}")
    print(
        "intervals: line={}ms control={}ms camera_fps={}".format(
            int(runtime_cfg.line_process_interval_ms),
            int(runtime_cfg.ai_control_interval_ms),
            float(args.camera_fps),
        )
    )

    try:
        while True:
            now = time.perf_counter()

            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)

            if (now - last_line_ts) >= line_interval_sec:
                line_detector.find_line(frame_bgr)
                cached_features = line_detector.get_latest_line_features()
                last_line_ts = now

            if (now - last_control_ts) >= control_interval_sec:
                left_speed, right_speed = ai_controller.predict_motor_speed(
                    line_features=cached_features,
                    current_left_speed=current_left_speed,
                    current_right_speed=current_right_speed,
                    base_speed=int(args.base_speed),
                )

                current_left_speed = int(np.clip(left_speed * MOTOR_LEFT_SIGN, -100, 100))
                current_right_speed = int(np.clip(right_speed * MOTOR_RIGHT_SIGN, -100, 100))

                set_left_motor(current_left_speed, pwm_b)
                set_right_motor(current_right_speed, pwm_a)
                last_control_ts = now

            if camera_sleep_sec > 0.0:
                time.sleep(camera_sleep_sec)
    except KeyboardInterrupt:
        print("Stopping local AI mode...")
    finally:
        stop_all_motors(pwm_a, pwm_b)
        picam2.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi runtime for robot car (remote or local AI mode)")
    parser.add_argument("--mode", type=str, default="remote", choices=["remote", "local_ai"])
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--onnx-path", type=str, default="")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--base-speed", type=int, default=60)
    parser.add_argument("--max-motor-speed", type=int, default=100)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--exposure-time", type=int, default=5000)
    parser.add_argument("--analogue-gain", type=float, default=1.0)
    parser.add_argument("--stop-on-no-line", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()

    if not args.config_path:
        args.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "comfig.txt"))
    if not args.onnx_path:
        args.onnx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot_MLP", "model.onnx"))

    pwm_a, pwm_b = setup_gpio()
    try:
        if args.mode == "local_ai":
            run_local_ai_mode(args, pwm_a, pwm_b)
        else:
            run_remote_mode(pwm_a, pwm_b, camera_fps=args.camera_fps)
    finally:
        stop_all_motors(pwm_a, pwm_b)
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()