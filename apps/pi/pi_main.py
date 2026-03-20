import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

_SHARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

import cv2
import numpy as np
import RPi.GPIO as GPIO
import zmq
from picamera2 import Picamera2

from ai_controller import LineTraceONNXController
from camera_receiver import CameraReceiver
from pid_learned_controller import LearnedPIDONNXController


# Network ports used by legacy remote mode.
CAMERA_PORT = 5556
LINE_FEATURE_PORT = 5557
MOTOR_PORT = 5555
CALIB_PORT = 5558
JPEG_QUALITY_REMOTE = 70

# Command sign correction in remote mode.
CMD_LEFT_SIGN = -1
CMD_RIGHT_SIGN = 1

# Motor output sign correction in local AI mode.
MOTOR_LEFT_SIGN = -1
MOTOR_RIGHT_SIGN = 1

# GPIO pin mapping.
MA_IN1, MA_IN2, MA_PWM = 19, 21, 23
MB_IN1, MB_IN2, MB_PWM = 15, 13, 11


def setup_gpio() -> Tuple[GPIO.PWM, GPIO.PWM]:
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


def safe_shutdown_gpio(pwm_a: Optional[GPIO.PWM], pwm_b: Optional[GPIO.PWM]) -> None:
    """Stop motors/PWM in a destructor-safe order before GPIO cleanup."""
    try:
        if pwm_a is not None and pwm_b is not None:
            stop_all_motors(pwm_a, pwm_b)
    except Exception:
        pass

    for pwm in (pwm_a, pwm_b):
        if pwm is None:
            continue
        try:
            pwm.stop()
        except Exception:
            pass

    # Drop PWM references before GPIO.cleanup() so __del__ does not run after
    # low-level handles are already released.
    try:
        del pwm_a
        del pwm_b
    except Exception:
        pass

    try:
        GPIO.cleanup()
    except Exception:
        pass


@dataclass
class PiRuntimeConfig:
    curve_slowdown_sensitivity: float = 0.70
    ai_smoothing_alpha: float = 0.35
    ai_no_line_hold_frames: int = 3
    ai_no_line_brake_frames: int = 8
    line_process_interval_ms: int = 70
    ai_control_interval_ms: int = 100
    line_color_space: str = "lab"
    line_detection_profile: str = "default"
    far_threshold: int = 100
    near_threshold: int = 70
    auto_threshold_enabled: bool = True
    pid_kp: float = 0.95
    pid_ki: float = 0.08
    pid_kd: float = 0.22
    pid_output_limit: float = 1.0
    pid_integral_limit: float = 1.5
    pid_stop_on_no_line: bool = True
    pid_gain_smoothing_alpha: float = 0.25
    pid_steer_rate_limit: float = 0.18
    motor_left_sign: int = -1
    motor_right_sign: int = 1


class FrameBuffer:
    """最新フレームのみ保持するスレッドセーフバッファ（ZMQ CONFLATE と同等）。
    カメラスレッドが書き込み、検出スレッドが読み出す。"""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame


def _parse_bool(text: str) -> Optional[bool]:
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float,
        integral_limit: float,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(max(1e-6, abs(output_limit)))
        self.integral_limit = float(max(0.0, abs(integral_limit)))
        self.integral = 0.0
        self.prev_error: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        dt = max(1e-4, float(dt))
        self.integral += float(error) * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))

        derivative = 0.0
        if self.prev_error is not None:
            derivative = (float(error) - float(self.prev_error)) / dt
        self.prev_error = float(error)

        out = (self.kp * float(error)) + (self.ki * self.integral) + (self.kd * derivative)
        return float(np.clip(out, -self.output_limit, self.output_limit))


def _extract_pid_error(line_features: dict) -> Optional[float]:
    # Near zone gets the highest weight because it is most relevant for immediate steering.
    weights = [
        ("line_detect_top", "line_offset_top", 0.20),
        ("line_detect_mid", "line_offset_mid", 0.35),
        ("line_detect_bottom", "line_offset_bottom", 0.45),
    ]

    weighted_sum = 0.0
    weight_total = 0.0
    for detect_key, offset_key, weight in weights:
        detected = float(line_features.get(detect_key, 0.0))
        if detected <= 0.0:
            continue
        offset = float(np.clip(line_features.get(offset_key, 0.0), -1.0, 1.0))
        weighted_sum += weight * offset
        weight_total += weight

    if weight_total <= 0.0:
        return None
    return float(np.clip(weighted_sum / weight_total, -1.0, 1.0))


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
                elif key == "line_color_space":
                    cfg.line_color_space = str(value).strip().lower()
                elif key == "line_detection_profile":
                    cfg.line_detection_profile = str(value).strip().lower()
                elif key == "far_threshold":
                    cfg.far_threshold = int(float(value))
                elif key == "near_threshold":
                    cfg.near_threshold = int(float(value))
                elif key == "auto_threshold_enabled":
                    parsed = _parse_bool(value)
                    if parsed is not None:
                        cfg.auto_threshold_enabled = parsed
                elif key == "pid_kp":
                    cfg.pid_kp = float(value)
                elif key == "pid_ki":
                    cfg.pid_ki = float(value)
                elif key == "pid_kd":
                    cfg.pid_kd = float(value)
                elif key == "pid_output_limit":
                    cfg.pid_output_limit = float(value)
                elif key == "pid_integral_limit":
                    cfg.pid_integral_limit = float(value)
                elif key == "pid_stop_on_no_line":
                    parsed = _parse_bool(value)
                    if parsed is not None:
                        cfg.pid_stop_on_no_line = parsed
                elif key == "pid_gain_smoothing_alpha":
                    cfg.pid_gain_smoothing_alpha = float(value)
                elif key == "pid_steer_rate_limit":
                    cfg.pid_steer_rate_limit = float(value)
                elif key == "motor_left_sign":
                    cfg.motor_left_sign = int(float(value))
                elif key == "motor_right_sign":
                    cfg.motor_right_sign = int(float(value))
            except ValueError:
                continue

    cfg.curve_slowdown_sensitivity = float(np.clip(cfg.curve_slowdown_sensitivity, 0.0, 2.0))
    cfg.ai_smoothing_alpha = float(np.clip(cfg.ai_smoothing_alpha, 0.0, 1.0))
    cfg.ai_no_line_hold_frames = int(np.clip(cfg.ai_no_line_hold_frames, 0, 60))
    cfg.ai_no_line_brake_frames = int(np.clip(cfg.ai_no_line_brake_frames, 1, 120))
    cfg.line_process_interval_ms = int(np.clip(cfg.line_process_interval_ms, 20, 300))
    cfg.ai_control_interval_ms = int(np.clip(cfg.ai_control_interval_ms, 20, 300))
    cfg.line_color_space = "hsv" if str(cfg.line_color_space).strip().lower() == "hsv" else "lab"
    profile = str(cfg.line_detection_profile).strip().lower()
    if profile not in {"default", "panel_seam", "glare", "panel_seam_glare"}:
        profile = "default"
    cfg.line_detection_profile = profile
    cfg.far_threshold = int(np.clip(cfg.far_threshold, 0, 255))
    cfg.near_threshold = int(np.clip(cfg.near_threshold, 0, 255))
    cfg.pid_output_limit = float(np.clip(cfg.pid_output_limit, 0.05, 2.5))
    cfg.pid_integral_limit = float(np.clip(cfg.pid_integral_limit, 0.0, 10.0))
    cfg.pid_gain_smoothing_alpha = float(np.clip(cfg.pid_gain_smoothing_alpha, 0.0, 1.0))
    cfg.pid_steer_rate_limit = float(np.clip(cfg.pid_steer_rate_limit, 0.02, 0.50))
    cfg.motor_left_sign = -1 if int(cfg.motor_left_sign) < 0 else 1
    cfg.motor_right_sign = -1 if int(cfg.motor_right_sign) < 0 else 1
    return cfg


def run_pid_mode(args: argparse.Namespace, pwm_a: GPIO.PWM, pwm_b: GPIO.PWM) -> None:
    logger = logging.getLogger("pi_pid")
    logger.setLevel(logging.INFO)

    print(f"[pid] Loading runtime config: {args.config_path}")
    runtime_cfg = load_runtime_config(args.config_path)
    motor_left_sign = int(runtime_cfg.motor_left_sign)
    motor_right_sign = int(runtime_cfg.motor_right_sign)

    line_detector = CameraReceiver("127.0.0.1", 0, logger)
    line_detector.set_debug_overlay_enabled(False)
    line_detector.set_line_detection_profile_name(runtime_cfg.line_detection_profile)
    line_detector.set_line_color_space(runtime_cfg.line_color_space)
    line_detector.set_auto_threshold_enabled(bool(runtime_cfg.auto_threshold_enabled))
    line_detector.set_thresholds(
        far_threshold=int(runtime_cfg.far_threshold),
        near_threshold=int(runtime_cfg.near_threshold),
    )

    kp = float(runtime_cfg.pid_kp if args.pid_kp is None else args.pid_kp)
    ki = float(runtime_cfg.pid_ki if args.pid_ki is None else args.pid_ki)
    kd = float(runtime_cfg.pid_kd if args.pid_kd is None else args.pid_kd)
    output_limit = float(runtime_cfg.pid_output_limit)
    integral_limit = float(runtime_cfg.pid_integral_limit)
    stop_on_no_line = bool(runtime_cfg.pid_stop_on_no_line)

    pid = PIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        output_limit=output_limit,
        integral_limit=integral_limit,
    )

    control_interval_sec = max(0.02, float(runtime_cfg.ai_control_interval_ms) / 1000.0)
    line_interval_sec = max(0.02, float(runtime_cfg.line_process_interval_ms) / 1000.0)
    camera_sleep_sec = max(0.0, 1.0 / max(1e-6, float(args.camera_fps)))

    print(f"[pid] gains: Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}")
    print(
        "[pid] intervals: line={}ms control={}ms camera_fps={}".format(
            int(runtime_cfg.line_process_interval_ms),
            int(runtime_cfg.ai_control_interval_ms),
            float(args.camera_fps),
        )
    )

    stop_event = threading.Event()
    frame_buf = FrameBuffer()

    def _camera_loop() -> None:
        print("[pid] Initializing camera...")
        picam2 = Picamera2()
        cam_conf = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
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
        print("[pid] Camera started.")
        next_frame_ts = time.perf_counter()
        try:
            while not stop_event.is_set():
                frame_rgb = picam2.capture_array()
                frame_buf.put(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if camera_sleep_sec > 0.0:
                    next_frame_ts += camera_sleep_sec
                    wait = next_frame_ts - time.perf_counter()
                    if wait > 0.0:
                        time.sleep(wait)
                    else:
                        next_frame_ts = time.perf_counter()
        finally:
            picam2.stop()

    def _detection_loop() -> None:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            frame = frame_buf.get()
            if frame is not None:
                line_detector.find_line(frame)
            elapsed = time.perf_counter() - t0
            remaining = line_interval_sec - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    cam_thread = threading.Thread(target=_camera_loop, daemon=True)
    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    cam_thread.start()
    det_thread.start()

    last_control_ts = time.perf_counter()
    last_log_ts = 0.0
    cmd_left = 0
    cmd_right = 0
    error: Optional[float] = None

    print("[pid] Started.")
    try:
        while True:
            now = time.perf_counter()
            if (now - last_control_ts) >= control_interval_sec:
                dt = now - last_control_ts
                last_control_ts = now

                cached_features = line_detector.get_latest_line_features()
                error = _extract_pid_error(cached_features)
                if error is None and stop_on_no_line:
                    pid.reset()
                    cmd_left = 0
                    cmd_right = 0
                else:
                    if error is None:
                        error = 0.0
                    steer = pid.update(error=error, dt=dt)
                    turn = int(np.round(steer * float(args.base_speed)))
                    left = int(np.clip(int(args.base_speed) + turn, -100, 100))
                    right = int(np.clip(int(args.base_speed) - turn, -100, 100))
                    cmd_left = int(np.clip(left * motor_left_sign, -100, 100))
                    cmd_right = int(np.clip(right * motor_right_sign, -100, 100))

                set_left_motor(cmd_left, pwm_b)
                set_right_motor(cmd_right, pwm_a)

                if (now - last_log_ts) >= 1.0:
                    e_disp = "None" if error is None else f"{float(error):+.3f}"
                    print(f"[pid] err={e_disp} cmd=({cmd_left},{cmd_right})")
                    last_log_ts = now

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Stopping pid mode...")
    finally:
        stop_event.set()
        cam_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        stop_all_motors(pwm_a, pwm_b)


def run_remote_mode(pwm_a: GPIO.PWM, pwm_b: GPIO.PWM, camera_fps: float, config_path: str) -> None:
    running = True
    logger = logging.getLogger("pi_remote")
    logger.setLevel(logging.INFO)
    runtime_cfg = load_runtime_config(config_path)

    line_detector = CameraReceiver("127.0.0.1", 0, logger)
    line_detector.set_debug_overlay_enabled(False)
    line_detector.set_line_detection_profile_name(runtime_cfg.line_detection_profile)
    line_detector.set_line_color_space(runtime_cfg.line_color_space)
    line_detector.set_auto_threshold_enabled(bool(runtime_cfg.auto_threshold_enabled))
    line_detector.set_thresholds(
        far_threshold=int(runtime_cfg.far_threshold),
        near_threshold=int(runtime_cfg.near_threshold),
    )
    line_interval_sec = max(0.02, float(runtime_cfg.line_process_interval_ms) / 1000.0)
    last_line_ts = 0.0
    cached_features = line_detector.get_latest_line_features()

    def camera_thread() -> None:
        nonlocal running, last_line_ts, cached_features
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.bind(f"tcp://*:{CAMERA_PORT}")

        feature_sock = ctx.socket(zmq.PUSH)
        feature_sock.setsockopt(zmq.CONFLATE, 1)
        feature_sock.bind(f"tcp://*:{LINE_FEATURE_PORT}")

        print("[remote] Initializing camera...")
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        picam2.set_controls(
            {
                "AeEnable": True,
                "AwbEnable": True,
            }
        )

        sleep_sec = max(0.0, 1.0 / max(1e-6, camera_fps))
        next_frame_ts = time.perf_counter()
        print("[remote] Camera stream thread started.")
        try:
            while running:
                now = time.perf_counter()
                img_rgb = picam2.capture_array()
                frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                if (now - last_line_ts) >= line_interval_sec:
                    line_detector.find_line(frame_bgr)
                    cached_features = line_detector.get_latest_line_features()
                    last_line_ts = now

                ok, enc = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY_REMOTE)],
                )
                if ok:
                    sock.send(enc.tobytes())
                else:
                    # エンコード失敗時のみ後方互換のraw送信にフォールバック
                    sock.send(img_rgb.tobytes())
                feature_sock.send_json({"line_features": cached_features})
                if sleep_sec > 0.0:
                    next_frame_ts += sleep_sec
                    wait_sec = next_frame_ts - time.perf_counter()
                    if wait_sec > 0.0:
                        time.sleep(wait_sec)
                    else:
                        next_frame_ts = time.perf_counter()
        except Exception as exc:
            print(f"Camera thread error: {exc}")
        finally:
            picam2.stop()
            sock.close()
            feature_sock.close()
            ctx.term()

    def calib_receive_thread() -> None:
        nonlocal running
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        try:
            sock.bind(f"tcp://*:{CALIB_PORT}")
        except zmq.ZMQError as e:
            print(f"[remote] Calibration socket bind failed (port={CALIB_PORT}): {e}")
            ctx.term()
            return
        print("[remote] Calibration receive thread started.")
        try:
            while running:
                try:
                    msg = sock.recv_json()
                    w_norm = msg.get("calib_width_norm")
                    if w_norm is None:
                        line_detector.clear_calibrated_width()
                        print("[remote] Calibration cleared.")
                    else:
                        line_detector.set_calibrated_width_norm(float(w_norm))
                        print(f"[remote] Calibration set: {w_norm:.4f}")
                except zmq.ZMQError:
                    pass
        except Exception as e:
            print(f"[remote] Calibration thread error: {e}")
        finally:
            sock.close()
            ctx.term()

    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    calib_thread = threading.Thread(target=calib_receive_thread, daemon=True)
    calib_thread.start()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{MOTOR_PORT}")
    print("[remote] Motor command server started.")

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

    print(f"[local_ai] Loading runtime config: {args.config_path}")
    runtime_cfg = load_runtime_config(args.config_path)
    motor_left_sign = int(runtime_cfg.motor_left_sign)
    motor_right_sign = int(runtime_cfg.motor_right_sign)

    onnx_path = os.path.abspath(os.path.expanduser(args.onnx_path))
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    print(f"[local_ai] ONNX found: {onnx_path}")

    line_detector = CameraReceiver("127.0.0.1", 0, logger)
    line_detector.set_debug_overlay_enabled(False)
    line_detector.set_line_detection_profile_name(runtime_cfg.line_detection_profile)
    line_detector.set_line_color_space(runtime_cfg.line_color_space)
    line_detector.set_auto_threshold_enabled(bool(runtime_cfg.auto_threshold_enabled))
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
    print("[local_ai] Loading ONNX model...")
    ai_controller.load_model(onnx_path)
    print("[local_ai] ONNX model loaded.")

    control_interval_sec = max(0.02, float(runtime_cfg.ai_control_interval_ms) / 1000.0)
    line_interval_sec = max(0.02, float(runtime_cfg.line_process_interval_ms) / 1000.0)
    camera_sleep_sec = max(0.0, 1.0 / max(1e-6, float(args.camera_fps)))

    print(f"[local_ai] ONNX={onnx_path}")
    print(
        "[local_ai] intervals: line={}ms control={}ms camera_fps={}".format(
            int(runtime_cfg.line_process_interval_ms),
            int(runtime_cfg.ai_control_interval_ms),
            float(args.camera_fps),
        )
    )

    stop_event = threading.Event()
    frame_buf = FrameBuffer()

    def _camera_loop() -> None:
        print("[local_ai] Initializing camera...")
        picam2 = Picamera2()
        cam_conf = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
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
        print("[local_ai] Camera started.")
        next_frame_ts = time.perf_counter()
        try:
            while not stop_event.is_set():
                frame_rgb = picam2.capture_array()
                frame_buf.put(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if camera_sleep_sec > 0.0:
                    next_frame_ts += camera_sleep_sec
                    wait = next_frame_ts - time.perf_counter()
                    if wait > 0.0:
                        time.sleep(wait)
                    else:
                        next_frame_ts = time.perf_counter()
        finally:
            picam2.stop()

    def _detection_loop() -> None:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            frame = frame_buf.get()
            if frame is not None:
                line_detector.find_line(frame)
            elapsed = time.perf_counter() - t0
            remaining = line_interval_sec - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    cam_thread = threading.Thread(target=_camera_loop, daemon=True)
    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    cam_thread.start()
    det_thread.start()

    model_left_speed = 0
    model_right_speed = 0
    last_control_ts = time.perf_counter()

    print("[local_ai] Started.")
    try:
        while True:
            now = time.perf_counter()
            if (now - last_control_ts) >= control_interval_sec:
                cached_features = line_detector.get_latest_line_features()
                left_speed, right_speed = ai_controller.predict_motor_speed(
                    line_features=cached_features,
                    current_left_speed=model_left_speed,
                    current_right_speed=model_right_speed,
                    base_speed=int(args.base_speed),
                )

                model_left_speed = int(np.clip(left_speed, -100, 100))
                model_right_speed = int(np.clip(right_speed, -100, 100))

                hw_left_speed = int(np.clip(model_left_speed * motor_left_sign, -100, 100))
                hw_right_speed = int(np.clip(model_right_speed * motor_right_sign, -100, 100))

                set_left_motor(hw_left_speed, pwm_b)
                set_right_motor(hw_right_speed, pwm_a)
                last_control_ts = now

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Stopping local AI mode...")
    finally:
        stop_event.set()
        cam_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        stop_all_motors(pwm_a, pwm_b)


def run_pid_learned_mode(args: argparse.Namespace, pwm_a: GPIO.PWM, pwm_b: GPIO.PWM) -> None:
    logger = logging.getLogger("pi_pid_learned")
    logger.setLevel(logging.INFO)

    print(f"[pid_learned] Loading runtime config: {args.config_path}")
    runtime_cfg = load_runtime_config(args.config_path)
    motor_left_sign = int(runtime_cfg.motor_left_sign)
    motor_right_sign = int(runtime_cfg.motor_right_sign)

    onnx_path = os.path.abspath(os.path.expanduser(args.onnx_path))
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"PID-gain ONNX model not found: {onnx_path}")

    line_detector = CameraReceiver("127.0.0.1", 0, logger)
    line_detector.set_debug_overlay_enabled(False)
    line_detector.set_line_detection_profile_name(runtime_cfg.line_detection_profile)
    line_detector.set_line_color_space(runtime_cfg.line_color_space)
    line_detector.set_auto_threshold_enabled(bool(runtime_cfg.auto_threshold_enabled))
    line_detector.set_thresholds(
        far_threshold=int(runtime_cfg.far_threshold),
        near_threshold=int(runtime_cfg.near_threshold),
    )

    controller = LearnedPIDONNXController(
        history=int(args.history),
        smoothing_alpha=float(runtime_cfg.ai_smoothing_alpha),
        max_motor_speed=int(args.max_motor_speed),
        stop_on_no_line=bool(runtime_cfg.pid_stop_on_no_line),
        steer_limit=float(runtime_cfg.pid_output_limit),
        no_line_hold_frames=int(runtime_cfg.ai_no_line_hold_frames),
        no_line_brake_frames=int(runtime_cfg.ai_no_line_brake_frames),
        gain_smoothing_alpha=float(runtime_cfg.pid_gain_smoothing_alpha),
        steer_rate_limit=float(runtime_cfg.pid_steer_rate_limit),
    )
    print("[pid_learned] Loading ONNX model...")
    controller.load_model(onnx_path)
    print("[pid_learned] ONNX model loaded.")

    control_interval_sec = max(0.02, float(runtime_cfg.ai_control_interval_ms) / 1000.0)
    line_interval_sec = max(0.02, float(runtime_cfg.line_process_interval_ms) / 1000.0)
    camera_sleep_sec = max(0.0, 1.0 / max(1e-6, float(args.camera_fps)))

    print(f"[pid_learned] ONNX={onnx_path}")

    stop_event = threading.Event()
    frame_buf = FrameBuffer()

    def _camera_loop() -> None:
        print("[pid_learned] Initializing camera...")
        picam2 = Picamera2()
        cam_conf = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
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
        print("[pid_learned] Camera started.")
        next_frame_ts = time.perf_counter()
        try:
            while not stop_event.is_set():
                frame_rgb = picam2.capture_array()
                frame_buf.put(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if camera_sleep_sec > 0.0:
                    next_frame_ts += camera_sleep_sec
                    wait = next_frame_ts - time.perf_counter()
                    if wait > 0.0:
                        time.sleep(wait)
                    else:
                        next_frame_ts = time.perf_counter()
        finally:
            picam2.stop()

    def _detection_loop() -> None:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            frame = frame_buf.get()
            if frame is not None:
                line_detector.find_line(frame)
            elapsed = time.perf_counter() - t0
            remaining = line_interval_sec - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    cam_thread = threading.Thread(target=_camera_loop, daemon=True)
    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    cam_thread.start()
    det_thread.start()

    model_left_speed = 0
    model_right_speed = 0
    last_control_ts = time.perf_counter()
    last_log_ts = 0.0

    print("[pid_learned] Started.")
    try:
        while True:
            now = time.perf_counter()
            if (now - last_control_ts) >= control_interval_sec:
                cached_features = line_detector.get_latest_line_features()
                left_speed, right_speed = controller.predict_motor_speed(
                    line_features=cached_features,
                    current_left_speed=model_left_speed,
                    current_right_speed=model_right_speed,
                    base_speed=int(args.base_speed),
                )

                model_left_speed = int(np.clip(left_speed, -100, 100))
                model_right_speed = int(np.clip(right_speed, -100, 100))

                hw_left_speed = int(np.clip(model_left_speed * motor_left_sign, -100, 100))
                hw_right_speed = int(np.clip(model_right_speed * motor_right_sign, -100, 100))

                set_left_motor(hw_left_speed, pwm_b)
                set_right_motor(hw_right_speed, pwm_a)
                last_control_ts = now

                if (now - last_log_ts) >= 1.0:
                    kp, ki, kd = controller.last_gains
                    print(
                        "[pid_learned] gain=(%.3f, %.3f, %.3f) model_cmd=(%d,%d) hw_cmd=(%d,%d)"
                        % (kp, ki, kd, model_left_speed, model_right_speed, hw_left_speed, hw_right_speed)
                    )
                    last_log_ts = now

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Stopping pid_learned mode...")
    finally:
        stop_event.set()
        cam_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        stop_all_motors(pwm_a, pwm_b)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi runtime for robot car (remote or local AI mode)")
    parser.add_argument("--mode", type=str, default="remote", choices=["remote", "local_ai", "pid", "pid_learned"])
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--onnx-path", type=str, default="")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--base-speed", type=int, default=60)
    parser.add_argument("--max-motor-speed", type=int, default=100)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--exposure-time", type=int, default=5000)
    parser.add_argument("--analogue-gain", type=float, default=1.0)
    parser.add_argument("--pid-kp", type=float, default=None)
    parser.add_argument("--pid-ki", type=float, default=None)
    parser.add_argument("--pid-kd", type=float, default=None)
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        parser.add_argument("--stop-on-no-line", action=bool_action, default=True)
    else:
        # Python 3.8 compatibility on older Raspberry Pi OS images.
        parser.add_argument("--stop-on-no-line", dest="stop_on_no_line", action="store_true", default=True)
        parser.add_argument("--no-stop-on-no-line", dest="stop_on_no_line", action="store_false")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()

    if not args.config_path:
        args.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "comfig.txt"))
    if not args.onnx_path:
        args.onnx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "robot_MLP", "model.onnx"))

    print(f"[boot] mode={args.mode}")
    print(f"[boot] config={args.config_path}")
    print(f"[boot] onnx={args.onnx_path}")

    # Validate ONNX path before touching GPIO to avoid noisy PWM cleanup errors.
    if args.mode in ("local_ai", "pid_learned"):
        onnx_abs = os.path.abspath(os.path.expanduser(args.onnx_path))
        if not os.path.isfile(onnx_abs):
            raise FileNotFoundError(
                "ONNX model not found: {}\nHint: check filename typo (e.g. compat vs compact) and file location.".format(
                    onnx_abs
                )
            )
        args.onnx_path = onnx_abs

    pwm_a, pwm_b = setup_gpio()
    try:
        if args.mode == "local_ai":
            run_local_ai_mode(args, pwm_a, pwm_b)
        elif args.mode == "pid":
            run_pid_mode(args, pwm_a, pwm_b)
        elif args.mode == "pid_learned":
            run_pid_learned_mode(args, pwm_a, pwm_b)
        else:
            run_remote_mode(
                pwm_a,
                pwm_b,
                camera_fps=args.camera_fps,
                config_path=args.config_path,
            )
    finally:
        safe_shutdown_gpio(pwm_a, pwm_b)