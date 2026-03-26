import logging
import os
import shutil
from dataclasses import dataclass
from typing import List

from path_utils import normalize_config_path


@dataclass
class AppConfig:
    save_base_dir: str
    drive_types: List[str]
    drive_speed_default: int
    dpad_drive_speed_scale: float
    dpad_turn_speed_scale: float
    curve_slowdown_sensitivity: float
    ai_smoothing_alpha: float
    ai_history: int
    ai_no_line_hold_frames: int
    ai_no_line_brake_frames: int
    pid_kp: float
    pid_ki: float
    pid_kd: float
    pid_output_limit: float
    pid_integral_limit: float
    pid_gain_smoothing_alpha: float
    pid_steer_rate_limit: float
    cnn_steer_rate_limit: float
    line_process_interval_ms: int
    ai_control_interval_ms: int
    ui_update_interval_ms: int
    line_color_space: str
    line_detection_profile: str
    far_threshold: int
    near_threshold: int
    auto_threshold_enabled: bool
    camera_ids: List[str]
    default_camera_id: str


def ensure_config_file(config_path: str, project_dir: str) -> None:
    if os.path.exists(config_path):
        return
    with open(config_path, mode="w", encoding="utf-8") as f:
        f.write("# Config file for data collection\n")
        f.write("# 各PCで、このプロジェクトフォルダのパスを save_base_dir に設定してください\n")
        f.write(f"save_base_dir={project_dir}\n")
        f.write("drive_speed_default=60\n")
        f.write("dpad_drive_speed_scale=1.0\n")
        f.write("dpad_turn_speed_scale=0.5\n")
        f.write("curve_slowdown_sensitivity=0.70\n")
        f.write("ai_smoothing_alpha=0.35\n")
        f.write("ai_no_line_hold_frames=3\n")
        f.write("ai_no_line_brake_frames=8\n")
        f.write("line_process_interval_ms=70\n")
        f.write("ai_control_interval_ms=100\n")
        f.write("ui_update_interval_ms=30\n")
        f.write("camera_id=camera_1\n")
        f.write("camera_id=camera_2\n")
        f.write("default_camera_id=camera_1\n")
        f.write("\n")
        f.write("drive_type=straight\n")
        f.write("drive_type=left_curve\n")
        f.write("drive_type=right_curve\n")
        f.write("drive_type=stop_and_go\n")


def migrate_legacy_config_if_needed(config_path: str, legacy_path: str) -> None:
    if (not os.path.exists(config_path)) and os.path.exists(legacy_path):
        shutil.copyfile(legacy_path, config_path)


def load_config(
    config_path: str,
    project_dir: str,
    logger: logging.Logger,
    default_drive_speed: int,
    default_dpad_drive_scale: float,
    default_dpad_turn_scale: float,
) -> AppConfig:
    configured_save_base_dir = project_dir
    configured_drive_speed = default_drive_speed
    configured_dpad_drive_scale = default_dpad_drive_scale
    configured_dpad_turn_scale = default_dpad_turn_scale
    configured_curve_slowdown_sensitivity = 0.70
    configured_ai_smoothing_alpha = 0.35
    configured_ai_history = 10
    configured_ai_no_line_hold_frames = 3
    configured_ai_no_line_brake_frames = 8
    configured_pid_kp = 0.95
    configured_pid_ki = 0.08
    configured_pid_kd = 0.22
    configured_pid_output_limit = 0.35
    configured_pid_integral_limit = 1.5
    configured_pid_gain_smoothing_alpha = 0.25
    configured_pid_steer_rate_limit = 0.18
    configured_cnn_steer_rate_limit = 0.15
    configured_line_process_interval_ms = 70
    configured_ai_control_interval_ms = 100
    configured_ui_update_interval_ms = 30
    configured_line_color_space = "lab"
    configured_line_detection_profile = "default"
    configured_far_threshold = 100
    configured_near_threshold = 70
    configured_auto_threshold_enabled = True
    configured_camera_ids: List[str] = []
    configured_default_camera_id = ""
    drive_types: List[str] = []

    with open(config_path, mode="r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "save_base_dir" and value:
                    configured_save_base_dir = normalize_config_path(value)
                elif key == "drive_speed_default" and value:
                    try:
                        configured_drive_speed = int(float(value))
                    except ValueError:
                        logger.warning("Invalid drive_speed_default: %s", value)
                elif key == "dpad_drive_speed_scale" and value:
                    try:
                        configured_dpad_drive_scale = float(value)
                    except ValueError:
                        logger.warning("Invalid dpad_drive_speed_scale: %s", value)
                elif key == "dpad_turn_speed_scale" and value:
                    try:
                        configured_dpad_turn_scale = float(value)
                    except ValueError:
                        logger.warning("Invalid dpad_turn_speed_scale: %s", value)
                elif key == "curve_slowdown_sensitivity" and value:
                    try:
                        configured_curve_slowdown_sensitivity = float(value)
                    except ValueError:
                        logger.warning("Invalid curve_slowdown_sensitivity: %s", value)
                elif key == "ai_smoothing_alpha" and value:
                    try:
                        configured_ai_smoothing_alpha = float(value)
                    except ValueError:
                        logger.warning("Invalid ai_smoothing_alpha: %s", value)
                elif key == "ai_history" and value:
                    try:
                        configured_ai_history = int(float(value))
                    except ValueError:
                        logger.warning("Invalid ai_history: %s", value)
                elif key == "ai_no_line_hold_frames" and value:
                    try:
                        configured_ai_no_line_hold_frames = int(float(value))
                    except ValueError:
                        logger.warning("Invalid ai_no_line_hold_frames: %s", value)
                elif key == "ai_no_line_brake_frames" and value:
                    try:
                        configured_ai_no_line_brake_frames = int(float(value))
                    except ValueError:
                        logger.warning("Invalid ai_no_line_brake_frames: %s", value)
                elif key == "pid_kp" and value:
                    try:
                        configured_pid_kp = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_kp: %s", value)
                elif key == "pid_ki" and value:
                    try:
                        configured_pid_ki = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_ki: %s", value)
                elif key == "pid_kd" and value:
                    try:
                        configured_pid_kd = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_kd: %s", value)
                elif key == "pid_output_limit" and value:
                    try:
                        configured_pid_output_limit = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_output_limit: %s", value)
                elif key == "pid_integral_limit" and value:
                    try:
                        configured_pid_integral_limit = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_integral_limit: %s", value)
                elif key == "pid_gain_smoothing_alpha" and value:
                    try:
                        configured_pid_gain_smoothing_alpha = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_gain_smoothing_alpha: %s", value)
                elif key == "pid_steer_rate_limit" and value:
                    try:
                        configured_pid_steer_rate_limit = float(value)
                    except ValueError:
                        logger.warning("Invalid pid_steer_rate_limit: %s", value)
                elif key == "cnn_steer_rate_limit" and value:
                    try:
                        configured_cnn_steer_rate_limit = float(value)
                    except ValueError:
                        logger.warning("Invalid cnn_steer_rate_limit: %s", value)
                elif key == "line_process_interval_ms" and value:
                    try:
                        configured_line_process_interval_ms = int(float(value))
                    except ValueError:
                        logger.warning("Invalid line_process_interval_ms: %s", value)
                elif key == "ai_control_interval_ms" and value:
                    try:
                        configured_ai_control_interval_ms = int(float(value))
                    except ValueError:
                        logger.warning("Invalid ai_control_interval_ms: %s", value)
                elif key == "ui_update_interval_ms" and value:
                    try:
                        configured_ui_update_interval_ms = int(float(value))
                    except ValueError:
                        logger.warning("Invalid ui_update_interval_ms: %s", value)
                elif key == "line_color_space" and value:
                    configured_line_color_space = str(value).strip().lower()
                elif key == "line_detection_profile" and value:
                    configured_line_detection_profile = str(value).strip().lower()
                elif key == "far_threshold" and value:
                    try:
                        configured_far_threshold = int(float(value))
                    except ValueError:
                        logger.warning("Invalid far_threshold: %s", value)
                elif key == "near_threshold" and value:
                    try:
                        configured_near_threshold = int(float(value))
                    except ValueError:
                        logger.warning("Invalid near_threshold: %s", value)
                elif key == "auto_threshold_enabled" and value:
                    lowered = str(value).strip().lower()
                    if lowered in {"1", "true", "yes", "on"}:
                        configured_auto_threshold_enabled = True
                    elif lowered in {"0", "false", "no", "off"}:
                        configured_auto_threshold_enabled = False
                    else:
                        logger.warning("Invalid auto_threshold_enabled: %s", value)
                elif key == "camera_id" and value:
                    configured_camera_ids.append(value)
                elif key == "default_camera_id" and value:
                    configured_default_camera_id = value
                elif key == "drive_type" and value:
                    drive_types.append(value)
                continue

            drive_types.append(line)

    if not drive_types:
        drive_types = ["straight"]
        logger.warning("No drive_type found in config; fallback to 'straight'.")

    if not configured_save_base_dir:
        configured_save_base_dir = project_dir
        logger.warning("save_base_dir is empty; fallback to project_dir.")

    configured_drive_speed = max(20, min(100, int(configured_drive_speed)))

    if configured_dpad_drive_scale <= 0:
        logger.warning("dpad_drive_speed_scale must be positive. fallback=%s", default_dpad_drive_scale)
        configured_dpad_drive_scale = default_dpad_drive_scale

    if configured_dpad_turn_scale <= 0:
        logger.warning("dpad_turn_speed_scale must be positive. fallback=%s", default_dpad_turn_scale)
        configured_dpad_turn_scale = default_dpad_turn_scale

    configured_curve_slowdown_sensitivity = max(0.0, min(2.0, float(configured_curve_slowdown_sensitivity)))
    configured_ai_smoothing_alpha = max(0.0, min(1.0, float(configured_ai_smoothing_alpha)))
    configured_ai_history = max(1, min(50, int(configured_ai_history)))
    configured_ai_no_line_hold_frames = max(0, min(60, int(configured_ai_no_line_hold_frames)))
    configured_ai_no_line_brake_frames = max(1, min(120, int(configured_ai_no_line_brake_frames)))
    configured_pid_kp = max(0.0, min(10.0, float(configured_pid_kp)))
    configured_pid_ki = max(0.0, min(5.0, float(configured_pid_ki)))
    configured_pid_kd = max(0.0, min(5.0, float(configured_pid_kd)))
    configured_pid_output_limit = max(0.01, min(1.0, float(configured_pid_output_limit)))
    configured_pid_integral_limit = max(0.0, min(10.0, float(configured_pid_integral_limit)))
    configured_pid_gain_smoothing_alpha = max(0.0, min(1.0, float(configured_pid_gain_smoothing_alpha)))
    configured_pid_steer_rate_limit = max(0.01, min(1.0, float(configured_pid_steer_rate_limit)))
    configured_cnn_steer_rate_limit = max(0.01, min(1.0, float(configured_cnn_steer_rate_limit)))
    configured_line_process_interval_ms = max(20, min(300, int(configured_line_process_interval_ms)))
    configured_ai_control_interval_ms = max(20, min(300, int(configured_ai_control_interval_ms)))
    configured_ui_update_interval_ms = max(10, min(100, int(configured_ui_update_interval_ms)))
    configured_line_color_space = "hsv" if configured_line_color_space == "hsv" else "lab"
    if configured_line_detection_profile not in {"default", "panel_seam", "glare", "panel_seam_glare"}:
        configured_line_detection_profile = "default"
    configured_far_threshold = max(0, min(255, int(configured_far_threshold)))
    configured_near_threshold = max(0, min(255, int(configured_near_threshold)))

    if not configured_camera_ids:
        configured_camera_ids = ["camera_1", "camera_2"]
        logger.warning("No camera_id found in config; fallback to default camera list.")

    if not configured_default_camera_id or configured_default_camera_id not in configured_camera_ids:
        configured_default_camera_id = configured_camera_ids[0]

    return AppConfig(
        save_base_dir=configured_save_base_dir,
        drive_types=drive_types,
        drive_speed_default=configured_drive_speed,
        dpad_drive_speed_scale=configured_dpad_drive_scale,
        dpad_turn_speed_scale=configured_dpad_turn_scale,
        curve_slowdown_sensitivity=configured_curve_slowdown_sensitivity,
        ai_smoothing_alpha=configured_ai_smoothing_alpha,
        ai_history=configured_ai_history,
        ai_no_line_hold_frames=configured_ai_no_line_hold_frames,
        ai_no_line_brake_frames=configured_ai_no_line_brake_frames,
        pid_kp=configured_pid_kp,
        pid_ki=configured_pid_ki,
        pid_kd=configured_pid_kd,
        pid_output_limit=configured_pid_output_limit,
        pid_integral_limit=configured_pid_integral_limit,
        pid_gain_smoothing_alpha=configured_pid_gain_smoothing_alpha,
        pid_steer_rate_limit=configured_pid_steer_rate_limit,
        cnn_steer_rate_limit=configured_cnn_steer_rate_limit,
        line_process_interval_ms=configured_line_process_interval_ms,
        ai_control_interval_ms=configured_ai_control_interval_ms,
        ui_update_interval_ms=configured_ui_update_interval_ms,
        line_color_space=configured_line_color_space,
        line_detection_profile=configured_line_detection_profile,
        far_threshold=configured_far_threshold,
        near_threshold=configured_near_threshold,
        auto_threshold_enabled=bool(configured_auto_threshold_enabled),
        camera_ids=configured_camera_ids,
        default_camera_id=configured_default_camera_id,
    )
