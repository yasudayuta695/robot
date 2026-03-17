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
        camera_ids=configured_camera_ids,
        default_camera_id=configured_default_camera_id,
    )
