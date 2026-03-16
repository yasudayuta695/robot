import csv
import datetime
import logging
import os
import shutil
import time
from enum import Enum
from typing import Dict, List

import cv2
import numpy as np


class RecorderState(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    RECORDING = "recording"


class DataRecorder:
    CSV_HEADER = [
        "image_path",
        "left_speed",
        "right_speed",
        "drive_type",
        "drive_speed_base",
        "control_mode",
        "line_detect_top",
        "line_detect_mid",
        "line_detect_bottom",
        "line_offset_top",
        "line_offset_mid",
        "line_offset_bottom",
        "current_left_norm",
        "current_right_norm",
        "base_speed_norm",
        "target_left_norm",
        "target_right_norm",
    ]

    def __init__(self, save_base_dir: str, logger: logging.Logger) -> None:
        self.logger = logger
        self.save_base_dir = save_base_dir
        self.temp_dir = os.path.join(self.save_base_dir, "temp_record")
        self.temp_img_dir = os.path.join(self.temp_dir, "images")
        self.temp_csv_path = os.path.join(self.temp_dir, "driving_log.csv")
        self.state: RecorderState = RecorderState.IDLE
        self.csv_file = None
        self.csv_writer = None
        self.last_save_time = 0.0
        self.recording_session_id = ""
        self.frame_sequence = 0
        self.output_folder_name = ""
        self.recording_started_at = None
        self.recording_stopped_at = None
        self.last_target_left_norm = 0.0
        self.last_target_right_norm = 0.0

    def arm(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_img_dir, exist_ok=True)

        self.csv_file = open(self.temp_csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.CSV_HEADER)

        self.state = RecorderState.ARMED
        self.last_save_time = 0.0
        self.recording_session_id = ""
        self.frame_sequence = 0
        self.output_folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.recording_started_at = None
        self.recording_stopped_at = None
        self.last_target_left_norm = 0.0
        self.last_target_right_norm = 0.0

    def start_recording_if_needed(self, left_speed: int, right_speed: int) -> bool:
        if self.state == RecorderState.ARMED and (left_speed != 0 or right_speed != 0):
            self.state = RecorderState.RECORDING
            self.last_save_time = 0.0
            self.recording_session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.frame_sequence = 0
            self.recording_started_at = time.time()
            self.recording_stopped_at = None
            return True
        return False

    def stop(self) -> None:
        if self.state == RecorderState.RECORDING and self.recording_started_at is not None:
            self.recording_stopped_at = time.time()
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        self.state = RecorderState.IDLE

    def get_output_folder_name(self) -> str:
        if not self.output_folder_name:
            self.output_folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.output_folder_name

    def get_recording_duration_sec(self) -> float:
        if self.recording_started_at is None:
            return 0.0
        end = self.recording_stopped_at if self.recording_stopped_at is not None else time.time()
        return max(0.0, end - self.recording_started_at)

    def has_temp_data(self) -> bool:
        return os.path.exists(self.temp_csv_path)

    def list_temp_images(self) -> List[str]:
        if not os.path.exists(self.temp_img_dir):
            return []
        images = [
            name
            for name in os.listdir(self.temp_img_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        images.sort()
        return images

    def record_frame(
        self,
        image_rgb: np.ndarray,
        left_speed: int,
        right_speed: int,
        line_features: Dict[str, float],
        drive_type: str,
        drive_speed_base: int,
        control_mode: str,
        interval_sec: float = 0.1,
    ) -> None:
        if self.state != RecorderState.RECORDING or self.csv_writer is None:
            return

        current_time = time.time()
        if current_time - self.last_save_time < interval_sec:
            return

        if not self.recording_session_id:
            self.recording_session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"{self.recording_session_id}_{self.frame_sequence:06d}.jpg"
        filepath = os.path.join(self.temp_img_dir, filename)

        bgr_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr_img)

        target_left_norm = float(np.clip(left_speed / 100.0, -1.0, 1.0))
        target_right_norm = float(np.clip(right_speed / 100.0, -1.0, 1.0))
        current_left_norm = float(self.last_target_left_norm)
        current_right_norm = float(self.last_target_right_norm)
        base_speed_norm = float(np.clip(drive_speed_base / 100.0, 0.0, 1.0))

        line_detect_top = float(line_features.get("line_detect_top", 0.0))
        line_detect_mid = float(line_features.get("line_detect_mid", 0.0))
        line_detect_bottom = float(line_features.get("line_detect_bottom", 0.0))
        line_offset_top = float(np.clip(line_features.get("line_offset_top", 0.0), -1.0, 1.0))
        line_offset_mid = float(np.clip(line_features.get("line_offset_mid", 0.0), -1.0, 1.0))
        line_offset_bottom = float(np.clip(line_features.get("line_offset_bottom", 0.0), -1.0, 1.0))

        self.csv_writer.writerow([
            f"image/{filename}",
            left_speed,
            right_speed,
            drive_type,
            drive_speed_base,
            control_mode,
            line_detect_top,
            line_detect_mid,
            line_detect_bottom,
            line_offset_top,
            line_offset_mid,
            line_offset_bottom,
            current_left_norm,
            current_right_norm,
            base_speed_norm,
            target_left_norm,
            target_right_norm,
        ])
        self.frame_sequence += 1
        self.last_save_time = current_time
        self.last_target_left_norm = target_left_norm
        self.last_target_right_norm = target_right_norm

    def commit_data(
        self,
        base_dir: str,
        drive_type: str,
        drive_speed_base: int,
        control_mode: str,
        camera_id: str,
        recording_duration_sec: float,
    ) -> None:
        final_img_dir = os.path.join(base_dir, "image")
        final_csv_path = os.path.join(base_dir, "driving_log.csv")

        os.makedirs(final_img_dir, exist_ok=True)

        moved_count = 0
        for img_file in os.listdir(self.temp_img_dir):
            src = os.path.join(self.temp_img_dir, img_file)
            dst = os.path.join(final_img_dir, img_file)
            shutil.move(src, dst)
            moved_count += 1

        file_exists = os.path.isfile(final_csv_path)
        with open(final_csv_path, mode="a", newline="") as f_out:
            writer = csv.writer(f_out)
            if not file_exists:
                writer.writerow(self.CSV_HEADER)

            with open(self.temp_csv_path, mode="r", newline="") as f_in:
                reader = csv.reader(f_in)
                next(reader)
                for row in reader:
                    if row and isinstance(row[0], str) and row[0].startswith("images/"):
                        row[0] = row[0].replace("images/", "image/", 1)
                    if len(row) == 3:
                        row.append(drive_type)
                        row.append(drive_speed_base)
                        row.append(control_mode)
                    elif len(row) == 4:
                        row.append(drive_speed_base)
                        row.append(control_mode)
                    elif len(row) == 5:
                        row.append(control_mode)
                    writer.writerow(row)

        summary_path = os.path.join(base_dir, "recording_summary.txt")
        with open(summary_path, mode="w", encoding="utf-8") as f:
            f.write(f"image_count={moved_count}\n")
            f.write(f"drive_type={drive_type}\n")
            f.write(f"camera_id={camera_id}\n")
            f.write(f"drive_speed_base={drive_speed_base}\n")
            f.write(f"recording_duration_sec={recording_duration_sec:.3f}\n")

        shutil.rmtree(self.temp_dir)

    def discard_data(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def close_and_discard_on_exit(self) -> None:
        if self.state in (RecorderState.ARMED, RecorderState.RECORDING):
            self.stop()
            self.discard_data()
