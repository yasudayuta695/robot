import csv
import datetime
import logging
import os
import shutil
import time
from enum import Enum
from typing import List

import cv2
import numpy as np


class RecorderState(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    RECORDING = "recording"


class DataRecorder:
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

    def arm(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_img_dir, exist_ok=True)

        self.csv_file = open(self.temp_csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "image_path",
            "left_speed",
            "right_speed",
            "drive_type",
            "drive_speed_base",
            "control_mode",
        ])

        self.state = RecorderState.ARMED
        self.last_save_time = 0.0

    def start_recording_if_needed(self, left_speed: int, right_speed: int) -> bool:
        if self.state == RecorderState.ARMED and (left_speed != 0 or right_speed != 0):
            self.state = RecorderState.RECORDING
            self.last_save_time = 0.0
            return True
        return False

    def stop(self) -> None:
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        self.state = RecorderState.IDLE

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

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(self.temp_img_dir, filename)

        bgr_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr_img)

        self.csv_writer.writerow([
            f"images/{filename}",
            left_speed,
            right_speed,
            drive_type,
            drive_speed_base,
            control_mode,
        ])
        self.last_save_time = current_time

    def commit_data(self, base_dir: str, drive_type: str, drive_speed_base: int, control_mode: str) -> None:
        final_img_dir = os.path.join(base_dir, "images")
        final_csv_path = os.path.join(base_dir, "driving_log.csv")

        os.makedirs(final_img_dir, exist_ok=True)

        for img_file in os.listdir(self.temp_img_dir):
            src = os.path.join(self.temp_img_dir, img_file)
            dst = os.path.join(final_img_dir, img_file)
            shutil.move(src, dst)

        file_exists = os.path.isfile(final_csv_path)
        with open(final_csv_path, mode="a", newline="") as f_out:
            writer = csv.writer(f_out)
            if not file_exists:
                writer.writerow([
                    "image_path",
                    "left_speed",
                    "right_speed",
                    "drive_type",
                    "drive_speed_base",
                    "control_mode",
                ])

            with open(self.temp_csv_path, mode="r", newline="") as f_in:
                reader = csv.reader(f_in)
                next(reader)
                for row in reader:
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

        shutil.rmtree(self.temp_dir)

    def discard_data(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def close_and_discard_on_exit(self) -> None:
        if self.state in (RecorderState.ARMED, RecorderState.RECORDING):
            self.stop()
            self.discard_data()
