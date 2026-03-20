from dataclasses import dataclass
from typing import Set, Tuple
import math


@dataclass(frozen=True)
class DriveParams:
    left_sign: int
    right_sign: int
    pivot_y_threshold: float
    pivot_turn_gain: float
    reverse_speed_scale: float
    reverse_straight_x_threshold: float
    outer_speed_scale: float
    inner_speed_scale: float


def clamp_speed(value: int) -> int:
    return max(-100, min(100, value))


def compute_joystick_command(
    dx: float,
    dy: float,
    radius: float,
    deadzone: float,
    drive_speed: int,
    params: DriveParams,
) -> Tuple[int, int]:
    distance = math.sqrt(dx * dx + dy * dy)

    if distance > radius:
        dx = dx * radius / distance
        dy = dy * radius / distance
        distance = radius

    if distance < deadzone:
        return 0, 0

    dir_mag = math.sqrt(dx * dx + dy * dy)
    dir_x = dx / max(dir_mag, 1e-6)
    dir_y = dy / max(dir_mag, 1e-6)

    base_speed = int((-dir_y) * drive_speed)
    if base_speed < 0 and abs(dir_x) < params.reverse_straight_x_threshold:
        base_speed = int(base_speed * params.reverse_speed_scale)

    if abs(dir_y) < params.pivot_y_threshold:
        turn = int((dir_x * abs(dir_x)) * drive_speed * params.pivot_turn_gain)
        left_speed = turn
        right_speed = -turn
    else:
        angle_scale = abs(dir_y)
        outer = int(base_speed * params.outer_speed_scale)
        inner = int(base_speed * angle_scale * params.inner_speed_scale)

        if dir_x > 0:
            left_speed = outer
            right_speed = inner
        else:
            left_speed = inner
            right_speed = outer

    left_speed *= params.left_sign
    right_speed *= params.right_sign

    return clamp_speed(left_speed), clamp_speed(right_speed)


def compute_dpad_command(
    active_dirs: Set[str],
    dpad_drive_speed: int,
    dpad_turn_speed: int,
    reverse_speed_scale: float,
) -> Tuple[int, int]:
    up = "up" in active_dirs
    down = "down" in active_dirs
    left = "left" in active_dirs
    right = "right" in active_dirs

    if up and not down:
        base = dpad_drive_speed
    elif down and not up:
        base = -int(dpad_drive_speed * reverse_speed_scale)
    else:
        base = 0

    if right and not left:
        turn = dpad_turn_speed
    elif left and not right:
        turn = -dpad_turn_speed
    else:
        turn = 0

    if base == 0 and turn == 0:
        left_speed, right_speed = 0, 0
    elif base == 0:
        left_speed, right_speed = turn, -turn
    else:
        left_speed = base + turn
        right_speed = base - turn

    return clamp_speed(left_speed), clamp_speed(right_speed)
