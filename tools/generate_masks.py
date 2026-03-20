#!/usr/bin/env python3
"""
generate_masks.py

走行ログの画像に対して、現在のルールベース検出で二値マスクを自動生成する。
アノテーション GUI (annotate_masks.py) の下書きとして使用する。

使用方法:
  # 特定セッションを処理
  python tools/generate_masks.py dataset/camera_1/straight/20260316_133810_941565

  # dataset/ 以下の全セッションを一括処理
  python tools/generate_masks.py --all dataset/

  # 既存マスクを上書きしたい場合
  python tools/generate_masks.py --all dataset/ --overwrite

出力:
  各セッションディレクトリ内に mask/ フォルダを作成し、
  image/*.jpg に対応する mask/*.png (0/255 グレースケール) を保存する。
"""

import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_ROOT, "apps", "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from camera_receiver import CameraReceiver


def build_receiver(
    far_threshold: int = 100,
    near_threshold: int = 70,
    color_space: str = "lab",
) -> CameraReceiver:
    logger = logging.getLogger("gen_masks")
    receiver = CameraReceiver("", 0, logger)
    receiver.set_thresholds(far_threshold=far_threshold, near_threshold=near_threshold)
    receiver.set_line_color_space(color_space)
    # 自動閾値補正を有効にして、各フレームの照明に適応させる
    receiver.set_auto_threshold_enabled(True)
    return receiver


def generate_for_session(
    session_dir: str,
    receiver: CameraReceiver,
    overwrite: bool = False,
) -> int:
    image_dir = os.path.join(session_dir, "image")
    mask_dir = os.path.join(session_dir, "mask")

    if not os.path.isdir(image_dir):
        return 0

    os.makedirs(mask_dir, exist_ok=True)

    img_paths: list[str] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    img_paths.sort()

    generated = 0
    for img_path in img_paths:
        mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        if not overwrite and os.path.exists(mask_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        mask = receiver.compute_line_mask(frame)
        cv2.imwrite(mask_path, mask)
        generated += 1

    return generated


def find_sessions(dataset_dir: str) -> list[str]:
    """image/ サブフォルダを持つディレクトリをセッションとして列挙する。"""
    sessions = []
    for root, dirs, _ in os.walk(dataset_dir):
        if "image" in dirs:
            sessions.append(root)
    return sorted(sessions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="走行ログ画像からセグメンテーションマスクを自動生成する"
    )
    parser.add_argument(
        "target",
        help="セッションディレクトリ、または --all 使用時はデータセットルート",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="target 以下の全セッションをまとめて処理する",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存マスクを上書きする（デフォルトはスキップ）",
    )
    parser.add_argument("--far-threshold", type=int, default=100, metavar="N")
    parser.add_argument("--near-threshold", type=int, default=70, metavar="N")
    parser.add_argument(
        "--color-space",
        default="lab",
        choices=["lab", "hsv"],
        help="輝度チャンネルの色空間 (default: lab)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    receiver = build_receiver(
        far_threshold=args.far_threshold,
        near_threshold=args.near_threshold,
        color_space=args.color_space,
    )

    if args.all:
        sessions = find_sessions(args.target)
        if not sessions:
            print(f"セッションが見つかりません: {args.target}")
            return
        print(f"{len(sessions)} セッションを処理します...")
        total = 0
        for session in sessions:
            n = generate_for_session(session, receiver, overwrite=args.overwrite)
            rel = os.path.relpath(session, args.target)
            print(f"  {n:4d} 枚生成: {rel}")
            total += n
        print(f"\n合計 {total} 枚のマスクを生成しました。")
    else:
        n = generate_for_session(args.target, receiver, overwrite=args.overwrite)
        print(f"{n} 枚のマスクを生成しました: {args.target}")


if __name__ == "__main__":
    main()
