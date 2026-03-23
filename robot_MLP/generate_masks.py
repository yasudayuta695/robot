"""
generate_masks.py — dataset の image/ から model_seg.onnx でマスク画像を一括生成する

各セッションの image/ と並列に mask/ フォルダを作成し、
画像と同じステム名の PNG ファイルを保存する。

Usage:
    python generate_masks.py \\
        --dataset ../apps/dataset/camera_4 \\
        --seg-model ../apps/shared/model_seg.onnx

    # 既存マスクも上書き
    python generate_masks.py \\
        --dataset ../apps/dataset \\
        --seg-model ../apps/shared/model_seg.onnx \\
        --overwrite
"""
import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np


SEG_INPUT_W = 160
SEG_INPUT_H = 120
SEG_THRESHOLD = 0.50
ROI_TOP_RATIO = 0.25  # 上部 25% はゼロマスク（find_line() と同じ）

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def build_seg_net(model_path: str) -> cv2.dnn.Net:
    if not os.path.isfile(model_path):
        print(f"[Error] Seg model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    net = cv2.dnn.readNetFromONNX(model_path)
    return net


def run_seg(net: cv2.dnn.Net, bgr: np.ndarray, threshold: float = SEG_THRESHOLD) -> np.ndarray:
    """
    BGR 画像からフルフレームマスク (uint8, 0/255) を生成する。
    上部 ROI_TOP_RATIO はゼロに固定（camera_receiver.find_line() と同じ挙動）。
    """
    h, w = bgr.shape[:2]
    roi_top = int(h * ROI_TOP_RATIO)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inp = cv2.resize(gray, (SEG_INPUT_W, SEG_INPUT_H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    blob = inp[np.newaxis, np.newaxis, :, :]

    net.setInput(blob)
    out = net.forward()

    if out.ndim == 4:
        prob = out[0, 0]
    elif out.ndim == 3:
        prob = out[0]
    else:
        raise RuntimeError(f"Unexpected output shape: {out.shape}")

    seg_small = (prob >= threshold).astype(np.uint8) * 255
    full_mask = cv2.resize(seg_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # ROI 外（上部）をゼロにする
    full_mask[:roi_top, :] = 0

    return full_mask


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def find_image_dirs(root: str) -> List[str]:
    """
    root 以下を再帰的に走査し、image/ フォルダのパスを収集する。
    """
    found: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        if "image" in dirnames:
            found.append(os.path.join(dirpath, "image"))
        # image/ と mask/ の内部には再帰しない
        dirnames[:] = [d for d in dirnames if d not in ("image", "mask")]
    found.sort()
    return found


def list_images(image_dir: str) -> List[str]:
    names = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    names.sort()
    return names


def mask_path_for(image_dir: str, image_name: str) -> Tuple[str, str]:
    """mask/ フォルダパスとマスクファイルパスを返す。"""
    session_dir = os.path.dirname(image_dir)
    mask_dir = os.path.join(session_dir, "mask")
    stem = os.path.splitext(image_name)[0]
    mask_file = os.path.join(mask_dir, stem + ".png")
    return mask_dir, mask_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> None:
    dataset_root = os.path.abspath(os.path.expanduser(args.dataset))
    seg_model_path = os.path.abspath(os.path.expanduser(args.seg_model))

    if not os.path.isdir(dataset_root):
        print(f"[Error] Dataset directory not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset : {dataset_root}")
    print(f"Seg model: {seg_model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Overwrite: {args.overwrite}")

    net = build_seg_net(seg_model_path)

    image_dirs = find_image_dirs(dataset_root)
    if not image_dirs:
        print("[Error] No image/ directories found under dataset root.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_dirs)} image/ folder(s).\n")

    total_saved = 0
    total_skipped = 0
    total_error = 0

    for image_dir in image_dirs:
        session_dir = os.path.dirname(image_dir)
        rel_session = os.path.relpath(session_dir, dataset_root)
        images = list_images(image_dir)

        if not images:
            print(f"  [Skip] {rel_session}/image/  (no images)")
            continue

        saved = 0
        skipped = 0
        errors = 0

        for img_name in images:
            mask_dir, mask_file = mask_path_for(image_dir, img_name)

            if not args.overwrite and os.path.isfile(mask_file):
                skipped += 1
                continue

            img_path = os.path.join(image_dir, img_name)
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"  [Warn] Cannot read: {img_path}")
                errors += 1
                continue

            try:
                mask = run_seg(net, bgr, threshold=args.threshold)
            except Exception as exc:
                print(f"  [Warn] Seg failed for {img_name}: {exc}")
                errors += 1
                continue

            os.makedirs(mask_dir, exist_ok=True)
            cv2.imwrite(mask_file, mask)
            saved += 1

        total_saved += saved
        total_skipped += skipped
        total_error += errors

        status = f"saved={saved}"
        if skipped:
            status += f"  skipped={skipped}"
        if errors:
            status += f"  errors={errors}"
        print(f"  {rel_session}/image/  ({len(images)} imgs)  → {status}")

    print(f"\nDone.  total saved={total_saved}  skipped={total_skipped}  errors={total_error}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate seg mask images from dataset image/ folders using model_seg.onnx"
    )
    p.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset root directory (e.g. ../apps/dataset/camera_4 or ../apps/dataset)",
    )
    p.add_argument(
        "--seg-model", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "apps", "shared", "model_seg.onnx"),
        help="Path to model_seg.onnx (default: ../apps/shared/model_seg.onnx)",
    )
    p.add_argument(
        "--threshold", type=float, default=SEG_THRESHOLD,
        help=f"Segmentation threshold (default: {SEG_THRESHOLD})",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing mask files (default: skip)",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate(args)
