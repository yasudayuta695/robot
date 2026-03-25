"""
End-to-End CNN evaluation script.

test.py の CNN 版。train_image.py で学習したモデルを
テストデータで評価し、MSE / MAE / RMSE を出力する。

Usage example:
    # split report を使う場合（推奨）
    python test_image.py \\
        --weights-path model_cnn.pt \\
        --split-report-path dataset_split_image.json \\
        --split-name test

    # CSV パスを直接指定する場合
    python test_image.py \\
        --weights-path model_cnn.pt \\
        --csv-path ../apps/dataset/camera_4

    # 予測結果を CSV に保存
    python test_image.py \\
        --weights-path model_cnn.pt \\
        --split-report-path dataset_split_image.json \\
        --pred-csv-path predictions_cnn.csv
"""
import argparse
import csv
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_image import ImageDrivingDataset, LightCNNWithScalars, resolve_csv_paths


def load_csv_paths_from_split_report(split_report_path: str, split_name: str) -> List[str]:
    with open(split_report_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if split_name not in payload:
        raise ValueError(f"split '{split_name}' is not found in split report: {split_report_path}")

    csv_paths = payload.get(split_name, [])
    if not isinstance(csv_paths, list):
        raise ValueError(f"split report field '{split_name}' must be a list")

    resolved: List[str] = []
    for p in csv_paths:
        p_abs = os.path.abspath(os.path.expanduser(str(p)))
        if not os.path.isfile(p_abs):
            raise ValueError(f"CSV from split report not found: {p_abs}")
        resolved.append(p_abs)

    if not resolved:
        raise ValueError(f"split '{split_name}' has no CSV paths in report: {split_report_path}")
    return resolved


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for img, scalars, target in loader:
            img = img.to(device)
            scalars = scalars.to(device)
            pred = model(img, scalars)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.numpy())

    if not preds:
        raise ValueError("No samples were evaluated. Check CSV content.")

    pred_arr = np.concatenate(preds, axis=0).astype(np.float32)
    target_arr = np.concatenate(targets, axis=0).astype(np.float32)

    err = pred_arr - target_arr
    abs_err = np.abs(err)
    sq_err = err ** 2

    metrics = {
        "num_samples": int(target_arr.shape[0]),
        "mse":         float(np.mean(sq_err)),
        "mse_left":    float(np.mean(sq_err[:, 0])),
        "mse_right":   float(np.mean(sq_err[:, 1])),
        "mae":         float(np.mean(abs_err)),
        "mae_left":    float(np.mean(abs_err[:, 0])),
        "mae_right":   float(np.mean(abs_err[:, 1])),
        "rmse":        float(np.sqrt(np.mean(sq_err))),
        "rmse_left":   float(np.sqrt(np.mean(sq_err[:, 0]))),
        "rmse_right":  float(np.sqrt(np.mean(sq_err[:, 1]))),
    }
    return metrics, pred_arr, target_arr


def save_predictions_csv(path: str, preds: np.ndarray, targets: np.ndarray) -> None:
    abs_err = np.abs(preds - targets)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "pred_left", "pred_right",
            "target_left", "target_right",
            "abs_err_left", "abs_err_right",
        ])
        for i in range(preds.shape[0]):
            writer.writerow([
                i,
                float(preds[i, 0]),   float(preds[i, 1]),
                float(targets[i, 0]), float(targets[i, 1]),
                float(abs_err[i, 0]), float(abs_err[i, 1]),
            ])
    print(f"Saved prediction details: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate trained CNN model (train_image.py) on driving_log CSV."
    )
    p.add_argument("--csv-path", type=str, default="",
                   help="driving_log.csv file or directory of session folders")
    p.add_argument("--weights-path", type=str, required=True,
                   help="Path to trained model .pt (state_dict)")
    p.add_argument("--split-report-path", type=str, default="",
                   help="dataset split JSON from train_image.py (dataset_split_image.json)")
    p.add_argument("--split-name", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate when split report is provided")
    p.add_argument("--img-h", type=int, default=60, help="Mask height (must match training)")
    p.add_argument("--img-w", type=int, default=80, help="Mask width  (must match training)")
    p.add_argument("--hidden", type=int, default=128, help="FC hidden units (must match training)")
    p.add_argument("--label-shift", type=int, default=0,
                   help="Label shift used during training (must match --label-shift in train_image.py)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="",
                   help='e.g. "cpu" or "cuda". Empty means auto.')
    p.add_argument("--pred-csv-path", type=str, default="",
                   help="Optional path to save per-sample predictions CSV")
    return p


def main(args: argparse.Namespace) -> None:
    # --- CSV 解決 ---
    if args.split_report_path:
        csv_paths = load_csv_paths_from_split_report(args.split_report_path, args.split_name)
        print(f"Using split report: split={args.split_name}, csvs={len(csv_paths)}")
    else:
        if not args.csv_path:
            raise ValueError("Either --csv-path or --split-report-path is required")
        csv_paths = resolve_csv_paths(args.csv_path)

    # --- Dataset ---
    dataset = ImageDrivingDataset(csv_paths, args.img_h, args.img_w, args.label_shift)
    print(f"Loaded {len(csv_paths)} csv file(s), total rows={len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # --- Model ---
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LightCNNWithScalars(img_h=args.img_h, img_w=args.img_w, hidden=args.hidden)
    state_dict = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  Model params: {n_params:,}")

    # --- Evaluate ---
    metrics, preds, targets = evaluate_model(model, loader, device)

    print("=== Test Metrics ===")
    print(f"samples    : {metrics['num_samples']}")
    print(f"MSE        : {metrics['mse']:.6f}")
    print(f"MSE(left)  : {metrics['mse_left']:.6f}")
    print(f"MSE(right) : {metrics['mse_right']:.6f}")
    print(f"MAE        : {metrics['mae']:.6f}")
    print(f"MAE(left)  : {metrics['mae_left']:.6f}")
    print(f"MAE(right) : {metrics['mae_right']:.6f}")
    print(f"RMSE       : {metrics['rmse']:.6f}")
    print(f"RMSE(left) : {metrics['rmse_left']:.6f}")
    print(f"RMSE(right): {metrics['rmse_right']:.6f}")

    if args.pred_csv_path:
        save_predictions_csv(args.pred_csv_path, preds, targets)


if __name__ == "__main__":
    main(build_parser().parse_args())
