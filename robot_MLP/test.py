import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import DrivingLogSequenceDataset, LineTraceMLP, resolve_csv_paths


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
    model: LineTraceMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            preds.append(pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())

    if len(preds) == 0:
        raise ValueError("No samples were evaluated. Check CSV content.")

    pred_arr = np.concatenate(preds, axis=0).astype(np.float32)
    target_arr = np.concatenate(targets, axis=0).astype(np.float32)

    err = pred_arr - target_arr
    abs_err = np.abs(err)
    sq_err = err ** 2

    mse_left = float(np.mean(sq_err[:, 0]))
    mse_right = float(np.mean(sq_err[:, 1]))
    mse = float(np.mean(sq_err))

    mae_left = float(np.mean(abs_err[:, 0]))
    mae_right = float(np.mean(abs_err[:, 1]))
    mae = float(np.mean(abs_err))

    rmse_left = float(np.sqrt(mse_left))
    rmse_right = float(np.sqrt(mse_right))
    rmse = float(np.sqrt(mse))

    metrics = {
        "num_samples": int(target_arr.shape[0]),
        "mse": mse,
        "mse_left": mse_left,
        "mse_right": mse_right,
        "mae": mae,
        "mae_left": mae_left,
        "mae_right": mae_right,
        "rmse": rmse,
        "rmse_left": rmse_left,
        "rmse_right": rmse_right,
    }
    return metrics, pred_arr, target_arr


def save_predictions_csv(path: str, preds: np.ndarray, targets: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "pred_left",
            "pred_right",
            "target_left",
            "target_right",
            "abs_err_left",
            "abs_err_right",
        ])

        abs_err = np.abs(preds - targets)
        for i in range(preds.shape[0]):
            writer.writerow([
                i,
                float(preds[i, 0]),
                float(preds[i, 1]),
                float(targets[i, 0]),
                float(targets[i, 1]),
                float(abs_err[i, 0]),
                float(abs_err[i, 1]),
            ])

    print(f"Saved prediction details: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained MLP model on driving_log CSV.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="Path to one test driving_log.csv file OR a directory containing multiple session folders",
    )
    parser.add_argument("--weights-path", type=str, required=True, help="Path to trained model .pt (state_dict)")
    parser.add_argument("--split-report-path", type=str, default="", help="Optional dataset split JSON from train.py")
    parser.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate when split report is provided")
    parser.add_argument("--history", type=int, default=10, help="Frame history length used during training")
    parser.add_argument("--hidden1", type=int, default=64, help="Hidden layer size 1 (must match training)")
    parser.add_argument("--hidden2", type=int, default=32, help="Hidden layer size 2 (must match training)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help='e.g. "cpu" or "cuda". Empty means auto.')
    parser.add_argument("--pred-csv-path", type=str, default="", help="Optional path to save per-sample predictions")
    return parser


def main(args: argparse.Namespace) -> None:
    if args.split_report_path:
        csv_paths = load_csv_paths_from_split_report(args.split_report_path, args.split_name)
        print(f"Using split report: split={args.split_name}, csvs={len(csv_paths)}")
    else:
        if not args.csv_path:
            raise ValueError("Either --csv-path or --split-report-path is required")
        csv_paths = resolve_csv_paths(args.csv_path)
    dataset = DrivingLogSequenceDataset(csv_paths=csv_paths, history=args.history)
    print(f"Loaded {len(csv_paths)} csv file(s), total rows={len(dataset)}")
    input_dim = args.history * 9

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = LineTraceMLP(input_dim=input_dim, hidden1=args.hidden1, hidden2=args.hidden2, output_dim=2)
    state_dict = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

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
