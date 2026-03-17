import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import DrivingLogSequenceDataset, LineTraceMLP, PIDGainMLP, motor_from_gains, resolve_csv_paths


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
    target_mode: str,
    history: int,
    steer_limit: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    gains_all: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if target_mode == "pid-gain":
                gains = model(x)
                pred = motor_from_gains(gains, x=x, history=history, steer_limit=steer_limit)
                gains_all.append(gains.detach().cpu().numpy())
            else:
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
    if len(gains_all) > 0:
        gain_arr = np.concatenate(gains_all, axis=0).astype(np.float32)
        metrics.update(
            {
                "kp_mean": float(np.mean(gain_arr[:, 0])),
                "ki_mean": float(np.mean(gain_arr[:, 1])),
                "kd_mean": float(np.mean(gain_arr[:, 2])),
            }
        )
    else:
        gain_arr = np.empty((0, 3), dtype=np.float32)
    return metrics, pred_arr, target_arr, gain_arr


def save_predictions_csv(path: str, preds: np.ndarray, targets: np.ndarray, gains: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "index",
            "pred_left",
            "pred_right",
            "target_left",
            "target_right",
            "abs_err_left",
            "abs_err_right",
        ]
        has_gain = gains.size > 0 and gains.shape[0] == preds.shape[0]
        if has_gain:
            header.extend(["kp", "ki", "kd"])
        writer.writerow(header)

        abs_err = np.abs(preds - targets)
        for i in range(preds.shape[0]):
            row = [
                i,
                float(preds[i, 0]),
                float(preds[i, 1]),
                float(targets[i, 0]),
                float(targets[i, 1]),
                float(abs_err[i, 0]),
                float(abs_err[i, 1]),
            ]
            if has_gain:
                row.extend([float(gains[i, 0]), float(gains[i, 1]), float(gains[i, 2])])
            writer.writerow(row)

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
    parser.add_argument("--onnx-path", type=str, default="", help="Optional ONNX path (accepted for workflow compatibility; not used in this script)")
    parser.add_argument("--split-report-path", type=str, default="", help="Optional dataset split JSON from train.py")
    parser.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate when split report is provided")
    parser.add_argument("--history", type=int, default=10, help="Frame history length used during training")
    parser.add_argument("--hidden1", type=int, default=64, help="Hidden layer size 1 (must match training)")
    parser.add_argument("--hidden2", type=int, default=32, help="Hidden layer size 2 (must match training)")
    parser.add_argument("--target-mode", type=str, default="pid-gain", choices=["motor", "pid-gain"], help="Evaluation mode matching train.py")
    parser.add_argument("--kp-max", type=float, default=2.5, help="Must match train.py when target-mode=pid-gain")
    parser.add_argument("--ki-max", type=float, default=1.0, help="Must match train.py when target-mode=pid-gain")
    parser.add_argument("--kd-max", type=float, default=1.5, help="Must match train.py when target-mode=pid-gain")
    parser.add_argument("--steer-limit", type=float, default=1.0, help="Steering clamp used for pid-gain evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help='e.g. "cpu" or "cuda". Empty means auto.')
    parser.add_argument("--pred-csv-path", type=str, default="", help="Optional path to save per-sample predictions")
    return parser


def main(args: argparse.Namespace) -> None:
    if args.onnx_path:
        print(f"[Info] --onnx-path is provided but ignored by test.py (PyTorch .pt evaluation): {args.onnx_path}")

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

    if args.target_mode == "pid-gain":
        model = PIDGainMLP(
            input_dim=input_dim,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            kp_max=args.kp_max,
            ki_max=args.ki_max,
            kd_max=args.kd_max,
        )
    else:
        model = LineTraceMLP(input_dim=input_dim, hidden1=args.hidden1, hidden2=args.hidden2, output_dim=2)
    state_dict = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    metrics, preds, targets, gains = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        target_mode=args.target_mode,
        history=args.history,
        steer_limit=args.steer_limit,
    )

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
    if args.target_mode == "pid-gain":
        print(f"Kp(mean)   : {metrics['kp_mean']:.6f}")
        print(f"Ki(mean)   : {metrics['ki_mean']:.6f}")
        print(f"Kd(mean)   : {metrics['kd_mean']:.6f}")

    if args.pred_csv_path:
        save_predictions_csv(args.pred_csv_path, preds, targets, gains)


if __name__ == "__main__":
    main(build_parser().parse_args())
