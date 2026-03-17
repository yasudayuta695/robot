import argparse
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from train import (
    FEATURE_COLUMNS,
    DrivingLogSequenceDataset,
    resolve_csv_paths,
    save_learning_curve,
    save_split_report,
    split_csv_paths_by_session,
    split_indices_sequential,
)


def pid_terms_from_input(x: torch.Tensor, history: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build PID terms from flattened sequence input.

    x shape: (batch, history*9)
    returns: error, integral, derivative, base_speed_norm (all shape: (batch,))
    """
    if history <= 0:
        raise ValueError("history must be >= 1")

    frame_dim = len(FEATURE_COLUMNS)
    seq = x.view(x.size(0), history, frame_dim)

    detect_mid = torch.clamp(seq[:, :, 1], 0.0, 1.0)
    offset_mid = torch.clamp(seq[:, :, 4], -1.0, 1.0)
    effective_error = detect_mid * offset_mid

    error = effective_error[:, -1]
    integral = torch.mean(effective_error, dim=1)
    if history >= 2:
        derivative = effective_error[:, -1] - effective_error[:, -2]
    else:
        derivative = torch.zeros_like(error)

    base_speed_norm = torch.clamp(seq[:, -1, 8], 0.0, 1.0)
    return error, integral, derivative, base_speed_norm


def motor_from_gains(
    gains: torch.Tensor,
    x: torch.Tensor,
    history: int,
    steer_limit: float,
) -> torch.Tensor:
    """
    Convert learned gains into motor outputs using PID formula.

    gains shape: (batch, 3) -> [kp, ki, kd]
    returns: (batch, 2) -> [left_norm, right_norm]
    """
    if gains.size(1) != 3:
        raise ValueError(f"Expected gains with 3 channels, got {gains.shape}")

    error, integral, derivative, base_speed_norm = pid_terms_from_input(x, history=history)

    kp = gains[:, 0]
    ki = gains[:, 1]
    kd = gains[:, 2]

    steer = (kp * error) + (ki * integral) + (kd * derivative)
    steer = torch.clamp(steer, -abs(float(steer_limit)), abs(float(steer_limit)))

    left = torch.clamp(base_speed_norm + steer, -1.0, 1.0)
    right = torch.clamp(base_speed_norm - steer, -1.0, 1.0)
    return torch.stack([left, right], dim=1)


class PIDGainMLP(nn.Module):
    """Predict bounded PID gains [kp, ki, kd] from flattened sequence input."""

    def __init__(
        self,
        input_dim: int = 90,
        hidden1: int = 64,
        hidden2: int = 32,
        kp_max: float = 2.5,
        ki_max: float = 1.0,
        kd_max: float = 1.5,
    ) -> None:
        super().__init__()
        self.kp_max = float(max(1e-6, kp_max))
        self.ki_max = float(max(1e-6, ki_max))
        self.kd_max = float(max(1e-6, kd_max))

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        # Avoid index-based gather ops to keep ONNX compatible with OpenCV DNN.
        scales = raw.new_tensor([self.kp_max, self.ki_max, self.kd_max])
        return torch.sigmoid(raw) * scales


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    history: int,
    steer_limit: float,
    criterion: nn.Module,
    gain_l2: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            gains = model(x)
            pred_motor = motor_from_gains(gains, x=x, history=history, steer_limit=steer_limit)
            loss = criterion(pred_motor, y)
            if gain_l2 > 0.0:
                loss = loss + (gain_l2 * torch.mean(gains * gains))

            bsz = x.size(0)
            total_loss += float(loss.item()) * bsz
            total_count += bsz

    if total_count == 0:
        return float("nan")
    return total_loss / total_count


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_paths = resolve_csv_paths(args.csv_path)
    is_directory_input = os.path.isdir(os.path.abspath(os.path.expanduser(args.csv_path)))

    if is_directory_input and len(csv_paths) > 1:
        train_csvs, val_csvs, test_csvs = split_csv_paths_by_session(
            csv_paths=csv_paths,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            holdout_unit=args.holdout_unit,
        )

        if len(train_csvs) == 0:
            raise ValueError("No training CSVs after split. Check dataset size and split ratios.")

        train_set = DrivingLogSequenceDataset(csv_paths=train_csvs, history=args.history)
        val_set = DrivingLogSequenceDataset(csv_paths=val_csvs, history=args.history) if len(val_csvs) > 0 else None
        print(
            f"Loaded {len(csv_paths)} csv file(s) from directory. "
            f"holdout_unit={args.holdout_unit} "
            f"train_csvs={len(train_csvs)} val_csvs={len(val_csvs)} test_csvs={len(test_csvs)}"
        )

        if args.split_report_path:
            save_split_report(args.split_report_path, train_csvs, val_csvs, test_csvs)
    else:
        dataset = DrivingLogSequenceDataset(csv_paths=csv_paths, history=args.history)
        print(f"Loaded {len(csv_paths)} csv file(s), total rows={len(dataset)}")

        train_idx, val_idx = split_indices_sequential(len(dataset), args.val_ratio)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

    input_dim = args.history * len(FEATURE_COLUMNS)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PIDGainMLP(
        input_dim=input_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        kp_max=args.kp_max,
        ki_max=args.ki_max,
        kd_max=args.kd_max,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: List[float] = []
    val_losses: List[float] = []
    use_validation = val_loader is not None
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    if args.early_stopping and not use_validation:
        print("[Warn] Early stopping requires validation data. Set --val-ratio > 0 to enable it.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            gains = model(x)
            pred_motor = motor_from_gains(gains, x=x, history=args.history, steer_limit=args.steer_limit)
            loss = criterion(pred_motor, y)
            if args.gain_l2 > 0.0:
                loss = loss + (args.gain_l2 * torch.mean(gains * gains))
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            running_loss += float(loss.item()) * bsz
            seen += bsz

        train_loss = running_loss / max(1, seen)
        train_losses.append(train_loss)

        if use_validation:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                history=args.history,
                steer_limit=args.steer_limit,
                criterion=criterion,
                gain_l2=args.gain_l2,
            )
            val_losses.append(val_loss)

            improved = val_loss < (best_val_loss - args.min_delta)
            if improved:
                best_val_loss = val_loss
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"no_improve={no_improve_count}/{args.patience}"
            )

            if args.early_stopping and no_improve_count >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        else:
            val_losses.append(float("nan"))
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] train_loss={train_loss:.6f}")

    if use_validation and best_state_dict is not None and not args.no_restore_best:
        model.load_state_dict(best_state_dict)
        print(f"Restored best validation model (best_val_loss={best_val_loss:.6f}).")

    torch.save(model.state_dict(), args.weights_path)
    print(f"Saved PyTorch weights: {args.weights_path}")

    if not args.no_curve:
        save_learning_curve(train_losses, val_losses, args.curve_path)

    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, input_dim, dtype=torch.float32)
    export_kwargs = dict(
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["pid_gains"],
    )
    try:
        torch.onnx.export(model_cpu, dummy, args.onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy, args.onnx_path, **export_kwargs)
    print(f"Exported PID-gain ONNX model: {args.onnx_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MLP to learn PID gains (Kp, Ki, Kd) and export ONNX.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--holdout-unit", type=str, default="hour", choices=["session", "hour"])
    parser.add_argument("--split-report-path", type=str, default="dataset_split.json")
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--weights-path", type=str, default="pid_gain_model.pt")
    parser.add_argument("--onnx-path", type=str, default="pid_gain_model.onnx")
    parser.add_argument("--kp-max", type=float, default=2.5)
    parser.add_argument("--ki-max", type=float, default=1.0)
    parser.add_argument("--kd-max", type=float, default=1.5)
    parser.add_argument("--steer-limit", type=float, default=1.0)
    parser.add_argument("--gain-l2", type=float, default=1e-4)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-restore-best", action="store_true")
    parser.add_argument("--curve-path", type=str, default="pid_gain_learning_curve.png")
    parser.add_argument("--no-curve", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
