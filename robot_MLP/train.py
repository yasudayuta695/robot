import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


FEATURE_COLUMNS = [
    "line_detect_top",
    "line_detect_mid",
    "line_detect_bottom",
    "line_offset_top",
    "line_offset_mid",
    "line_offset_bottom",
    "current_left_norm",
    "current_right_norm",
    "base_speed_norm",
]

TARGET_COLUMNS = ["target_left_norm", "target_right_norm"]


@dataclass
class Sample:
    feature: np.ndarray  # shape: (9,)
    target: np.ndarray   # shape: (2,)


def _clip_feature(name: str, value: float) -> float:
    if name.startswith("line_detect"):
        return float(np.clip(value, 0.0, 1.0))
    if name.startswith("line_offset"):
        return float(np.clip(value, -1.0, 1.0))
    if name in ("current_left_norm", "current_right_norm"):
        return float(np.clip(value, -1.0, 1.0))
    if name == "base_speed_norm":
        return float(np.clip(value, 0.0, 1.0))
    return float(value)


def _clip_target(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


class DrivingLogSequenceDataset(Dataset):
    """
    One sample = concatenated 10-frame input (90,) ending at current index
    plus current target (2,).

    For early indices with missing history, this uses edge padding by
    replicating frame 0 (no zero padding).
    """

    def __init__(self, csv_path: str, history: int = 10) -> None:
        if history <= 0:
            raise ValueError("history must be >= 1")
        self.history = history
        self.samples: List[Sample] = self._load_csv(csv_path)
        if len(self.samples) == 0:
            raise ValueError(f"No rows found in csv: {csv_path}")

    def _load_csv(self, csv_path: str) -> List[Sample]:
        loaded: List[Sample] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            missing_cols = [c for c in (FEATURE_COLUMNS + TARGET_COLUMNS) if c not in (reader.fieldnames or [])]
            if missing_cols:
                raise ValueError(f"CSV is missing required columns: {missing_cols}")

            for row_idx, row in enumerate(reader):
                try:
                    feature_vals = [_clip_feature(c, float(row[c])) for c in FEATURE_COLUMNS]
                    target_vals = [_clip_target(float(row[c])) for c in TARGET_COLUMNS]
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid numeric value at row {row_idx + 2}: {e}") from e

                loaded.append(
                    Sample(
                        feature=np.asarray(feature_vals, dtype=np.float32),
                        target=np.asarray(target_vals, dtype=np.float32),
                    )
                )
        return loaded

    def __len__(self) -> int:
        return len(self.samples)

    def _window_indices(self, index: int) -> List[int]:
        start = index - (self.history - 1)
        idxs: List[int] = []
        for i in range(start, index + 1):
            idxs.append(0 if i < 0 else i)
        return idxs

    def __getitem__(self, index: int):
        if isinstance(index, torch.Tensor):
            index = int(index.item())

        idxs = self._window_indices(index)
        stacked = np.concatenate([self.samples[i].feature for i in idxs], axis=0)  # (history*9,)
        target = self.samples[index].target  # (2,)

        x = torch.from_numpy(stacked)
        y = torch.from_numpy(target)
        return x, y


class LineTraceMLP(nn.Module):
    def __init__(self, input_dim: int = 90, hidden1: int = 64, hidden2: int = 32, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_indices_sequential(n_total: int, val_ratio: float) -> tuple[Sequence[int], Sequence[int]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples for training after split")

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, n_total))
    return train_indices, val_indices


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz

    if total_count == 0:
        return float("nan")
    return total_loss / total_count


def save_learning_curve(train_losses: Sequence[float], val_losses: Sequence[float], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn] matplotlib is not installed. Skipping learning curve plot.")
        return

    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int32)
    train_arr = np.asarray(train_losses, dtype=np.float32)
    val_arr = np.asarray(val_losses, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_arr, label="train_loss", linewidth=2)

    # Plot validation loss only when at least one finite value exists.
    if val_arr.size == train_arr.size:
        valid_mask = np.isfinite(val_arr)
        if np.any(valid_mask):
            ax.plot(epochs[valid_mask], val_arr[valid_mask], label="val_loss", linewidth=2)

    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved learning curve: {out_path}")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DrivingLogSequenceDataset(csv_path=args.csv_path, history=args.history)
    input_dim = args.history * len(FEATURE_COLUMNS)

    if input_dim != 90:
        print(f"[Info] input_dim={input_dim}. Requirement is 90 when history=10.")

    train_idx, val_idx = split_indices_sequential(len(dataset), args.val_ratio)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LineTraceMLP(input_dim=input_dim, hidden1=args.hidden1, hidden2=args.hidden2, output_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: List[float] = []
    val_losses: List[float] = []
    use_validation = len(val_set) > 0
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
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            running_loss += loss.item() * bsz
            seen += bsz

        train_loss = running_loss / max(1, seen)
        train_losses.append(train_loss)

        if use_validation:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            improved = val_loss < (best_val_loss - args.min_delta)
            if improved:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
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
        output_names=["motor_output"],
        dynamic_axes={"input": {0: "batch"}, "motor_output": {0: "batch"}},
    )
    try:
        # Prefer legacy exporter to avoid hard dependency on onnxscript.
        torch.onnx.export(model_cpu, dummy, args.onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions do not support `dynamo` argument.
        torch.onnx.export(model_cpu, dummy, args.onnx_path, **export_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name in ("onnx", "onnxscript"):
            raise ModuleNotFoundError(
                "ONNX export requires optional packages. Run: pip install onnx onnxscript"
            ) from exc
        raise
    except torch.onnx.OnnxExporterError as exc:
        if "onnx" in str(exc).lower():
            raise ModuleNotFoundError(
                "ONNX export requires optional packages. Run: pip install onnx onnxscript"
            ) from exc
        raise
    print(f"Exported ONNX model: {args.onnx_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MLP for line-trace motor output regression and export ONNX.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to driving_log.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=10, help="Number of frames to stack (10 -> input 90)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help='e.g. "cpu" or "cuda". Empty means auto.')
    parser.add_argument("--weights-path", type=str, default="model.pt")
    parser.add_argument("--onnx-path", type=str, default="model.onnx")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping with validation loss")
    parser.add_argument("--patience", type=int, default=8, help="Epochs to wait for val improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val-loss improvement to reset patience")
    parser.add_argument("--no-restore-best", action="store_true", help="Do not restore best validation checkpoint")
    parser.add_argument("--curve-path", type=str, default="learning_curve.png", help="Output path for learning curve image")
    parser.add_argument("--no-curve", action="store_true", help="Disable saving learning curve image")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
