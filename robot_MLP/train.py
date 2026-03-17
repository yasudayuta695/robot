import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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


def resolve_csv_paths(csv_path: str) -> List[str]:
    """
    Accept either:
    - one CSV file path
    - one directory path containing many session folders

    For directory input, recursively collect all driving_log.csv files.
    """
    if not csv_path:
        raise ValueError("csv_path is empty")

    raw_path = os.path.expanduser(csv_path)
    norm_path = os.path.abspath(raw_path)

    # Convenience fallback: when running from robot_MLP/, allow paths written
    # relative to the project root (e.g. ./dataset/camera_2).
    candidate_paths = [norm_path]
    if not os.path.isabs(raw_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        raw_rel = raw_path[2:] if raw_path.startswith("./") else raw_path
        candidate_paths.append(os.path.abspath(os.path.join(project_root, raw_rel)))

    resolved_path = ""
    for candidate in candidate_paths:
        if os.path.isfile(candidate) or os.path.isdir(candidate):
            resolved_path = candidate
            break

    if resolved_path:
        norm_path = resolved_path

    if os.path.isfile(norm_path):
        return [norm_path]

    if os.path.isdir(norm_path):
        collected: List[str] = []
        for root, _dirs, files in os.walk(norm_path):
            if "driving_log.csv" in files:
                collected.append(os.path.join(root, "driving_log.csv"))
        collected.sort()
        if not collected:
            raise ValueError(f"No driving_log.csv found under directory: {norm_path}")
        return collected

    msg = [f"csv path does not exist: {csv_path}"]
    msg.append("Checked paths:")
    msg.extend([f"- {p}" for p in dict.fromkeys(candidate_paths)])
    msg.append("Hint: from robot_MLP/, dataset is usually ../dataset/<camera_id>")
    raise ValueError("\n".join(msg))


def infer_session_and_type(csv_file: str) -> Tuple[str, str]:
    norm = os.path.normpath(csv_file)
    session_id = os.path.basename(os.path.dirname(norm))
    drive_type = os.path.basename(os.path.dirname(os.path.dirname(norm)))
    return session_id, drive_type


def infer_holdout_group_key(session_id: str, holdout_unit: str) -> str:
    # session format: YYYYMMDD_HHMMSS_microsec
    parts = session_id.split("_")
    if holdout_unit == "session":
        return session_id

    if holdout_unit == "hour":
        if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) >= 2:
            return f"{parts[0]}_{parts[1][:2]}"
        return session_id

    raise ValueError(f"Unsupported holdout_unit: {holdout_unit}")


def _allocate_counts_per_type(n_items: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        n_val = 1 if val_ratio > 0.0 else 0
        return 2 - n_val, n_val, 0

    n_val = int(round(n_items * max(0.0, val_ratio)))
    n_test = int(round(n_items * max(0.0, test_ratio)))

    if val_ratio > 0.0 and n_val == 0:
        n_val = 1
    if test_ratio > 0.0 and n_test == 0:
        n_test = 1

    while (n_val + n_test) > (n_items - 1):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    n_train = n_items - n_val - n_test
    if n_train <= 0:
        n_train = 1
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
    return n_train, n_val, n_test


def split_csv_paths_by_session(
    csv_paths: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    holdout_unit: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split by session (not by frame), stratified per drive_type.
    This keeps all rows from one session in exactly one split.
    """
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for csv_file in csv_paths:
        session_id, drive_type = infer_session_and_type(csv_file)
        holdout_key = infer_holdout_group_key(session_id, holdout_unit)
        grouped.setdefault(drive_type, {}).setdefault(holdout_key, []).append(csv_file)

    rng = random.Random(seed)
    train_csvs: List[str] = []
    val_csvs: List[str] = []
    test_csvs: List[str] = []

    for drive_type in sorted(grouped.keys()):
        group_items = [(key, sorted(paths)) for key, paths in grouped[drive_type].items()]
        rng.shuffle(group_items)

        n_train, n_val, n_test = _allocate_counts_per_type(len(group_items), val_ratio, test_ratio)

        test_part = group_items[:n_test]
        val_part = group_items[n_test:n_test + n_val]
        train_part = group_items[n_test + n_val:n_test + n_val + n_train]

        for _group, paths in train_part:
            train_csvs.extend(paths)
        for _group, paths in val_part:
            val_csvs.extend(paths)
        for _group, paths in test_part:
            test_csvs.extend(paths)

    train_csvs.sort()
    val_csvs.sort()
    test_csvs.sort()
    return train_csvs, val_csvs, test_csvs


def save_split_report(path: str, train_csvs: Sequence[str], val_csvs: Sequence[str], test_csvs: Sequence[str]) -> None:
    payload = {
        "train": list(train_csvs),
        "val": list(val_csvs),
        "test": list(test_csvs),
    }
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved split report: {path}")


class DrivingLogSequenceDataset(Dataset):
    """
    One sample = concatenated 10-frame input (90,) ending at current index
    plus current target (2,).

    For early indices with missing history, this uses edge padding by
    replicating frame 0 (no zero padding).
    """

    def __init__(self, csv_paths: Sequence[str], history: int = 10) -> None:
        if history <= 0:
            raise ValueError("history must be >= 1")
        if not csv_paths:
            raise ValueError("csv_paths is empty")

        self.history = history
        self.sequence_sources: List[str] = []
        self.sequences: List[List[Sample]] = []
        self.index_map: List[Tuple[int, int]] = []

        for src in csv_paths:
            seq = self._load_csv(src)
            if len(seq) == 0:
                continue
            seq_idx = len(self.sequences)
            self.sequence_sources.append(src)
            self.sequences.append(seq)
            self.index_map.extend((seq_idx, row_idx) for row_idx in range(len(seq)))

        if len(self.index_map) == 0:
            raise ValueError("No usable rows found in provided csv path(s)")

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
        return len(self.index_map)

    def _window_indices(self, index: int) -> List[int]:
        start = index - (self.history - 1)
        idxs: List[int] = []
        for i in range(start, index + 1):
            idxs.append(0 if i < 0 else i)
        return idxs

    def __getitem__(self, index: int):
        if isinstance(index, torch.Tensor):
            index = int(index.item())

        seq_idx, row_idx = self.index_map[index]
        seq = self.sequences[seq_idx]

        idxs = self._window_indices(row_idx)
        stacked = np.concatenate([seq[i].feature for i in idxs], axis=0)  # (history*9,)
        target = seq[row_idx].target  # (2,)

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
        kp = torch.sigmoid(raw[:, 0]) * self.kp_max
        ki = torch.sigmoid(raw[:, 1]) * self.ki_max
        kd = torch.sigmoid(raw[:, 2]) * self.kd_max
        return torch.stack([kp, ki, kd], dim=1)


def pid_terms_from_input(x: torch.Tensor, history: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_dim = len(FEATURE_COLUMNS)
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(x.shape)}")
    if x.size(1) != history * frame_dim:
        raise ValueError(f"Input feature size mismatch: got {x.size(1)}, expected {history * frame_dim}")

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


def motor_from_gains(gains: torch.Tensor, x: torch.Tensor, history: int, steer_limit: float) -> torch.Tensor:
    if gains.dim() != 2 or gains.size(1) != 3:
        raise ValueError(f"Expected gain tensor shape=(N,3), got {tuple(gains.shape)}")

    error, integral, derivative, base_speed_norm = pid_terms_from_input(x, history=history)
    kp = gains[:, 0]
    ki = gains[:, 1]
    kd = gains[:, 2]

    steer = (kp * error) + (ki * integral) + (kd * derivative)
    steer = torch.clamp(steer, -abs(float(steer_limit)), abs(float(steer_limit)))
    left = torch.clamp(base_speed_norm + steer, -1.0, 1.0)
    right = torch.clamp(base_speed_norm - steer, -1.0, 1.0)
    return torch.stack([left, right], dim=1)


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mode: str,
    history: int,
    steer_limit: float,
    gain_l2: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if target_mode == "pid-gain":
                gains = model(x)
                pred = motor_from_gains(gains, x=x, history=history, steer_limit=steer_limit)
                loss = criterion(pred, y)
                if gain_l2 > 0.0:
                    loss = loss + (gain_l2 * torch.mean(gains * gains))
            else:
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

        if is_directory_input and len(csv_paths) == 1:
            print("[Warn] Only one CSV found under directory; session-level split is not applicable.")

    input_dim = args.history * len(FEATURE_COLUMNS)

    if input_dim != 90:
        print(f"[Info] input_dim={input_dim}. Requirement is 90 when history=10.")

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

    if args.target_mode == "pid-gain":
        model = PIDGainMLP(
            input_dim=input_dim,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            kp_max=args.kp_max,
            ki_max=args.ki_max,
            kd_max=args.kd_max,
        ).to(device)
    else:
        model = LineTraceMLP(input_dim=input_dim, hidden1=args.hidden1, hidden2=args.hidden2, output_dim=2).to(device)
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
            if args.target_mode == "pid-gain":
                gains = model(x)
                pred = motor_from_gains(gains, x=x, history=args.history, steer_limit=args.steer_limit)
                loss = criterion(pred, y)
                if args.gain_l2 > 0.0:
                    loss = loss + (args.gain_l2 * torch.mean(gains * gains))
            else:
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
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                target_mode=args.target_mode,
                history=args.history,
                steer_limit=args.steer_limit,
                gain_l2=args.gain_l2,
            )
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
        output_names=["pid_gains" if args.target_mode == "pid-gain" else "motor_output"],
        dynamic_axes={
            "input": {0: "batch"},
            ("pid_gains" if args.target_mode == "pid-gain" else "motor_output"): {0: "batch"},
        },
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
    parser = argparse.ArgumentParser(description="Train MLP for line-trace (motor regression or PID-gain learning) and export ONNX.")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to one driving_log.csv file OR a directory containing multiple session folders",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=10, help="Number of frames to stack (10 -> input 90)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Session-level test split ratio when csv-path is a directory")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for session-level splitting")
    parser.add_argument("--holdout-unit", type=str, default="hour", choices=["session", "hour"], help="Group unit for split holdout when csv-path is a directory")
    parser.add_argument("--split-report-path", type=str, default="dataset_split.json", help="Output JSON path of train/val/test csv split")
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help='e.g. "cpu" or "cuda". Empty means auto.')
    parser.add_argument("--weights-path", type=str, default="model.pt")
    parser.add_argument("--onnx-path", type=str, default="model.onnx")
    parser.add_argument("--target-mode", type=str, default="pid-gain", choices=["motor", "pid-gain"], help="Training target: direct motor output or learned PID gains")
    parser.add_argument("--kp-max", type=float, default=2.5, help="Upper bound for learned Kp when target-mode=pid-gain")
    parser.add_argument("--ki-max", type=float, default=1.0, help="Upper bound for learned Ki when target-mode=pid-gain")
    parser.add_argument("--kd-max", type=float, default=1.5, help="Upper bound for learned Kd when target-mode=pid-gain")
    parser.add_argument("--steer-limit", type=float, default=1.0, help="Clamp limit of PID steering term when target-mode=pid-gain")
    parser.add_argument("--gain-l2", type=float, default=1e-4, help="L2 regularization weight applied to learned gains when target-mode=pid-gain")
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
