"""
End-to-End CNN training script.

Input 1 : segmentation mask image  (1 × H × W, float32 in [0, 1])
          mask/ フォルダの PNG（セグメンテーション処理済み2値画像）を使用
Input 2 : scalar state             (3,) = [current_left_norm, current_right_norm, base_speed_norm]
Output  : [target_left_norm, target_right_norm]  ∈ [-1, 1]

Usage example:
    python train_image.py \\
        --csv-path ../apps/dataset/camera_4 \\
        --img-h 60 --img-w 80 \\
        --epochs 50 --early-stopping \\
        --onnx-path model_cnn.onnx

ONNX inputs (OpenCV DNN):
    net.setInput(img_blob,    "image")    # (1, 1, H, W) float32 in [0, 1]
    net.setInput(scalar_blob, "scalars")  # (1, 3)       float32
    output = net.forward()               # (1, 2) float32 in [-1, 1]
"""
import argparse
import csv
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


TARGET_COLUMNS = ["target_left_norm", "target_right_norm"]
SCALAR_COLUMNS = ["current_left_norm", "current_right_norm", "base_speed_norm"]
N_SCALARS = len(SCALAR_COLUMNS)  # 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_path_to_mask_path(img_abs: str) -> str:
    """
    image/20260320_134133_000000.jpg
        → mask/20260320_134133_000000.png
    """
    session_dir = os.path.dirname(os.path.dirname(img_abs))
    stem = os.path.splitext(os.path.basename(img_abs))[0]
    return os.path.join(session_dir, "mask", stem + ".png")


def _clip_scalar(col: str, value: float) -> float:
    if col == "base_speed_norm":
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(value, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _Row:
    __slots__ = ("mask_path", "scalars", "target")

    def __init__(self, mask_path: str, scalars: np.ndarray, target: np.ndarray) -> None:
        self.mask_path = mask_path
        self.scalars = scalars    # float32 (N_SCALARS,)
        self.target = target      # float32 (2,)


class ImageDrivingDataset(Dataset):
    """
    One sample = (img_tensor, scalar_tensor, target_tensor)
        img_tensor    : float32 (1, H, W) in [0, 1]
        scalar_tensor : float32 (3,)   [left_norm, right_norm, base_speed_norm]
        target_tensor : float32 (2,)   [target_left_norm, target_right_norm]
    """

    def __init__(
        self,
        csv_paths: Sequence[str],
        img_h: int = 60,
        img_w: int = 80,
        label_shift: int = 0,
    ) -> None:
        if not csv_paths:
            raise ValueError("csv_paths is empty")
        if label_shift < 0:
            raise ValueError("label_shift must be >= 0")

        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.label_shift = int(label_shift)

        self.sequences: List[List[_Row]] = []
        self.index_map: List[Tuple[int, int]] = []

        for src in csv_paths:
            seq = self._load_csv(src)
            if not seq:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            self.index_map.extend((seq_idx, i) for i in range(len(seq)))

        if not self.index_map:
            raise ValueError("No usable rows found in provided csv path(s)")

    def _load_csv(self, csv_path: str) -> List[_Row]:
        base_dir = os.path.dirname(csv_path)
        rows: List[_Row] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required = ["image_path"] + TARGET_COLUMNS + SCALAR_COLUMNS
            missing = [c for c in required if c not in fieldnames]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}  ({csv_path})")

            for row_idx, row in enumerate(reader):
                try:
                    img_abs = os.path.join(base_dir, row["image_path"].strip())
                    mask_abs = _img_path_to_mask_path(img_abs)
                    scalars = np.asarray(
                        [_clip_scalar(c, float(row[c])) for c in SCALAR_COLUMNS],
                        dtype=np.float32,
                    )
                    target = np.asarray(
                        [float(np.clip(float(row[c]), -1.0, 1.0)) for c in TARGET_COLUMNS],
                        dtype=np.float32,
                    )
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid value at row {row_idx + 2} in {csv_path}: {e}") from e
                rows.append(_Row(mask_abs, scalars, target))
        return rows

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        seq_idx, row_idx = self.index_map[index]
        seq = self.sequences[seq_idx]
        row = seq[row_idx]
        target_idx = min(row_idx + self.label_shift, len(seq) - 1)
        target = seq[target_idx].target

        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found: {row.mask_path}")
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        img_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
        scalar_tensor = torch.from_numpy(row.scalars.copy())
        target_tensor = torch.from_numpy(target.copy())
        return img_tensor, scalar_tensor, target_tensor


class _FlipAugDataset(Dataset):
    """
    50% probability left-right flip:
      - image  : horizontal flip
      - scalars: swap current_left_norm ↔ current_right_norm (indices 0, 1)
      - target : swap left ↔ right motor (indices 0, 1)
    """

    def __init__(self, base: Dataset) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        img, scalars, target = self._base[index]
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            scalars = torch.stack([scalars[1], scalars[0], scalars[2]])  # left↔right, base unchanged
            target = torch.stack([target[1], target[0]])
        return img, scalars, target


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LightCNNWithScalars(nn.Module):
    """
    Lightweight End-to-End CNN with scalar motor state input.

    Architecture:
        image (1, H, W) → Conv×4(stride=2) → Flatten(flat_dim)
                                                               ↘
                                                        cat(flat_dim + 3) → FC → Tanh → (2,)
                                                               ↗
        scalars (3,) ─────────────────────────────────────────

    Stride-2 convolutions (no MaxPool) for clean ONNX/OpenCV DNN compatibility.
    """

    def __init__(self, img_h: int = 60, img_w: int = 80, hidden: int = 128) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # H/2  W/2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # H/4  W/4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/8  W/8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # H/16 W/16
            nn.ReLU(),
        )

        fh = (img_h + 15) // 16
        fw = (img_w + 15) // 16
        flat_dim = 64 * fh * fw

        self.fc = nn.Sequential(
            nn.Linear(flat_dim + N_SCALARS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        feat = self.conv(image).flatten(1)             # (B, flat_dim)
        combined = torch.cat([feat, scalars], dim=1)   # (B, flat_dim + 3)
        return self.fc(combined)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_csv_paths(csv_path: str) -> List[str]:
    raw_path = os.path.expanduser(csv_path)
    norm_path = os.path.abspath(raw_path)

    candidate_paths = [norm_path]
    if not os.path.isabs(raw_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        raw_rel = raw_path[2:] if raw_path.startswith("./") else raw_path
        candidate_paths.append(os.path.abspath(os.path.join(project_root, raw_rel)))

    for candidate in candidate_paths:
        if os.path.isfile(candidate) or os.path.isdir(candidate):
            norm_path = candidate
            break

    if os.path.isfile(norm_path):
        return [norm_path]

    if os.path.isdir(norm_path):
        collected: List[str] = []
        for root, _dirs, files in os.walk(norm_path):
            if "driving_log.csv" in files:
                collected.append(os.path.join(root, "driving_log.csv"))
        collected.sort()
        if not collected:
            raise ValueError(f"No driving_log.csv found under: {norm_path}")
        return collected

    raise ValueError(f"Path does not exist: {csv_path}")


def infer_session_and_type(csv_file: str) -> Tuple[str, str]:
    norm = os.path.normpath(csv_file)
    session_id = os.path.basename(os.path.dirname(norm))
    drive_type = os.path.basename(os.path.dirname(os.path.dirname(norm)))
    return session_id, drive_type


def _allocate_counts_per_type(n: int, val_r: float, test_r: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        nv = 1 if val_r > 0.0 else 0
        return 2 - nv, nv, 0
    nv = int(round(n * max(0.0, val_r)))
    nt = int(round(n * max(0.0, test_r)))
    if val_r > 0.0 and nv == 0:
        nv = 1
    if test_r > 0.0 and nt == 0:
        nt = 1
    while nv + nt > n - 1:
        if nt >= nv and nt > 0:
            nt -= 1
        elif nv > 0:
            nv -= 1
        else:
            break
    ntr = n - nv - nt
    if ntr <= 0:
        ntr = 1
        if nt > 0:
            nt -= 1
        elif nv > 0:
            nv -= 1
    return ntr, nv, nt


def split_csv_paths_by_session(
    csv_paths: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    grouped: Dict[str, List[str]] = {}
    for csv_file in csv_paths:
        _, drive_type = infer_session_and_type(csv_file)
        grouped.setdefault(drive_type, []).append(csv_file)

    rng = random.Random(seed)
    train_csvs: List[str] = []
    val_csvs: List[str] = []
    test_csvs: List[str] = []

    for drive_type in sorted(grouped.keys()):
        sessions = sorted(grouped[drive_type])
        rng.shuffle(sessions)
        ntr, nv, nt = _allocate_counts_per_type(len(sessions), val_ratio, test_ratio)
        test_csvs.extend(sessions[:nt])
        val_csvs.extend(sessions[nt:nt + nv])
        train_csvs.extend(sessions[nt + nv:nt + nv + ntr])

    train_csvs.sort()
    val_csvs.sort()
    test_csvs.sort()
    return train_csvs, val_csvs, test_csvs


def save_split_report(path: str, train_csvs: Sequence[str], val_csvs: Sequence[str], test_csvs: Sequence[str]) -> None:
    payload = {"train": list(train_csvs), "val": list(val_csvs), "test": list(test_csvs)}
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved split report: {path}")


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for img, scalars, target in loader:
            img = img.to(device)
            scalars = scalars.to(device)
            target = target.to(device)
            pred = model(img, scalars)
            loss = criterion(pred, target)
            total_loss += loss.item() * img.size(0)
            total_count += img.size(0)
    return total_loss / max(1, total_count)


def save_learning_curve(train_losses: Sequence[float], val_losses: Sequence[float], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn] matplotlib not installed. Skipping learning curve.")
        return

    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train_loss", linewidth=2)
    val_arr = np.asarray(val_losses, dtype=np.float32)
    if val_arr.size == epochs.size:
        valid = np.isfinite(val_arr)
        if np.any(valid):
            ax.plot(epochs[valid], val_arr[valid], label="val_loss", linewidth=2)
    ax.set_title("Learning Curve (CNN+Scalar)")
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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_paths = resolve_csv_paths(args.csv_path)
    is_directory = os.path.isdir(os.path.abspath(os.path.expanduser(args.csv_path)))

    if is_directory and len(csv_paths) > 1:
        train_csvs, val_csvs, test_csvs = split_csv_paths_by_session(
            csv_paths=csv_paths,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
        if not train_csvs:
            raise ValueError("No training CSVs after split. Dataset too small?")

        train_set: Dataset = ImageDrivingDataset(train_csvs, args.img_h, args.img_w, args.label_shift)
        val_set: Optional[Dataset] = (
            ImageDrivingDataset(val_csvs, args.img_h, args.img_w, args.label_shift)
            if val_csvs else None
        )
        print(
            f"Loaded {len(csv_paths)} csv(s).  "
            f"train={len(train_csvs)} val={len(val_csvs)} test={len(test_csvs)} sessions"
        )
        if args.split_report_path:
            save_split_report(args.split_report_path, train_csvs, val_csvs, test_csvs)
    else:
        full_set = ImageDrivingDataset(csv_paths, args.img_h, args.img_w, args.label_shift)
        n = len(full_set)
        val_size = int(n * args.val_ratio)
        train_size = n - val_size
        if train_size <= 0:
            raise ValueError("Not enough samples for training after split.")
        train_set = Subset(full_set, list(range(train_size)))
        val_set = Subset(full_set, list(range(train_size, n))) if val_size > 0 else None
        print(f"Loaded {len(csv_paths)} csv(s), total rows={n}  train={train_size} val={val_size}")

    if not args.no_augment:
        train_set = _FlipAugDataset(train_set)
        print("[Info] 左右反転データ拡張: ON (--no-augment で無効化)")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader: Optional[DataLoader] = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  Mask image: {args.img_h}x{args.img_w}  Scalars: {SCALAR_COLUMNS}")

    model = LightCNNWithScalars(img_h=args.img_h, img_w=args.img_w, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0
    use_validation = val_loader is not None

    if args.early_stopping and not use_validation:
        print("[Warn] Early stopping requires validation data. Set --val-ratio > 0.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for img, scalars, target in train_loader:
            img = img.to(device)
            scalars = scalars.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(img, scalars)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
            seen += img.size(0)

        train_loss = running_loss / max(1, seen)
        train_losses.append(train_loss)

        if use_validation:
            val_loss = evaluate(model, val_loader, criterion, device)  # type: ignore[arg-type]
            val_losses.append(val_loss)

            improved = val_loss < (best_val_loss - args.min_delta)
            if improved:
                best_val_loss = val_loss
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}]  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"no_improve={no_improve_count}/{args.patience}"
            )

            if args.early_stopping and no_improve_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        else:
            val_losses.append(float("nan"))
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}]  train={train_loss:.6f}")

    if use_validation and best_state_dict is not None and not args.no_restore_best:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val_loss={best_val_loss:.6f}).")

    torch.save(model.state_dict(), args.weights_path)
    print(f"Saved PyTorch weights: {args.weights_path}")

    if not args.no_curve:
        save_learning_curve(train_losses, val_losses, args.curve_path)

    # ONNX export  (2 inputs: "image", "scalars")
    model_cpu = model.to("cpu").eval()
    dummy_img = torch.zeros(1, 1, args.img_h, args.img_w, dtype=torch.float32)
    dummy_scalars = torch.zeros(1, N_SCALARS, dtype=torch.float32)

    export_kwargs = dict(
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image", "scalars"],
        output_names=["motor_output"],
        dynamic_axes={
            "image": {0: "batch"},
            "scalars": {0: "batch"},
            "motor_output": {0: "batch"},
        },
    )
    try:
        torch.onnx.export(model_cpu, (dummy_img, dummy_scalars), args.onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, (dummy_img, dummy_scalars), args.onnx_path, **export_kwargs)

    print(f"Exported ONNX model: {args.onnx_path}")
    print(f"  Input 'image'  : (1, 1, {args.img_h}, {args.img_w})  float32 in [0, 1]")
    print(f"  Input 'scalars': (1, {N_SCALARS})  float32  {SCALAR_COLUMNS}")
    print(f"  Output 'motor_output': (1, 2)  float32 in [-1, 1]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-End CNN: mask image + motor state → left/right motor regression → ONNX"
    )
    p.add_argument("--csv-path", type=str, required=True,
                   help="driving_log.csv file or directory of session folders")
    p.add_argument("--img-h", type=int, default=60, help="Resize height (default: 60)")
    p.add_argument("--img-w", type=int, default=80, help="Resize width  (default: 80)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128, help="FC hidden units after CNN flatten (default: 128)")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-report-path", type=str, default="dataset_split_image.json")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--weights-path", type=str, default="model_cnn.pt")
    p.add_argument("--onnx-path", type=str, default="model_cnn.onnx")
    p.add_argument("--label-shift", type=int, default=0,
                   help="Shift target N frames into the future (dpad delay compensation)")
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--no-restore-best", action="store_true")
    p.add_argument("--curve-path", type=str, default="learning_curve_cnn.png")
    p.add_argument("--no-curve", action="store_true")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable left-right flip augmentation")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
