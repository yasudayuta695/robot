"""
LSTM-based line-trace model trainer.

入力形状: (batch, seq_len=history, feature_dim=12)  ← MLPと違い、flattenしない
出力形状: (batch, 2)  motor mode, or (batch, 3)  pid-gain mode

motor mode  : target_left_norm, target_right_norm を直接回帰
pid-gain mode: Kp, Ki, Kd を学習し、PID計算でモーター出力に変換してMSE

Usage:
  cd robot_MLP/lstm

  # motor モード
  python train.py --csv-path ../../dataset/camera_1 --target-mode motor

  # pid-gain モード
  python train.py --csv-path ../../dataset/camera_1 --target-mode pid-gain
"""

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


# ---------------------------------------------------------------------------
# Feature / target column definitions  (parent の train.py と同一)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "line_detect_top",
    "line_detect_mid",
    "line_detect_bottom",
    "line_offset_top",
    "line_offset_mid",
    "line_offset_bottom",
    "line_width_top",
    "line_width_mid",
    "line_width_bottom",
    "current_left_norm",
    "current_right_norm",
    "base_speed_norm",
]

TARGET_COLUMNS = ["target_left_norm", "target_right_norm"]

FEATURE_COLUMNS_NO_MOTOR = [
    c for c in FEATURE_COLUMNS if c not in ("current_left_norm", "current_right_norm")
]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    feature: np.ndarray  # shape: (feature_dim,)
    target: np.ndarray   # shape: (2,)


def _clip_feature(name: str, value: float) -> float:
    if name.startswith("line_detect"):
        return float(np.clip(value, 0.0, 1.0))
    if name.startswith("line_offset"):
        return float(np.clip(value, -1.0, 1.0))
    if name.startswith("line_width"):
        return float(np.clip(value, 0.0, 1.0))
    if name in ("current_left_norm", "current_right_norm"):
        return float(np.clip(value, -1.0, 1.0))
    if name == "base_speed_norm":
        return float(np.clip(value, 0.0, 1.0))
    return float(value)


def _clip_target(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _apply_flip_lr(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_columns: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """左右反転: x shape=(seq_len, feature_dim)"""
    offset_indices = [
        feature_columns.index(c)
        for c in ("line_offset_top", "line_offset_mid", "line_offset_bottom")
    ]
    has_motor = "current_left_norm" in feature_columns
    left_idx = feature_columns.index("current_left_norm") if has_motor else -1
    right_idx = feature_columns.index("current_right_norm") if has_motor else -1

    x = x.clone()
    y = y.clone()

    # 全タイムステップで一括処理 (seq_len, feature_dim)
    for oi in offset_indices:
        x[:, oi] = -x[:, oi]
    if has_motor:
        left_val = x[:, left_idx].clone()
        x[:, left_idx] = x[:, right_idx]
        x[:, right_idx] = left_val

    y[0], y[1] = y[1].clone(), y[0].clone()
    return x, y


class _FlipAugDataset(Dataset):
    """訓練用ラッパー: 50% の確率で左右反転を適用する。"""

    def __init__(self, base: Dataset, feature_columns: List[str]) -> None:
        self._base = base
        self._feature_columns = feature_columns

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        x, y = self._base[index]
        if torch.rand(1).item() < 0.5:
            x, y = _apply_flip_lr(x, y, self._feature_columns)
        return x, y


# ---------------------------------------------------------------------------
# CSV path resolution  (parent と同一)
# ---------------------------------------------------------------------------

def resolve_csv_paths(csv_path: str) -> List[str]:
    if not csv_path:
        raise ValueError("csv_path is empty")

    raw_path = os.path.expanduser(csv_path)
    norm_path = os.path.abspath(raw_path)

    candidate_paths = [norm_path]
    if not os.path.isabs(raw_path):
        # robot_MLP/lstm/ から実行した場合も ../../dataset/... を解決できる
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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
    raise ValueError("\n".join(msg))


def infer_session_and_type(csv_file: str) -> Tuple[str, str]:
    norm = os.path.normpath(csv_file)
    session_id = os.path.basename(os.path.dirname(norm))
    drive_type = os.path.basename(os.path.dirname(os.path.dirname(norm)))
    return session_id, drive_type


def infer_holdout_group_key(session_id: str, holdout_unit: str) -> str:
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

    return sorted(train_csvs), sorted(val_csvs), sorted(test_csvs)


def save_split_report(path: str, train_csvs: Sequence[str], val_csvs: Sequence[str], test_csvs: Sequence[str]) -> None:
    payload = {"train": list(train_csvs), "val": list(val_csvs), "test": list(test_csvs)}
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved split report: {path}")


# ---------------------------------------------------------------------------
# Dataset  ← MLPと唯一の大きな違い: __getitem__ が 2D テンソルを返す
# ---------------------------------------------------------------------------

class DrivingLogSequenceDataset(Dataset):
    """
    One sample = sequence tensor (history, feature_dim) + target (2,).

    MLPの DrivingLogSequenceDataset と違い、flatten しない。
    LSTM は (batch, seq_len, input_size) を期待するため。
    """

    _OPTIONAL_COLUMNS = frozenset(["line_width_top", "line_width_mid", "line_width_bottom"])

    def __init__(
        self,
        csv_paths: Sequence[str],
        history: int = 10,
        feature_columns: List[str] = FEATURE_COLUMNS,
    ) -> None:
        if history <= 0:
            raise ValueError("history must be >= 1")
        if not csv_paths:
            raise ValueError("csv_paths is empty")

        self.history = history
        self.feature_columns = feature_columns
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
            fieldnames = set(reader.fieldnames or [])
            required = [c for c in (self.feature_columns + TARGET_COLUMNS) if c not in self._OPTIONAL_COLUMNS]
            missing_required = [c for c in required if c not in fieldnames]
            if missing_required:
                raise ValueError(f"CSV is missing required columns: {missing_required}")

            missing_optional = [c for c in self.feature_columns if c in self._OPTIONAL_COLUMNS and c not in fieldnames]
            if missing_optional:
                print(f"[Warn] {os.path.basename(os.path.dirname(csv_path))}: optional columns missing (filled with 0): {missing_optional}")

            for row_idx, row in enumerate(reader):
                try:
                    feature_vals = [
                        _clip_feature(c, float(row[c]) if c in fieldnames else 0.0)
                        for c in self.feature_columns
                    ]
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

    def __getitem__(self, index: int):
        if isinstance(index, torch.Tensor):
            index = int(index.item())

        seq_idx, row_idx = self.index_map[index]
        seq = self.sequences[seq_idx]

        start = row_idx - (self.history - 1)
        frames = []
        for i in range(start, row_idx + 1):
            frames.append(seq[0 if i < 0 else i].feature)

        # ↓ MLPと唯一の違い: stack して (history, feature_dim) にする
        stacked = np.stack(frames, axis=0)  # (history, feature_dim)
        target = seq[row_idx].target         # (2,)

        x = torch.from_numpy(stacked)   # (history, feature_dim)
        y = torch.from_numpy(target)    # (2,)
        return x, y


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LineTraceLSTM(nn.Module):
    """
    LSTMでフレーム列を処理し、最後のタイムステップからモーター出力を回帰。

    input:  (batch, seq_len, input_dim)
    output: (batch, 2)  [target_left_norm, target_right_norm]
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden_dim)
        last = out[:, -1, :]        # 最後のタイムステップ: (batch, hidden_dim)
        return self.head(last)      # (batch, 2)


class PIDGainLSTM(nn.Module):
    """
    LSTMでフレーム列を処理し、最後のタイムステップからPIDゲインを予測。

    input:  (batch, seq_len, input_dim)
    output: (batch, 3)  [Kp, Ki, Kd]
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        kp_max: float = 2.5,
        ki_max: float = 1.0,
        kd_max: float = 1.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kp_max = float(max(1e-6, kp_max))
        self.ki_max = float(max(1e-6, ki_max))
        self.kd_max = float(max(1e-6, kd_max))
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        raw = self.head(last)
        scales = raw.new_tensor([self.kp_max, self.ki_max, self.kd_max])
        return torch.sigmoid(raw) * scales


# ---------------------------------------------------------------------------
# PID計算ユーティリティ  (LSTM版: x が 3D テンソル)
# ---------------------------------------------------------------------------

def pid_terms_from_sequence(
    x: torch.Tensor,
    feature_columns: List[str] = FEATURE_COLUMNS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: (batch, seq_len, feature_dim)
    returns: error, integral, derivative, base_speed_norm  各 shape=(batch,)
    """
    base_speed_idx = feature_columns.index("base_speed_norm")

    detect = torch.clamp(x[:, :, 0:3], 0.0, 1.0)   # (batch, seq_len, 3)
    offset = torch.clamp(x[:, :, 3:6], -1.0, 1.0)  # (batch, seq_len, 3)
    zone_weights = x.new_tensor([0.20, 0.35, 0.45]).view(1, 1, 3)

    weighted_detect = detect * zone_weights
    weighted_sum = weighted_detect.sum(dim=2)        # (batch, seq_len)
    weighted_error_num = (weighted_detect * offset).sum(dim=2)

    eps = 1e-6
    effective_error = torch.where(
        weighted_sum > 0.01,
        weighted_error_num / (weighted_sum + eps),
        torch.zeros_like(weighted_sum),
    )  # (batch, seq_len)

    error = effective_error[:, -1]

    valid = (weighted_sum > 0.01).float()
    integral = (effective_error * valid).sum(dim=1) / (valid.sum(dim=1) + eps)

    seq_len = x.size(1)
    if seq_len >= 2:
        derivative = effective_error[:, -1] - effective_error[:, -2]
    else:
        derivative = torch.zeros_like(error)

    base_speed_norm = torch.clamp(x[:, -1, base_speed_idx], 0.0, 1.0)
    return error, integral, derivative, base_speed_norm


def motor_from_gains(
    gains: torch.Tensor,
    x: torch.Tensor,
    steer_limit: float,
    feature_columns: List[str] = FEATURE_COLUMNS,
) -> torch.Tensor:
    """gains: (batch, 3), x: (batch, seq_len, feature_dim) → (batch, 2)"""
    error, integral, derivative, base_speed_norm = pid_terms_from_sequence(x, feature_columns)
    kp, ki, kd = gains[:, 0], gains[:, 1], gains[:, 2]

    steer = kp * error + ki * integral + kd * derivative
    steer = torch.clamp(steer, -abs(float(steer_limit)), abs(float(steer_limit)))
    left = torch.clamp(base_speed_norm + steer, -1.0, 1.0)
    right = torch.clamp(base_speed_norm - steer, -1.0, 1.0)
    return torch.stack([left, right], dim=1)


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def split_indices_sequential(n_total: int, val_ratio: float) -> Tuple[Sequence[int], Sequence[int]]:
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples for training after split")
    return list(range(0, train_size)), list(range(train_size, n_total))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mode: str,
    steer_limit: float,
    gain_l2: float,
    feature_columns: List[str],
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if target_mode == "pid-gain":
                gains = model(x)
                pred = motor_from_gains(gains, x, steer_limit, feature_columns)
                loss = criterion(pred, y)
                if gain_l2 > 0.0:
                    loss = loss + gain_l2 * torch.mean(gains * gains)
            else:
                pred = model(x)
                loss = criterion(pred, y)
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz

    return total_loss / max(1, total_count)


def save_learning_curve(train_losses: Sequence[float], val_losses: Sequence[float], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn] matplotlib is not installed. Skipping learning curve plot.")
        return

    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, np.asarray(train_losses, dtype=np.float32), label="train_loss", linewidth=2)

    val_arr = np.asarray(val_losses, dtype=np.float32)
    if val_arr.size == epochs.size:
        valid_mask = np.isfinite(val_arr)
        if np.any(valid_mask):
            ax.plot(epochs[valid_mask], val_arr[valid_mask], label="val_loss", linewidth=2)

    ax.set_title("Learning Curve (LSTM)")
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
    feature_columns = FEATURE_COLUMNS_NO_MOTOR if args.no_motor_feedback else FEATURE_COLUMNS
    feature_dim = len(feature_columns)
    print(f"[Info] feature_columns ({feature_dim}次元/フレーム × {args.history}フレーム): {feature_columns}")

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

        train_set = DrivingLogSequenceDataset(csv_paths=train_csvs, history=args.history, feature_columns=feature_columns)
        val_set = (
            DrivingLogSequenceDataset(csv_paths=val_csvs, history=args.history, feature_columns=feature_columns)
            if val_csvs else None
        )
        print(
            f"Loaded {len(csv_paths)} csv(s) from directory. "
            f"train={len(train_csvs)} val={len(val_csvs)} test={len(test_csvs)}"
        )
        if args.split_report_path:
            save_split_report(args.split_report_path, train_csvs, val_csvs, test_csvs)
    else:
        dataset = DrivingLogSequenceDataset(csv_paths=csv_paths, history=args.history, feature_columns=feature_columns)
        print(f"Loaded {len(csv_paths)} csv(s), total rows={len(dataset)}")
        train_idx, val_idx = split_indices_sequential(len(dataset), args.val_ratio)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx) if val_idx else None

    if not args.no_augment:
        train_set = _FlipAugDataset(train_set, feature_columns=feature_columns)
        print("[Info] 左右反転データ拡張: ON (--no-augment で無効化)")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Info] device={device}")

    if args.target_mode == "pid-gain":
        model = PIDGainLSTM(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            kp_max=args.kp_max,
            ki_max=args.ki_max,
            kd_max=args.kd_max,
            dropout=args.dropout,
        ).to(device)
    else:
        model = LineTraceLSTM(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if args.target_mode == "pid-gain":
                gains = model(x)
                pred = motor_from_gains(gains, x, args.steer_limit, feature_columns)
                loss = criterion(pred, y)
                if args.gain_l2 > 0.0:
                    loss = loss + args.gain_l2 * torch.mean(gains * gains)
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

        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                target_mode=args.target_mode,
                steer_limit=args.steer_limit,
                gain_l2=args.gain_l2,
                feature_columns=feature_columns,
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
                f"train={train_loss:.6f} val={val_loss:.6f} "
                f"no_improve={no_improve_count}/{args.patience}"
            )

            if args.early_stopping and no_improve_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        else:
            val_losses.append(float("nan"))
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] train={train_loss:.6f}")

    if val_loader is not None and best_state_dict is not None and not args.no_restore_best:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val_loss={best_val_loss:.6f}).")

    torch.save(model.state_dict(), args.weights_path)
    print(f"Saved weights: {args.weights_path}")

    if not args.no_curve:
        save_learning_curve(train_losses, val_losses, args.curve_path)

    # -----------------------------------------------------------------------
    # ONNX エクスポート
    # MLPと違い、入力は 3D: (batch, seq_len, feature_dim)
    # -----------------------------------------------------------------------
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, args.history, feature_dim, dtype=torch.float32)
    output_name = "pid_gains" if args.target_mode == "pid-gain" else "motor_output"

    export_kwargs = dict(
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={
            "input": {0: "batch"},
            output_name: {0: "batch"},
        },
    )
    try:
        torch.onnx.export(model_cpu, dummy, args.onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy, args.onnx_path, **export_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name in ("onnx", "onnxscript"):
            raise ModuleNotFoundError(
                "ONNX export requires optional packages. Run: pip install onnx onnxscript"
            ) from exc
        raise

    print(f"Exported ONNX: {args.onnx_path}")
    print(f"  Input shape : (1, {args.history}, {feature_dim})")
    print(f"  Output name : {output_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train LSTM for line-trace (motor regression or PID-gain) and export ONNX."
    )
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to driving_log.csv or directory containing session folders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=10,
                        help="Sequence length fed into LSTM (フレーム数)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="LSTM hidden state dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of stacked LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout between LSTM layers (num_layers > 1 のときのみ有効)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--holdout-unit", type=str, default="hour", choices=["session", "hour"])
    parser.add_argument("--split-report-path", type=str, default="dataset_split.json")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="",
                        help='e.g. "cpu" or "cuda". Empty means auto.')
    parser.add_argument("--weights-path", type=str, default="model_lstm.pt")
    parser.add_argument("--onnx-path", type=str, default="model_lstm.onnx")
    parser.add_argument("--target-mode", type=str, default="pid-gain",
                        choices=["motor", "pid-gain"])
    parser.add_argument("--kp-max", type=float, default=2.5)
    parser.add_argument("--ki-max", type=float, default=1.0)
    parser.add_argument("--kd-max", type=float, default=1.5)
    parser.add_argument("--steer-limit", type=float, default=1.0)
    parser.add_argument("--gain-l2", type=float, default=1e-4)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-restore-best", action="store_true")
    parser.add_argument("--curve-path", type=str, default="learning_curve_lstm.png")
    parser.add_argument("--no-curve", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-motor-feedback", action="store_true",
                        help="Exclude current_left/right_norm from features (feature_dim: 12→10)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
