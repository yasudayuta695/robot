"""
train_seg.py  ―  軽量 U-Net によるコース領域セグメンテーション学習スクリプト

使用方法:
  python robot_seg/train_seg.py --dataset apps/dataset
  python robot_seg/train_seg.py --dataset apps/dataset --epochs 30 --batch 8

出力:
  robot_seg/model_seg.pt          学習済みモデル (PyTorch)
  robot_seg/model_seg.onnx        ONNX エクスポート済みモデル
  robot_seg/learning_curve_seg.png  損失・IoU 曲線

ディレクトリ構成の前提:
  apps/dataset/
    <camera_id>/
      <drive_type>/
        <timestamp>/
          image/  ← *.jpg (元フレーム)
          mask/   ← *.png (アノテーション済み二値マスク: 0 or 255)
"""

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────
IMG_W, IMG_H = 160, 120   # モデル入力サイズ (width, height)
SEED = 42


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def find_pairs(dataset_root: str) -> dict[str, list[tuple[str, str]]]:
    """
    dataset_root 以下のセッションをスキャンし、
    {session_dir: [(image_path, mask_path), ...]} を返す。

    セッション単位でまとめることで、後の train/val 分割時に
    同じセッションのフレームが train/val に混在しない (leakage 防止)。
    """
    sessions: dict[str, list[tuple[str, str]]] = {}

    # <camera_id>/<drive_type>/<timestamp>/ 階層を想定
    pattern = os.path.join(dataset_root, "*", "*", "*")
    for session_dir in sorted(glob.glob(pattern)):
        image_dir = os.path.join(session_dir, "image")
        mask_dir  = os.path.join(session_dir, "mask")
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            continue

        pairs: list[tuple[str, str]] = []
        for img_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, stem + ".png")
            if os.path.isfile(mask_path):
                pairs.append((img_path, mask_path))

        if pairs:
            sessions[session_dir] = pairs

    return sessions


def split_sessions(
    sessions: dict[str, list[tuple[str, str]]],
    val_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """セッション単位で train / val に分割する。"""
    session_keys = sorted(sessions.keys())
    random.seed(seed)
    random.shuffle(session_keys)

    n_val = max(1, int(len(session_keys) * val_ratio))
    val_keys   = set(session_keys[:n_val])
    train_keys = set(session_keys[n_val:])

    train_pairs: list[tuple[str, str]] = []
    val_pairs:   list[tuple[str, str]] = []
    for key in session_keys:
        if key in val_keys:
            val_pairs.extend(sessions[key])
        else:
            train_pairs.extend(sessions[key])

    return train_pairs, val_pairs


class SegDataset(Dataset):
    """
    (image_path, mask_path) のペアリストから PyTorch Dataset を作る。

    前処理:
      - 画像: グレースケール → 160×120 リサイズ → float32 に正規化 (÷255)
      - マスク: グレースケール → 160×120 リサイズ → 0/1 の float32 (閾値 127)

    出力:
      image : Tensor (1, H, W) = (1, 120, 160)
      mask  : Tensor (1, H, W) = (1, 120, 160)
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # ── 画像読み込み ──
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"画像が読めません: {img_path}")
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0  # (H, W), 0〜1

        # ── マスク読み込み ──
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"マスクが読めません: {mask_path}")
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)  # (H, W), 0 or 1

        # ── データ拡張（train のみ） ──
        if self.augment:
            img, mask = self._augment(img, mask)

        image_tensor = torch.from_numpy(img[np.newaxis])    # (1, H, W)
        mask_tensor  = torch.from_numpy(mask[np.newaxis])   # (1, H, W)
        return image_tensor, mask_tensor

    @staticmethod
    def _augment(
        img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """左右反転のみの軽量拡張（コースの左右対称性を利用）。"""
        if random.random() < 0.5:
            img  = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        return img, mask


# ──────────────────────────────────────────────
# モデル定義: 軽量 U-Net
# ──────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU の基本ブロック。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightUNet(nn.Module):
    """
    軽量 U-Net (Encoder 3 段 + Decoder 3 段)

    入力:  (B, 1, 120, 160)
    出力:  (B, 1, 120, 160)  sigmoid 済み

    チャンネル数の目安: 8→16→32→64 (推論時間重視で最小限に)
    Pi 4 での速度が要件に合わない場合は ch= を半分にして再試。
    """

    def __init__(self, base_ch: int = 8) -> None:
        super().__init__()
        c = base_ch  # 8

        # ── Encoder ──
        self.enc1 = ConvBnRelu(1,     c)       # → (B, c,   H,   W)
        self.enc2 = ConvBnRelu(c,     c * 2)   # → (B, 2c,  H/2, W/2)
        self.enc3 = ConvBnRelu(c * 2, c * 4)   # → (B, 4c,  H/4, W/4)

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = ConvBnRelu(c * 4, c * 8)  # → (B, 8c, H/8, W/8)

        # ── Decoder ──
        # 各 Up ブロック: Upsample(×2) → Concat(skip) → Conv
        self.up3 = ConvBnRelu(c * 8 + c * 4, c * 4)
        self.up2 = ConvBnRelu(c * 4 + c * 2, c * 2)
        self.up1 = ConvBnRelu(c * 2 + c,     c)

        # ── Output ──
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)                  # (B, c,   H,   W)
        s2 = self.enc2(self.pool(s1))      # (B, 2c,  H/2, W/2)
        s3 = self.enc3(self.pool(s2))      # (B, 4c,  H/4, W/4)
        b  = self.bottleneck(self.pool(s3))  # (B, 8c,  H/8, W/8)

        # Decoder (bilinear upsample + skip concat)
        d3 = self._up(b,  s3, self.up3)   # (B, 4c,  H/4, W/4)
        d2 = self._up(d3, s2, self.up2)   # (B, 2c,  H/2, W/2)
        d1 = self._up(d2, s1, self.up1)   # (B, c,   H,   W)

        return torch.sigmoid(self.out_conv(d1))

    @staticmethod
    def _up(
        x: torch.Tensor,
        skip: torch.Tensor,
        conv: nn.Module,
    ) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return conv(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────
# 損失関数
# ──────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice Loss。コース領域が少ないピクセル不均衡に有効。"""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """BCE Loss + Dice Loss の加重和。"""
    bce  = nn.functional.binary_cross_entropy(pred, target)
    dice = DiceLoss()(pred, target)
    return bce_weight * bce + (1.0 - bce_weight) * dice


# ──────────────────────────────────────────────
# メトリクス
# ──────────────────────────────────────────────
def compute_iou(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    """バッチ単位の平均 IoU (threshold=0.5 で二値化済みを想定)。"""
    inter = (pred_bin * target).sum().item()
    union = (pred_bin + target).clamp(0, 1).sum().item()
    return inter / (union + 1e-6)


# ──────────────────────────────────────────────
# 学習ループ
# ──────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """1 epoch を実行し (avg_loss, avg_iou) を返す。"""
    model.train(is_train)
    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = combined_loss(preds, masks)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_bin = (preds >= 0.5).float()
            total_loss += loss.item()
            total_iou  += compute_iou(pred_bin, masks)
            n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches


# ──────────────────────────────────────────────
# ONNX エクスポート
# ──────────────────────────────────────────────
def export_onnx(model: nn.Module, save_path: str, device: torch.device) -> None:
    """
    ONNX 形式でエクスポートする。

    ダミー入力: (1, 1, 120, 160) = (batch, ch, H, W)
    opset_version=11 (NCNN onnx2ncnn の対応版)
    """
    model.eval()
    dummy = torch.zeros(1, 1, IMG_H, IMG_W, device=device)
    torch.onnx.export(
        model,
        dummy,
        save_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"[export] ONNX 保存: {save_path}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="軽量 U-Net セグメンテーション学習")
    parser.add_argument("--dataset",  default="apps/dataset",    help="データセットルートディレクトリ")
    parser.add_argument("--out_dir",  default="robot_seg",        help="モデル・ログ出力先")
    parser.add_argument("--epochs",   type=int,   default=20,     help="学習エポック数")
    parser.add_argument("--batch",    type=int,   default=8,      help="バッチサイズ")
    parser.add_argument("--lr",       type=float, default=1e-3,   help="学習率 (Adam)")
    parser.add_argument("--base_ch",  type=int,   default=8,      help="U-Net 基本チャンネル数 (推論速度に直結)")
    parser.add_argument("--val_ratio",type=float, default=0.2,    help="バリデーション割合")
    parser.add_argument("--no_augment", action="store_true",      help="データ拡張を無効化")
    parser.add_argument("--seed",     type=int,   default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── データ収集 ──
    sessions = find_pairs(args.dataset)
    if not sessions:
        sys.exit(f"[error] dataset が見つかりません: {args.dataset}")
    print(f"[data] セッション数: {len(sessions)}")

    train_pairs, val_pairs = split_sessions(sessions, args.val_ratio, args.seed)
    print(f"[data] train: {len(train_pairs)} ペア / val: {len(val_pairs)} ペア")

    # セッション分割をファイルに保存（test_seg.py で再利用）
    split_info = {
        "train": [list(p) for p in train_pairs],
        "val":   [list(p) for p in val_pairs],
    }
    split_path = os.path.join(args.out_dir, "dataset_split_seg.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[data] 分割情報保存: {split_path}")

    train_ds = SegDataset(train_pairs, augment=not args.no_augment)
    val_ds   = SegDataset(val_pairs,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # ── モデル ──
    model     = LightUNet(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] パラメータ数: {total_params:,}")

    # ── 学習 ──
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_iou = 0.0
    best_model_path = os.path.join(args.out_dir, "model_seg.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou = run_epoch(model, train_loader, optimizer, device, is_train=True)
        va_loss, va_iou = run_epoch(model, val_loader,   None,      device, is_train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train loss={tr_loss:.4f} IoU={tr_iou:.4f} | "
            f"val loss={va_loss:.4f} IoU={va_iou:.4f}"
        )

        # ベストモデルを保存
        if va_iou > best_val_iou:
            best_val_iou = va_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> best model 更新 (val IoU={va_iou:.4f})")

    print(f"\n[result] best val IoU: {best_val_iou:.4f}")

    # ── 学習曲線の保存 ──
    _save_learning_curve(history, os.path.join(args.out_dir, "learning_curve_seg.png"))

    # ── ONNX エクスポート ──
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    onnx_path = os.path.join(args.out_dir, "model_seg.onnx")
    export_onnx(model, onnx_path, device)

    print("\n変換コマンド (NCNN):")
    print(f"  onnx2ncnn {onnx_path} model_seg.param model_seg.bin")
    print("  ncnnoptimize model_seg.param model_seg.bin model_seg_opt.param model_seg_opt.bin 65536")


def _save_learning_curve(history: dict, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["train_loss"], label="train")
        ax1.plot(epochs, history["val_loss"],   label="val")
        ax1.set_title("Loss (BCE + Dice)")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, history["train_iou"], label="train")
        ax2.plot(epochs, history["val_iou"],   label="val")
        ax2.set_title("IoU")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[curve] 保存: {save_path}")
    except ImportError:
        print("[curve] matplotlib が無いためスキップ")


if __name__ == "__main__":
    main()
