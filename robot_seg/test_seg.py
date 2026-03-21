"""
test_seg.py  ―  学習済みセグメンテーションモデルの評価スクリプト

使用方法:
  # val セットで評価（train_seg.py が生成した分割情報を使用）
  python robot_seg/test_seg.py

  # 任意のディレクトリを指定して評価
  python robot_seg/test_seg.py --image_dir apps/dataset/camera_1/straight/20260320_134132_715450/image

  # 可視化画像を保存する
  python robot_seg/test_seg.py --save_vis

出力:
  端末: フレームごとの IoU / Dice / Precision / Recall と全体の統計
  robot_seg/vis/  (--save_vis 時) 可視化画像 (入力 | GT マスク | 予測マスク)
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

# train_seg.py のモデル定義を再利用
from train_seg import IMG_H, IMG_W, LightUNet, SegDataset


# ──────────────────────────────────────────────
# メトリクス計算
# ──────────────────────────────────────────────
def compute_metrics(pred_bin: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    pred_bin, gt : 0/1 の numpy 配列 (H, W)
    返値: IoU, Dice, Precision, Recall
    """
    tp = float((pred_bin & gt).sum())
    fp = float((pred_bin & ~gt).sum())
    fn = float((~pred_bin & gt).sum())

    iou       = tp / (tp + fp + fn + 1e-6)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# ──────────────────────────────────────────────
# 可視化
# ──────────────────────────────────────────────
def make_vis(
    img_gray: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """
    3 枚を横に並べた BGR 画像を返す。
      左: 入力グレースケール
      中: GT マスク (緑)
      右: 予測マスク (赤) + 正解との差分表示
    """
    h, w = img_gray.shape
    bg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # GT: 緑でオーバーレイ
    gt_vis = bg.copy()
    gt_vis[gt_mask == 1] = (0, 200, 0)

    # 予測: TP=緑, FP=赤, FN=青
    pred_vis = bg.copy()
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    pred_vis[tp] = (0, 200, 0)   # 正解 → 緑
    pred_vis[fp] = (0, 0, 200)   # 過検出 → 赤
    pred_vis[fn] = (200, 0, 0)   # 未検出 → 青

    return np.hstack([bg, gt_vis, pred_vis])


# ──────────────────────────────────────────────
# 推論ユーティリティ
# ──────────────────────────────────────────────
def infer_single(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1 枚の画像に対して推論を行う。

    返値:
      img_gray  : (H, W) uint8 グレースケール (リサイズ後)
      pred_mask : (H, W) bool 二値マスク
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {img_path}")
    img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)   # (1, 1, H, W), sigmoid済み

    pred_np = pred[0, 0].cpu().numpy()  # (H, W), 0〜1
    pred_mask = pred_np >= threshold
    return img_resized, pred_mask


# ──────────────────────────────────────────────
# 評価ループ
# ──────────────────────────────────────────────
def evaluate(
    model: torch.nn.Module,
    pairs: list[tuple[str, str]],
    device: torch.device,
    save_vis: bool,
    vis_dir: str,
    threshold: float = 0.5,
    max_vis: int = 30,
) -> dict[str, float]:
    """
    val ペア全件を推論し、メトリクスを集計して返す。
    save_vis=True の場合、vis_dir に可視化画像を保存する。
    """
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    all_metrics: list[dict[str, float]] = []
    n_saved = 0

    for i, (img_path, mask_path) in enumerate(pairs):
        # ── 推論 ──
        img_gray, pred_mask = infer_single(model, img_path, device, threshold)

        # ── GT 読み込み ──
        gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            print(f"  [skip] マスクが読めません: {mask_path}")
            continue
        gt_resized = cv2.resize(gt_raw, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        gt_mask = gt_resized > 127  # bool

        # ── メトリクス ──
        m = compute_metrics(pred_mask, gt_mask)
        all_metrics.append(m)

        print(
            f"[{i+1:04d}/{len(pairs)}] "
            f"IoU={m['iou']:.3f}  Dice={m['dice']:.3f}  "
            f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
            f"| {os.path.basename(img_path)}"
        )

        # ── 可視化保存 ──
        if save_vis and n_saved < max_vis:
            vis = make_vis(img_gray, gt_mask.astype(np.uint8), pred_mask.astype(np.uint8))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(vis_dir, f"{i:04d}_{stem}.png")
            cv2.imwrite(save_path, vis)
            n_saved += 1

    if not all_metrics:
        return {}

    # ── 統計 ──
    agg: dict[str, float] = {}
    for key in ("iou", "dice", "precision", "recall"):
        vals = [m[key] for m in all_metrics]
        agg[f"mean_{key}"] = float(np.mean(vals))
        agg[f"std_{key}"]  = float(np.std(vals))
        agg[f"min_{key}"]  = float(np.min(vals))

    return agg


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="セグメンテーションモデル評価")
    parser.add_argument("--model",     default="robot_seg/model_seg.pt",              help=".pt モデルパス")
    parser.add_argument("--split",     default="robot_seg/dataset_split_seg.json",    help="train_seg.py が出力した分割 JSON")
    parser.add_argument("--split_key", default="val",                                  help="評価する分割キー (val / train)")
    parser.add_argument("--image_dir", default=None, help="直接指定する画像ディレクトリ (指定時は --split を無視)")
    parser.add_argument("--mask_dir",  default=None, help="--image_dir 使用時のマスクディレクトリ (省略可)")
    parser.add_argument("--base_ch",   type=int, default=8,    help="U-Net 基本チャンネル数 (train 時と合わせること)")
    parser.add_argument("--threshold", type=float, default=0.5, help="二値化閾値")
    parser.add_argument("--save_vis",  action="store_true",    help="可視化画像を保存する")
    parser.add_argument("--vis_dir",   default="robot_seg/vis", help="可視化保存先")
    parser.add_argument("--max_vis",   type=int, default=30,   help="可視化画像の最大保存枚数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── モデルロード ──
    model = LightUNet(base_ch=args.base_ch).to(device)
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"モデルが見つかりません: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[model] ロード完了: {args.model}")

    # ── ペアリスト作成 ──
    if args.image_dir is not None:
        # 直接指定モード
        import glob as _glob
        img_paths = sorted(_glob.glob(os.path.join(args.image_dir, "*.jpg")))
        if not img_paths:
            raise ValueError(f"jpg が見つかりません: {args.image_dir}")

        pairs: list[tuple[str, str]] = []
        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if args.mask_dir:
                mask_path = os.path.join(args.mask_dir, stem + ".png")
                if os.path.isfile(mask_path):
                    pairs.append((img_path, mask_path))
            else:
                # mask_dir が無ければメトリクス無しで推論だけ行う
                pairs.append((img_path, ""))

        print(f"[data] {len(pairs)} 枚 (直接指定モード)")

        # GT がない場合は推論のみ
        if all(m == "" for _, m in pairs):
            _infer_only(model, [p for p, _ in pairs], device, args)
            return

    else:
        # 分割 JSON からロード
        if not os.path.isfile(args.split):
            raise FileNotFoundError(
                f"分割ファイルが見つかりません: {args.split}\n"
                "先に train_seg.py を実行してください。"
            )
        with open(args.split) as f:
            split_data = json.load(f)

        if args.split_key not in split_data:
            raise ValueError(f"'{args.split_key}' が分割ファイルに存在しません。")

        pairs = [tuple(p) for p in split_data[args.split_key]]
        print(f"[data] {len(pairs)} ペア ({args.split_key} セット)")

    # ── 評価 ──
    agg = evaluate(
        model, pairs, device,
        save_vis=args.save_vis,
        vis_dir=args.vis_dir,
        threshold=args.threshold,
        max_vis=args.max_vis,
    )

    if agg:
        print("\n" + "=" * 50)
        print("  評価結果サマリ")
        print("=" * 50)
        print(f"  IoU        mean={agg['mean_iou']:.4f}  std={agg['std_iou']:.4f}  min={agg['min_iou']:.4f}")
        print(f"  Dice       mean={agg['mean_dice']:.4f}  std={agg['std_dice']:.4f}")
        print(f"  Precision  mean={agg['mean_precision']:.4f}")
        print(f"  Recall     mean={agg['mean_recall']:.4f}")
        print("=" * 50)

    if args.save_vis:
        print(f"\n[vis] 保存先: {args.vis_dir}")
        print("      左=入力  中=GT(緑)  右=予測(緑=TP 赤=FP 青=FN)")


def _infer_only(
    model: torch.nn.Module,
    img_paths: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """GT マスクなしで推論だけ行い、結果を可視化保存する。"""
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)

    for i, img_path in enumerate(img_paths):
        img_gray, pred_mask = infer_single(model, img_path, device, args.threshold)

        # 予測のみ表示 (GT なし → 赤単色)
        pred_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        pred_vis[pred_mask] = (0, 0, 200)
        result = np.hstack([cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), pred_vis])

        if args.save_vis and i < args.max_vis:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(args.vis_dir, f"{i:04d}_{stem}.png"), result)

        print(f"[{i+1:04d}/{len(img_paths)}] {os.path.basename(img_path)}")

    print(f"[done] {len(img_paths)} 枚 推論完了")


if __name__ == "__main__":
    main()
