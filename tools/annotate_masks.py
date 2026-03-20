#!/usr/bin/env python3
"""
annotate_masks.py

走行ログのフレームに対してセグメンテーションマスクをアノテーション（修正）するGUI。
generate_masks.py で自動生成したマスクを下書きとして読み込み、
左ドラッグで塗り、右ドラッグで消す。

使用方法:
  python tools/annotate_masks.py <session_dir>

  例:
  python tools/annotate_masks.py dataset/camera_1/straight/20260316_133810_941565

操作:
  左ドラッグ       : コース領域を塗る
  右ドラッグ       : マスクを消す
  ← / →          : 前後のフレームへ移動（自動保存）
  S               : 現在のマスクを保存
  P               : 塗るモードに切り替え
  E               : 消すモードに切り替え
  [ / ]           : ブラシサイズを小さく / 大きく
  Ctrl+Z          : 1ステップ元に戻す
"""

import argparse
import glob
import os
import sys
from typing import Optional

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"依存ライブラリが不足しています: {e}")
    print("pip install pillow")
    sys.exit(1)

# 表示スケール: 320×240 → 960×720
DISPLAY_SCALE = 3
# マスクのオーバーレイ色 (BGR)
MASK_COLOR_BGR = (0, 200, 0)
# Undo バッファの最大サイズ
UNDO_MAX = 20


class AnnotatorApp:
    def __init__(self, root: tk.Tk, session_dir: str) -> None:
        self.root = root
        self.session_dir = session_dir
        self.image_dir = os.path.join(session_dir, "image")
        self.mask_dir = os.path.join(session_dir, "mask")
        os.makedirs(self.mask_dir, exist_ok=True)

        # フレームリスト
        self.image_paths: list[str] = []
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, pat)))
        self.image_paths.sort()

        if not self.image_paths:
            messagebox.showerror("エラー", f"画像が見つかりません:\n{self.image_dir}")
            root.destroy()
            return

        # 状態
        self.current_idx: int = 0
        self.current_image: Optional[np.ndarray] = None  # BGR (H, W, 3)
        self.current_mask: Optional[np.ndarray] = None   # uint8 0/255 (H, W)
        self.dirty: bool = False
        self.draw_mode: str = "paint"   # "paint" / "erase"
        self.drawing: bool = False
        self.last_xy: Optional[tuple[int, int]] = None
        self._undo_stack: list[np.ndarray] = []

        self._build_ui()
        self._load_frame(0)

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title("Mask Annotator")
        self.root.resizable(False, False)

        # --- トップバー ---
        top = tk.Frame(self.root, pady=4)
        top.pack(side=tk.TOP, fill=tk.X, padx=6)

        tk.Button(top, text="◀ Prev", command=self._prev_frame, width=8).pack(side=tk.LEFT, padx=2)
        self.frame_label = tk.Label(top, text="0/0", width=10, anchor="center")
        self.frame_label.pack(side=tk.LEFT)
        tk.Button(top, text="Next ▶", command=self._next_frame, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # モード切り替え
        self.mode_var = tk.StringVar(value="paint")
        tk.Radiobutton(
            top, text="塗る (P)", variable=self.mode_var,
            value="paint", command=self._on_mode_change,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            top, text="消す (E)", variable=self.mode_var,
            value="erase", command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # ブラシサイズ
        tk.Label(top, text="ブラシ:").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=8)
        tk.Scale(
            top, variable=self.brush_var,
            from_=1, to=40, orient=tk.HORIZONTAL,
            length=120, showvalue=True,
        ).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 透明度
        tk.Label(top, text="透明度:").pack(side=tk.LEFT)
        self.opacity_var = tk.DoubleVar(value=0.45)
        tk.Scale(
            top, variable=self.opacity_var,
            from_=0.1, to=0.9, resolution=0.05,
            orient=tk.HORIZONTAL, length=80, showvalue=False,
            command=lambda _: self._render(),
        ).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        tk.Button(
            top, text="保存 (S)", command=self._save_mask,
            bg="#4CAF50", fg="white", width=8,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            top, text="元に戻す (Z)", command=self._undo, width=12,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            top, text="マスク消去", command=self._clear_mask,
            bg="#f44336", fg="white", width=10,
        ).pack(side=tk.LEFT, padx=2)

        # --- キャンバス (サイズは最初のフレーム読み込み時に確定) ---
        self.canvas = tk.Canvas(
            self.root, width=320 * DISPLAY_SCALE, height=240 * DISPLAY_SCALE,
            cursor="crosshair", bg="black",
        )
        self.canvas.pack(side=tk.TOP, padx=6, pady=4)

        self.canvas.bind("<ButtonPress-1>", self._on_press_paint)
        self.canvas.bind("<B1-Motion>", self._on_drag_paint)
        self.canvas.bind("<ButtonPress-3>", self._on_press_erase)
        self.canvas.bind("<B3-Motion>", self._on_drag_erase)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)

        # --- ステータスバー ---
        self.status_var = tk.StringVar(value="")
        tk.Label(
            self.root, textvariable=self.status_var,
            anchor=tk.W, fg="gray40",
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=2)

        # --- キーバインド ---
        self.root.bind("<Left>", lambda _: self._prev_frame())
        self.root.bind("<Right>", lambda _: self._next_frame())
        self.root.bind("s", lambda _: self._save_mask())
        self.root.bind("p", lambda _: self._set_mode("paint"))
        self.root.bind("e", lambda _: self._set_mode("erase"))
        self.root.bind("[", lambda _: self._adjust_brush(-2))
        self.root.bind("]", lambda _: self._adjust_brush(2))
        self.root.bind("<Control-z>", lambda _: self._undo())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # フレーム読み込み / 描画
    # ------------------------------------------------------------------
    def _load_frame(self, idx: int) -> None:
        self.current_idx = int(np.clip(idx, 0, len(self.image_paths) - 1))
        img_path = self.image_paths[self.current_idx]

        frame = cv2.imread(img_path)
        if frame is None:
            self.status_var.set(f"読み込み失敗: {img_path}")
            return
        self.current_image = frame
        h, w = frame.shape[:2]

        # キャンバスサイズを更新
        self.canvas.config(width=w * DISPLAY_SCALE, height=h * DISPLAY_SCALE)

        # マスク読み込み（なければ空）
        mask_path = self._mask_path_for(img_path)
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == (h, w):
                self.current_mask = (m > 127).astype(np.uint8) * 255
            else:
                self.current_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self.current_mask = np.zeros((h, w), dtype=np.uint8)

        self._undo_stack.clear()
        self.dirty = False
        n = len(self.image_paths)
        self.frame_label.config(text=f"{self.current_idx + 1}/{n}")
        self.status_var.set(os.path.basename(img_path))
        self._render()

    def _render(self) -> None:
        if self.current_image is None or self.current_mask is None:
            return
        alpha = float(self.opacity_var.get())
        base = self.current_image.copy()
        overlay = base.copy()
        overlay[self.current_mask > 0] = MASK_COLOR_BGR
        blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)

        h, w = blended.shape[:2]
        display = cv2.resize(
            blended, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    # ------------------------------------------------------------------
    # 描画ヘルパー
    # ------------------------------------------------------------------
    def _canvas_to_img(self, cx: int, cy: int) -> tuple[int, int]:
        return cx // DISPLAY_SCALE, cy // DISPLAY_SCALE

    def _push_undo(self) -> None:
        if self.current_mask is None:
            return
        self._undo_stack.append(self.current_mask.copy())
        if len(self._undo_stack) > UNDO_MAX:
            self._undo_stack.pop(0)

    def _apply_brush(self, cx: int, cy: int, mode: str) -> None:
        if self.current_mask is None:
            return
        x, y = self._canvas_to_img(cx, cy)
        r = max(1, self.brush_var.get())
        val = 255 if mode == "paint" else 0
        cv2.circle(self.current_mask, (x, y), r, val, -1)
        self.dirty = True

    def _apply_stroke(self, cx0: int, cy0: int, cx1: int, cy1: int, mode: str) -> None:
        """ドラッグ中の2点間を補間してブラシ塗り。"""
        if self.current_mask is None:
            return
        x0, y0 = self._canvas_to_img(cx0, cy0)
        x1, y1 = self._canvas_to_img(cx1, cy1)
        r = max(1, self.brush_var.get())
        val = 255 if mode == "paint" else 0
        cv2.line(self.current_mask, (x0, y0), (x1, y1), val, r * 2)
        self.dirty = True

    # ------------------------------------------------------------------
    # マウスイベント
    # ------------------------------------------------------------------
    def _on_press_paint(self, event: tk.Event) -> None:
        self._push_undo()
        self.drawing = True
        self.last_xy = (event.x, event.y)
        self._apply_brush(event.x, event.y, "paint")
        self._render()

    def _on_drag_paint(self, event: tk.Event) -> None:
        if self.drawing and self.last_xy:
            self._apply_stroke(self.last_xy[0], self.last_xy[1], event.x, event.y, "paint")
            self._render()
        self.last_xy = (event.x, event.y)

    def _on_press_erase(self, event: tk.Event) -> None:
        self._push_undo()
        self.drawing = True
        self.last_xy = (event.x, event.y)
        self._apply_brush(event.x, event.y, "erase")
        self._render()

    def _on_drag_erase(self, event: tk.Event) -> None:
        if self.drawing and self.last_xy:
            self._apply_stroke(self.last_xy[0], self.last_xy[1], event.x, event.y, "erase")
            self._render()
        self.last_xy = (event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        self.drawing = False
        self.last_xy = None

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------
    def _save_mask(self) -> None:
        if self.current_mask is None or not self.image_paths:
            return
        mask_path = self._mask_path_for(self.image_paths[self.current_idx])
        cv2.imwrite(mask_path, self.current_mask)
        self.dirty = False
        self.status_var.set(f"保存済み: {os.path.basename(mask_path)}")

    def _clear_mask(self) -> None:
        if self.current_mask is not None:
            self._push_undo()
            self.current_mask[:] = 0
            self.dirty = True
            self._render()

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self.current_mask = self._undo_stack.pop()
        self.dirty = True
        self._render()
        self.status_var.set("元に戻しました")

    def _prev_frame(self) -> None:
        if self.dirty:
            self._save_mask()
        if self.current_idx > 0:
            self._load_frame(self.current_idx - 1)

    def _next_frame(self) -> None:
        if self.dirty:
            self._save_mask()
        if self.current_idx < len(self.image_paths) - 1:
            self._load_frame(self.current_idx + 1)

    def _set_mode(self, mode: str) -> None:
        self.draw_mode = mode
        self.mode_var.set(mode)

    def _on_mode_change(self) -> None:
        self.draw_mode = self.mode_var.get()

    def _adjust_brush(self, delta: int) -> None:
        self.brush_var.set(int(np.clip(self.brush_var.get() + delta, 1, 40)))

    def _on_close(self) -> None:
        if self.dirty:
            self._save_mask()
        self.root.destroy()

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------
    def _mask_path_for(self, img_path: str) -> str:
        name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        return os.path.join(self.mask_dir, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="セグメンテーションマスクアノテーションGUI")
    parser.add_argument(
        "session_dir",
        help="セッションディレクトリ (image/ フォルダを含む)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.session_dir):
        print(f"ディレクトリが存在しません: {args.session_dir}")
        sys.exit(1)

    root = tk.Tk()
    AnnotatorApp(root, args.session_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
