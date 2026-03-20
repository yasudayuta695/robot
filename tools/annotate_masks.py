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
  ホイール         : ブラシサイズを変更
  中クリック       : 元に戻す（Undo）
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"依存ライブラリが不足しています: {e}")
    print("pip install pillow")
    sys.exit(1)


def _setup_japanese_font(root: "tk.Tk") -> None:
    """日本語表示可能なフォントを検出してデフォルトに設定する。"""
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAGothic",
        "IPAPGothic",
        "VL Gothic",
        "TakaoGothic",
        "Takao",
        "Meiryo",
        "MS Gothic",
    ]
    available = set(tkfont.families(root))
    for name in candidates:
        if name in available:
            # tkinter の全名前付きフォントを変更する
            for font_name in (
                "TkDefaultFont", "TkTextFont", "TkMenuFont",
                "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                "TkIconFont", "TkTooltipFont",
            ):
                try:
                    tkfont.nametofont(font_name).configure(family=name, size=10)
                except Exception:
                    pass
            return
    # 見つからなければ何もしない（デフォルトのまま）

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

        # ブラシカーソル
        self._cursor_id: Optional[int] = None
        self._cursor_xy: Optional[tuple[int, int]] = None

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

        # モード切り替え（大きなトグルボタン）
        self.mode_var = tk.StringVar(value="paint")
        self._btn_paint = tk.Button(
            top, text="✏ 塗る (P)", width=10, relief=tk.SUNKEN,
            bg="#2196F3", fg="white", font=("", 10, "bold"),
            command=lambda: self._set_mode("paint"),
        )
        self._btn_paint.pack(side=tk.LEFT, padx=2)
        self._btn_erase = tk.Button(
            top, text="✕ 消す (E)", width=10, relief=tk.RAISED,
            bg="#cccccc", fg="black", font=("", 10, "bold"),
            command=lambda: self._set_mode("erase"),
        )
        self._btn_erase.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # ブラシサイズ
        tk.Label(top, text="ブラシ:").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=8)
        self.brush_var.trace_add("write", lambda *_: self._redraw_cursor())
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
            cursor="none", bg="black",
        )
        self.canvas.pack(side=tk.TOP, padx=6, pady=4)

        # 描画イベント
        self.canvas.bind("<ButtonPress-1>", self._on_press_paint)
        self.canvas.bind("<B1-Motion>", self._on_drag_paint)
        self.canvas.bind("<ButtonPress-3>", self._on_press_erase)
        self.canvas.bind("<B3-Motion>", self._on_drag_erase)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)

        # カーソル追跡
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # マウスホイール（ブラシサイズ）
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)        # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._wheel_brush(1))   # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self._wheel_brush(-1))  # Linux scroll down

        # 中クリック → Undo
        self.canvas.bind("<ButtonPress-2>", lambda _: self._undo())

        # --- ステータスバー ---
        self.status_var = tk.StringVar(value="")
        tk.Label(
            self.root, textvariable=self.status_var,
            anchor=tk.W, fg="gray40",
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=2)

        # --- キーバインド ---
        self.root.bind("<Left>", lambda _: self._prev_frame())
        self.root.bind("<Right>", lambda _: self._next_frame())
        self.root.bind("<KP_Left>", lambda _: self._prev_frame())
        self.root.bind("<KP_Right>", lambda _: self._next_frame())
        self.root.bind("<KP_Up>", lambda _: self._adjust_brush(2))
        self.root.bind("<KP_Down>", lambda _: self._adjust_brush(-2))
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

        # 元画像をフル輝度で表示し、マスク領域だけ半透明の色を重ねる
        base_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        h, w = base_rgb.shape[:2]

        # PIL で alpha composite（元画像 100% + 色レイヤー alpha%）
        base_pil = Image.fromarray(base_rgb).convert("RGBA")
        color_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        color_arr = np.zeros((h, w, 4), dtype=np.uint8)
        color_arr[self.current_mask > 0] = (0, 200, 0, int(alpha * 255))
        color_layer = Image.fromarray(color_arr, mode="RGBA")
        blended_pil = Image.alpha_composite(base_pil, color_layer).convert("RGB")
        blended = np.array(blended_pil)

        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # マスク境界を白線で描画（輪郭が見やすい）
        contours, _ = cv2.findContours(
            self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended_bgr, contours, -1, (255, 255, 255), 1)

        # ROI 境界線と上部の無効領域を表示
        roi_top = int(h * 0.25)
        # 上部 25% を暗く表示（アノテーション不要領域）
        blended_bgr[:roi_top] = (blended_bgr[:roi_top] * 0.4).astype(np.uint8)
        # ROI 境界を赤の点線で描画
        for x in range(0, w, 8):
            cv2.line(blended_bgr, (x, roi_top), (min(x + 4, w), roi_top), (0, 0, 220), 1)

        blended = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

        display = cv2.resize(
            blended, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(display))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        self._redraw_cursor()

    # ------------------------------------------------------------------
    # ブラシカーソル
    # ------------------------------------------------------------------
    def _redraw_cursor(self) -> None:
        """カーソル円を再描画する。"""
        if self._cursor_id is not None:
            self.canvas.delete(self._cursor_id)
            self._cursor_id = None
        if self._cursor_xy is None:
            return
        cx, cy = self._cursor_xy
        r = max(1, self.brush_var.get()) * DISPLAY_SCALE
        color = "#00dd00" if self.draw_mode == "paint" else "#ff3333"
        self._cursor_id = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=color, width=2, fill="",
        )

    def _on_canvas_motion(self, event: tk.Event) -> None:
        self._cursor_xy = (event.x, event.y)
        self._redraw_cursor()

    def _on_canvas_leave(self, event: tk.Event) -> None:
        self._cursor_xy = None
        if self._cursor_id is not None:
            self.canvas.delete(self._cursor_id)
            self._cursor_id = None

    # ------------------------------------------------------------------
    # マウスホイール
    # ------------------------------------------------------------------
    def _on_mousewheel(self, event: tk.Event) -> None:
        # Windows: delta=±120, macOS: delta=±1〜
        steps = 1 if event.delta > 0 else -1
        self._adjust_brush(steps * 2)

    def _wheel_brush(self, direction: int) -> None:
        self._adjust_brush(direction * 2)

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
        self._apply_brush(event.x, event.y, self.draw_mode)
        self._render()

    def _on_drag_paint(self, event: tk.Event) -> None:
        self._cursor_xy = (event.x, event.y)
        if self.drawing and self.last_xy:
            self._apply_stroke(self.last_xy[0], self.last_xy[1], event.x, event.y, self.draw_mode)
            self._render()
        self.last_xy = (event.x, event.y)

    def _on_press_erase(self, event: tk.Event) -> None:
        self._push_undo()
        self.drawing = True
        self.last_xy = (event.x, event.y)
        self._apply_brush(event.x, event.y, "erase")
        self._render()

    def _on_drag_erase(self, event: tk.Event) -> None:
        self._cursor_xy = (event.x, event.y)
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
        if mode == "paint":
            self._btn_paint.config(relief=tk.SUNKEN, bg="#2196F3", fg="white")
            self._btn_erase.config(relief=tk.RAISED, bg="#cccccc", fg="black")
        else:
            self._btn_paint.config(relief=tk.RAISED, bg="#cccccc", fg="black")
            self._btn_erase.config(relief=tk.SUNKEN, bg="#f44336", fg="white")
        self._redraw_cursor()

    def _on_mode_change(self) -> None:
        self._set_mode(self.mode_var.get())

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

    # パスが存在しない場合、apps/ 以下を自動で試す
    if not os.path.isdir(args.session_dir):
        alt = os.path.join(_ROOT, "apps", args.session_dir)
        if os.path.isdir(alt):
            print(f"パスを自動補完: {args.session_dir} → {alt}")
            args.session_dir = alt
        else:
            print(f"ディレクトリが存在しません: {args.session_dir}")
            sys.exit(1)

    root = tk.Tk()
    _setup_japanese_font(root)
    AnnotatorApp(root, args.session_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
