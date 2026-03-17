# robot_MLP

ライントレース用モーター制御MLPの学習・評価手順です。

このフォルダには次が含まれます。
- `train.py`: 学習実行（`.pt`保存 + `.onnx`出力）
- `test.py`: 学習済み`.pt`モデルの評価

## 1. モデル概要

- タスク: 回帰（`target_left_norm`, `target_right_norm`）
- 1フレーム入力: 9要素
- 時系列窓: 10フレーム
- 最終入力次元: `90`（=`9 * 10`）
- 出力次元: `2`

`train.py` の初期フレームはゼロ埋めではなく、エッジパディング（先頭フレーム複製）です。

## 2. 必要なCSV列

最低限、以下の列が必要です。

- `line_detect_top`
- `line_detect_mid`
- `line_detect_bottom`
- `line_offset_top`
- `line_offset_mid`
- `line_offset_bottom`
- `current_left_norm`
- `current_right_norm`
- `base_speed_norm`
- `target_left_norm`
- `target_right_norm`

列仕様は `apps/recorder.py` の出力と一致しています。

## 3. VS Code前提の環境準備

以下は「VS Codeでこのリポジトリを開いた状態」で、統合ターミナルから実行します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy matplotlib onnx onnxscript
```

補足:
- `matplotlib` は学習曲線画像を出さないなら省略可。
- `onnx` と `onnxscript` はONNX出力に必要。
- GPUが使える環境では自動利用（`--device cpu`指定時はCPU固定）。

## 4. 学習（VS Code統合ターミナル）

以下はワークスペースルート（`Robot_car/`）で実行する想定です。

### Linux / WSL (bash)

```bash
python robot_MLP/train.py \
  --csv-path dataset/camera_1 \
  --holdout-unit hour \
  --split-report-path dataset_split.json \
  --epochs 80 \
  --batch-size 64 \
  --lr 1e-3 \
  --history 10 \
  --val-ratio 0.1 \
  --test-ratio 0.15 \
  --early-stopping \
  --patience 10 \
  --min-delta 1e-4 \
  --weights-path robot_MLP/model.pt \
  --onnx-path robot_MLP/model.onnx \
  --curve-path robot_MLP/learning_curve.png
```

### Windows PowerShell

```powershell
python robot_MLP/train.py `
  --csv-path dataset/camera_1 `
  --holdout-unit hour `
  --split-report-path dataset_split.json `
  --epochs 80 `
  --batch-size 64 `
  --lr 1e-3 `
  --history 10 `
  --val-ratio 0.1 `
  --test-ratio 0.15 `
  --early-stopping `
  --patience 10 `
  --min-delta 1e-4 `
  --weights-path robot_MLP/model.pt `
  --onnx-path robot_MLP/model.onnx `
  --curve-path robot_MLP/learning_curve.png
```

主な出力:
- `robot_MLP/model.pt`
- `robot_MLP/model.onnx`
- `robot_MLP/learning_curve.png`（`--no-curve`時は未出力）

便利オプション:
- `--val-ratio 0`: 検証分割なし
- `--no-curve`: 学習曲線画像を作らない
- `--no-restore-best`: ベスト重みではなく最終epoch重みを採用
- `--test-ratio 0.15`: 未知セッションをテストに確保
- `--split-report-path dataset_split.json`: 分割結果を保存
- `--holdout-unit hour`: 同一時刻帯セッションを同一splitへ固定

分割仕様:
- 分割単位はフレームではなくセッションディレクトリ
- `drive_type` ごとに層化
- `test` splitは学習に使わない

## 5. 評価（VS Code統合ターミナル）

### 全CSVを対象に評価

```bash
python robot_MLP/test.py \
  --csv-path dataset/camera_1 \
  --weights-path robot_MLP/model.pt \
  --history 10 \
  --hidden1 64 \
  --hidden2 32 \
  --pred-csv-path robot_MLP/test_predictions.csv
```

### 推奨: holdoutされたtest splitで評価

```bash
python robot_MLP/test.py \
  --split-report-path dataset_split.json \
  --split-name test \
  --weights-path robot_MLP/model.pt \
  --history 10 \
  --hidden1 64 \
  --hidden2 32 \
  --pred-csv-path robot_MLP/test_predictions.csv
```

出力指標:
- MSE（overall/left/right）
- MAE（overall/left/right）
- RMSE（overall/left/right）

注意:
- `--hidden1` / `--hidden2` は学習時の構成と一致させる
- `--history` は学習時と一致させる

## 6. アプリで推論実行（ONNX）

`apps/pi_pc_app.py` には `AI (ONNX)` モードがあります。

```bash
python apps/pi_pc_app.py
```

手順:
1. `train.py`で`robot_MLP/model.onnx`を出力
2. `python apps/pi_pc_app.py` で起動
3. 操作モードを `AI (ONNX)` に切り替え
4. 必要ならONNXパスを指定して `Load`
5. 緊急停止は `Space` または `Esc` 長押し

## 7. 推奨ワークフロー

1. `--val-ratio 0.1` + 早期終了で学習
2. `test.py` で誤差確認
3. ハイパーパラメータ調整
4. ONNXを書き出して実機アプリで検証
5. 新規データを収集して再学習

## 8. トラブルシュート

- `ModuleNotFoundError: torch`
  - 仮想環境を有効化して依存を再インストール
- ONNX読込済みなのに動かない
  - ライン特徴量が取得できているか確認
  - 学習時と推論時のモデル構成が一致しているか確認
  - `pi_main.py` 側のモーター受信サーバー起動を確認
- 起動直後の挙動が不安定
  - 履歴窓の立ち上がりで感度が高くなるため、数フレームは様子を見る
