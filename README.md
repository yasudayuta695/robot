# Robot_car

## コード構成

- 実運用のメインスクリプトは `apps/` フォルダに配置しています（`pi_pc_app.py`, `pi_main.py`）。
- テスト/補助スクリプトは `test/` フォルダに配置しています（`pc_receiver.py`, `pi_mortor.py`, `pi_pc.py`, `pi_sender.py`）。

## pi_pc_app.py のデータ保存フロー

- 録画停止時に「保存前プレビュー」ウィンドウが表示されます。
- サムネイルと枚数を確認して、`保存する` / `破棄する` / `キャンセル` を選べます。

## 走行種別と保存先の設定

- 走行種別と保存先は `comfig.txt` を Config ファイルとして管理します。
- コマンドプロンプト（Windows）での実行を前提にする場合、`save_base_dir=` は UNC パスを設定してください。
- 各PCで `save_base_dir=` に、そのPC上のプロジェクトフォルダパスを書いてください。
- 走行種別は `drive_type=` を複数行追加してください。
- UIの速度基準初期値は `drive_speed_default=` で変更できます。
- 十字キー速度は `dpad_drive_speed_scale=` / `dpad_turn_speed_scale=` で変更できます。
- カメラ選択肢は `camera_id=` を複数行で設定できます。
- 起動時の選択カメラは `default_camera_id=` で設定できます。

例:

```txt
save_base_dir=/home/ryuryu/lab/Robot_car
drive_speed_default=60
dpad_drive_speed_scale=1.0
dpad_turn_speed_scale=0.4
camera_id=camera_1
camera_id=camera_2
default_camera_id=camera_1
drive_type=straight
drive_type=left_curve
drive_type=right_curve
drive_type=stop_and_go
```

Windows から編集する場合の例:

```txt
save_base_dir=\\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car
```

あなたの環境例:

```txt
save_base_dir=\\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\
```

注:

- 旧形式（1行1走行種別）も後方互換で読み込めます。

## 保存先

- 保存先は次の形式です。

```txt
datasets/1_raw_data/<camera_id>/<drive_type>/
```

- 実行時に、UI とログへ UNC 形式のパス（`\\wsl.localhost\...`）を表示します。

## Raspberry Pi側: 自動運転の動かし方

この章は「Pi単体で走らせる」ための手順です。
実行エントリは `apps/pi_main.py` で、`--mode` で制御方式を切り替えます。

### 0. 事前準備

- Pi上でこのリポジトリを配置しておく
- カメラとモータードライバ配線を接続しておく
- 初回はタイヤを浮かせた状態で確認する（暴走防止）

### 1. 依存パッケージを入れる（Pi）

Raspberry Pi OS (Bookworm) なら apt で揃えるのが安全です。

```bash
sudo apt update
sudo apt install -y \
	python3-opencv \
	python3-numpy \
	python3-zmq \
	python3-picamera2 \
	python3-rpi.gpio
```

### 2. 設定ファイルを確認

- `comfig.txt` の以下をまず確認
- `motor_left_sign`, `motor_right_sign`（直進方向の符号補正）
- `line_color_space`, `far_threshold`, `near_threshold`, `auto_threshold_enabled`
- `ai_control_interval_ms`, `line_process_interval_ms`

### 3. モデルファイルを配置

- MLP自動運転: `robot_MLP/model.onnx`
- 学習PIDゲイン制御: `robot_MLP/pid_gain_model_compat.onnx`

`pi_main.py` は `--onnx-path` 省略時に `robot_MLP/model.onnx` を使います。

### 4. 起動コマンド（Pi）

プロジェクトルートで実行します。

```bash
cd ~/ss2025robot
```

#### A. ローカルAI（ONNXで左右モーターを直接推論）

```bash
sudo python3 apps/pi_main.py \
	--mode local_ai \
	--config-path comfig.txt \
	--onnx-path robot_MLP/model.onnx \
	--base-speed 60 \
	--camera-fps 30
```

#### B. PID（固定ゲイン）

```bash
sudo python3 apps/pi_main.py \
	--mode pid \
	--config-path comfig.txt \
	--base-speed 55 \
	--camera-fps 30
```

必要なら `--pid-kp`, `--pid-ki`, `--pid-kd` で一時上書きできます。

#### C. PID Learned（ONNXでPIDゲインを推論）

```bash
sudo python3 apps/pi_main.py \
	--mode pid_learned \
	--config-path comfig.txt \
	--onnx-path robot_MLP/pid_gain_model_compat.onnx \
	--base-speed 55 \
	--camera-fps 30
```

#### D. PC操作用サーバーモード（参考）

PCの `apps/pi_pc_app.py` から操作する場合は Pi 側を `remote` で待ち受けます。

```bash
sudo python3 apps/pi_main.py --mode remote --camera-fps 30
```

使用ポート: Camera `5556`, Motor `5555`

### 5. 停止方法

- ターミナルで `Ctrl+C`
- 例外終了時も `pi_main.py` 側でモーター停止とGPIO解放を実行します

### 6. うまく動かないとき

- `unrecognized arguments: --model ...`: 引数名は `--model` ではなく `--mode`。また ONNX ファイル名は単体で書かず `--onnx-path` の後ろに書く
- `ONNX model not found`: `--onnx-path` と実ファイル名を再確認
- 直進でその場回転する: `comfig.txt` の `motor_left_sign`, `motor_right_sign` を見直す
- ラインを見失いやすい: `line_color_space` を `lab` / `hsv` で切替、`auto_threshold_enabled=true` を試す、`far_threshold`, `near_threshold` を調整
- 動きが荒い/遅い: `base-speed`, `ai_control_interval_ms`, `line_process_interval_ms` を見直す

#comfig.txt(下地用)
# Config file for data collection
# 各PCで、このプロジェクトフォルダのパスを save_base_dir に設定してください
# Linux例: save_base_dir=/home/ryuryu/lab/Robot_car
# UNC例: save_base_dir=\\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car

save_base_dir=\\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car

# 速度基準（UI初期値）
drive_speed_default=60
# 十字キー速度の比率（drive_speed_defaultに対する倍率）
dpad_drive_speed_scale=1.0
dpad_turn_speed_scale=0.4
# カーブ時の自動減速感度（0.0:無効、推奨0.4〜1.0）
curve_slowdown_sensitivity=0.70
# 推論出力の平滑化（0.0:無効 / 1.0:強い追従）
ai_smoothing_alpha=0.80
# AI制御モデルの履歴フレーム数（学習時の --history と合わせること）
ai_history=5
# ライン見失い時の保持フレーム数（長いほど復帰が遅くなる）
ai_no_line_hold_frames=8
# ライン見失い後の減速に使うフレーム数
ai_no_line_brake_frames=10
# ライン処理周期(ms)。重いfind_lineを毎フレーム走らせないための間隔
line_process_interval_ms=60
# カメラ配信プロファイル（low_latency / high_quality）
# 低遅延重視は low_latency，画質重視は high_quality
camera_stream_profile=low_latency
# ライン抽出の色空間（lab または hsv）
line_color_space=lab
# ライン検出プロファイル（default=panel_seam_glare / panel_seam / glare / panel_seam_glare）
line_detection_profile=glare
# しきい値自動補正（見失いが多い場合は true 推奨）
auto_threshold_enabled=true
# 固定しきい値（auto_threshold_enabled=false のとき有効）
far_threshold=84
near_threshold=76
# AI制御周期(ms)。学習データ(0.1秒間隔)に合わせるなら100
ai_control_interval_ms=100
# UI更新周期(ms)
ui_update_interval_ms=30

# PID制御ゲイン（--mode pid で使用）
pid_kp=0.95
pid_ki=0.08
pid_kd=0.22
# PID出力の上限（操舵量）
pid_output_limit=0.35
# 積分項の上限（windup対策）
pid_integral_limit=1.5
# ライン未検出時に停止するか
pid_stop_on_no_line=true
# 学習ゲインの時間平滑化（0.0:平滑なし, 1.0:非常に強い追従）
pid_gain_smoothing_alpha=0.40
# 操舵量のフレーム間変化上限（小さいほど暴れにくい）
pid_steer_rate_limit=0.5
# モータ符号補正（直進でその場回転する場合は見直す）
motor_left_sign=-1
motor_right_sign=1

# カメラ選択肢
camera_id=camera_1
camera_id=camera_2
camera_id=camera_4
default_camera_id=camera_4

# 走行種別は drive_type= で複数行書けます
drive_type=straight
drive_type=recovery
drive_type=left_curve
drive_type=right_curve
drive_type=stop_and_go
