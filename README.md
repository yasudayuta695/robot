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