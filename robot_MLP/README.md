# robot_MLP

Minimal training/evaluation workflow for line-trace motor control MLP.

This folder contains:
- `train.py`: Train the model (`.pt`) and export ONNX (`.onnx`)
- `test.py`: Evaluate a trained `.pt` model on CSV and print metrics

## 1. What The Model Learns

- Task: regression (`target_left_norm`, `target_right_norm`)
- Input per frame: 9 values
- Time window: 10 frames
- Final input size: `90` (=`9 * 10`)
- Output size: `2`

`train.py` uses edge padding for early frames (replicate frame 0), not zero padding.

## 2. Required CSV Columns

The CSV must include at least these columns:

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

These match the recorder output format used in `apps/recorder.py`.

## 3. Environment Setup

Example (venv):

```bash
cd /home/cr7_yas97/ss2025robot
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy matplotlib onnx onnxscript
```

Notes:
- `matplotlib` is optional (only for learning curve image).
- `onnx` and `onnxscript` are required for ONNX export.
- GPU is auto-used when available unless `--device cpu` is specified.

## 4. Train

Run from project root or from `robot_MLP/`.

### Yasuda Example (bash)

```bash
python /home/cr7_yas97/ss2025robot/robot_MLP/train.py \
  --csv-path /home/cr7_yas97/ss2025robot/dataset/camera_1 ^
  --holdout-unit hour ^
  --split-report-path /home/cr7_yas97/ss2025robot/dataset_split.json ^
  --epochs 80 \
  --batch-size 64 \
  --lr 1e-3 \
  --history 10 \
  --val-ratio 0.1 \
  --early-stopping \
  --patience 10 \
  --min-delta 1e-4 \
  --weights-path /home/cr7_yas97/ss2025robot/robot_MLP/model.pt \
  --onnx-path /home/cr7_yas97/ss2025robot/robot_MLP/model.onnx \
  --curve-path /home/cr7_yas97/ss2025robot/robot_MLP/learning_curve.png
```

Main outputs:
- `model.pt` (PyTorch state_dict)
- `model.onnx` (for C++ / OpenCV DNN runtime)
- `learning_curve.png` (if not disabled)

You can also pass a directory to `--csv-path`.
In that case, `train.py` recursively collects all `driving_log.csv` files under it.

### Ryuryu Example (Windows, all sessions under camera_1)

```bat
python \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\train.py ^
  --csv-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\dataset\camera_1 ^
  --holdout-unit hour ^
  --split-report-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\dataset_split.json ^
  --epochs 80 ^
  --batch-size 64 ^
  --lr 1e-3 ^
  --history 10 ^
  --val-ratio 0.1 ^
  --test-ratio 0.15 ^
  --early-stopping ^
  --patience 10 ^
  --min-delta 1e-4 ^
  --weights-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\model.pt ^
  --onnx-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\model.onnx ^
  --curve-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\learning_curve.png
```

Useful options:
- `--val-ratio 0`: train without validation split
- `--no-curve`: skip curve image generation
- `--no-restore-best`: keep final epoch model instead of best validation model
- `--test-ratio 0.15`: hold out unseen sessions for final test
- `--split-report-path dataset_split.json`: save train/val/test CSV assignment
- `--holdout-unit hour`: hold out by hour block (same hour sessions stay together)

Session split behavior (implemented):
- Split unit is session directory (not frame).
- Split is stratified by `drive_type` folder.
- `test` split is never used in training.
- With `--holdout-unit hour`, same-hour sessions are never split across train/val/test.

## 5. Test / Evaluate

### Yasuda Example (bash)

```bash
python /home/cr7_yas97/ss2025robot/robot_MLP/test.py \
  --split-report-path /home/cr7_yas97/ss2025robot/dataset_split.json ^
  --split-name test ^
  --csv-path /path/to/test_driving_log.csv \
  --weights-path /home/cr7_yas97/ss2025robot/robot_MLP/model.pt \
  --history 10 \
  --hidden1 64 \
  --hidden2 32 \
  --pred-csv-path /home/cr7_yas97/ss2025robot/robot_MLP/test_predictions.csv
```

### Ryuryu Example (Windows, evaluate all CSVs under camera_1)

```bat
python \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\test.py ^
  --csv-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\dataset\camera_1 ^
  --weights-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\model.pt ^
  --history 10 ^
  --hidden1 64 ^
  --hidden2 32 ^
  --pred-csv-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\test_predictions.csv
```

### Ryuryu Recommended (Windows, unseen held-out test split)

```bat
python \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\test.py ^
  --split-report-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\dataset_split.json ^
  --split-name test ^
  --weights-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\model.pt ^
  --history 10 ^
  --hidden1 64 ^
  --hidden2 32 ^
  --pred-csv-path \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\robot_MLP\test_predictions.csv
```

  
Printed metrics:
- MSE (overall/left/right)
- MAE (overall/left/right)
- RMSE (overall/left/right)

Important:
- `--hidden1` and `--hidden2` must match training architecture.
- `--history` must match training.

## 6. Deploy To Robot App (ONNX)

`apps/pi_pc_app.py` includes an `AI (ONNX)` mode.

Expected default model path:
- `/home/cr7_yas97/ss2025robot/robot_MLP/model.onnx`

Steps:
1. Train and export `model.onnx` to `robot_MLP/`.
2. Launch app:

### Yasuda Example (bash)

```bash
python /home/cr7_yas97/ss2025robot/apps/pi_pc_app.py

python \\wsl.localhost\Ubuntu-24.04\home\ryuryu\lab\Robot_car\apps\pi_pc_app.py
```


3. Switch control mode to `AI (ONNX)`.
4. Load ONNX model (auto-load attempts default path).
5. Use emergency stop if needed (`Space` or `Esc`, hold to force `0,0`).

## 7. Recommended Workflow

1. Train with validation (`--val-ratio 0.1`) and early stopping.
2. Evaluate with `test.py` and inspect errors.
3. Tune hyperparameters and retrain.
4. Export ONNX and test in `AI (ONNX)` mode.
5. Record new driving data and iterate.

## 8. Troubleshooting

- `ModuleNotFoundError: torch`
  - Install dependencies in active venv.
- ONNX loads but robot does not move
  - Check if line features are detected.
  - Verify model path and architecture consistency.
  - Confirm motor server (`pi_main.py`) is running and reachable.
- Bad early behavior at startup
  - This is expected to be sensitive in first few frames; history is edge-padded.
