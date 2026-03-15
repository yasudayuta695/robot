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