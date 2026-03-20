"""PC(WSL)側で実行: キャリブ値の送信を確認するデバッグスクリプト
使い方: python debug_calib_send.py [Pi_IP] [値]
例:     python debug_calib_send.py 192.168.23.162 0.178
"""
import sys
import time
import zmq

PI_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.23.162"
VALUE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.178
CALIB_PORT = 5558

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.setsockopt(zmq.SNDHWM, 4)
sock.connect(f"tcp://{PI_IP}:{CALIB_PORT}")

print(f"[PC] {PI_IP}:{CALIB_PORT} に接続中...")
time.sleep(0.5)  # 接続確立を待つ

msg = {"calib_width_norm": VALUE}
try:
    sock.send_json(msg, flags=zmq.NOBLOCK)
    print(f"[PC] 送信成功: {msg}")
except zmq.ZMQError as e:
    print(f"[PC] 送信失敗: {e}")

time.sleep(0.3)  # 送信完了を待つ
sock.close()
ctx.term()
