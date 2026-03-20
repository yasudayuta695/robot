"""Pi側で実行: キャリブ値の受信を確認するデバッグスクリプト
使い方: python debug_calib_recv.py
"""
import zmq

CALIB_PORT = 5558

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒タイムアウト

try:
    sock.bind(f"tcp://*:{CALIB_PORT}")
    print(f"[Pi] ポート {CALIB_PORT} で待機中...")

    while True:
        try:
            msg = sock.recv_json()
            print(f"[Pi] 受信: {msg}")
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                print("[Pi] タイムアウト（10秒間受信なし）")
            else:
                print(f"[Pi] エラー: {e}")
                break
except zmq.ZMQError as e:
    print(f"[Pi] バインド失敗 (port={CALIB_PORT}): {e}")
finally:
    sock.close()
    ctx.term()
