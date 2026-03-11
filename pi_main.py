# --- カメラ配信スレッド ---
def camera_thread():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.CONFLATE, 1)  # ★追加：古い映像を捨てて最新1枚だけを送る！
    sock.bind(f"tcp://*:{CAMERA_PORT}")
    
    picam2 = Picamera2()
    # ★変更：サイズを 320x240 に下げて通信量を劇的に軽くする！
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    print("カメラ配信スレッド起動（PCからの接続待機中...）")
    try:
        while running:
            img = picam2.capture_array()
            height, width = img.shape[:2]
            ndim = img.ndim

            data = [np.array([height]), np.array([width]), np.array([ndim]), img.data]
            sock.send_multipart(data)
            
            # ★追加：少しだけお休みを入れてWi-Fiのパンクを防ぐ (約30FPSに制限)
            time.sleep(0.03) 
            
    except Exception as e:
        print(f"カメラ処理でエラー: {e}")
    finally:
        picam2.stop()
        sock.close()