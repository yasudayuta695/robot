# --- カメラ受信スレッド ---
def receiver_thread():
    global image_data, running
    
    ctx = zmq.Context()
    cam_socket = ctx.socket(zmq.PULL)
    cam_socket.setsockopt(zmq.CONFLATE, 1) # ★追加：渋滞している古い映像を読み捨てる！
    cam_socket.connect(f"tcp://{PI_IP}:{CAMERA_PORT}")
    print("カメラ受信スレッド起動...")

    while running:
        try:
            byte_rows, byte_cols, byte_mat_type, data = cam_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            time.sleep(0.01)
            continue
            
        row = struct.unpack("q", byte_rows)[0]
        cols = struct.unpack("q", byte_cols)
        mat_type = struct.unpack("q", byte_mat_type)
        
        if mat_type[0] == 0:
            img = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0]))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0], 3))
            # ★削除：色が変な原因だった「cvtColor」の行を丸ごと消しました！
            
        # ★追加：ラズパイから来た軽い映像(320x240)を、画面サイズ(640x480)に引き伸ばす
        img = cv2.resize(img, (640, 480))
        image_data = img