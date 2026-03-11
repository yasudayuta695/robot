import zmq
import numpy as np
from picamera2 import Picamera2

conn_str="tcp://192.168.23.204:5555"

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect(conn_str)

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)

picam2.start()

try:
    while True:
        img = picam2.capture_array()
        height, width = img.shape[:2]
        ndim = img.ndim

        data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), img.data ]
        sock.send_multipart(data)
        recv = sock.recv_string()
        print('recv:', recv)
except KeyboardInterrupt:
    print('quit')

picam2.stop()

