import zmq
import cv2
import struct
import numpy as np
import threading
import time
import socket
import os
import platform
import ctypes.util
import subprocess
import sys
import shutil

image = np.zeros((480, 640, 3), dtype=np.uint8)
running = True


def configure_qt_fontdir():
    # OpenCV's bundled Qt plugin may point to a non-existent fonts folder.
    if platform.system() != "Linux":
        return
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    candidates = [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/freefont",
        "/usr/share/fonts",
    ]
    for font_dir in candidates:
        if os.path.isdir(font_dir):
            os.environ["QT_QPA_FONTDIR"] = font_dir
            return


def ensure_cv2_qt_fonts_dir():
    # Some OpenCV Qt builds still probe cv2/qt/fonts directly and warn if missing.
    if platform.system() != "Linux":
        return

    cv2_dir = os.path.dirname(cv2.__file__)
    qt_fonts_dir = os.path.join(cv2_dir, "qt", "fonts")
    os.makedirs(qt_fonts_dir, exist_ok=True)

    # If any font already exists, nothing to do.
    if any(name.lower().endswith((".ttf", ".otf")) for name in os.listdir(qt_fonts_dir)):
        return

    source_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for src in source_fonts:
        if not os.path.isfile(src):
            continue
        dst = os.path.join(qt_fonts_dir, os.path.basename(src))
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return


def linux_gui_precheck():
    """Return (ok, reason). Runs checks that avoid hard crash in cv2 highgui."""
    if platform.system() != "Linux":
        return True, "GUI is available on non-Linux platform."

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, "No DISPLAY/WAYLAND display server found."

    missing = []
    for lib in ("SM", "ICE"):
        if ctypes.util.find_library(lib) is None:
            missing.append(lib)
    if missing:
        return False, f"Missing shared libraries: {', '.join(missing)}"

    # Probe cv2 highgui in a child process so parent won't crash on Qt/xcb errors.
    probe = [
        sys.executable,
        "-c",
        (
            "import cv2; "
            "cv2.namedWindow('probe', cv2.WINDOW_AUTOSIZE); "
            "cv2.destroyAllWindows(); "
            "print('ok')"
        ),
    ]
    result = subprocess.run(probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip().splitlines()
        hint = stderr[-1] if stderr else "cv2 highgui probe failed."
        return False, hint
    return True, "cv2 highgui probe passed."


def receiver_thread():
    global image
    # Open ZMQ Connection
    port = 5555
    conn_str = f"tcp://*:{port}"  # Connection String
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(conn_str)
    print("Receiver start.")

    count = 0
    while running:
        # Receve Data
        try:
            byte_rows, byte_cols, byte_mat_type, data = sock.recv_multipart(
                flags=zmq.NOBLOCK
            )
        except zmq.ZMQError:
            time.sleep(0.01)
            continue
        count += 1
        sock.send_string(f"ok {count}")

        # Convert byte to integer
        row = struct.unpack("q", byte_rows)[0]
        cols = struct.unpack("q", byte_cols)
        mat_type = struct.unpack("q", byte_mat_type)
        # Convert byte buffer to nparray
        if mat_type[0] == 0:  # Gray Scale
            image = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0]))
        else:  # BGR Color
            image = np.frombuffer(data, dtype=np.uint8).reshape((row, cols[0], 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def gui():
    # Initialize
    global running
    window_name = "receiver"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("Press [ESC] to quit.")

    # GUI loop
    while running:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(30)
        if key == 27:
            running = False

    # Closing
    cv2.destroyAllWindows()


if __name__ == "__main__":
    configure_qt_fontdir()
    ensure_cv2_qt_fonts_dir()

    ip = socket.gethostbyname(socket.gethostname())
    print(ip)

    thread1 = threading.Thread(target=receiver_thread)
    thread1.start()

    can_gui, reason = linux_gui_precheck()
    if can_gui:
        gui()
    else:
        print("GUI is not available, running headless.")
        print(f"Reason: {reason}")
        print("Install on Ubuntu/WSL: sudo apt update && sudo apt install -y libsm6 libice6")
        print("Press Ctrl+C to quit.")
        try:
            while running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            running = False

    thread1.join()
