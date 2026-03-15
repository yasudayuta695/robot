import logging

import zmq


class MotorClient:
    def __init__(self, pi_ip: str, motor_port: int, logger: logging.Logger) -> None:
        self.logger = logger
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUSH)
        self._socket.connect(f"tcp://{pi_ip}:{motor_port}")
        self._current_cmd = "0,0"

    def send(self, left_speed: int, right_speed: int) -> None:
        cmd = f"{left_speed},{right_speed}"
        if cmd == self._current_cmd:
            return
        self._socket.send_string(cmd)
        self._current_cmd = cmd

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
