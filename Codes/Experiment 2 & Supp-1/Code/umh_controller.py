import struct
import threading
import time

import serial
import serial.tools.list_ports


class UMHController:
    FRAME_HEADER = b"\xAA\x55"
    FRAME_TAIL = b"\x0D\x0A"

    CMD_ENABLE_DISABLE = 0x01
    CMD_PING = 0x02
    CMD_SET_STIMULATION = 0x05
    CMD_SET_DEMO = 0x07

    RSP_ACK = 0x80
    RSP_PING_ACK = 0x82
    RSP_SACK = 0x85
    RSP_DEMO_ACK = 0x86
    RSP_ERROR_CODE = 0xFF

    STIM_TYPE_POINT = 0
    STIM_TYPE_DISCRETE = 1
    STIM_TYPE_LINEAR = 2
    STIM_TYPE_CIRCULAR = 3

    def __init__(self):
        self.ser = None
        self.is_connected = False
        self.lock = threading.Lock()

    def find_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def find_device(self):
        for port in self.find_ports():
            try:
                self.connect(port)
                if self.ping():
                    return port
                self.disconnect()
            except Exception:
                self.disconnect()
        return None

    def connect(self, port):
        self.disconnect()
        self.ser = serial.Serial(
            port=port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5,
        )
        self.is_connected = True

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False
        self.ser = None

    def _calculate_checksum(self, cmd_type, data_length, data):
        checksum = cmd_type + data_length
        for byte in data:
            checksum += byte
        return checksum & 0xFF

    def _send_frame(self, cmd_type, data=None):
        if not self.is_connected:
            raise RuntimeError("Device not connected")
        payload = data or []
        checksum = self._calculate_checksum(cmd_type, len(payload), payload)
        frame = bytearray()
        frame.extend(self.FRAME_HEADER)
        frame.append(cmd_type)
        frame.append(len(payload))
        frame.extend(payload)
        frame.append(checksum)
        frame.extend(self.FRAME_TAIL)
        with self.lock:
            self.ser.write(frame)

    def _read_response(self, expected_cmd_type=None, timeout=1.0):
        if not self.is_connected:
            return None

        start_time = time.time()
        buffer = bytearray()
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                buffer.extend(self.ser.read(self.ser.in_waiting))
                while len(buffer) >= 7:
                    if buffer[:2] != self.FRAME_HEADER:
                        buffer.pop(0)
                        continue
                    data_len = buffer[3]
                    total_len = 2 + 1 + 1 + data_len + 1 + 2
                    if len(buffer) < total_len:
                        break
                    frame = buffer[:total_len]
                    cmd_type = frame[2]
                    data = frame[4:4 + data_len]
                    checksum = frame[4 + data_len]
                    tail = frame[5 + data_len:]
                    calc = self._calculate_checksum(cmd_type, data_len, data)
                    buffer = buffer[total_len:]
                    if calc != checksum or tail != self.FRAME_TAIL:
                        continue
                    if expected_cmd_type is None or cmd_type == expected_cmd_type:
                        return cmd_type, bytes(data)
            time.sleep(0.005)
        return None

    def ping(self):
        try:
            payload = [0x12, 0x34, 0x56]
            self._send_frame(self.CMD_PING, payload)
            response = self._read_response(self.RSP_PING_ACK)
            return bool(response and list(response[1]) == payload)
        except Exception:
            return False

    def enable_output(self, enable=True):
        self._send_frame(self.CMD_ENABLE_DISABLE, [1 if enable else 0])
        response = self._read_response()
        return bool(response and response[0] == self.RSP_ACK)

    def set_demo(self, demo_index):
        self._send_frame(self.CMD_SET_DEMO, [int(demo_index)])
        response = self._read_response(timeout=1.0)
        if not response:
            return None
        cmd, data = response
        if cmd == self.RSP_DEMO_ACK:
            return data.decode("utf-8", errors="ignore").strip("\x00")
        return None

    def scan_demo_names(self, max_demo_count=16):
        result = {}
        for idx in range(max_demo_count):
            name = self.set_demo(idx)
            if name:
                result[idx] = name
        self.enable_output(False)
        return result

    def _append_float32(self, payload, value):
        payload.extend(struct.pack("<f", float(value)))

    def _append_int32(self, payload, value):
        payload.extend(struct.pack("<i", int(value)))


    def set_discrete_stimulation(self, position, normal_vector, radius, segments, strength, frequency):
        payload = bytearray([self.STIM_TYPE_DISCRETE])
        for value in position:
            self._append_float32(payload, value)
        for value in normal_vector:
            self._append_float32(payload, value)
        self._append_float32(payload, radius)
        self._append_int32(payload, segments)
        self._append_float32(payload, strength)
        self._append_float32(payload, frequency)
        self._send_frame(self.CMD_SET_STIMULATION, payload)
        response = self._read_response(timeout=1.0)
        return bool(response and response[0] == self.RSP_SACK)

    def set_linear_stimulation(self, start_point, end_point, segments, strength, frequency):
        payload = bytearray([self.STIM_TYPE_LINEAR])
        for value in list(start_point) + list(end_point):
            self._append_float32(payload, value)
        self._append_int32(payload, segments)
        self._append_float32(payload, strength)
        self._append_float32(payload, frequency)
        self._send_frame(self.CMD_SET_STIMULATION, payload)
        response = self._read_response(timeout=1.0)
        return bool(response and response[0] == self.RSP_SACK)

    def set_circular_stimulation(self, position, normal_vector, radius, strength, frequency):
        payload = bytearray([self.STIM_TYPE_CIRCULAR])
        for value in position:
            self._append_float32(payload, value)
        for value in normal_vector:
            self._append_float32(payload, value)
        self._append_float32(payload, radius)
        self._append_float32(payload, strength)
        self._append_float32(payload, frequency)
        self._send_frame(self.CMD_SET_STIMULATION, payload)
        response = self._read_response(timeout=1.0)
        return bool(response and response[0] == self.RSP_SACK)

    def play_condition(self, condition):
        condition_type = condition.get("type")
        label = condition.get("label", "condition")


        if condition_type == "linear":
            ok = self.set_linear_stimulation(
                condition["start_point"],
                condition["end_point"],
                condition.get("segments", 1),
                condition["strength"],
                condition["frequency"],
            )
            if not ok:
                raise RuntimeError(f"Failed to configure linear stimulation for {label}")
            return True

        if condition_type == "discrete":
            ok = self.set_discrete_stimulation(
                condition["position"],
                condition["normal_vector"],
                condition["radius"],
                condition.get("segments", 1),
                condition["strength"],
                condition["frequency"],
            )
            if not ok:
                raise RuntimeError(f"Failed to configure discrete stimulation for {label}")
            return True

        if condition_type == "circular":
            ok = self.set_circular_stimulation(
                condition["position"],
                condition["normal_vector"],
                condition["radius"],
                condition["strength"],
                condition["frequency"],
            )
            if not ok:
                raise RuntimeError(f"Failed to configure circular stimulation for {label}")
            return True

        raise RuntimeError(f"Unsupported condition type: {condition_type}")
