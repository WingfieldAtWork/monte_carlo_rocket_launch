"""Hardware-in-the-Loop (HITL) binary UDP telemetry packet broadcaster.

100 Hz UDP binary telemetry streaming serialized using big-endian struct:
Format: >d3f3f4f3fff
- Time: double (8 bytes)
- Position [x, y, z]: 3 floats (12 bytes)
- Velocity [vx, vy, vz]: 3 floats (12 bytes)
- Attitude Quaternion [q0, q1, q2, q3]: 4 floats (16 bytes)
- Body Rates [wx, wy, wz]: 3 floats (12 bytes)
- Instantaneous Mass: float (4 bytes)
- Dynamic Pressure: float (4 bytes)
Total Packet Size: 68 bytes
"""

from __future__ import annotations
import socket
import struct
import numpy as np

TELEMETRY_STRUCT_FMT = ">d3f3f4f3fff"
PACKET_SIZE = struct.calcsize(TELEMETRY_STRUCT_FMT)  # 68 bytes


class HITLTelemetryBroadcaster:
    """Streams vehicle states over UDP socket at configurable rate (default 100 Hz)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9099, enabled: bool = True):
        self.host = host
        self.port = int(port)
        self.enabled = enabled
        self._sock: socket.socket | None = None

        if self.enabled:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception as e:
                print(f"[Telemetry Warning] Failed to initialize UDP socket: {e}")
                self._sock = None

    @staticmethod
    def pack(
        t: float,
        pos_i: np.ndarray,
        vel_i: np.ndarray,
        quat: np.ndarray,
        rates_b: np.ndarray,
        mass: float,
        dyn_pressure: float,
    ) -> bytes:
        """Serialize state data into big-endian binary packet."""
        return struct.pack(
            TELEMETRY_STRUCT_FMT,
            float(t),
            float(pos_i[0]), float(pos_i[1]), float(pos_i[2]),
            float(vel_i[0]), float(vel_i[1]), float(vel_i[2]),
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
            float(rates_b[0]), float(rates_b[1]), float(rates_b[2]),
            float(mass),
            float(dyn_pressure),
        )

    @staticmethod
    def unpack(data: bytes) -> dict:
        """Deserialize big-endian binary packet into dictionary."""
        unpacked = struct.unpack(TELEMETRY_STRUCT_FMT, data)
        return {
            "time": unpacked[0],
            "pos": np.array(unpacked[1:4], dtype=np.float32),
            "vel": np.array(unpacked[4:7], dtype=np.float32),
            "quat": np.array(unpacked[7:11], dtype=np.float32),
            "rates": np.array(unpacked[11:14], dtype=np.float32),
            "mass": unpacked[14],
            "dyn_pressure": unpacked[15],
        }

    def broadcast(
        self,
        t: float,
        pos_i: np.ndarray,
        vel_i: np.ndarray,
        quat: np.ndarray,
        rates_b: np.ndarray,
        mass: float,
        dyn_pressure: float,
    ) -> bool:
        """Pack and transmit state over UDP socket. Returns True on success."""
        if not self.enabled or self._sock is None:
            return False

        try:
            packet = self.pack(t, pos_i, vel_i, quat, rates_b, mass, dyn_pressure)
            self._sock.sendto(packet, (self.host, self.port))
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close socket handle."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

