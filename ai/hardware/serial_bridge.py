"""
serial_bridge.py — Non-blocking ESP32 hardware integration layer.

Reads JSON sensor packets from ESP32 over UART.
Runs in a background thread — never blocks the AI pipeline.

Expected JSON format from ESP32 firmware (115200 baud):
    {
      "vibration": false,
      "ultrasonic_cm": [120, 85, 200, 160],
      "radar_present": [true, false, true, false],
      "gps_lat": 13.0827,
      "gps_lon": 80.2707,
      "timestamp_ms": 45231
    }

Usage:
    bridge = SerialBridge(cfg)
    bridge.start()

    # In pipeline loop:
    signals = bridge.latest()
    if signals.vibration:
        anomaly_engine.flag_vibration()

    bridge.stop()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareSignals:
    """Snapshot of latest hardware sensor readings from ESP32."""
    vibration: bool = False
    ultrasonic_cm: List[float] = field(default_factory=lambda: [999.0] * 4)
    radar_present: List[bool] = field(default_factory=lambda: [False] * 4)
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    timestamp_ms: int = 0
    received_at: float = field(default_factory=time.time)
    # Derived: True if vibration was seen in the last N seconds
    vibration_recent: bool = False

    @property
    def has_gps(self) -> bool:
        return self.gps_lat != 0.0 or self.gps_lon != 0.0

    @property
    def gps_tuple(self):
        return (self.gps_lat, self.gps_lon) if self.has_gps else None

    def vehicle_count_by_ultrasonic(self, threshold_cm: float = 150.0) -> int:
        """Count lanes with a vehicle detected (distance below threshold)."""
        return sum(1 for d in self.ultrasonic_cm if d < threshold_cm)


# Sentinel: returned when hardware is disabled or no data received yet
_NULL_SIGNALS = HardwareSignals()


class SerialBridge:
    """
    Background serial reader for ESP32 sensor data.
    Thread-safe. Pipeline reads `bridge.latest()` each frame at zero cost.

    If hardware.enabled = false in settings.yaml, this class is a no-op:
    `latest()` always returns the null HardwareSignals object.
    """

    # How long a vibration flag stays "recent" after the last spike
    VIBRATION_DECAY_S: float = 2.0

    def __init__(self, cfg: dict):
        hw_cfg = cfg.get("hardware", {})
        self.enabled = hw_cfg.get("enabled", False)
        self.port = hw_cfg.get("port", "COM3")
        self.baud = hw_cfg.get("baud_rate", 115200)
        self.timeout = hw_cfg.get("timeout_s", 0.05)

        self._latest: HardwareSignals = HardwareSignals()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_vibration_time: float = 0.0

        if not self.enabled:
            logger.info("SerialBridge: hardware.enabled=false — running in software-only mode.")

    # ── Public API ────────────────────────────────────────────────

    def start(self) -> None:
        """Start background reader thread. No-op if hardware disabled."""
        if not self.enabled:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="serial-bridge"
        )
        self._thread.start()
        logger.info("SerialBridge started: port=%s baud=%d", self.port, self.baud)

    def stop(self) -> None:
        """Signal background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("SerialBridge stopped.")

    def latest(self) -> HardwareSignals:
        """
        Returns a copy of the latest hardware signals. Thread-safe.
        Returns HardwareSignals() (all defaults) if hardware is disabled.
        """
        if not self.enabled:
            return _NULL_SIGNALS
        with self._lock:
            sig = HardwareSignals(
                vibration=self._latest.vibration,
                ultrasonic_cm=list(self._latest.ultrasonic_cm),
                radar_present=list(self._latest.radar_present),
                gps_lat=self._latest.gps_lat,
                gps_lon=self._latest.gps_lon,
                timestamp_ms=self._latest.timestamp_ms,
                received_at=self._latest.received_at,
            )
        # Decay: vibration stays "recent" for VIBRATION_DECAY_S seconds
        sig.vibration_recent = (
            sig.vibration or
            (time.time() - self._last_vibration_time) < self.VIBRATION_DECAY_S
        )
        return sig

    def inject_test_vibration(self) -> None:
        """For simulation/testing: manually trigger a vibration signal."""
        with self._lock:
            self._latest.vibration = True
            self._last_vibration_time = time.time()
        logger.debug("SerialBridge: TEST vibration injected.")

    # ── Internal ──────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Background thread: continuously read lines from serial port."""
        try:
            import serial
        except ImportError:
            logger.error(
                "pyserial not installed. Run: pip install pyserial\n"
                "Falling back to software-only mode."
            )
            self.enabled = False
            return

        ser = None
        while not self._stop_event.is_set():
            try:
                if ser is None or not ser.is_open:
                    ser = serial.Serial(
                        self.port, self.baud,
                        timeout=self.timeout
                    )
                    logger.info("SerialBridge: connected to %s", self.port)

                line = ser.readline()
                if not line:
                    continue

                self._parse_and_store(line.decode("utf-8", errors="ignore").strip())

            except serial.SerialException as e:
                logger.warning("SerialBridge connection lost: %s — retrying in 2s", e)
                if ser:
                    try:
                        ser.close()
                    except Exception:
                        pass
                ser = None
                time.sleep(2.0)

            except Exception as e:
                logger.debug("SerialBridge parse error: %s", e)

        if ser and ser.is_open:
            ser.close()

    def _parse_and_store(self, line: str) -> None:
        """Parse a JSON line from ESP32 and update _latest."""
        if not line or not line.startswith("{"):
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return

        vib = bool(data.get("vibration", False))
        now = time.time()

        with self._lock:
            self._latest = HardwareSignals(
                vibration=vib,
                ultrasonic_cm=data.get("ultrasonic_cm", [999.0] * 4),
                radar_present=data.get("radar_present", [False] * 4),
                gps_lat=float(data.get("gps_lat", 0.0)),
                gps_lon=float(data.get("gps_lon", 0.0)),
                timestamp_ms=int(data.get("timestamp_ms", 0)),
                received_at=now,
            )
            if vib:
                self._last_vibration_time = now

        logger.debug(
            "HW: vib=%s ultra=%s gps=(%.4f,%.4f)",
            vib,
            self._latest.ultrasonic_cm,
            self._latest.gps_lat,
            self._latest.gps_lon,
        )