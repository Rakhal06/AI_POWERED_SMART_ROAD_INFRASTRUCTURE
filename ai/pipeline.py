"""
TrafficIntelligencePipeline — Main orchestrator.

v3 changes vs original:
  [FIX]  KeyError on cfg["lanes"] — now uses cfg.get("lanes", {})
  [FIX]  Single speed source — all speeds from TrafficAnalyzer.speed_estimator
         (tracker.py no longer computes speed; no duplicate/conflicting values)
  [NEW]  SerialBridge — non-blocking ESP32 hardware integration thread
  [NEW]  AnomalyEngine — multi-signal accident detection
         (camera aspect-ratio + sudden-stop + hw vibration scoring)
  [NEW]  Anomaly events logged immediately via EventLogger.log_anomaly_batch()
  [NEW]  Tripwire counter in HUD (from VehicleTracker)
  [NEW]  Hardware status indicator in HUD (shows live GPS + vibration state)
  [NEW]  Frame shape passed to tracker.update() for tripwire line-crossing

Data flow (per frame):
  cap.read()
    → AutoCalibrator (every N frames)
      → H injected into TrafficAnalyzer.speed_estimator.H
    → DetectorFactory (RT-DETR / YOLO / ONNX)
    → VehicleTracker.update(detections, frame_shape)
    → LaneManager.assign_lane() per detection
    → TrafficAnalyzer.update(detections, lane_assignments)
    → TrafficAnalyzer.get_speed_map()         ← single speed source
    → AnomalyEngine.update(
          detections, tracker_states, speed_map, hw_signals, frame_index
      )
    → EventLogger.maybe_log(analytics)
    → EventLogger.log_anomaly_batch(events)
    → HUD render → cv2.imshow()
"""

import cv2
import time
import logging
import os
import sys
import yaml
import numpy as np

from ai.core.detector_factory import DetectorFactory
from ai.core.tracker import VehicleTracker
from ai.core.lane_manager import LaneManager
from ai.core.anomaly_engine import AnomalyEngine
from ai.analytics.traffic_analyzer import TrafficAnalyzer, DENSITY_COLORS
from ai.analytics.event_logger import EventLogger
from ai.calibration.auto_calibrator import AutoCalibrator
from ai.hardware.serial_bridge import SerialBridge

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────

def setup_logging(cfg: dict) -> None:
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"session_{int(time.time())}.log")
    level    = getattr(logging, cfg.get("logging", {}).get("level", "INFO"))

    # Force UTF-8 on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    try:
        sh = logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w",
                 encoding="utf-8", errors="replace",
                 closefd=False, buffering=1)
        )
    except Exception:
        sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logger.info("=" * 64)
    logger.info("   Smart RSU — AI Traffic Intelligence Engine  Starting")
    logger.info("   IIT Madras National Road Safety Hackathon 2026")
    logger.info("=" * 64)


# ─────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────

class TrafficIntelligencePipeline:

    def __init__(self, config_path: str = "config/settings.yaml"):

        # ── Config ────────────────────────────────────────────────
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        setup_logging(self.cfg)

        # ── Output dirs ───────────────────────────────────────────
        os.makedirs(self.cfg["analytics"]["output_dir"], exist_ok=True)
        os.makedirs(self.cfg.get("logging", {}).get("log_dir", "logs"), exist_ok=True)

        # ── Detector ──────────────────────────────────────────────
        self.detector = DetectorFactory.create(self.cfg)
        self.detector.warmup()

        # ── Auto-calibrator ───────────────────────────────────────
        self.auto_calibrator = AutoCalibrator(self.cfg)
        self._show_cal_debug = (
            self.cfg.get("calibration", {}).get("debug_overlay", False)
            and self.cfg.get("display", {}).get("show_calibration_debug", False)
        )
        logger.info(
            "AutoCalibrator ready (mode=%s)",
            self.cfg.get("calibration", {}).get("mode", "hybrid"),
        )

        # ── Tracker ───────────────────────────────────────────────
        # [FIX] Use .get() — "lanes" key may not exist in settings.yaml
        self.tracker = VehicleTracker(self.cfg)

        # ── Lane manager ──────────────────────────────────────────
        # [FIX] Use .get() with default dict — was crashing on missing "lanes" key
        self.lane_mgr = LaneManager(self.cfg.get("lanes", {}))

        # ── Traffic analyzer ──────────────────────────────────────
        self.analyzer = TrafficAnalyzer(self.cfg)

        # ── Anomaly engine ────────────────────────────────────────
        # [NEW] Multi-signal accident + anomaly detection
        self.anomaly_engine = AnomalyEngine(self.cfg)

        # ── Event logger ──────────────────────────────────────────
        self.event_logger = EventLogger(self.cfg)

        # ── Hardware serial bridge ────────────────────────────────
        # [NEW] Non-blocking ESP32 reader thread
        self.hw_bridge = SerialBridge(self.cfg)

        # ── State ─────────────────────────────────────────────────
        self.fps         = 0.0
        self.frame_count = 0
        self.start_time  = None
        self._writer     = None

        logger.info("All modules initialized. Ready.")

    # ─── CAMERA ───────────────────────────────────────────────────

    def _open_camera(self):
        src = self.cfg["camera"]["source"]

        if src == "simulation":
            from ai.simulation.mock_camera import MockCamera
            logger.info("SIMULATION MODE — MockCamera active")
            res = self.cfg["camera"].get("resolution", [640, 360])
            return MockCamera(width=res[0], height=res[1])

        import platform
        if platform.system() == "Windows" and isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)

        res = self.cfg["camera"].get("resolution", [640, 360])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

        if not cap.isOpened():
            raise RuntimeError(f"Camera source '{src}' unavailable.")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera opened | %s | %dx%d", src, actual_w, actual_h)
        return cap

    # ─── VIDEO WRITER ─────────────────────────────────────────────

    def _init_writer(self, frame: np.ndarray) -> None:
        disp_cfg = self.cfg.get("display", {})
        if not disp_cfg.get("record_output", False):
            return
        out_path = disp_cfg.get("output_video_path", "outputs/recorded.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        h, w = frame.shape[:2]
        self._writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
        )
        logger.info("Recording → %s", out_path)

    # ─── FPS ──────────────────────────────────────────────────────

    def _compute_fps(self, prev_time: float) -> float:
        elapsed = time.time() - prev_time
        return 1.0 / elapsed if elapsed > 0 else 0.0

    # ─── CALIBRATION UPDATE ───────────────────────────────────────

    def _update_calibration(self, frame: np.ndarray):
        """Run auto-calibrator and push updated H into SpeedEstimator."""
        result = self.auto_calibrator.calibrate(frame)
        if result is not None and result.H is not None:
            self.analyzer.speed_estimator.H = result.H
        return result

    # ─── HUD RENDERER ─────────────────────────────────────────────

    def _render_hud(
        self,
        frame:          np.ndarray,
        analytics:      dict,
        anomaly_events: list,
        hw_signals,
        detections:     list = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── Top-left panel: core stats ────────────────────────────
        total   = analytics.get("total_vehicles", 0)
        density = analytics.get("density_level", "LOW")
        d_color = DENSITY_COLORS.get(density, (0, 255, 0))

        def put(text, y, color=(220, 220, 220), scale=0.58, thick=1):
            cv2.putText(
                frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA,
            )

        put(f"Vehicles  : {total}", 30, d_color, 0.65, 2)
        put(f"Density   : {density}", 55, d_color)
        put(f"Avg Speed : {analytics.get('avg_speed_kmh', 0):.1f} km/h", 78)
        put(f"Med Speed : {analytics.get('median_speed_kmh', 0):.1f} km/h", 101)
        put(f"Flow      : {analytics.get('flow_rate_per_min', 0):.1f} veh/min", 124)
        put(f"FPS       : {analytics.get('fps', 0):.1f}", 147)
        put(f"Tripwire  : {self.tracker.tripwire_count} crossed", 170)
        put(f"Uptime    : {analytics.get('uptime_seconds', 0):.0f}s", 193)

        # ── Top-right panel: lane breakdown ───────────────────────
        y_off = 30
        for lane, count in analytics.get("lane_counts", {}).items():
            lane_spd = analytics.get("lane_speeds_kmh", {}).get(lane, 0)
            cv2.putText(
                frame,
                f"{lane}: {count} veh  {lane_spd:.0f}km/h",
                (w - 230, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA,
            )
            y_off += 26

        # ── Hardware status panel ─────────────────────────────────
        if self.cfg.get("hardware", {}).get("enabled", False):
            hw_color = (0, 180, 255)
            vib_str  = "VIBRATION!" if hw_signals.vibration_recent else "ok"
            vib_col  = (0, 0, 255) if hw_signals.vibration_recent else (0, 200, 0)
            gps_str  = (
                f"{hw_signals.gps_lat:.4f},{hw_signals.gps_lon:.4f}"
                if hw_signals.has_gps else "no fix"
            )
            ult_str  = f"{hw_signals.vehicle_count_by_ultrasonic()} lanes active"
            y_hw = h - 90
            cv2.putText(frame, f"HW Vibration: {vib_str}", (10, y_hw),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, vib_col, 1, cv2.LINE_AA)
            cv2.putText(frame, f"HW Ultrasonic: {ult_str}", (10, y_hw + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, hw_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"GPS: {gps_str}", (10, y_hw + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, hw_color, 1, cv2.LINE_AA)

        # ── Per-vehicle speed badges (drawn on detection bbox) ────
        vehicle_speeds = analytics.get("vehicle_speeds_kmh", {})
        if detections:
            for det in detections:
                tid   = det.track_id
                speed = vehicle_speeds.get(tid, 0.0)
                bbox  = getattr(det, "bbox", None)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Speed colour: green < 40 | orange < 70 | red ≥ 70
                if speed <= 0:
                    spd_color = (160, 160, 160)
                    label = f"ID:{tid}"
                elif speed < 40:
                    spd_color = (0, 220, 0)
                    label = f"{speed:.0f} km/h"
                elif speed < 70:
                    spd_color = (0, 165, 255)
                    label = f"{speed:.0f} km/h"
                else:
                    spd_color = (0, 60, 255)
                    label = f"{speed:.0f} km/h !"

                # Measure text so we can draw a tight background pill
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                thickness  = 1
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Pill position: just above the top of the bbox
                px = x1
                py = max(y1 - 6, th + 4)

                # Dark background rectangle for readability
                cv2.rectangle(
                    frame,
                    (px - 2, py - th - 3),
                    (px + tw + 4, py + baseline),
                    (20, 20, 20), -1,
                )
                # Coloured text
                cv2.putText(
                    frame, label, (px + 1, py),
                    font, font_scale, spd_color, thickness, cv2.LINE_AA,
                )

        # ── Anomaly overlays ──────────────────────────────────────
        for ev in anomaly_events:
            if not ev.alert:
                continue
            if ev.location_px:
                px, py = ev.location_px
                cv2.circle(frame, (int(px), int(py)), 20, (0, 0, 255), 3)
                cv2.putText(
                    frame, ev.event_type,
                    (int(px) - 30, int(py) - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA,
                )

        # ── Wrong-way markers ─────────────────────────────────────
        wrong_way = self.tracker.get_wrong_way_vehicles(
            self.cfg.get("lanes", {}).get("expected_direction", "RIGHT")
        )
        for tid in wrong_way:
            state = self.tracker.get_track_state(tid)
            if state and len(state.history) > 0:
                x, y, _ = state.history[-1]
                cv2.putText(
                    frame, "WRONG WAY!",
                    (int(x) - 20, int(y) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
                )

        # ── Accident alert banner (most critical — full-width) ─────
        accident_alerts = [
            ev for ev in anomaly_events
            if ev.alert and ev.event_type == "ACCIDENT"
        ]
        if accident_alerts:
            ev = accident_alerts[0]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 56), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
            gps_part = (
                f"GPS:{ev.gps[0]:.4f},{ev.gps[1]:.4f}"
                if ev.gps else "GPS:N/A"
            )
            cv2.putText(
                frame,
                f" !! ACCIDENT DETECTED  conf={ev.confidence:.0f}%  {gps_part}",
                (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # ── Density banner (when no accident) ─────────────────────
        elif analytics.get("alert"):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 48), (w, h), (0, 0, 140), -1)
            cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
            cv2.putText(
                frame,
                " WARNING: HIGH TRAFFIC DENSITY — CONGESTION ALERT",
                (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # ── Wrong-way system alert ────────────────────────────────
        if wrong_way:
            cv2.putText(
                frame,
                f"!! WRONG-WAY VEHICLE — IDs: {wrong_way}",
                (15, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 60, 255), 2, cv2.LINE_AA,
            )

        return frame

    # ─── MAIN LOOP ────────────────────────────────────────────────

    def run(self) -> None:
        cap = self._open_camera()
        self.hw_bridge.start()   # starts background serial thread (no-op if disabled)

        prev_time        = time.time()
        self.start_time  = time.time()
        writer_init_done = False

        show_window = self.cfg.get("display", {}).get("show_window", True)

        logger.info("Pipeline running — press Q to stop.")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    src = self.cfg["camera"]["source"]
                    if isinstance(src, str) and os.path.isfile(src):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("Frame capture failed — retrying.")
                    time.sleep(0.05)
                    continue

                self.frame_count += 1
                frame_shape = frame.shape

                # ── 1. AUTO-CALIBRATION ────────────────────────────────
                cal_result = self._update_calibration(frame)

                # ── 2. DETECT ─────────────────────────────────────────
                detections, raw_result = self.detector.infer(frame)

                # ── 3. FPS ────────────────────────────────────────────
                self.fps  = self._compute_fps(prev_time)
                prev_time = time.time()

                # ── 4. TRACK ──────────────────────────────────────────
                # [FIX v3] Pass frame_shape for tripwire line-crossing
                self.tracker.update(detections, frame_shape=frame_shape)

                # ── 5. LANE ASSIGNMENT ────────────────────────────────
                lane_assignments = {
                    det.track_id: self.lane_mgr.assign_lane(det.center)
                    for det in detections
                }

                # ── 6. ANALYZE ────────────────────────────────────────
                self.analyzer.update(detections, lane_assignments)
                analytics = self.analyzer.to_dict(self.fps, lane_assignments)

                # Merge tracker summary (active_tracks, tripwire_count)
                analytics.update(self.tracker.summary())

                # ── 7. HARDWARE SIGNALS ───────────────────────────────
                # [NEW v3] Read latest ESP32 sensor data (non-blocking)
                hw_signals = self.hw_bridge.latest()

                # ── 8. ANOMALY DETECTION ──────────────────────────────
                # [NEW v3] Single authoritative speed map from TrafficAnalyzer
                speed_map = self.analyzer.get_speed_map()

                anomaly_events = self.anomaly_engine.update(
                    detections=detections,
                    tracker_states=self.tracker.get_all_states(),
                    speed_map=speed_map,
                    hw_signals=hw_signals,
                    frame_index=self.frame_count,
                )

                if anomaly_events:
                    alert_events = [e for e in anomaly_events if e.alert]
                    if alert_events:
                        for ev in alert_events:
                            logger.warning(
                                "ANOMALY | %s | track=%d | conf=%.0f | %s",
                                ev.event_type, ev.track_id,
                                ev.confidence, ev.message,
                            )

                # ── 9. LOG ────────────────────────────────────────────
                self.event_logger.maybe_log(analytics)
                self.event_logger.log_anomaly_batch(anomaly_events)

                # ── 10. RENDER ────────────────────────────────────────
                annotated = (
                    raw_result.plot()
                    if raw_result is not None and hasattr(raw_result, "plot")
                    else frame.copy()
                )

                annotated = self.lane_mgr.draw_lanes(annotated)

                if self._show_cal_debug and cal_result is not None:
                    annotated = self.auto_calibrator.draw_debug(annotated, cal_result)

                annotated = self._render_hud(
                    annotated, analytics, anomaly_events, hw_signals,
                    detections=detections,
                )

                # ── 11. DISPLAY ───────────────────────────────────────
                if show_window:
                    win_title = self.cfg.get("display", {}).get(
                        "window_title",
                        "Smart RSU — AI Traffic Intelligence",
                    )
                    cv2.imshow(win_title, annotated)

                # ── 12. RECORD ────────────────────────────────────────
                if not writer_init_done:
                    self._init_writer(annotated)
                    writer_init_done = True
                if self._writer is not None:
                    self._writer.write(annotated)

                # ── 13. EXIT ──────────────────────────────────────────
                if show_window and (cv2.waitKey(1) & 0xFF == ord("q")):
                    logger.info("Shutdown requested (Q key).")
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down.")

        finally:
            self.hw_bridge.stop()
            cap.release()
            if self._writer is not None:
                self._writer.release()
            if show_window:
                cv2.destroyAllWindows()

            elapsed = time.time() - (self.start_time or time.time())
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                "Session ended | frames=%d | duration=%.1fs | avg_fps=%.1f",
                self.frame_count, elapsed, avg_fps,
            )