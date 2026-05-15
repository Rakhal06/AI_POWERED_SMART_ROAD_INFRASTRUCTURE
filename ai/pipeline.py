"""
TrafficIntelligencePipeline — Main orchestrator.
Connects: Detector → Tracker → LaneManager → Analyzer → Logger → HUD
"""

import cv2
import time
import logging
import os
import yaml
import numpy as np
from collections import defaultdict

from ai.core.detector import VehicleDetector
from ai.core.lane_manager import LaneManager
from ai.core.tracker import VehicleTracker
from ai.analytics.traffic_analyzer import TrafficAnalyzer, DENSITY_COLORS
from ai.analytics.event_logger import EventLogger

logger = logging.getLogger(__name__)


# ─── LOGGING SETUP ────────────────────────────────────────────────────────────

def setup_logging(cfg: dict):
    log_dir = cfg["logging"]["log_dir"]

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"session_{int(time.time())}.log"
    )

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 60)
    logger.info("   AI Traffic Intelligence Engine — Starting")
    logger.info("=" * 60)


# ─── MAIN PIPELINE CLASS ──────────────────────────────────────────────────────

class TrafficIntelligencePipeline:

    def __init__(self, config_path: str = "config/settings.yaml"):

        # ── Load config ───────────────────────────────────────────
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found: {config_path}"
            )

        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        setup_logging(self.cfg)

        # ── Create output dirs ────────────────────────────────────
        os.makedirs(
            self.cfg["analytics"]["output_dir"],
            exist_ok=True
        )

        os.makedirs(
            self.cfg["logging"]["log_dir"],
            exist_ok=True
        )

        # ── Init modules ──────────────────────────────────────────
        self.detector = VehicleDetector(self.cfg)

        self.tracker = VehicleTracker(self.cfg)

        self.lane_mgr = LaneManager(
            self.cfg["lanes"]
        )

        self.analyzer = TrafficAnalyzer(self.cfg)

        self.event_logger = EventLogger(self.cfg)

        # ── Performance tracking ──────────────────────────────────
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = None

        logger.info("All modules initialized successfully.")

    # ─── CAMERA ───────────────────────────────────────────────────────────────

    def _open_camera(self):

        src = self.cfg["camera"]["source"]

        # ── SIMULATION MODE ──────────────────────────────────────
        if src == "simulation":

            from ai.simulation.mock_camera import MockCamera

            logger.info(
                "SIMULATION MODE — using MockCamera"
            )

            return MockCamera(
                width=self.cfg["camera"]["width"],
                height=self.cfg["camera"]["height"]
            )

        # ── WINDOWS CAMERA FIX ───────────────────────────────────
        import platform

        if (
            platform.system() == "Windows"
            and isinstance(src, int)
        ):
            cap = cv2.VideoCapture(
                src,
                cv2.CAP_DSHOW
            )
        else:
            cap = cv2.VideoCapture(src)

        # ── CAMERA SETTINGS ──────────────────────────────────────
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.cfg["camera"]["width"]
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.cfg["camera"]["height"]
        )

        # ── VALIDATION ───────────────────────────────────────────
        if not cap.isOpened():

            logger.error(
                f"Cannot open camera source: {src}"
            )

            raise RuntimeError(
                f"Camera source '{src}' unavailable."
            )

        actual_w = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_h = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        logger.info(
            f"Camera opened | source={src} | "
            f"resolution={actual_w}x{actual_h}"
        )

        return cap

    # ─── FPS ──────────────────────────────────────────────────────────────────

    def _compute_fps(self, prev_time: float) -> float:

        now = time.time()

        elapsed = now - prev_time

        return 1.0 / elapsed if elapsed > 0 else 0.0

    # ─── HUD RENDERER ─────────────────────────────────────────────────────────

    def _render_hud(
        self,
        frame: np.ndarray,
        analytics: dict,
        sudden_stops: list,
        wrong_way: list
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        density = analytics["density_level"]

        color = DENSITY_COLORS.get(
            density,
            (255, 255, 255)
        )

        # ── Top bar ──────────────────────────────────────────────
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 90),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.6,
            frame,
            0.4,
            0,
            frame
        )

        # ── Line 1 ───────────────────────────────────────────────
        cv2.putText(
            frame,
            f"VEHICLES: {analytics['total_vehicles']}  |  "
            f"DENSITY: {density}  |  "
            f"FLOW: {analytics['flow_rate_per_min']} veh/min  |  "
            f"FPS: {analytics['fps']}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        # ── Line 2 ───────────────────────────────────────────────
        type_str = "  ".join(
            f"{k.upper()}: {v}"
            for k, v in analytics.get(
                "vehicle_types",
                {}
            ).items()
        )

        cv2.putText(
            frame,
            type_str,
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 200, 200),
            1
        )

        # ── Line 3 ───────────────────────────────────────────────
        avg_speed = analytics.get(
            "avg_speed_kmh",
            0.0
        )

        n_tracks = analytics.get(
            "active_tracks",
            0
        )

        cv2.putText(
            frame,
            f"TRACKS: {n_tracks}  |  "
            f"AVG SPEED: {avg_speed} km/h  |  "
            f"UPTIME: "
            f"{int(analytics.get('uptime_seconds', 0))}s",
            (15, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1
        )

        # ── Lane counts ──────────────────────────────────────────
        y_off = 115

        for lane, cnt in analytics.get(
            "lane_counts",
            {}
        ).items():

            cv2.putText(
                frame,
                f"{lane}: {cnt} veh",
                (w - 210, y_off),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (0, 220, 255),
                1
            )

            y_off += 26

        # ── Speed badges ─────────────────────────────────────────
        speed_map = analytics.get(
            "speed_estimates_kmh",
            {}
        )

        for track_id, speed in speed_map.items():

            state = self.tracker.get_track_state(
                int(track_id)
            )

            if state and len(state.history) > 0:

                x, y, _ = state.history[-1]

                spd_color = (
                    (0, 255, 0)
                    if speed < 40 else
                    (0, 165, 255)
                    if speed < 70 else
                    (0, 0, 255)
                )

                cv2.putText(
                    frame,
                    f"{speed}km/h",
                    (int(x) + 5, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    spd_color,
                    1
                )

        # ── Wrong-way warnings ───────────────────────────────────
        for tid in wrong_way:

            state = self.tracker.get_track_state(tid)

            if state and len(state.history) > 0:

                x, y, _ = state.history[-1]

                cv2.putText(
                    frame,
                    "WRONG WAY!",
                    (int(x) - 20, int(y) - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

        # ── Sudden-stop markers ──────────────────────────────────
        for tid in sudden_stops:

            state = self.tracker.get_track_state(tid)

            if state and len(state.history) > 0:

                x, y, _ = state.history[-1]

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    14,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "STOP",
                    (int(x) - 16, int(y) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1
                )

        # ── Density alert ────────────────────────────────────────
        if analytics.get("alert"):

            banner_overlay = frame.copy()

            cv2.rectangle(
                banner_overlay,
                (0, h - 48),
                (w, h),
                (0, 0, 180),
                -1
            )

            cv2.addWeighted(
                banner_overlay,
                0.75,
                frame,
                0.25,
                0,
                frame
            )

            cv2.putText(
                frame,
                " WARNING: HIGH TRAFFIC DENSITY — ALERT ACTIVE",
                (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2
            )

        # ── Wrong-way system alert ───────────────────────────────
        if wrong_way:

            cv2.putText(
                frame,
                f"!! WRONG-WAY VEHICLE DETECTED — IDs: "
                f"{wrong_way}",
                (15, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 60, 255),
                2
            )

        return frame

    # ─── MAIN LOOP ────────────────────────────────────────────────────────────

    def run(self):

        cap = self._open_camera()

        prev_time = time.time()

        self.start_time = time.time()

        logger.info(
            "Pipeline running — press Q to stop."
        )

        try:

            while True:

                ret, frame = cap.read()

                if not ret:

                    logger.warning(
                        "Frame capture failed — retrying..."
                    )

                    time.sleep(0.05)

                    continue

                self.frame_count += 1

                # ── 1. DETECT ──────────────────────────────────
                detections, raw_result = \
                    self.detector.infer(frame)

                # ── 2. TRACK ───────────────────────────────────
                self.fps = self._compute_fps(prev_time)

                prev_time = time.time()

                self.tracker.update(
                    detections,
                    self.fps
                )

                # ── 3. LANE ASSIGNMENT ─────────────────────────
                lane_assignments = {}

                for det in detections:

                    lane_assignments[
                        det.track_id
                    ] = self.lane_mgr.assign_lane(
                        det.center
                    )

                # ── 4. ANALYZE ─────────────────────────────────
                self.analyzer.update(
                    detections,
                    lane_assignments
                )

                analytics = self.analyzer.to_dict(
                    self.fps
                )

                tracker_summary = self.tracker.summary()

                analytics.update(tracker_summary)

                # ── 5. ANOMALY CHECK ───────────────────────────
                sudden_stops = \
                    self.tracker.get_sudden_stops()

                wrong_way = \
                    self.tracker.get_wrong_way_vehicles(
                        expected_direction=
                        self.cfg.get(
                            "lanes",
                            {}
                        ).get(
                            "expected_direction",
                            "RIGHT"
                        )
                    )

                if sudden_stops:

                    logger.warning(
                        f"Sudden stop detected — "
                        f"track IDs: {sudden_stops}"
                    )

                if wrong_way:

                    logger.warning(
                        f"Wrong-way vehicle — "
                        f"track IDs: {wrong_way}"
                    )

                # ── 6. LOG ─────────────────────────────────────
                self.event_logger.maybe_log(
                    analytics
                )

                # ── 7. RENDER ──────────────────────────────────
                annotated = raw_result.plot()

                annotated = self.lane_mgr.draw_lanes(
                    annotated
                )

                annotated = self._render_hud(
                    annotated,
                    analytics,
                    sudden_stops,
                    wrong_way
                )

                cv2.imshow(
                    "AI Traffic Intelligence Engine — Smart RSU",
                    annotated
                )

                # ── 8. EXIT ────────────────────────────────────
                if cv2.waitKey(1) & 0xFF == ord('q'):

                    logger.info(
                        "Shutdown requested."
                    )

                    break

        except KeyboardInterrupt:

            logger.info(
                "KeyboardInterrupt — shutting down."
            )

        finally:

            cap.release()

            cv2.destroyAllWindows()

            elapsed = time.time() - self.start_time

            logger.info(
                f"Session ended | "
                f"frames={self.frame_count} | "
                f"duration={elapsed:.1f}s | "
                f"avg_fps={self.frame_count/elapsed:.1f}"
            )