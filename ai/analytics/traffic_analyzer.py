"""
TrafficAnalyzer — Real-time traffic intelligence layer.
Computes:
  - per-lane vehicle counts
  - traffic density classification
  - vehicle type distribution
  - flow rate (vehicles/minute)
  - accurate real-world speed (km/h) via homography + Kalman smoothing
  - alert triggers

Speed Pipeline:
  det.center (pixels) -> homography (bird's-eye) -> real-world metres -> m/s -> km/h
  -> Kalman filter per track -> smoothed speed -> avg/lane speed

Detection contract (from detector.py):
    det.track_id   : int
    det.class_name : str
    det.center     : (int, int)   # set by Detection.__post_init__
    det.bbox       : [x1, y1, x2, y2]
    det.confidence : float
"""

import time
import logging
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# DENSITY DISPLAY COLOURS
# ─────────────────────────────────────────────────────────────────
DENSITY_COLORS = {
    "LOW":      (0, 255, 0),
    "MODERATE": (0, 165, 255),
    "HIGH":     (0, 0, 255),
    "CRITICAL": (128, 0, 128),
}


# ─────────────────────────────────────────────────────────────────
# HOMOGRAPHY CALIBRATION
#
# HOW TO CALIBRATE FOR YOUR CAMERA:
# 1. Pause your video on a clear frame.
# 2. Pick 4 road points whose real distances you know.
#    UK motorway: lane width = 3.65 m, dashed-line gap = 10 m.
# 3. Note their pixel coords (col, row).
# 4. Note their real-world ground coords (X_m, Y_m).
# 5. Replace the arrays below, OR put them in settings.yaml:
#
#   analytics:
#     homography:
#       src_pts: [[col,row], [col,row], [col,row], [col,row]]
#       dst_pts: [[X,Y],     [X,Y],     [X,Y],     [X,Y]   ]
#
# Calibrated for forward-facing motorway overbridge camera (~855x500 frame).
# Points measured from actual video frame — UK motorway, 3 lanes visible.
#
# To recalibrate for a different camera:
#   python calibrate_homography.py --video datasets/traffic.mp4
# ─────────────────────────────────────────────────────────────────

_DEFAULT_SRC_PTS = np.float32([
    [19, 318],    # near-left
    [613, 318],   # near-right
    [293, 129],   # far-left
    [359, 132],   # far-right
])

_DEFAULT_DST_PTS = np.float32([
    [0.0, 0.0],
    [3.65, 0.0],
    [0.0, 10.0],
    [3.65, 10.0],
])


def _find_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    try:
        import cv2
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("cv2.findHomography returned None")
        n = int(status.sum()) if status is not None else len(src)
        logger.info("Homography built (inliers=%d/%d)", n, len(src))
        return H
    except ImportError:
        pass

    # Pure-numpy SVD fallback (no cv2 required)
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src, dst)):
        A[2*i]   = [-sx, -sy, -1,   0,   0,  0, dx*sx, dx*sy, dx]
        A[2*i+1] = [  0,   0,  0, -sx, -sy, -1, dy*sx, dy*sy, dy]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    logger.info("Homography built via SVD fallback")
    return H.astype(np.float32)


def build_homography(
    src_pts: np.ndarray = _DEFAULT_SRC_PTS,
    dst_pts: np.ndarray = _DEFAULT_DST_PTS,
) -> np.ndarray:
    if len(src_pts) < 4:
        raise ValueError("Need at least 4 point pairs.")
    return _find_homography(src_pts, dst_pts)


def pixel_to_world(px: float, py: float, H: np.ndarray) -> Tuple[float, float]:
    try:
        import cv2
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)
        return float(world[0][0][0]), float(world[0][0][1])
    except ImportError:
        p = np.array([px, py, 1.0], dtype=np.float64)
        w = H.astype(np.float64) @ p
        return float(w[0] / w[2]), float(w[1] / w[2])


# ─────────────────────────────────────────────────────────────────
# 1-D KALMAN FILTER
# ─────────────────────────────────────────────────────────────────

class KalmanSpeed1D:
    """
    Scalar Kalman filter tracking one vehicle's speed.
    R = measurement noise (higher = smoother but slower to react)
    Q = process noise    (higher = reacts faster to real speed changes)
    """

    def __init__(self, initial_speed: float = 0.0, R: float = 25.0, Q: float = 1.0):
        self.x = initial_speed
        self.P = 100.0
        self.R = R
        self.Q = Q

    def update(self, z: float) -> float:
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1.0 - K)
        return self.x


# ─────────────────────────────────────────────────────────────────
# SPEED ESTIMATOR
# ─────────────────────────────────────────────────────────────────

@dataclass
class _TrackSpeedState:
    world_x: float
    world_y: float
    timestamp: float
    kalman: KalmanSpeed1D = field(default_factory=KalmanSpeed1D)
    smoothed_speed_kmh: float = 0.0
    raw_speed_kmh: float = 0.0
    speed_history: deque = field(default_factory=lambda: deque(maxlen=30))


class SpeedEstimator:
    """
    Converts per-frame pixel centroids to accurate km/h via homography + Kalman.

    Call estimator.update(track_id, cx, cy) every frame.
    Retrieve speed with estimator.get_speed(track_id).
    """

    def __init__(
        self,
        H: np.ndarray,
        min_dt: float = 0.033,
        max_dt: float = 2.0,
        max_speed_kmh: float = 220.0,
        min_displacement_m: float = 0.05,
    ):
        self.H = H
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.max_speed_kmh = max_speed_kmh
        self.min_displacement_m = min_displacement_m
        self._tracks: Dict[int, _TrackSpeedState] = {}

    def update(self, track_id: int, pixel_cx: float, pixel_cy: float) -> Optional[float]:
        wx, wy = pixel_to_world(pixel_cx, pixel_cy, self.H)
        now = time.time()

        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackSpeedState(world_x=wx, world_y=wy, timestamp=now)
            return None

        state = self._tracks[track_id]
        dt = now - state.timestamp

        # Track was lost too long — reset anchor
        if dt > self.max_dt:
            state.world_x, state.world_y, state.timestamp = wx, wy, now
            return state.smoothed_speed_kmh or None

        # Update came too fast — return cached value
        if dt < self.min_dt:
            return state.smoothed_speed_kmh or None

        dist_m = float(np.sqrt((wx - state.world_x)**2 + (wy - state.world_y)**2))

        if dist_m < self.min_displacement_m:
            # Stationary / jitter: decay speed gently toward 0
            smoothed = state.kalman.update(0.0)
            state.smoothed_speed_kmh = round(max(smoothed, 0.0), 1)
            state.world_x, state.world_y, state.timestamp = wx, wy, now
            return state.smoothed_speed_kmh

        speed_kmh = (dist_m / dt) * 3.6

        if speed_kmh > self.max_speed_kmh:
            # Impossible jump (tracker glitch) — reset anchor, keep last speed
            state.world_x, state.world_y, state.timestamp = wx, wy, now
            return state.smoothed_speed_kmh or None

        smoothed = state.kalman.update(speed_kmh)
        state.raw_speed_kmh = round(speed_kmh, 1)
        state.smoothed_speed_kmh = round(max(smoothed, 0.0), 1)
        state.speed_history.append(state.smoothed_speed_kmh)
        state.world_x, state.world_y, state.timestamp = wx, wy, now
        return state.smoothed_speed_kmh

    def get_speed(self, track_id: int) -> float:
        s = self._tracks.get(track_id)
        return s.smoothed_speed_kmh if s else 0.0

    def get_raw_speed(self, track_id: int) -> float:
        s = self._tracks.get(track_id)
        return s.raw_speed_kmh if s else 0.0

    def average_speed(self, track_ids: Optional[List[int]] = None) -> float:
        ids = track_ids if track_ids is not None else list(self._tracks.keys())
        speeds = [self._tracks[i].smoothed_speed_kmh for i in ids
                  if i in self._tracks and self._tracks[i].smoothed_speed_kmh > 0]
        return round(sum(speeds) / len(speeds), 1) if speeds else 0.0

    def median_speed(self, track_ids: Optional[List[int]] = None) -> float:
        ids = track_ids if track_ids is not None else list(self._tracks.keys())
        speeds = sorted(self._tracks[i].smoothed_speed_kmh for i in ids
                        if i in self._tracks and self._tracks[i].smoothed_speed_kmh > 0)
        if not speeds:
            return 0.0
        m = len(speeds) // 2
        return round(speeds[m] if len(speeds) % 2 else (speeds[m-1] + speeds[m]) / 2, 1)

    def purge_stale(self, max_age_s: float = 5.0) -> None:
        cutoff = time.time() - max_age_s
        for tid in [t for t, s in self._tracks.items() if s.timestamp < cutoff]:
            del self._tracks[tid]


# ─────────────────────────────────────────────────────────────────
# TRAFFIC ANALYZER
# ─────────────────────────────────────────────────────────────────

class TrafficAnalyzer:
    """
    Main analytics class. Drop-in replacement — same public API as before.

    Works directly with the Detection dataclass from detector.py:
        det.track_id   : int
        det.class_name : str
        det.center     : (int, int)   <- primary centroid source
        det.bbox       : [x1,y1,x2,y2] <- fallback
    """

    def __init__(self, cfg: dict):
        acfg = cfg.get("analytics", {})
        self.thresholds = acfg.get(
            "density_thresholds", {"low": 3, "medium": 7, "high": 12}
        )
        self.speed_alert_kmh: float = float(acfg.get("speed_alert_kmh", 20.0))

        hcfg = acfg.get("homography", {})
        if hcfg:
            self.H = build_homography(
                np.float32(hcfg["src_pts"]),
                np.float32(hcfg["dst_pts"]),
            )
        else:
            self.H = build_homography()

        self.speed_estimator = SpeedEstimator(self.H)
        self.flow_window: deque = deque(maxlen=60)

        self.lane_counts: Dict[str, int] = defaultdict(int)
        self.type_distribution: Dict[str, int] = defaultdict(int)
        self.active_track_ids: List[int] = []

        self.session_start = time.time()
        self._frame_count = 0

    # ── update ────────────────────────────────────────────────────

    def update(self, detections: list, lane_assignments: Dict[int, str]) -> None:
        """
        Call once per frame.
        detections       : List[Detection] from VehicleDetector.infer()
        lane_assignments : {track_id: lane_name} from LaneManager
        """
        self._frame_count += 1
        self.lane_counts = defaultdict(int)
        self.type_distribution = defaultdict(int)
        self.active_track_ids = []

        for det in detections:
            lane = lane_assignments.get(det.track_id, "ALL")
            self.lane_counts[lane] += 1
            self.type_distribution[det.class_name] += 1
            self.active_track_ids.append(det.track_id)

            # ── centroid resolution ───────────────────────────────
            # 1st choice: det.center  (tuple set by Detection.__post_init__)
            # 2nd choice: det.bbox    ([x1,y1,x2,y2] also on Detection)
            cx = cy = None

            if getattr(det, "center", None) is not None:
                cx, cy = float(det.center[0]), float(det.center[1])
            elif getattr(det, "bbox", None) is not None and len(det.bbox) == 4:
                cx = (float(det.bbox[0]) + float(det.bbox[2])) / 2.0
                cy = (float(det.bbox[1]) + float(det.bbox[3])) / 2.0
            elif hasattr(det, "x1"):
                cx = (float(det.x1) + float(det.x2)) / 2.0
                cy = (float(det.y1) + float(det.y2)) / 2.0

            if cx is None:
                logger.warning("Detection id=%s has no usable bbox — skipping speed", det.track_id)
                continue

            self.speed_estimator.update(det.track_id, cx, cy)

        self.flow_window.append((time.time(), len(detections)))

        if self._frame_count % 100 == 0:
            self.speed_estimator.purge_stale()

    # ── density ───────────────────────────────────────────────────

    def get_density_level(self, count: int) -> str:
        if count <= self.thresholds["low"]:
            return "LOW"
        elif count <= self.thresholds["medium"]:
            return "MODERATE"
        elif count <= self.thresholds["high"]:
            return "HIGH"
        return "CRITICAL"

    # ── flow rate ─────────────────────────────────────────────────

    def get_flow_rate(self) -> float:
        if len(self.flow_window) < 2:
            return 0.0
        elapsed = time.time() - self.flow_window[0][0]
        total = sum(v for _, v in self.flow_window)
        return round((total / elapsed) * 60, 2) if elapsed > 0 else 0.0

    # ── lane speeds ───────────────────────────────────────────────

    def get_lane_speeds(self, lane_assignments: Dict[int, str]) -> Dict[str, float]:
        lane_tids: Dict[str, List[int]] = defaultdict(list)
        for tid, lane in lane_assignments.items():
            lane_tids[lane].append(tid)
        return {lane: self.speed_estimator.average_speed(tids) for lane, tids in lane_tids.items()}

    # ── snapshot ──────────────────────────────────────────────────

    def to_dict(
        self,
        fps: float,
        lane_assignments: Optional[Dict[int, str]] = None,
    ) -> dict:
        total = sum(self.lane_counts.values())
        density = self.get_density_level(total)
        avg_speed    = self.speed_estimator.average_speed(self.active_track_ids)
        median_speed = self.speed_estimator.median_speed(self.active_track_ids)

        result = {
            "timestamp":          round(time.time(), 2),
            "fps":                round(fps, 2),
            "total_vehicles":     total,
            "density_level":      density,
            "flow_rate_per_min":  self.get_flow_rate(),
            "avg_speed_kmh":      avg_speed,
            "median_speed_kmh":   median_speed,
            "vehicle_speeds_kmh": {t: self.speed_estimator.get_speed(t) for t in self.active_track_ids},
            "lane_counts":        dict(self.lane_counts),
            "vehicle_types":      dict(self.type_distribution),
            "uptime_seconds":     round(time.time() - self.session_start, 1),
            "alert":              density in ("HIGH", "CRITICAL"),
            "speed_alert":        0 < avg_speed < self.speed_alert_kmh,
        }
        if lane_assignments:
            result["lane_speeds_kmh"] = self.get_lane_speeds(lane_assignments)
        return result


# ─────────────────────────────────────────────────────────────────
# OVERLAY HELPER
# ─────────────────────────────────────────────────────────────────

def draw_speed_overlay(frame, detections: list, analyzer: "TrafficAnalyzer"):
    """Draw per-vehicle speed labels + avg-speed banner. Call AFTER analyzer.update()."""
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not available — skipping overlay")
        return frame

    for det in detections:
        speed = analyzer.speed_estimator.get_speed(det.track_id)
        label = f"id:{det.track_id} {speed:.1f}km/h" if speed > 0 else f"id:{det.track_id} ---"
        color = (180, 180, 180) if speed <= 0 else ((0, 255, 100) if speed > 30 else (0, 180, 255))

        bbox = getattr(det, "bbox", None)
        lx = int(bbox[0]) if bbox else 0
        ly = int(bbox[1]) if bbox else 20
        cv2.putText(frame, label, (lx, max(ly - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    avg = analyzer.speed_estimator.average_speed(analyzer.active_track_ids)
    cv2.putText(frame, f"AVG SPEED: {avg:.1f} km/h", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────
# SELF-TEST  —  python traffic_analyzer.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from dataclasses import dataclass as _dc, field as _f

    @_dc
    class _Det:
        """Exact replica of your Detection dataclass."""
        track_id:   int
        class_id:   int
        class_name: str
        confidence: float
        bbox:       list
        center:     tuple = _f(init=False)

        def __post_init__(self):
            x1, y1, x2, y2 = self.bbox
            self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    print("=" * 60)
    print("TrafficAnalyzer self-test  (exact Detection dataclass)")
    print("=" * 60)

    analyzer = TrafficAnalyzer({"analytics": {
        "density_thresholds": {"low": 3, "medium": 7, "high": 12},
        "speed_alert_kmh": 20,
    }})

    # Simulate 5 vehicles at 80 km/h (31.2 px/frame at 30 FPS)
    # After Kalman warm-up (~10 frames) avg should read ~80 km/h
    vehicles = {10: [160.0, 330.0], 11: [220.0, 310.0], 12: [300.0, 300.0],
                13: [130.0, 345.0], 14: [420.0, 290.0]}

    FRAMES, FPS, PX = 90, 30, 31.2
    for fi in range(FRAMES):
        dets, lmap = [], {}
        for tid, pos in vehicles.items():
            pos[0] += PX
            x1, y1, x2, y2 = pos[0]-25, pos[1]-15, pos[0]+25, pos[1]+15
            dets.append(_Det(tid, 2, "bus" if tid == 12 else "car", 0.9, [x1,y1,x2,y2]))
            lmap[tid] = f"Lane_{(tid%2)+1}"
        analyzer.update(dets, lmap)
        time.sleep(1.0 / FPS)
        if (fi + 1) % 10 == 0:
            s = analyzer.to_dict(FPS, lmap)
            print(f"Frame {fi+1:3d} | avg={s['avg_speed_kmh']:.1f} km/h | "
                  f"median={s['median_speed_kmh']:.1f} km/h | density={s['density_level']}")

    s = analyzer.to_dict(FPS, lmap)
    print("\nPer-vehicle speeds:")
    for tid, spd in s["vehicle_speeds_kmh"].items():
        print(f"  track {tid}: {spd} km/h")
    print(f"\nLane speeds: {s.get('lane_speeds_kmh')}")
    result = "PASSED" if s["avg_speed_kmh"] > 60 else "FAILED"
    print(f"\nTest {result}  (avg={s['avg_speed_kmh']} km/h, expected ~80)")