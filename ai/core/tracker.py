"""
VehicleTracker — Trajectory history and speed/direction analysis.

Wraps ByteTrack (built into YOLOv8 via model.track(persist=True)).
This module manages per-track state AFTER YOLO assigns IDs:
  - stores centroid history per track_id
  - computes speed estimate from pixel displacement
  - computes dominant motion direction
  - detects sudden stops
  - detects wrong-way movement
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Direction buckets (degrees, 0 = right, 90 = up, 180 = left, 270 = down)
DIRECTION_LABELS = {
    (0,   45):  "RIGHT",
    (45,  135): "UP",
    (135, 225): "LEFT",
    (225, 315): "DOWN",
    (315, 360): "RIGHT",
}


def _angle_deg(dx: float, dy: float) -> float:
    """Returns angle in degrees [0, 360) from pixel displacement vector."""
    angle = math.degrees(math.atan2(-dy, dx))  # negate dy: pixel y grows downward
    return angle % 360


def _direction_label(angle: float) -> str:
    for (lo, hi), label in DIRECTION_LABELS.items():
        if lo <= angle < hi:
            return label
    return "RIGHT"


@dataclass
class TrackState:
    """Per-vehicle persistent state."""
    track_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=30))  # (x, y, timestamp_idx)
    speed_kmh: float = 0.0
    direction_angle: float = 0.0
    direction_label: str = "UNKNOWN"
    is_stopped: bool = False
    wrong_way: bool = False
    age_frames: int = 0          # how many frames this track has existed
    last_seen_frame: int = 0


class VehicleTracker:
    """
    Manages per-vehicle trajectory state using YOLO ByteTrack IDs.

    Usage:
        tracker = VehicleTracker(cfg)
        # After detector.infer(frame) returns detections with track_ids:
        tracker.update(detections, frame_index, fps)
        speed = tracker.get_speed(track_id)
        direction = tracker.get_direction(track_id)
        stops = tracker.get_sudden_stops()
        wrong_way = tracker.get_wrong_way_vehicles(expected_direction="RIGHT")
    """

    def __init__(self, cfg: dict):
        """
        cfg: the full settings.yaml dict.
        Uses camera.fps_target and analytics.density_thresholds for calibration.
        """
        # Pixel-to-meter calibration.
        # Default: assume lane width = 3.5m visible across half frame width (640/2 = 320px)
        # Override in settings.yaml under tracker.pixels_per_meter
        tracker_cfg = cfg.get("tracker", {})
        self.pixels_per_meter: float = tracker_cfg.get("pixels_per_meter", 320 / 3.5)
        self.stop_speed_threshold: float = tracker_cfg.get("stop_speed_kmh", 3.0)
        self.wrong_way_angle_gap: float = tracker_cfg.get("wrong_way_angle_gap_deg", 150.0)
        self.min_track_age: int = tracker_cfg.get("min_track_age_frames", 5)

        self._tracks: Dict[int, TrackState] = {}
        self._frame_index: int = 0

        logger.info(
            f"VehicleTracker initialized | "
            f"px/m={self.pixels_per_meter:.1f} | "
            f"stop_threshold={self.stop_speed_threshold} km/h"
        )

    # ── PUBLIC API ────────────────────────────────────────────────

    def update(self, detections: list, fps: float) -> None:
        """
        Call once per frame with the current list of Detection objects.
        detections: List[Detection] from VehicleDetector.infer()
        fps: current measured FPS (used for speed calculation)
        """
        self._frame_index += 1
        active_ids = set()

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue  # untracked detection (ByteTrack failed to assign)

            active_ids.add(tid)

            if tid not in self._tracks:
                self._tracks[tid] = TrackState(track_id=tid)
                logger.debug(f"New track: ID={tid}")

            state = self._tracks[tid]
            state.history.append((det.center[0], det.center[1], self._frame_index))
            state.age_frames += 1
            state.last_seen_frame = self._frame_index

            # Only analyze tracks old enough to have meaningful history
            if state.age_frames >= self.min_track_age and len(state.history) >= 2:
                self._compute_kinematics(state, fps)

        # Mark stale tracks (not seen for >30 frames) for cleanup
        stale = [
            tid for tid, st in self._tracks.items()
            if self._frame_index - st.last_seen_frame > 30
        ]
        for tid in stale:
            logger.debug(f"Track expired: ID={tid}")
            del self._tracks[tid]

    def get_speed(self, track_id: int) -> float:
        """Returns latest speed estimate in km/h for a track ID. 0.0 if unknown."""
        state = self._tracks.get(track_id)
        return state.speed_kmh if state else 0.0

    def get_direction(self, track_id: int) -> str:
        """Returns direction label: RIGHT / LEFT / UP / DOWN / UNKNOWN."""
        state = self._tracks.get(track_id)
        return state.direction_label if state else "UNKNOWN"

    def get_all_speeds(self) -> Dict[int, float]:
        """Returns {track_id: speed_kmh} for all active tracks."""
        return {
            tid: st.speed_kmh
            for tid, st in self._tracks.items()
            if st.age_frames >= self.min_track_age
        }

    def get_sudden_stops(self) -> List[int]:
        """
        Returns list of track IDs that just stopped (were moving, now stopped).
        Useful for accident detection.
        """
        return [
            tid for tid, st in self._tracks.items()
            if st.is_stopped and st.age_frames >= self.min_track_age
        ]

    def get_wrong_way_vehicles(self, expected_direction: str = "RIGHT") -> List[int]:
        """
        Returns track IDs moving against the expected traffic direction.
        expected_direction: one of RIGHT / LEFT / UP / DOWN
        """
        expected_angle = {"RIGHT": 0, "UP": 90, "LEFT": 180, "DOWN": 270}.get(
            expected_direction.upper(), 0
        )
        wrong = []
        for tid, st in self._tracks.items():
            if st.age_frames < self.min_track_age:
                continue
            angle_diff = abs(st.direction_angle - expected_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > self.wrong_way_angle_gap:
                st.wrong_way = True
                wrong.append(tid)
            else:
                st.wrong_way = False
        return wrong

    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def active_track_count(self) -> int:
        return len(self._tracks)

    # ── INTERNAL ──────────────────────────────────────────────────

    def _compute_kinematics(self, state: TrackState, fps: float) -> None:
        """
        Computes speed and direction from the last 2 history points.
        Speed = pixel displacement / fps * pixels_per_meter * 3.6
        Direction = atan2 of displacement vector.
        """
        pts = list(state.history)
        if len(pts) < 2:
            return

        # Use latest two points
        x1, y1, f1 = pts[-2]
        x2, y2, f2 = pts[-1]

        dx = x2 - x1
        dy = y2 - y1
        frame_gap = max(f2 - f1, 1)

        # Pixel distance per frame → meters per second → km/h
        pixel_dist = math.hypot(dx, dy)
        meters_per_frame = pixel_dist / self.pixels_per_meter
        speed_ms = meters_per_frame * fps / frame_gap
        state.speed_kmh = round(speed_ms * 3.6, 1)

        # Direction
        state.direction_angle = _angle_deg(dx, dy)
        state.direction_label = _direction_label(state.direction_angle)

        # Sudden stop: was moving (>5 km/h), now below threshold
        state.is_stopped = state.speed_kmh < self.stop_speed_threshold

    def summary(self) -> dict:
        """Returns a JSON-serializable summary for analytics output."""
        speeds = self.get_all_speeds()
        avg_speed = round(sum(speeds.values()) / len(speeds), 1) if speeds else 0.0
        return {
            "active_tracks": self.active_track_count(),
            "speed_estimates_kmh": speeds,
            "avg_speed_kmh": avg_speed,
            "sudden_stops": self.get_sudden_stops(),
        }