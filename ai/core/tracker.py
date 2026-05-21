"""
tracker.py — Vehicle trajectory state manager.

DESIGN CHANGE (v2):
  Speed computation has been REMOVED from this module.
  TrafficAnalyzer owns all speed estimation via homography + Kalman filter.
  This module owns:
    - Centroid trajectory history (for direction + AnomalyEngine)
    - ByteTrack ID management (via YOLO's built-in tracker)
    - Direction analysis (RIGHT / LEFT / UP / DOWN / UNKNOWN)
    - Wrong-way flag (set by AnomalyEngine or pipeline based on analyzer speed)
    - Line-crossing vehicle counter (virtual tripwire)
    - Stale track cleanup

  Speed-dependent anomaly detection (sudden stop) is now in AnomalyEngine,
  which receives speeds directly from TrafficAnalyzer.speed_estimator.

Usage:
    tracker = VehicleTracker(cfg)

    # Each frame:
    tracker.update(detections)
    direction = tracker.get_direction(track_id)
    wrong_way = tracker.get_wrong_way_vehicles(expected_direction="RIGHT")
    count_crossed = tracker.tripwire_count
    all_states = tracker.get_all_states()
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Direction buckets (angle in degrees: 0=right, 90=up, 180=left, 270=down)
_DIRECTION_MAP = {
    (0,   45):  "RIGHT",
    (45,  135): "UP",
    (135, 225): "LEFT",
    (225, 315): "DOWN",
    (315, 361): "RIGHT",
}


def _angle_deg(dx: float, dy: float) -> float:
    """Angle [0, 360) from a pixel displacement vector. Pixel Y grows downward."""
    return math.degrees(math.atan2(-dy, dx)) % 360


def _dir_label(angle: float) -> str:
    for (lo, hi), label in _DIRECTION_MAP.items():
        if lo <= angle < hi:
            return label
    return "RIGHT"


# ─────────────────────────────────────────────────────────────────
# PER-TRACK STATE
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Trajectory + direction state for one tracked vehicle."""
    track_id:       int
    # (x, y, frame_index) — latest first via appendleft
    history:        deque = field(default_factory=lambda: deque(maxlen=30))
    direction_angle: float = 0.0
    direction_label: str   = "UNKNOWN"
    wrong_way:      bool   = False
    age_frames:     int    = 0
    last_seen_frame: int   = 0
    # tripwire: True once this track has crossed the virtual line
    crossed_tripwire: bool = False


# ─────────────────────────────────────────────────────────────────
# LINE-CROSSING COUNTER (virtual tripwire)
# ─────────────────────────────────────────────────────────────────

class TripwireCounter:
    """
    Counts vehicles that cross a horizontal line at y = frame_height * y_pct.
    A vehicle "crosses" when its centroid transitions from above to below the line.
    Only counts each track_id once.
    """

    def __init__(self, y_pct: float = 0.75):
        self.y_pct  = y_pct            # fraction of frame height
        self._count = 0
        self._seen:  Dict[int, bool] = {}  # track_id -> was_above

    def update(self, track_id: int, cy: float, frame_h: int) -> bool:
        """Returns True if this call triggered a new crossing."""
        line_y = frame_h * self.y_pct
        above  = cy < line_y
        was_above = self._seen.get(track_id, above)  # assume no crossing on first frame

        if was_above and not above:
            # Crossed from above to below — count it
            self._count += 1
            self._seen[track_id] = above
            logger.debug("Tripwire crossed by track %d | total=%d", track_id, self._count)
            return True

        self._seen[track_id] = above
        return False

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0
        self._seen.clear()


# ─────────────────────────────────────────────────────────────────
# VEHICLE TRACKER
# ─────────────────────────────────────────────────────────────────

class VehicleTracker:
    """
    Manages per-vehicle trajectory state on top of ByteTrack IDs
    assigned by the YOLO/RT-DETR detector.

    This module does NOT compute speed — that belongs to TrafficAnalyzer.
    """

    def __init__(self, cfg: dict):
        tracker_cfg = cfg.get("tracker", {})
        self.min_track_age          = tracker_cfg.get("min_track_age_frames", 5)
        self.wrong_way_angle_gap    = float(tracker_cfg.get("wrong_way_angle_gap_deg", 150.0))
        self._direction_history_len = 10   # frames of direction smoothing

        # Tripwire
        tripwire_cfg = cfg.get("lanes", {}).get("tripwire", {})
        tw_enabled   = tripwire_cfg.get("enabled", False)
        tw_y_pct     = float(tripwire_cfg.get("y_pct", 0.75))
        self.tripwire = TripwireCounter(y_pct=tw_y_pct) if tw_enabled else None

        self._tracks:       Dict[int, TrackState] = {}
        self._frame_index:  int = 0

        logger.info(
            "VehicleTracker v2 | min_age=%d | wrong_way_gap=%.0f° | tripwire=%s",
            self.min_track_age,
            self.wrong_way_angle_gap,
            f"enabled at y={tw_y_pct:.0%}" if tw_enabled else "disabled",
        )

    # ── Public API ────────────────────────────────────────────────

    def update(self, detections: list, frame_shape: tuple = None) -> None:
        """
        Call once per frame with the current List[Detection].
        frame_shape: (height, width) — required for tripwire.
        """
        self._frame_index += 1
        active_ids = set()

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue
            active_ids.add(tid)

            if tid not in self._tracks:
                self._tracks[tid] = TrackState(track_id=tid)

            state = self._tracks[tid]
            cx, cy = float(det.center[0]), float(det.center[1])
            state.history.append((cx, cy, self._frame_index))
            state.age_frames += 1
            state.last_seen_frame = self._frame_index

            # Direction analysis (requires at least min_age frames)
            if state.age_frames >= self.min_track_age and len(state.history) >= 4:
                self._compute_direction(state)

            # Tripwire
            if self.tripwire is not None and frame_shape is not None:
                self.tripwire.update(tid, cy, frame_shape[0])

        # Purge stale tracks (not seen for >30 frames)
        stale = [
            tid for tid, st in self._tracks.items()
            if self._frame_index - st.last_seen_frame > 30
        ]
        for tid in stale:
            del self._tracks[tid]

    def get_direction(self, track_id: int) -> str:
        st = self._tracks.get(track_id)
        return st.direction_label if st else "UNKNOWN"

    def get_wrong_way_vehicles(self, expected_direction: str = "RIGHT") -> List[int]:
        """
        Returns track IDs moving against expected_direction.
        Marks state.wrong_way accordingly.
        """
        expected_angle = {
            "RIGHT": 0.0, "UP": 90.0, "LEFT": 180.0, "DOWN": 270.0
        }.get(expected_direction.upper(), 0.0)

        wrong = []
        for tid, st in self._tracks.items():
            if st.age_frames < self.min_track_age:
                continue
            diff = abs(st.direction_angle - expected_angle)
            if diff > 180.0:
                diff = 360.0 - diff
            st.wrong_way = diff > self.wrong_way_angle_gap
            if st.wrong_way:
                wrong.append(tid)
        return wrong

    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def get_all_states(self) -> Dict[int, TrackState]:
        """Returns all active track states for AnomalyEngine consumption."""
        return dict(self._tracks)

    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def tripwire_count(self) -> int:
        """Total vehicles that have crossed the virtual tripwire line."""
        return self.tripwire.count if self.tripwire else 0

    def summary(self) -> dict:
        """JSON-serializable summary for analytics merging."""
        return {
            "active_tracks":  self.active_track_count(),
            "tripwire_count": self.tripwire_count,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _compute_direction(self, state: TrackState) -> None:
        """
        Computes direction from the displacement across the last N history points.
        Using N=6 frames of averaging suppresses jitter without adding latency.
        """
        pts = list(state.history)
        if len(pts) < 4:
            return

        # Use last 6 points for smoothed direction vector
        window = pts[-min(6, len(pts)):]
        x1, y1, _ = window[0]
        x2, y2, _ = window[-1]

        dx = x2 - x1
        dy = y2 - y1

        # Require meaningful movement to update direction
        if math.hypot(dx, dy) < 3.0:
            return

        state.direction_angle = _angle_deg(dx, dy)
        state.direction_label = _dir_label(state.direction_angle)