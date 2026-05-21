"""
anomaly_engine.py — Multi-signal accident and anomaly detection.

Architecture:
  Camera signals  ──┐
  Tracker state   ──┼──► AnomalyEngine.update() ──► List[AnomalyEvent]
  Speed estimates ──┤
  HW signals      ──┘   (vibration, ultrasonic from ESP32)

Confidence scoring (accident detection):
  Bounding box inversion (overturned vehicle)   → +60 pts
  Sudden stop (speed dropped to near-zero)       → +30 pts
  Hardware vibration spike (SW-420 sensor)       → +40 pts
  ---------------------------------------------------------
  Alert threshold: 70/100

Other anomalies:
  Wrong-way vehicle         → immediate alert
  Prolonged stopped vehicle → alert after N frames
  Overspeeding              → alert if speed > mean * multiplier

Usage:
    engine = AnomalyEngine(cfg)
    events = engine.update(
        detections, tracker_states, speed_map, hw_signals, frame_index
    )
    for ev in events:
        if ev.alert:
            logger.warning("ANOMALY: %s id=%d conf=%.0f", ev.event_type, ev.track_id, ev.confidence)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────────────────────────

ACCIDENT          = "ACCIDENT"
WRONG_WAY         = "WRONG_WAY"
VEHICLE_STOPPED   = "VEHICLE_STOPPED"
OVERSPEED         = "OVERSPEED"
CONGESTION        = "CONGESTION"


# ─────────────────────────────────────────────────────────────────
# ANOMALY EVENT
# ─────────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    """A single detected anomaly. JSON-serializable."""
    event_type:   str
    track_id:     int
    confidence:   float          # 0–100
    alert:        bool           # True if confidence >= threshold
    frame_index:  int
    timestamp:    float = field(default_factory=time.time)
    speed_kmh:    float = 0.0
    location_px:  Optional[Tuple[int, int]] = None   # pixel centroid
    gps:          Optional[Tuple[float, float]] = None
    hw_vibration: bool = False
    message:      str = ""

    def to_dict(self) -> dict:
        return {
            "event_type":   self.event_type,
            "track_id":     self.track_id,
            "confidence":   round(self.confidence, 1),
            "alert":        self.alert,
            "frame_index":  self.frame_index,
            "timestamp":    round(self.timestamp, 2),
            "speed_kmh":    round(self.speed_kmh, 1),
            "location_px":  self.location_px,
            "gps":          self.gps,
            "hw_vibration": self.hw_vibration,
            "message":      self.message,
        }


# ─────────────────────────────────────────────────────────────────
# PER-TRACK ANOMALY STATE
# ─────────────────────────────────────────────────────────────────

@dataclass
class _TrackAnomalyState:
    track_id:          int
    accident_score:    float = 0.0
    stopped_frames:    int   = 0
    was_moving:        bool  = False
    prev_speed_kmh:    float = 0.0
    overspeed_frames:  int   = 0
    alerted_accident:  bool  = False   # prevent alert spam per track
    alerted_stopped:   bool  = False
    alerted_overspeed: bool  = False


# ─────────────────────────────────────────────────────────────────
# ANOMALY ENGINE
# ─────────────────────────────────────────────────────────────────

class AnomalyEngine:
    """
    Stateful anomaly detector. Call update() once per frame.
    Returns a list of AnomalyEvent objects for the pipeline to log and display.
    """

    def __init__(self, cfg: dict):
        acfg = cfg.get("anomaly", {})
        self.enabled = acfg.get("enabled", True)

        # Accident scoring
        self.accident_threshold    = float(acfg.get("accident_confidence_threshold", 70))
        self.score_aspect_inv      = float(acfg.get("aspect_ratio_inversion_score", 60))
        self.score_vibration       = float(acfg.get("vibration_spike_score", 40))
        self.score_sudden_stop     = float(acfg.get("sudden_stop_score", 30))

        # Stopped vehicle
        self.stopped_speed_kmh     = float(acfg.get("stopped_speed_kmh", 4.0))
        self.stopped_min_frames    = int(acfg.get("stopped_min_frames", 45))

        # Wrong-way
        self.wrong_way_enabled     = bool(acfg.get("wrong_way_enabled", True))
        self.expected_dir          = cfg.get("lanes", {}).get("expected_direction", "RIGHT")

        # Overspeed
        self.overspeed_multiplier  = float(acfg.get("overspeed_multiplier", 1.8))

        # Internal state
        self._states: Dict[int, _TrackAnomalyState] = {}
        self._frame_index = 0

        logger.info(
            "AnomalyEngine ready | accident_threshold=%.0f | stopped_frames=%d",
            self.accident_threshold, self.stopped_min_frames,
        )

    # ── PUBLIC ────────────────────────────────────────────────────

    def update(
        self,
        detections:    list,         # List[Detection]
        tracker_states: dict,        # {track_id: TrackState} from VehicleTracker
        speed_map:     Dict[int, float],  # {track_id: speed_kmh} from TrafficAnalyzer
        hw_signals,                  # HardwareSignals from SerialBridge
        frame_index:   int,
    ) -> List[AnomalyEvent]:
        """
        Run anomaly detection for one frame.
        Returns list of AnomalyEvent (may be empty).
        """
        if not self.enabled:
            return []

        self._frame_index = frame_index
        events: List[AnomalyEvent] = []

        # Cleanup stale states for tracks no longer visible
        active_ids = {d.track_id for d in detections}
        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]

        # Per-vehicle analysis
        mean_speed = self._mean_speed(speed_map, active_ids)
        gps = hw_signals.gps_tuple if hw_signals else None
        vib = hw_signals.vibration_recent if hw_signals else False

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue

            speed = speed_map.get(tid, 0.0)
            state = self._states.setdefault(tid, _TrackAnomalyState(track_id=tid))
            center = det.center if hasattr(det, "center") else None

            # 1. Accident detection
            acc_ev = self._check_accident(det, state, speed, vib, gps, center)
            if acc_ev:
                events.append(acc_ev)

            # 2. Stopped vehicle
            stop_ev = self._check_stopped(tid, state, speed, gps, center)
            if stop_ev:
                events.append(stop_ev)

            # 3. Overspeed
            over_ev = self._check_overspeed(tid, state, speed, mean_speed, gps, center)
            if over_ev:
                events.append(over_ev)

            # Update previous speed
            state.prev_speed_kmh = speed

        # 4. Wrong-way (uses tracker direction data)
        if self.wrong_way_enabled:
            for det in detections:
                tid = det.track_id
                ts = tracker_states.get(tid)
                if ts and getattr(ts, "wrong_way", False):
                    center = det.center if hasattr(det, "center") else None
                    events.append(AnomalyEvent(
                        event_type=WRONG_WAY,
                        track_id=tid,
                        confidence=95.0,
                        alert=True,
                        frame_index=frame_index,
                        speed_kmh=speed_map.get(tid, 0.0),
                        location_px=center,
                        gps=gps,
                        message=f"Vehicle ID {tid} moving against traffic flow",
                    ))

        return events

    def get_state(self, track_id: int) -> Optional[_TrackAnomalyState]:
        return self._states.get(track_id)

    def reset_alert(self, track_id: int) -> None:
        """Reset alert flags for a track (e.g. after operator acknowledgement)."""
        state = self._states.get(track_id)
        if state:
            state.alerted_accident  = False
            state.alerted_stopped   = False
            state.alerted_overspeed = False

    # ── INTERNAL ──────────────────────────────────────────────────

    def _check_accident(
        self, det, state: _TrackAnomalyState,
        speed: float, vibration: bool,
        gps, center
    ) -> Optional[AnomalyEvent]:
        """
        Multi-signal accident confidence scoring.

        Signal                           Score
        ──────────────────────────────── ─────
        Bounding box aspect inversion    +60
        Sudden stop (was moving >15)     +30
        HW vibration spike               +40
        ──────────────────────────────── ─────
        Max possible                      130 (capped at 100)
        Alert threshold                    70
        """
        score = 0.0

        # Signal 1: bounding box aspect ratio inversion
        # Normal car: width > height (aspect > 1).
        # Overturned/sideways: height >> width (aspect < 0.5 for vehicles).
        aspect = det.aspect_ratio if hasattr(det, "aspect_ratio") else (
            (det.bbox[2] - det.bbox[0]) / max(det.bbox[3] - det.bbox[1], 1)
            if hasattr(det, "bbox") else 1.0
        )
        # Bikes/motorcycles are naturally tall — skip for class_id == 3
        class_id = getattr(det, "class_id", 2)
        if class_id != 3 and aspect < 0.6:
            score += self.score_aspect_inv

        # Signal 2: sudden stop (was moving >15 km/h, now <4 km/h)
        was_fast = state.prev_speed_kmh > 15.0
        now_slow = speed < self.stopped_speed_kmh
        if was_fast and now_slow:
            score += self.score_sudden_stop

        # Signal 3: hardware vibration
        if vibration:
            score += self.score_vibration

        score = min(score, 100.0)
        state.accident_score = score

        alert = score >= self.accident_threshold
        if alert and state.alerted_accident:
            return None   # already raised this alert
        if not alert:
            return None

        state.alerted_accident = True
        logger.warning(
            "ACCIDENT detected | track=%d | score=%.0f | vib=%s | aspect=%.2f",
            det.track_id, score, vibration, aspect,
        )
        return AnomalyEvent(
            event_type=ACCIDENT,
            track_id=det.track_id,
            confidence=score,
            alert=True,
            frame_index=self._frame_index,
            speed_kmh=speed,
            location_px=center,
            gps=gps,
            hw_vibration=vibration,
            message=(
                f"Possible accident — vehicle ID {det.track_id} "
                f"[score={score:.0f}/100"
                f"{', vibration confirmed' if vibration else ''}]"
            ),
        )

    def _check_stopped(
        self, tid: int, state: _TrackAnomalyState,
        speed: float, gps, center
    ) -> Optional[AnomalyEvent]:
        """Detect vehicle stopped for an extended period on the road."""
        if speed < self.stopped_speed_kmh:
            state.stopped_frames += 1
        else:
            state.stopped_frames = 0
            state.was_moving = True
            state.alerted_stopped = False  # reset if vehicle moves again

        if (
            state.was_moving
            and state.stopped_frames >= self.stopped_min_frames
            and not state.alerted_stopped
        ):
            state.alerted_stopped = True
            logger.warning(
                "STOPPED VEHICLE | track=%d | stopped_frames=%d",
                tid, state.stopped_frames,
            )
            return AnomalyEvent(
                event_type=VEHICLE_STOPPED,
                track_id=tid,
                confidence=85.0,
                alert=True,
                frame_index=self._frame_index,
                speed_kmh=speed,
                location_px=center,
                gps=gps,
                message=f"Vehicle ID {tid} stopped on road for >{self.stopped_min_frames} frames",
            )
        return None

    def _check_overspeed(
        self, tid: int, state: _TrackAnomalyState,
        speed: float, mean_speed: float, gps, center
    ) -> Optional[AnomalyEvent]:
        """Detect vehicle moving significantly faster than traffic mean."""
        if mean_speed < 5.0:
            return None   # not enough reliable mean speed yet

        threshold = mean_speed * self.overspeed_multiplier
        if speed > threshold:
            state.overspeed_frames += 1
        else:
            state.overspeed_frames = 0
            state.alerted_overspeed = False

        # Require 10 consecutive overspeed frames to avoid false positives
        if state.overspeed_frames >= 10 and not state.alerted_overspeed:
            state.alerted_overspeed = True
            logger.info(
                "OVERSPEED | track=%d | speed=%.1f km/h | mean=%.1f km/h",
                tid, speed, mean_speed,
            )
            return AnomalyEvent(
                event_type=OVERSPEED,
                track_id=tid,
                confidence=75.0,
                alert=True,
                frame_index=self._frame_index,
                speed_kmh=speed,
                location_px=center,
                gps=gps,
                message=(
                    f"Vehicle ID {tid} overspeeding: "
                    f"{speed:.1f} km/h vs mean {mean_speed:.1f} km/h"
                ),
            )
        return None

    @staticmethod
    def _mean_speed(speed_map: Dict[int, float], active_ids: set) -> float:
        """Mean speed of active vehicles with >0 speed estimate."""
        speeds = [v for k, v in speed_map.items() if k in active_ids and v > 3.0]
        return round(sum(speeds) / len(speeds), 1) if speeds else 0.0