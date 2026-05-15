"""
LaneManager — Assigns detections to defined lane regions.
Supports arbitrary polygon ROIs for Indian multi-lane roads.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict

logger = logging.getLogger(__name__)


class LaneManager:
    def __init__(self, lane_cfg: dict):
        self.enabled = lane_cfg.get("enabled", False)
        self.lanes: List[Dict] = []

        if self.enabled:
            for lane in lane_cfg.get("regions", []):
                pts = np.array(lane["points"], dtype=np.int32)
                self.lanes.append({"name": lane["name"], "polygon": pts})
            logger.info(f"LaneManager initialized with {len(self.lanes)} lanes.")

    def assign_lane(self, center: tuple) -> str:
        """Returns which lane a vehicle center belongs to."""
        if not self.enabled:
            return "ALL"
        for lane in self.lanes:
            if cv2.pointPolygonTest(lane["polygon"], center, False) >= 0:
                return lane["name"]
        return "OUT_OF_ROI"

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw lane polygons on the frame."""
        if not self.enabled:
            return frame
        for lane in self.lanes:
            cv2.polylines(frame, [lane["polygon"]], True, (255, 165, 0), 2)
            cx, cy = lane["polygon"].mean(axis=0).astype(int)
            cv2.putText(frame, lane["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        return frame