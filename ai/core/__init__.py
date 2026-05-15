"""
AI Traffic Intelligence — Core Package
Exposes: VehicleDetector, Detection, LaneManager, VehicleTracker
"""

from .detector import VehicleDetector, Detection
from .lane_manager import LaneManager
from .tracker import VehicleTracker

__all__ = ["VehicleDetector", "Detection", "LaneManager", "VehicleTracker"]