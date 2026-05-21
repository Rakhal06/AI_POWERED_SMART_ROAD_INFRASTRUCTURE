"""
AI Traffic Intelligence — Core Package
Exposes: VehicleDetector, Detection, LaneManager, VehicleTracker, AnomalyEngine
"""

from .detector_factory import DetectorFactory, BaseDetector, Detection
from .lane_manager import LaneManager
from .tracker import VehicleTracker
from .anomaly_engine import AnomalyEngine, AnomalyEvent

__all__ = [
    "DetectorFactory", "BaseDetector", "Detection",
    "LaneManager",
    "VehicleTracker",
    "AnomalyEngine", "AnomalyEvent",
]