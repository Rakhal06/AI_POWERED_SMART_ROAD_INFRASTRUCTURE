"""
VehicleDetector — YOLOv8-powered inference engine.
Wraps YOLO with confidence filtering, class filtering,
and structured detection output.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single vehicle detection result."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2]
    center: tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))


class VehicleDetector:
    """
    Core AI inference engine.
    Encapsulates YOLOv8 with:
    - confidence threshold filtering
    - vehicle-class filtering
    - tracking support (ByteTrack)
    - structured Detection output
    """

    def __init__(self, cfg: dict):
        self.model_path = cfg["model"]["path"]
        self.conf = cfg["model"]["confidence_threshold"]
        self.iou = cfg["model"]["iou_threshold"]
        self.device = cfg["model"]["device"]
        self.vehicle_class_map: Dict[int, str] = {
            int(k): v for k, v in cfg["vehicle_classes"].items()
        }

        logger.info(f"Loading YOLO model: {self.model_path} on device={self.device}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        logger.info("Model loaded successfully.")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.
        Returns list of Detection objects for vehicles only.
        Uses ByteTrack for ID persistence across frames.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=list(self.vehicle_class_map.keys()),
            verbose=False,
            tracker="bytetrack.yaml"
        )

        detections = []
        if results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_class_map:
                continue

            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append(Detection(
                track_id=track_id,
                class_id=cls_id,
                class_name=self.vehicle_class_map[cls_id],
                confidence=conf,
                bbox=xyxy
            ))

        return detections, results[0]