"""
detector.py — RT-DETRv2 (primary) + YOLOv8 (fallback) detectors.

RT-DETRv2 advantages over YOLOv8 for highway ITS:
  - Global attention: transformer encoder sees full scene context
  - NMS-free: deterministic, no suppression of close vehicles
  - Deformable attention: accurate multi-scale (distant small vehicles)
  - +2-3% mAP on crowded/occluded highway scenes
  - Better occlusion handling via persistent object queries

Both detectors self-register into DetectorFactory.
Select via settings.yaml: model.backend = rtdetr | yolo
"""

import logging
import numpy as np
from typing import List, Tuple

from .detector_factory import BaseDetector, Detection, DetectorFactory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# RT-DETRv2 DETECTOR  (primary)
# ─────────────────────────────────────────────────────────────────

@DetectorFactory.register("rtdetr")
class RTDETRDetector(BaseDetector):
    """
    RT-DETRv2 — Real-Time Detection Transformer v2 (Baidu, 2024).

    Model options (via settings.yaml model.name):
      rtdetr-l    : 53.0 mAP, 114 FPS on T4   ← recommended default
      rtdetr-x    : 54.8 mAP,  74 FPS on T4
      rtdetrv2-s  : 48.1 mAP, real-time on Pi  ← edge deployment
      rtdetrv2-l  : 55.1 mAP                   ← best balance
      rtdetrv2-x  : 56.4 mAP                   ← max accuracy

    Key technical differences vs YOLO:
      - Hybrid encoder: CNN (local) + Transformer (global) features
      - Object queries: 300 learned slots, each "attends" to one object
      - Bipartite matching loss (Hungarian): eliminates need for NMS
      - Deformable attention: focuses on relevant feature map regions
    """

    # Map friendly names to model file names
    MODEL_MAP = {
        "rtdetr-l":   "rtdetr-l.pt",
        "rtdetr-x":   "rtdetr-x.pt",
        "rtdetrv2-s": "rtdetrv2-s.pt",
        "rtdetrv2-m": "rtdetrv2-m.pt",
        "rtdetrv2-l": "rtdetrv2-l.pt",
        "rtdetrv2-x": "rtdetrv2-x.pt",
    }

    def __init__(self, cfg: dict):
        self.conf   = cfg["model"]["confidence_threshold"]
        self.iou    = cfg["model"]["iou_threshold"]
        self.device = cfg["model"]["device"]
        self.vehicle_class_map = {
            int(k): v for k, v in cfg["vehicle_classes"].items()
        }

        model_name = cfg["model"].get("name", "rtdetr-l")
        model_path = cfg["model"].get("path") or self.MODEL_MAP.get(model_name, "rtdetr-l.pt")

        logger.info(
            "Loading RT-DETR: %s on device=%s | NMS-free=True",
            model_path, self.device,
        )

        try:
            from ultralytics import RTDETR
            self.model = RTDETR(model_path)
            self.model.to(self.device)
        except ImportError:
            raise ImportError(
                "ultralytics>=8.2.0 required for RT-DETR. "
                "Run: pip install ultralytics>=8.2.0"
            )

        logger.info("RT-DETR loaded successfully. No NMS — deterministic output.")

    def warmup(self) -> None:
        logger.info("RT-DETR warmup...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.infer(dummy)
        logger.info("RT-DETR warmup complete.")

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], object]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=list(self.vehicle_class_map.keys()),
            verbose=False,
            tracker="bytetrack.yaml",
        )

        detections: List[Detection] = []
        if results[0].boxes is None:
            return detections, results[0]

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_class_map:
                continue
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append(Detection(
                track_id=track_id,
                class_id=cls_id,
                class_name=self.vehicle_class_map[cls_id],
                confidence=float(box.conf[0]),
                bbox=box.xyxy[0].tolist(),
            ))

        return detections, results[0]

    def __repr__(self):
        return (f"RTDETRDetector(device={self.device}, "
                f"conf={self.conf}, nms_free=True)")


# ─────────────────────────────────────────────────────────────────
# YOLOv8 FALLBACK DETECTOR
# ─────────────────────────────────────────────────────────────────

@DetectorFactory.register("yolo")
class YOLODetector(BaseDetector):
    """
    YOLOv8 fallback detector.
    Use for: edge deployment, Pi, compatibility, quick testing.
    Select via: model.backend = yolo in settings.yaml
    """

    def __init__(self, cfg: dict):
        self.conf   = cfg["model"]["confidence_threshold"]
        self.iou    = cfg["model"]["iou_threshold"]
        self.device = cfg["model"]["device"]
        self.vehicle_class_map = {
            int(k): v for k, v in cfg["vehicle_classes"].items()
        }
        model_path = cfg["model"].get("path", "yolov8n.pt")

        logger.info("Loading YOLOv8: %s on device=%s", model_path, self.device)

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except ImportError:
            raise ImportError("ultralytics required: pip install ultralytics")

        logger.info("YOLOv8 loaded.")

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.infer(dummy)

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], object]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=list(self.vehicle_class_map.keys()),
            verbose=False,
            tracker="bytetrack.yaml",
        )

        detections: List[Detection] = []
        if results[0].boxes is None:
            return detections, results[0]

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_class_map:
                continue
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append(Detection(
                track_id=track_id,
                class_id=cls_id,
                class_name=self.vehicle_class_map[cls_id],
                confidence=float(box.conf[0]),
                bbox=box.xyxy[0].tolist(),
            ))

        return detections, results[0]


# ─────────────────────────────────────────────────────────────────
# ONNX RUNTIME DETECTOR  (edge deployment)
# ─────────────────────────────────────────────────────────────────

@DetectorFactory.register("onnx")
class ONNXDetector(BaseDetector):
    """
    ONNX Runtime detector for edge deployment (Raspberry Pi, Jetson, etc.)
    No PyTorch/CUDA dependency — runs on CPU via onnxruntime.

    Export your RT-DETR model first:
        python tools/export_onnx.py --model models/rtdetr-l.pt
    Then set: model.backend = onnx, model.path = models/rtdetr-l.onnx
    """

    def __init__(self, cfg: dict):
        model_path = cfg["model"]["path"]
        self.conf  = cfg["model"]["confidence_threshold"]
        self.vehicle_class_map = {
            int(k): v for k, v in cfg["vehicle_classes"].items()
        }
        self.input_size = cfg["model"].get("input_size", 640)

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required for ONNX backend. "
                "Run: pip install onnxruntime  (CPU) "
                "or: pip install onnxruntime-gpu  (GPU)"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active = self.session.get_providers()[0]
        logger.info("ONNX detector loaded: %s | provider=%s", model_path, active)

        # Simple tracker state for ONNX (no ByteTrack without ultralytics)
        self._next_id = 0
        self._track_memory = {}   # basic IoU matching

    def warmup(self) -> None:
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], object]:
        import cv2

        # Preprocess
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

        # Inference
        outputs = self.session.run(None, {self.input_name: blob})

        # Post-process (RT-DETR ONNX output: [1, 300, 6] → [x1,y1,x2,y2,conf,cls])
        detections: List[Detection] = []
        if len(outputs) == 0:
            return detections, None

        preds = outputs[0][0]  # [300, 6]
        sx, sy = w / self.input_size, h / self.input_size

        for pred in preds:
            x1, y1, x2, y2, conf, cls_id = pred
            if conf < self.conf:
                continue
            cls_id = int(cls_id)
            if cls_id not in self.vehicle_class_map:
                continue

            bbox = [
                float(x1 * sx), float(y1 * sy),
                float(x2 * sx), float(y2 * sy),
            ]
            track_id = self._assign_id(bbox)
            detections.append(Detection(
                track_id=track_id,
                class_id=cls_id,
                class_name=self.vehicle_class_map[cls_id],
                confidence=float(conf),
                bbox=bbox,
            ))

        return detections, None

    def _assign_id(self, bbox: list) -> int:
        """Simple IoU-based ID assignment for ONNX (no ByteTrack dependency)."""
        best_iou, best_id = 0.3, -1
        for tid, prev_bbox in self._track_memory.items():
            iou = self._iou(bbox, prev_bbox)
            if iou > best_iou:
                best_iou, best_id = iou, tid

        if best_id == -1:
            best_id = self._next_id
            self._next_id += 1

        self._track_memory[best_id] = bbox
        # Prune stale tracks (keep last 30)
        if len(self._track_memory) > 50:
            oldest = list(self._track_memory.keys())[0]
            del self._track_memory[oldest]

        return best_id

    @staticmethod
    def _iou(a: list, b: list) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(ua, 1e-6)