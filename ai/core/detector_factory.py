"""
detector_factory.py — Detector-agnostic base + registry.

Swap RT-DETR / YOLO / ONNX via settings.yaml without touching pipeline.py.

Usage:
    detector = DetectorFactory.create(cfg)
    detections, raw = detector.infer(frame)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single vehicle detection. Compatible with all pipeline consumers."""
    track_id:   int
    class_id:   int
    class_name: str
    confidence: float
    bbox:       list        # [x1, y1, x2, y2]
    center:     tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def width(self):  return self.bbox[2] - self.bbox[0]
    @property
    def height(self): return self.bbox[3] - self.bbox[1]
    @property
    def area(self):   return self.width * self.height
    @property
    def aspect_ratio(self): return self.width / max(self.height, 1)


class BaseDetector(ABC):
    """
    Detector interface. Any backend (RT-DETR, YOLO, ONNX, TensorRT)
    must implement these two methods.
    """

    @abstractmethod
    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], object]:
        """
        Run inference on a single BGR frame.
        Returns: (List[Detection], raw_result)
        raw_result can be None for backends that don't need it downstream.
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Pre-load CUDA kernels / model weights with a dummy forward pass."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class DetectorFactory:
    """
    Registry pattern. Detectors self-register with @DetectorFactory.register("name").
    Pipeline calls DetectorFactory.create(cfg) — never imports a concrete class.
    """
    _registry: dict = {}

    @classmethod
    def register(cls, name: str):
        """Decorator: @DetectorFactory.register("rtdetr")"""
        def decorator(detector_cls):
            cls._registry[name] = detector_cls
            logger.debug("Detector registered: %s -> %s", name, detector_cls.__name__)
            return detector_cls
        return decorator

    @classmethod
    def create(cls, cfg: dict) -> BaseDetector:
        """
        Instantiate the configured detector backend.
        Reads cfg["model"]["backend"]: "rtdetr" | "yolo" | "onnx"
        """
        # Import all detectors so they self-register
        from ai.core import detector as _  # noqa: F401

        backend = cfg["model"].get("backend", "rtdetr")
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown detector backend '{backend}'. "
                f"Available: {available}"
            )
        logger.info("Creating detector: backend=%s", backend)
        return cls._registry[backend](cfg)

    @classmethod
    def available_backends(cls) -> list:
        return list(cls._registry.keys())