"""Document field detection (YOLO11n)."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from ultralytics import YOLO

from app.schemas import Detection

log = logging.getLogger(__name__)

CLASS_NAMES = {0: "text", 1: "photo", 2: "signature"}


class FieldDetector:
    """YOLO11n field detection on dewarped document images."""

    def __init__(self, model_path: str, device: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        log.info("FieldDetector loaded: %s on %s", model_path, device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect fields on a dewarped document image."""
        results = self.model(image, conf=self.conf, verbose=False)
        result = results[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            detections.append(Detection(
                label=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                confidence=round(conf, 4),
                bbox=[round(v, 1) for v in [x1, y1, x2, y2]],
            ))

        return detections
