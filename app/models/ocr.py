"""OCR-based field extraction for lite pipeline mode."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.schemas import Detection

log = logging.getLogger(__name__)

MIN_CROP_PX = 10


def _spatial_sort_detections(detections: list[Detection]) -> list[Detection]:
    """Sort detections: top-to-bottom rows, left-to-right within each row."""
    if not detections:
        return []

    heights = [d.bbox[3] - d.bbox[1] for d in detections]
    row_height = max(float(np.median(heights)), 1.0)

    def sort_key(d: Detection) -> tuple[int, float]:
        y_center = (d.bbox[1] + d.bbox[3]) / 2
        x_center = (d.bbox[0] + d.bbox[2]) / 2
        row_idx = int(y_center / row_height)
        return (row_idx, x_center)

    return sorted(detections, key=sort_key)


class OCREngine:
    """EasyOCR wrapper for field-level text extraction."""

    def __init__(self, lang: list[str] | None = None, device: str = "cpu"):
        import easyocr

        lang = lang or ["en"]
        self._reader = easyocr.Reader(lang, gpu=(device != "cpu"))
        log.info("OCREngine loaded: easyocr (lang=%s, device=%s)", lang, device)

    def extract_fields_from_detections(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> dict[str, str]:
        """Crop each 'text' detection, OCR it, return spatially sorted fields."""
        text_dets = [d for d in detections if d.label == "text"]
        sorted_dets = _spatial_sort_detections(text_dets)

        h, w = image.shape[:2]
        fields: dict[str, str] = {}

        for i, det in enumerate(sorted_dets, 1):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                continue

            crop = image[y1:y2, x1:x2]
            text = " ".join(self._reader.readtext(crop, detail=0)).strip()
            if text:
                fields[f"field_{i}"] = text

        return fields
