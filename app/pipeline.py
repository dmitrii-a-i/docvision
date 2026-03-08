"""Orchestration: image bytes → ProcessResponse."""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

from app.config import settings
from app.models.corner import CornerDetector
from app.models.fields import FieldDetector
from app.schemas import ProcessResponse

log = logging.getLogger(__name__)


class Pipeline:
    """Full document processing pipeline."""

    def __init__(self):
        device = settings.device
        mode = settings.PIPELINE_MODE
        log.info("Initializing pipeline: mode=%s, device=%s", mode, device)

        self.corner_detector = CornerDetector(
            settings.CORNER_MODEL, device, settings.CORNER_CONF,
        )
        self.field_detector = FieldDetector(
            settings.FIELD_MODEL, device, settings.FIELD_CONF,
        )

        self.ocr_engine = None
        self.vlm = None

        if mode == "lite":
            from app.models.ocr import OCREngine

            self.ocr_engine = OCREngine(
                lang=settings.ocr_languages, device=device,
            )
        elif mode == "api":
            from app.models.vlm import APIClient

            self.vlm = APIClient(
                settings.VLM_API_KEY, settings.VLM_BASE_URL, settings.VLM_MODEL,
            )
        else:  # standard
            from app.models.vlm import LocalVLM

            self.vlm = LocalVLM(settings.VLM_MODEL_ID, device)

        self._models_loaded = ["corner_detect", "field_detect"]
        if mode == "lite":
            self._models_loaded.append("ocr:easyocr")
        elif mode == "api":
            self._models_loaded.append(f"vlm_api:{settings.VLM_MODEL}")
        else:
            self._models_loaded.append(f"vlm_local:{settings.VLM_MODEL_ID}")

    @property
    def models_loaded(self) -> list[str]:
        return self._models_loaded

    def process(self, image_bytes: bytes) -> ProcessResponse:
        """Run the full pipeline on raw image bytes."""
        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")

        # Corner detection + dewarp
        result = self.corner_detector.detect_and_dewarp(image)
        if result is None:
            raise ValueError("No document detected in image")
        dewarped, quad = result

        # Field detection
        detections = self.field_detector.detect(dewarped)

        # Text extraction
        if self.ocr_engine is not None:
            fields = self.ocr_engine.extract_fields_from_detections(
                dewarped, detections,
            )
        else:
            fields = self.vlm.extract_fields(dewarped)

        # Encode dewarped image as base64 JPEG
        _, buf = cv2.imencode(".jpg", dewarped, [cv2.IMWRITE_JPEG_QUALITY, 90])
        dewarped_b64 = base64.b64encode(buf.tobytes()).decode()

        return ProcessResponse(
            fields=fields,
            detections=detections,
            dewarped_image=dewarped_b64,
        )
