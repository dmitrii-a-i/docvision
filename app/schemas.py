from __future__ import annotations

from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixels


class ProcessResponse(BaseModel):
    fields: dict[str, str]
    detections: list[Detection]
    dewarped_image: str  # base64-encoded JPEG
    annotated_image: str  # base64-encoded JPEG with detection boxes drawn


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: list[str]
