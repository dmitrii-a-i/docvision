"""Document corner detection + perspective dewarping (YOLO11n-pose)."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points to TL, TR, BR, BL via centroid + angle sort."""
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_pts = pts[order]

    s = sorted_pts.sum(axis=1)
    d = np.diff(sorted_pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = sorted_pts[np.argmin(s)]   # TL
    ordered[1] = sorted_pts[np.argmin(d)]   # TR
    ordered[2] = sorted_pts[np.argmax(s)]   # BR
    ordered[3] = sorted_pts[np.argmax(d)]   # BL
    return ordered


def compute_output_size(quad: np.ndarray) -> tuple[int, int]:
    """Compute output (width, height) from quad edge lengths."""
    tl, tr, br, bl = quad
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    return int(round(max(w_top, w_bot))), int(round(max(h_left, h_right)))


def dewarp(
    image: np.ndarray, quad: np.ndarray, output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Perspective-transform so that quad maps to a rectangle."""
    if output_size is None:
        output_size = compute_output_size(quad)
    w, h = output_size
    src = quad.astype(np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)


class CornerDetector:
    """YOLO11n-pose corner detection + dewarping."""

    def __init__(self, model_path: str, device: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        log.info("CornerDetector loaded: %s on %s", model_path, device)

    def detect_and_dewarp(
        self, image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect document corners and dewarp.

        Returns (dewarped_image, ordered_quad) or None if no document found.
        """
        results = self.model(image, conf=self.conf, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None

        # Take the first (highest confidence) detection
        kps = result.keypoints.data[0].cpu().numpy()
        quad = order_corners(kps[:, :2].astype(np.float32))
        dewarped = dewarp(image, quad)
        return dewarped, quad
