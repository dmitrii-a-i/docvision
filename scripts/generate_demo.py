#!/usr/bin/env python3
"""Generate pipeline demo images for README.

Creates horizontal pipeline visualizations:
  Original → Corners → Dewarped → Fields

Usage:
    python scripts/generate_demo.py
    python scripts/generate_demo.py --midv /path/to/midv-500/dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Reuse project constants
CLASS_NAMES = {0: "text", 1: "photo", 2: "signature"}
FIELD_COLORS = {
    "text": (50, 200, 50),       # green
    "photo": (200, 50, 50),      # blue
    "signature": (50, 50, 200),  # red
}

# Document types to use for demo (code, human label)
DOC_TYPES = [
    ("01_alb_id", "ID card — Albania"),
    ("12_deu_drvlic_new", "Driver's license — Germany"),
    ("39_rus_internalpassport", "Internal passport — Russia"),
]


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points to TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[2] = pts[np.argmax(s)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def dewarp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-transform image so that quad maps to a rectangle."""
    tl, tr, br, bl = quad
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)


def draw_corners(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Draw detected corners and quad on image."""
    vis = image.copy()
    pts = quad.astype(np.int32)

    # Semi-transparent fill
    overlay = vis.copy()
    cv2.fillConvexPoly(overlay, pts, (255, 255, 0))
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    # Quad edges
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 255), 2)

    # Corner dots
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    labels = ["TL", "TR", "BR", "BL"]
    r = max(6, min(vis.shape[:2]) // 80)
    for i, (label, color) in enumerate(zip(labels, colors)):
        cx, cy = int(pts[i][0]), int(pts[i][1])
        cv2.circle(vis, (cx, cy), r, color, -1)
        cv2.putText(vis, label, (cx + r + 2, cy - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis


def draw_fields(image: np.ndarray, field_model: YOLO, conf: float = 0.25) -> np.ndarray:
    """Draw detected field bounding boxes on image."""
    vis = image.copy()
    results = field_model(image, conf=conf, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return vis

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        color = FIELD_COLORS.get(label, (128, 128, 128))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def add_label(image: np.ndarray, text: str, height: int = 40) -> np.ndarray:
    """Add a text label bar above the image."""
    h, w = image.shape[:2]
    bar = np.full((height, w, 3), 255, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = (w - tw) // 2
    cv2.putText(bar, text, (x, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return np.vstack([bar, image])


def resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to target height, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = target_h / h
    return cv2.resize(image, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def find_sample_image(midv_root: Path, doc_code: str) -> Path | None:
    """Find a sample image from MIDV-500 for given doc type."""
    doc_dir = midv_root / doc_code / doc_code / "images"
    # Prefer TA (table) conditions — clearest
    for condition in ["TA", "TS", "HA", "KA"]:
        cond_dir = doc_dir / condition
        if cond_dir.exists():
            images = sorted(cond_dir.glob("*.*"))
            if images:
                # Pick a middle frame (not first/last which may be blurry)
                return images[len(images) // 2]
    return None


def make_pipeline_image(
    image_path: Path,
    corner_model: YOLO,
    field_model: YOLO,
    target_h: int = 500,
) -> np.ndarray | None:
    """Create a single pipeline strip: Original → Corners → Dewarped → Fields."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Corner detection
    results = corner_model(str(image_path), conf=0.25, verbose=False)
    result = results[0]
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    kps = result.keypoints.data[0].cpu().numpy()
    quad = order_corners(kps[:, :2].astype(np.float32))

    # Generate stages
    corners_vis = draw_corners(image, quad)
    dewarped = dewarp(image, quad)
    fields_vis = draw_fields(dewarped, field_model)

    # Resize all to same height
    stages = [
        ("Original", image),
        ("Corner Detection", corners_vis),
        ("Dewarped", dewarped),
        ("Field Detection", fields_vis),
    ]

    resized = []
    for label, img in stages:
        img_r = resize_to_height(img, target_h)
        img_l = add_label(img_r, label)
        resized.append(img_l)

    # Arrow separators
    arrow_w = 50
    strip_h = resized[0].shape[0]
    parts = []
    for i, img in enumerate(resized):
        parts.append(img)
        if i < len(resized) - 1:
            arrow = np.full((strip_h, arrow_w, 3), 255, dtype=np.uint8)
            cy = strip_h // 2
            # Draw arrow: line + head
            cv2.arrowedLine(arrow, (8, cy), (arrow_w - 8, cy),
                            (100, 100, 100), 2, tipLength=0.4)
            parts.append(arrow)

    return np.hstack(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pipeline demo images")
    parser.add_argument("--midv", type=Path,
                        default=Path("/mnt/B/Data/docvision/midv-500/dataset"),
                        help="Path to MIDV-500 dataset root")
    parser.add_argument("--output", type=Path,
                        default=Path("docs/examples"),
                        help="Output directory")
    parser.add_argument("--corner-model", type=Path,
                        default=Path("models/corner_detect.pt"))
    parser.add_argument("--field-model", type=Path,
                        default=Path("models/field_detect.pt"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    corner_model = YOLO(str(args.corner_model))
    field_model = YOLO(str(args.field_model))

    strips = []
    for doc_code, doc_label in DOC_TYPES:
        img_path = find_sample_image(args.midv, doc_code)
        if img_path is None:
            print(f"  SKIP {doc_code}: no images found")
            continue

        print(f"  Processing {doc_code} ({img_path.name})...")
        strip = make_pipeline_image(img_path, corner_model, field_model)
        if strip is None:
            print(f"  SKIP {doc_code}: detection failed")
            continue

        strips.append(strip)

    if not strips:
        print("ERROR: No pipeline images generated")
        return

    # Pad all strips to the same width
    max_w = max(s.shape[1] for s in strips)
    padded = []
    for s in strips:
        if s.shape[1] < max_w:
            pad = np.full((s.shape[0], max_w - s.shape[1], 3), 255, dtype=np.uint8)
            s = np.hstack([s, pad])
        padded.append(s)

    # Vertical stack with separator
    sep_h = 10
    parts = []
    for i, s in enumerate(padded):
        parts.append(s)
        if i < len(padded) - 1:
            sep = np.full((sep_h, max_w, 3), 240, dtype=np.uint8)
            parts.append(sep)

    combined = np.vstack(parts)
    out_path = args.output / "pipeline.jpg"
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved: {out_path} ({combined.shape[1]}x{combined.shape[0]})")


if __name__ == "__main__":
    main()
