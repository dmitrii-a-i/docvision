#!/usr/bin/env python3
"""Perspective-transform pipeline: YOLO corner detection → dewarp → save.

Usage:
    # Single image
    python scripts/dewarp.py --input photo.jpg --output dewarped/

    # Folder
    python scripts/dewarp.py --input photos/ --output dewarped/

    # With side-by-side visualization
    python scripts/dewarp.py --input photos/ --output dewarped/ --viz

    # Custom output size (default: auto from quad geometry)
    python scripts/dewarp.py --input photo.jpg --output dewarped/ --size 800x600
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

log = logging.getLogger("dewarp")


# ── Core functions ──────────────────────────────────────────────────────────


def detect_corners(
    model: YOLO, image_path: str | Path, conf: float = 0.25,
) -> list[np.ndarray] | None:
    """Run YOLO inference and return list of quads (each 4×2 float32, pixels).

    Returns None if no document detected.
    """
    results = model(str(image_path), conf=conf, verbose=False)
    result = results[0]

    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    quads: list[np.ndarray] = []
    for det_idx in range(len(result.keypoints.data)):
        kps = result.keypoints.data[det_idx].cpu().numpy()  # (4, 3)
        quad = kps[:, :2].astype(np.float32)  # (4, 2) — drop confidence
        quads.append(quad)
    return quads


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points to TL, TR, BR, BL via centroid + angle sort."""
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_pts = pts[order]

    # After angle sort the order is roughly: left→bottom→right→top
    # We need TL, TR, BR, BL.  Remap via sums/diffs:
    s = sorted_pts.sum(axis=1)       # x+y
    d = np.diff(sorted_pts, axis=1).ravel()  # y-x

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = sorted_pts[np.argmin(s)]   # TL: smallest x+y
    ordered[1] = sorted_pts[np.argmin(d)]   # TR: smallest y-x
    ordered[2] = sorted_pts[np.argmax(s)]   # BR: largest x+y
    ordered[3] = sorted_pts[np.argmax(d)]   # BL: largest y-x
    return ordered


def compute_output_size(quad: np.ndarray) -> tuple[int, int]:
    """Compute output (width, height) from quad edge lengths."""
    tl, tr, br, bl = quad
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    w = int(round(max(w_top, w_bot)))
    h = int(round(max(h_left, h_right)))
    return w, h


def dewarp(
    image: np.ndarray, quad: np.ndarray, output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Perspective-transform *image* so that *quad* maps to a rectangle.

    Args:
        image: BGR image (OpenCV format).
        quad: 4×2 float32 corners in TL, TR, BR, BL order.
        output_size: (width, height) of the output. Auto-computed from quad if None.

    Returns:
        Dewarped image of shape (height, width, 3).
    """
    if output_size is None:
        output_size = compute_output_size(quad)
    w, h = output_size
    src = quad.astype(np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)


def make_viz(
    original: np.ndarray, quad: np.ndarray, dewarped: np.ndarray,
) -> np.ndarray:
    """Create a side-by-side visualization: original with quad → dewarped."""
    vis = original.copy()
    pts_int = quad.astype(np.int32)

    # Draw filled semi-transparent quad
    overlay = vis.copy()
    cv2.fillConvexPoly(overlay, pts_int, (255, 255, 0))
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    # Draw quad edges
    for i in range(4):
        cv2.line(vis, tuple(pts_int[i]), tuple(pts_int[(i + 1) % 4]), (0, 255, 255), 2)

    # Draw corner circles with labels
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # TL TR BR BL
    labels = ["TL", "TR", "BR", "BL"]
    r = max(6, min(vis.shape[:2]) // 80)
    for i, (label, color) in enumerate(zip(labels, colors)):
        cx, cy = int(pts_int[i][0]), int(pts_int[i][1])
        cv2.circle(vis, (cx, cy), r, color, -1)
        cv2.putText(vis, label, (cx + r + 2, cy - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Scale dewarped to match original height
    oh = vis.shape[0]
    scale = oh / dewarped.shape[0]
    dw_resized = cv2.resize(
        dewarped, (int(dewarped.shape[1] * scale), oh), interpolation=cv2.INTER_AREA,
    )

    # Horizontal concat with a thin separator
    sep = np.full((oh, 4, 3), 128, dtype=np.uint8)
    return np.hstack([vis, sep, dw_resized])


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_size(s: str) -> tuple[int, int]:
    """Parse 'WxH' string into (width, height)."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid size format: {s!r}, expected WxH")
    return int(parts[0]), int(parts[1])


def collect_images(path: Path) -> list[Path]:
    """Return list of image files from a file or directory."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    raise FileNotFoundError(f"Input path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dewarp documents using YOLO corner detection + perspective transform",
    )
    parser.add_argument("--model", type=Path, default=Path("data/models/best.pt"))
    parser.add_argument("--input", type=Path, required=True,
                        help="Image file or directory")
    parser.add_argument("--output", type=Path, default=Path("dewarped"),
                        help="Output directory")
    parser.add_argument("--size", type=parse_size, default=None,
                        help="Output size WxH (default: auto from quad geometry)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--viz", action="store_true",
                        help="Save side-by-side visualization (*_viz.png)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    images = collect_images(args.input)

    if not images:
        log.warning("No images found in %s", args.input)
        return

    size_label = f"{args.size[0]}x{args.size[1]}" if args.size else "auto"
    log.info("Processing %d image(s) → %s  [size=%s]",
             len(images), args.output, size_label)

    saved = 0
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("Cannot read %s, skipping", img_path)
            continue

        quads = detect_corners(model, img_path, conf=args.conf)
        if quads is None:
            log.warning("No document detected in %s, skipping", img_path.name)
            continue

        stem = img_path.stem
        multi = len(quads) > 1

        for idx, raw_quad in enumerate(quads):
            quad = order_corners(raw_quad)
            dewarped = dewarp(image, quad, args.size)

            suffix = f"_{idx}" if multi else ""
            out_path = args.output / f"{stem}{suffix}.png"
            cv2.imwrite(str(out_path), dewarped)
            saved += 1

            if args.viz:
                viz = make_viz(image, quad, dewarped)
                viz_path = args.output / f"{stem}{suffix}_viz.png"
                cv2.imwrite(str(viz_path), viz)

    log.info("Done — saved %d dewarped image(s)", saved)


if __name__ == "__main__":
    main()
