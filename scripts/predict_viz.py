#!/usr/bin/env python3
"""Run inference on val set and produce visualization grids.

Draws predicted keypoints + quad polygon + GT overlay for comparison.

Usage:
    python scripts/predict_viz.py \
        --model runs/pose/doc_corners_300ep/best.pt \
        --data /mnt/B/Data/docvision/yolo_corners \
        --n 24 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

KP_COLORS = ["red", "lime", "blue", "yellow"]  # TL, TR, BR, BL
KP_LABELS = ["TL", "TR", "BR", "BL"]


def parse_gt_label(label_path: Path, img_w: int, img_h: int):
    """Parse YOLO keypoints label file and return denormalized quad."""
    text = label_path.read_text().strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 17:
        return None
    quad = []
    for i in range(4):
        base = 5 + i * 3
        x = float(parts[base]) * img_w
        y = float(parts[base + 1]) * img_h
        quad.append((x, y))
    return quad


def draw_pred(img: Image.Image, keypoints, conf: float) -> None:
    """Draw predicted keypoints and quad on image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(8, min(w, h) // 60)

    pts = []
    for i, (kx, ky, kconf) in enumerate(keypoints):
        pts.append((kx, ky))
        color = KP_COLORS[i]
        draw.ellipse(
            [kx - r, ky - r, kx + r, ky + r],
            fill=color, outline="white", width=2,
        )
        draw.text((kx + r + 4, ky - r), KP_LABELS[i], fill=color)

    # Draw quad polygon
    if len(pts) == 4:
        draw.polygon(pts, outline="cyan")
        for i in range(4):
            draw.line([pts[i], pts[(i + 1) % 4]], fill="cyan", width=3)

    # Confidence label
    draw.text((10, 10), f"conf={conf:.3f}", fill="lime")


def draw_gt(img: Image.Image, quad) -> None:
    """Draw GT quad as dashed magenta overlay."""
    draw = ImageDraw.Draw(img)
    r = max(5, min(img.size) // 80)
    for i in range(4):
        draw.line([quad[i], quad[(i + 1) % 4]], fill="magenta", width=2)
        x, y = quad[i]
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            outline="magenta", width=2,
        )


def make_grid(images: list[Image.Image], cols: int, thumb_size: int) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (30, 30, 30))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        ox = c * thumb_size + (thumb_size - img.width) // 2
        oy = r * thumb_size + (thumb_size - img.height) // 2
        grid.paste(img, (ox, oy))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("runs/pose/doc_corners_300ep/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("/mnt/B/Data/docvision/yolo_corners"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=24, help="Number of samples")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("val_predictions.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(str(args.model))

    img_dir = args.data / "images" / args.split
    lbl_dir = args.data / "labels" / args.split

    images = sorted(img_dir.glob("*.jpg"))
    rng = random.Random(args.seed)
    chosen = rng.sample(images, min(args.n, len(images)))

    panels = []
    for img_path in chosen:
        # Run inference
        results = model(str(img_path), conf=args.conf, verbose=False)
        result = results[0]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Draw GT first (magenta)
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            gt_quad = parse_gt_label(lbl_path, w, h)
            if gt_quad:
                draw_gt(img, gt_quad)

        # Draw predictions (cyan + colored keypoints)
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            for det_idx in range(len(result.keypoints.data)):
                kps = result.keypoints.data[det_idx].cpu().numpy()  # (4, 3)
                conf = float(result.boxes.conf[det_idx].cpu())
                keypoints = [(float(kps[i][0]), float(kps[i][1]), float(kps[i][2])) for i in range(4)]
                draw_pred(img, keypoints, conf)
        else:
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "NO DETECTION", fill="red")

        panels.append(img)

    grid = make_grid(panels, args.cols, args.thumb)
    grid.save(args.output, quality=95)
    print(f"Saved {len(panels)} predictions to {args.output}")


if __name__ == "__main__":
    main()
