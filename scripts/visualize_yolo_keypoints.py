#!/usr/bin/env python3
"""Visualize YOLO keypoint labels overlaid on images.

Picks random samples from the dataset and draws bounding boxes + keypoints.
Saves a grid image for quick visual verification.

Usage:
    python scripts/visualize_yolo_keypoints.py \
        --dataset /mnt/B/Data/docvision/yolo_corners \
        --split train --n 12 --output viz_check.jpg
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


KP_COLORS = ["red", "lime", "blue", "yellow"]  # TL, TR, BR, BL
KP_LABELS = ["TL", "TR", "BR", "BL"]
BBOX_COLOR = "cyan"


def parse_yolo_label(label_path: Path) -> list[dict] | None:
    """Parse a YOLO keypoints label file."""
    text = label_path.read_text().strip()
    if not text:
        return None

    objects = []
    for line in text.splitlines():
        parts = line.split()
        # 0 cx cy w h kp1_x kp1_y kp1_v ... kp4_x kp4_y kp4_v
        if len(parts) < 17:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        keypoints = []
        for i in range(4):
            base = 5 + i * 3
            kx = float(parts[base])
            ky = float(parts[base + 1])
            kv = int(parts[base + 2])
            keypoints.append((kx, ky, kv))
        objects.append({"bbox": (cx, cy, bw, bh), "keypoints": keypoints})

    return objects if objects else None


def draw_annotation(img: Image.Image, ann: dict) -> None:
    """Draw bbox and keypoints on an image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    cx, cy, bw, bh = ann["bbox"]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=3)

    # Draw keypoints and connect them
    pts = []
    for i, (kx, ky, kv) in enumerate(ann["keypoints"]):
        px, py = kx * w, ky * h
        pts.append((px, py))
        r = max(6, min(w, h) // 80)
        color = KP_COLORS[i]
        if kv == 1:
            # Outside frame — draw as hollow circle
            draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=3)
        else:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=2)
        draw.text((px + r + 4, py - r), KP_LABELS[i], fill=color)

    # Draw quad polygon
    if len(pts) == 4:
        draw.polygon(pts, outline="magenta")


def make_grid(images: list[Image.Image], cols: int, thumb_size: int) -> Image.Image:
    """Arrange images into a grid."""
    rows = math.ceil(len(images) / cols)
    grid_w = cols * thumb_size
    grid_h = rows * thumb_size
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        # Resize maintaining aspect ratio
        img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        # Center in cell
        ox = c * thumb_size + (thumb_size - img.width) // 2
        oy = r * thumb_size + (thumb_size - img.height) // 2
        grid.paste(img, (ox, oy))

    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO keypoint labels")
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=12, help="Number of samples to show")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns")
    parser.add_argument("--thumb", type=int, default=640, help="Thumbnail size in px")
    parser.add_argument("--output", type=Path, default=Path("viz_check.jpg"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split

    label_files = sorted(lbl_dir.glob("*.txt"))
    if not label_files:
        print(f"No labels found in {lbl_dir}")
        return

    rng = random.Random(args.seed)
    chosen = rng.sample(label_files, min(args.n, len(label_files)))

    panels = []
    for lbl_file in chosen:
        img_file = img_dir / lbl_file.with_suffix(".jpg").name
        if not img_file.is_file():
            # Try resolving symlink
            continue

        ann_list = parse_yolo_label(lbl_file)
        if not ann_list:
            continue

        img = Image.open(img_file).convert("RGB")
        for ann in ann_list:
            draw_annotation(img, ann)
        panels.append(img)

    if not panels:
        print("No valid samples found")
        return

    grid = make_grid(panels, args.cols, args.thumb)
    grid.save(args.output, quality=95)
    print(f"Saved {len(panels)} annotated samples to {args.output}")


if __name__ == "__main__":
    main()
