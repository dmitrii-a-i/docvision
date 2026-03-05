#!/usr/bin/env python3
"""Prepare YOLO keypoint dataset for document corner detection.

Converts MIDV-500, MIDV-2019, and MIDV-2020 datasets into YOLO keypoints format
for training YOLOv8n pose estimation models.

Output label format (per image):
    0 cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v kp4_x kp4_y kp4_v

Keypoints: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left).
All coordinates normalized to [0, 1]. Visibility: 2=visible, 1=outside frame.

Usage:
    python scripts/prepare_yolo_dataset.py \
        --output /mnt/B/Data/docvision/yolo_corners \
        --datasets midv500 midv2019 midv2020 \
        --val-ratio 0.15 --subsample 5 --workers 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Known image dimensions (width, height) per dataset
MIDV500_SIZE = (1080, 1920)
MIDV2019_SIZE = (2160, 3840)
MIDV2020_SIZE = (2160, 3840)

# Default dataset root paths
MIDV500_ROOT = Path("/mnt/B/Data/docvision/midv-500/dataset")
MIDV2019_ROOT = Path("/mnt/B/Data/docvision/midv-2019/dataset")
MIDV2020_ROOT = Path("/mnt/B/Data/docvision/midv-2020/dataset")


@dataclass
class Sample:
    """A single annotated image sample."""

    source_image: Path
    image_width: int
    image_height: int
    quad: list[tuple[int, int]]  # 4 corners: TL, TR, BR, BL
    doc_type: str  # canonical document type name
    source_dataset: str  # midv500, midv2019, midv2020
    output_stem: str  # unique output filename (without extension)


def canonical_doctype(dirname: str) -> str:
    """Extract canonical document type name.

    '01_alb_id' -> 'alb_id', 'alb_id' -> 'alb_id'
    """
    parts = dirname.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return dirname


# ---------------------------------------------------------------------------
# Parsers — each yields Iterator[Sample]
# ---------------------------------------------------------------------------


def parse_midv500(root: Path) -> Iterator[Sample]:
    """Parse MIDV-500. Double-nested: NN_type/NN_type/{images,ground_truth}/sub/."""
    if not root.is_dir():
        logger.warning("MIDV-500 root not found: %s", root)
        return

    w, h = MIDV500_SIZE

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.endswith(".zip"):
            continue

        doc_type = canonical_doctype(doc_dir.name)
        inner = doc_dir / doc_dir.name  # double nesting
        gt_dir = inner / "ground_truth"
        img_dir = inner / "images"

        if not gt_dir.is_dir():
            logger.warning("Missing ground_truth: %s", gt_dir)
            continue

        for gt_file in sorted(gt_dir.rglob("*.json")):
            # Skip root-level GT (field-level annotations, no per-frame quad)
            rel = gt_file.relative_to(gt_dir)
            if len(rel.parts) < 2:
                continue

            img_file = img_dir / rel.with_suffix(".tif")
            if not img_file.is_file():
                logger.warning("Missing image: %s", img_file)
                continue

            try:
                with open(gt_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Bad GT %s: %s", gt_file, e)
                continue

            if "quad" not in data:
                continue

            quad = [tuple(pt) for pt in data["quad"]]
            stem = f"midv500_{doc_dir.name}_{rel.parent.name}_{rel.stem}"

            yield Sample(
                source_image=img_file,
                image_width=w,
                image_height=h,
                quad=quad,
                doc_type=doc_type,
                source_dataset="midv500",
                output_stem=stem,
            )


def parse_midv2019(root: Path) -> Iterator[Sample]:
    """Parse MIDV-2019. Single-nested: NN_type/{images,ground_truth}/sub/."""
    if not root.is_dir():
        logger.warning("MIDV-2019 root not found: %s", root)
        return

    w, h = MIDV2019_SIZE

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.endswith(".zip"):
            continue

        doc_type = canonical_doctype(doc_dir.name)
        gt_dir = doc_dir / "ground_truth"
        img_dir = doc_dir / "images"

        if not gt_dir.is_dir():
            logger.warning("Missing ground_truth: %s", gt_dir)
            continue

        for gt_file in sorted(gt_dir.rglob("*.json")):
            rel = gt_file.relative_to(gt_dir)
            if len(rel.parts) < 2:
                continue

            img_file = img_dir / rel.with_suffix(".tif")
            if not img_file.is_file():
                logger.warning("Missing image: %s", img_file)
                continue

            try:
                with open(gt_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Bad GT %s: %s", gt_file, e)
                continue

            if "quad" not in data:
                continue

            quad = [tuple(pt) for pt in data["quad"]]
            stem = f"midv2019_{doc_dir.name}_{rel.parent.name}_{rel.stem}"

            yield Sample(
                source_image=img_file,
                image_width=w,
                image_height=h,
                quad=quad,
                doc_type=doc_type,
                source_dataset="midv2019",
                output_stem=stem,
            )


def parse_midv2020(root: Path, subsample_rate: int = 5) -> Iterator[Sample]:
    """Parse MIDV-2020. VIA 2.0.11 annotations, clip frames as JPEG."""
    if not root.is_dir():
        logger.warning("MIDV-2020 root not found: %s", root)
        return

    w, h = MIDV2020_SIZE
    ann_root = root / "annotations"
    img_root = root / "images"

    if not ann_root.is_dir():
        logger.warning("Missing annotations dir: %s", ann_root)
        return

    for doc_dir in sorted(ann_root.iterdir()):
        if not doc_dir.is_dir():
            continue

        doc_type = doc_dir.name  # already canonical (e.g. 'alb_id')

        for ann_file in sorted(doc_dir.glob("*.json")):
            clip_id = ann_file.stem  # e.g. '00'
            clip_img_dir = img_root / doc_type / clip_id

            if not clip_img_dir.is_dir():
                logger.warning("Missing clip dir: %s", clip_img_dir)
                continue

            try:
                with open(ann_file) as f:
                    via_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Bad annotation %s: %s", ann_file, e)
                continue

            img_metadata = via_data.get("_via_img_metadata", {})

            # Collect frames with doc_quad annotations
            frames: list[tuple[str, list[tuple[int, int]]]] = []
            for _key, meta in img_metadata.items():
                filename = meta.get("filename", "")
                if not filename.endswith(".jpg"):
                    continue

                # Find doc_quad region
                quad = None
                for region in meta.get("regions", []):
                    attrs = region.get("region_attributes", {})
                    if attrs.get("field_name") == "doc_quad":
                        shape = region.get("shape_attributes", {})
                        xs = shape.get("all_points_x", [])
                        ys = shape.get("all_points_y", [])
                        if len(xs) == 4 and len(ys) == 4:
                            quad = list(zip(xs, ys))
                        break

                if quad is not None:
                    frames.append((filename, quad))

            # Sort by filename, then subsample
            frames.sort(key=lambda x: x[0])
            for i, (filename, quad) in enumerate(frames):
                if i % subsample_rate != 0:
                    continue

                img_path = clip_img_dir / filename
                if not img_path.is_file():
                    logger.warning("Missing image: %s", img_path)
                    continue

                frame_stem = Path(filename).stem
                stem = f"midv2020_{doc_type}_{clip_id}_{frame_stem}"

                yield Sample(
                    source_image=img_path,
                    image_width=w,
                    image_height=h,
                    quad=quad,
                    doc_type=doc_type,
                    source_dataset="midv2020",
                    output_stem=stem,
                )


# ---------------------------------------------------------------------------
# Conversion & splitting
# ---------------------------------------------------------------------------


def quad_to_yolo_keypoints(
    quad: list[tuple[int, int]], img_w: int, img_h: int
) -> str | None:
    """Convert quad corners to YOLO keypoints label string.

    Returns None for degenerate bounding boxes (all corners on one side).
    """
    if len(quad) != 4:
        return None

    keypoints = []
    clamped_xs = []
    clamped_ys = []

    for x, y in quad:
        nx = x / img_w
        ny = y / img_h

        outside = nx < 0 or nx > 1 or ny < 0 or ny > 1
        visibility = 1 if outside else 2

        cnx = max(0.0, min(1.0, nx))
        cny = max(0.0, min(1.0, ny))

        clamped_xs.append(cnx)
        clamped_ys.append(cny)
        keypoints.append((cnx, cny, visibility))

    # Bounding box from clamped keypoints
    x_min, x_max = min(clamped_xs), max(clamped_xs)
    y_min, y_max = min(clamped_ys), max(clamped_ys)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    if bbox_w < 1e-6 or bbox_h < 1e-6:
        return None

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    parts = [f"0 {cx:.6f} {cy:.6f} {bbox_w:.6f} {bbox_h:.6f}"]
    for kx, ky, kv in keypoints:
        parts.append(f"{kx:.6f} {ky:.6f} {kv}")

    return " ".join(parts)


def split_by_doctype(
    samples: list[Sample], val_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split samples by document type to avoid data leakage.

    All samples of the same canonical doc type go to the same split,
    even if they come from different source datasets.
    """
    by_type: dict[str, list[Sample]] = {}
    for s in samples:
        by_type.setdefault(s.doc_type, []).append(s)

    doc_types = sorted(by_type.keys())
    n_val = max(1, round(len(doc_types) * val_ratio))

    rng = random.Random(seed)
    shuffled = list(doc_types)
    rng.shuffle(shuffled)

    val_types = set(shuffled[:n_val])

    logger.info(
        "Doc type split: %d train, %d val (%d total)",
        len(doc_types) - n_val,
        n_val,
        len(doc_types),
    )
    logger.info("Val doc types: %s", sorted(val_types))

    train: list[Sample] = []
    val: list[Sample] = []
    for dt in sorted(by_type):
        target = val if dt in val_types else train
        target.extend(by_type[dt])

    return train, val


# ---------------------------------------------------------------------------
# Writing output
# ---------------------------------------------------------------------------


def _process_sample(args: tuple) -> str | None:
    """Process one sample: convert/link image + write label. Returns error or None."""
    sample: Sample
    output_dir: Path
    split: str
    use_symlinks: bool
    jpeg_quality: int
    sample, output_dir, split, use_symlinks, jpeg_quality = args

    img_out_dir = output_dir / "images" / split
    lbl_out_dir = output_dir / "labels" / split

    label = quad_to_yolo_keypoints(sample.quad, sample.image_width, sample.image_height)
    if label is None:
        return f"Skipped degenerate: {sample.output_stem}"

    # Write label
    lbl_file = lbl_out_dir / f"{sample.output_stem}.txt"
    lbl_file.write_text(label + "\n")

    # Handle image
    img_out = img_out_dir / f"{sample.output_stem}.jpg"

    if sample.source_image.suffix.lower() in (".tif", ".tiff"):
        try:
            with Image.open(sample.source_image) as img:
                img = img.convert("RGB")
                img.save(img_out, "JPEG", quality=jpeg_quality)
        except Exception as e:
            lbl_file.unlink(missing_ok=True)
            return f"TIFF convert failed {sample.source_image}: {e}"
    else:
        if use_symlinks:
            try:
                img_out.symlink_to(sample.source_image.resolve())
            except FileExistsError:
                pass
        else:
            shutil.copy2(sample.source_image, img_out)

    return None


def write_dataset(
    samples: list[Sample],
    output_dir: Path,
    split: str,
    use_symlinks: bool,
    jpeg_quality: int,
    workers: int,
) -> None:
    """Write all samples for a split to disk."""
    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    task_args = [
        (s, output_dir, split, use_symlinks, jpeg_quality) for s in samples
    ]

    errors = 0
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.imap_unordered(
                _process_sample, task_args, chunksize=32
            )
            for result in tqdm(results, total=len(task_args), desc=f"Writing {split}"):
                if result is not None:
                    logger.warning(result)
                    errors += 1
    else:
        for a in tqdm(task_args, desc=f"Writing {split}"):
            result = _process_sample(a)
            if result is not None:
                logger.warning(result)
                errors += 1

    written = len(task_args) - errors
    logger.info("%s: %d written, %d errors", split, written, errors)


def write_yaml(output_dir: Path) -> None:
    """Generate dataset.yaml for Ultralytics."""
    content = (
        "# YOLO Keypoints dataset for document corner detection\n"
        "# Auto-generated by prepare_yolo_dataset.py\n"
        "\n"
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "# 4 keypoints (TL, TR, BR, BL), each with (x, y, visibility)\n"
        "kpt_shape: [4, 3]\n"
        "\n"
        "# Flip indices for horizontal augmentation: TL<->TR, BL<->BR\n"
        "flip_idx: [1, 0, 3, 2]\n"
        "\n"
        "names:\n"
        "  0: document\n"
    )
    yaml_file = output_dir / "dataset.yaml"
    yaml_file.write_text(content)
    logger.info("Wrote %s", yaml_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO keypoint dataset for document corner detection"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the YOLO dataset",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["midv500", "midv2019", "midv2020"],
        choices=["midv500", "midv2019", "midv2020"],
        help="Datasets to include (default: all three)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of doc types for validation (default: 0.15)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=5,
        help="MIDV-2020 clip subsample rate (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for image conversion (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy JPEG files instead of creating symlinks",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for TIFF conversion (default: 95)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ---- Parse datasets ----
    all_samples: list[Sample] = []

    if "midv500" in args.datasets:
        logger.info("Parsing MIDV-500...")
        samples = list(parse_midv500(MIDV500_ROOT))
        logger.info("MIDV-500: %d samples", len(samples))
        all_samples.extend(samples)

    if "midv2019" in args.datasets:
        logger.info("Parsing MIDV-2019...")
        samples = list(parse_midv2019(MIDV2019_ROOT))
        logger.info("MIDV-2019: %d samples", len(samples))
        all_samples.extend(samples)

    if "midv2020" in args.datasets:
        logger.info("Parsing MIDV-2020 (subsample=%d)...", args.subsample)
        samples = list(parse_midv2020(MIDV2020_ROOT, args.subsample))
        logger.info("MIDV-2020: %d samples", len(samples))
        all_samples.extend(samples)

    if not all_samples:
        logger.error("No samples found!")
        sys.exit(1)

    logger.info("Total: %d samples", len(all_samples))

    # ---- Split by doc type ----
    train_samples, val_samples = split_by_doctype(
        all_samples, args.val_ratio, args.seed
    )
    logger.info("Train: %d, Val: %d", len(train_samples), len(val_samples))

    # ---- Write output ----
    args.output.mkdir(parents=True, exist_ok=True)
    use_symlinks = not args.copy

    write_dataset(
        train_samples, args.output, "train", use_symlinks, args.jpeg_quality, args.workers
    )
    write_dataset(
        val_samples, args.output, "val", use_symlinks, args.jpeg_quality, args.workers
    )

    write_yaml(args.output)

    logger.info("Done! Dataset at %s", args.output)
    logger.info("  Train: %d images", len(train_samples))
    logger.info("  Val:   %d images", len(val_samples))


if __name__ == "__main__":
    main()
