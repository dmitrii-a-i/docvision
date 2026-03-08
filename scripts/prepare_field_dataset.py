#!/usr/bin/env python3
"""Prepare YOLO detection dataset for document field detection.

Converts MIDV-500, MIDV-2019, and MIDV-2020 datasets into YOLO detection format
with dewarping for training document field detectors.

Output label format (per dewarped image):
    class_id cx cy w h

Classes: 0=text, 1=photo, 2=signature

Usage:
    python scripts/prepare_field_dataset.py \
        --output /mnt/B/Data/docvision/yolo_fields \
        --datasets midv500 midv2019 midv2020 \
        --val-ratio 0.15 --max-per-type 50 --workers 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dewarp import compute_output_size, dewarp, order_corners

logger = logging.getLogger(__name__)

# Default dataset root paths
MIDV500_ROOT = Path("/mnt/B/Data/docvision/midv-500/dataset")
MIDV2019_ROOT = Path("/mnt/B/Data/docvision/midv-2019/dataset")
MIDV2020_ROOT = Path("/mnt/B/Data/docvision/midv-2020/dataset")

EXCLUDED_FIELDS = {"doc_quad", "face"}
MIN_SIDE_PX = 32   # minimum doc quad side length to accept
MIN_FIELD_PX = 2   # minimum field bbox dimension after scaling


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class FieldAnnotation:
    """A single field on the document template."""

    field_name: str
    class_id: int  # 0=text, 1=photo, 2=signature
    quad: list[tuple[float, float]]  # 4 points in template coordinates


@dataclass
class TemplateInfo:
    """Template-level info shared across frames of the same document."""

    doc_type: str
    template_width: int
    template_height: int
    fields: list[FieldAnnotation]


@dataclass
class FrameSample:
    """A single frame to be dewarped and labelled."""

    source_image: Path
    doc_quad: list[tuple[float, float]]  # 4 corners in frame coordinates
    template: TemplateInfo
    doc_type: str
    source_dataset: str  # "midv2019" | "midv500" | "midv2020"
    output_stem: str


# ── Field classification ─────────────────────────────────────────────────────


def classify_field(name: str) -> int | None:
    """Map field name to class ID. Returns None for excluded fields."""
    if name in EXCLUDED_FIELDS:
        return None
    if name == "photo":
        return 1
    if name == "signature":
        return 2
    return 0  # all text fields


# ── Helpers ──────────────────────────────────────────────────────────────────


def _canonical_doctype(dirname: str) -> str:
    """'01_alb_id' -> 'alb_id'."""
    parts = dirname.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return dirname


def _parse_template_fields_legacy(gt_json: Path) -> list[FieldAnnotation]:
    """Parse field annotations from MIDV-2019/500 template JSON."""
    with open(gt_json) as f:
        data = json.load(f)

    fields: list[FieldAnnotation] = []
    for name, info in data.items():
        cls = classify_field(name)
        if cls is None:
            continue
        quad = info.get("quad")
        if quad is None or len(quad) != 4:
            continue
        fields.append(FieldAnnotation(
            field_name=name,
            class_id=cls,
            quad=[(float(p[0]), float(p[1])) for p in quad],
        ))
    return fields


# ── Parsers ──────────────────────────────────────────────────────────────────


def _parse_midv_legacy(
    root: Path, dataset_name: str, *, double_nested: bool,
) -> Iterator[FrameSample]:
    """Common parser for MIDV-2019 and MIDV-500.

    MIDV-500 has an extra directory nesting level (double_nested=True).
    """
    if not root.is_dir():
        logger.warning("%s root not found: %s", dataset_name, root)
        return

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.endswith(".zip"):
            continue

        doc_type = _canonical_doctype(doc_dir.name)
        base = doc_dir / doc_dir.name if double_nested else doc_dir
        gt_dir = base / "ground_truth"
        img_dir = base / "images"

        if not gt_dir.is_dir():
            logger.warning("Missing ground_truth: %s", gt_dir)
            continue

        # Read template-level field annotations
        template_json = gt_dir / f"{doc_dir.name}.json"
        if not template_json.is_file():
            logger.warning("Missing template JSON: %s", template_json)
            continue

        fields = _parse_template_fields_legacy(template_json)
        if not fields:
            logger.warning("No valid fields in %s", template_json)
            continue

        # Get template image dimensions
        template_img = img_dir / f"{doc_dir.name}.tif"
        if not template_img.is_file():
            logger.warning("Missing template image: %s", template_img)
            continue

        with Image.open(template_img) as img:
            tw, th = img.size

        template = TemplateInfo(
            doc_type=doc_type,
            template_width=tw,
            template_height=th,
            fields=fields,
        )

        # Iterate per-frame JSONs (in subdirectories, skip root-level template JSON)
        for gt_file in sorted(gt_dir.rglob("*.json")):
            rel = gt_file.relative_to(gt_dir)
            if len(rel.parts) < 2:
                continue

            img_file = img_dir / rel.with_suffix(".tif")
            if not img_file.is_file():
                continue

            try:
                with open(gt_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if "quad" not in data:
                continue

            quad = [(float(p[0]), float(p[1])) for p in data["quad"]]
            stem = f"{dataset_name}_{doc_dir.name}_{rel.parent.name}_{rel.stem}"

            yield FrameSample(
                source_image=img_file,
                doc_quad=quad,
                template=template,
                doc_type=doc_type,
                source_dataset=dataset_name,
                output_stem=stem,
            )


def parse_midv2020(root: Path) -> Iterator[FrameSample]:
    """Parse MIDV-2020 dataset (VIA annotation format)."""
    if not root.is_dir():
        logger.warning("MIDV-2020 root not found: %s", root)
        return

    ann_root = root / "annotations"
    img_root = root / "images"

    if not ann_root.is_dir():
        logger.warning("Missing annotations dir: %s", ann_root)
        return

    for doc_ann_file in sorted(ann_root.glob("*.json")):
        doc_type = doc_ann_file.stem

        try:
            with open(doc_ann_file) as f:
                via_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Bad template annotation %s: %s", doc_ann_file, e)
            continue

        # Build per-variant (clip) template info from field annotations
        variant_templates: dict[str, TemplateInfo] = {}
        for _key, meta in via_data.get("_via_img_metadata", {}).items():
            filename = meta.get("filename", "")
            if not filename.endswith(".jpg"):
                continue

            clip_id = Path(filename).stem  # "00.jpg" → "00"

            template_img = img_root / doc_type / filename
            if not template_img.is_file():
                continue

            with Image.open(template_img) as img:
                tw, th = img.size

            fields: list[FieldAnnotation] = []
            for region in meta.get("regions", []):
                attrs = region.get("region_attributes", {})
                field_name = attrs.get("field_name", "")
                cls = classify_field(field_name)
                if cls is None:
                    continue

                shape = region.get("shape_attributes", {})
                if shape.get("name") != "rect":
                    continue

                x = float(shape["x"])
                y = float(shape["y"])
                w = float(shape["width"])
                h = float(shape["height"])
                quad = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                fields.append(FieldAnnotation(
                    field_name=field_name,
                    class_id=cls,
                    quad=quad,
                ))

            if fields:
                variant_templates[clip_id] = TemplateInfo(
                    doc_type=doc_type,
                    template_width=tw,
                    template_height=th,
                    fields=fields,
                )

        # Iterate per-clip annotation files for frame-level doc_quads
        clip_ann_dir = ann_root / doc_type
        if not clip_ann_dir.is_dir():
            continue

        for clip_ann_file in sorted(clip_ann_dir.glob("*.json")):
            clip_id = clip_ann_file.stem

            template = variant_templates.get(clip_id)
            if template is None:
                continue

            try:
                with open(clip_ann_file) as f:
                    clip_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            clip_img_dir = img_root / doc_type / clip_id
            if not clip_img_dir.is_dir():
                continue

            for _key, meta in clip_data.get("_via_img_metadata", {}).items():
                filename = meta.get("filename", "")
                if not filename.endswith(".jpg"):
                    continue

                # Extract doc_quad from polygon annotation
                doc_quad = None
                for region in meta.get("regions", []):
                    attrs = region.get("region_attributes", {})
                    if attrs.get("field_name") == "doc_quad":
                        shape = region.get("shape_attributes", {})
                        xs = shape.get("all_points_x", [])
                        ys = shape.get("all_points_y", [])
                        if len(xs) == 4 and len(ys) == 4:
                            doc_quad = [
                                (float(x), float(y))
                                for x, y in zip(xs, ys)
                            ]
                        break

                if doc_quad is None:
                    continue

                img_path = clip_img_dir / filename
                if not img_path.is_file():
                    continue

                frame_stem = Path(filename).stem
                stem = f"midv2020_{doc_type}_{clip_id}_{frame_stem}"

                yield FrameSample(
                    source_image=img_path,
                    doc_quad=doc_quad,
                    template=template,
                    doc_type=doc_type,
                    source_dataset="midv2020",
                    output_stem=stem,
                )


# ── Processing ───────────────────────────────────────────────────────────────


def _process_sample(args: tuple) -> str | None:
    """Process one frame: dewarp + scale fields + write .jpg/.txt. For multiprocessing."""
    sample: FrameSample
    output_dir: Path
    split: str
    sample, output_dir, split = args

    # 1. Read source image
    image = cv2.imread(str(sample.source_image))
    if image is None:
        return f"Cannot read: {sample.source_image}"

    # 2. Order corners and check for degenerate quad
    quad = np.array(sample.doc_quad, dtype=np.float32)
    ordered = order_corners(quad)

    for i in range(4):
        side = float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i]))
        if side < MIN_SIDE_PX:
            return f"Degenerate quad ({side:.0f}px side): {sample.output_stem}"

    # 3. Dewarp
    dw, dh = compute_output_size(ordered)
    dewarped = dewarp(image, ordered, (dw, dh))

    # 4. Scale template fields → dewarped coordinates, convert to YOLO format
    tw = sample.template.template_width
    th = sample.template.template_height
    sx = dw / tw
    sy = dh / th

    lines: list[str] = []
    for field in sample.template.fields:
        xs = [p[0] * sx for p in field.quad]
        ys = [p[1] * sy for p in field.quad]

        # Axis-aligned bbox, clamped to image bounds
        x_min = max(0.0, min(xs))
        y_min = max(0.0, min(ys))
        x_max = min(float(dw), max(xs))
        y_max = min(float(dh), max(ys))

        if (x_max - x_min) < MIN_FIELD_PX or (y_max - y_min) < MIN_FIELD_PX:
            continue

        # Normalize to [0, 1]
        cx = ((x_min + x_max) / 2) / dw
        cy = ((y_min + y_max) / 2) / dh
        bw = (x_max - x_min) / dw
        bh = (y_max - y_min) / dh

        lines.append(f"{field.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return f"No valid fields: {sample.output_stem}"

    # 5. Save dewarped image + label
    img_out = output_dir / "images" / split / f"{sample.output_stem}.jpg"
    lbl_out = output_dir / "labels" / split / f"{sample.output_stem}.txt"

    cv2.imwrite(str(img_out), dewarped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lbl_out.write_text("\n".join(lines) + "\n")

    return None


# ── Split & output ───────────────────────────────────────────────────────────


def split_by_doctype(
    samples: list[FrameSample], val_ratio: float, seed: int,
) -> tuple[list[FrameSample], list[FrameSample]]:
    """Split by document type (not individual frames) to avoid data leakage."""
    by_type: dict[str, list[FrameSample]] = {}
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
        len(doc_types) - n_val, n_val, len(doc_types),
    )
    logger.info("Val doc types: %s", sorted(val_types))

    train: list[FrameSample] = []
    val: list[FrameSample] = []
    for dt in sorted(by_type):
        target = val if dt in val_types else train
        target.extend(by_type[dt])

    return train, val


def write_dataset(
    samples: list[FrameSample],
    output_dir: Path,
    split: str,
    workers: int,
) -> None:
    """Write all samples for a split to disk (dewarp + labels)."""
    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    task_args = [(s, output_dir, split) for s in samples]

    errors = 0
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.imap_unordered(
                _process_sample, task_args, chunksize=32,
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
    """Generate dataset.yaml for Ultralytics YOLO."""
    content = (
        "# YOLO detection dataset for document field detection\n"
        "# Auto-generated by prepare_field_dataset.py\n"
        "\n"
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: text\n"
        "  1: photo\n"
        "  2: signature\n"
    )
    yaml_file = output_dir / "dataset.yaml"
    yaml_file.write_text(content)
    logger.info("Wrote %s", yaml_file)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO detection dataset for document field detection",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for the YOLO dataset",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["midv500", "midv2019", "midv2020"],
        choices=["midv500", "midv2019", "midv2020"],
        help="Datasets to include (default: all three)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of doc types for validation (default: 0.15)",
    )
    parser.add_argument(
        "--max-per-type", type=int, default=0,
        help="Max samples per doc type, 0=unlimited (default: 0)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for image processing (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ── Parse datasets ──
    all_samples: list[FrameSample] = []

    if "midv2019" in args.datasets:
        logger.info("Parsing MIDV-2019...")
        samples = list(_parse_midv_legacy(
            MIDV2019_ROOT, "midv2019", double_nested=False,
        ))
        logger.info("MIDV-2019: %d samples", len(samples))
        all_samples.extend(samples)

    if "midv500" in args.datasets:
        logger.info("Parsing MIDV-500...")
        samples = list(_parse_midv_legacy(
            MIDV500_ROOT, "midv500", double_nested=True,
        ))
        logger.info("MIDV-500: %d samples", len(samples))
        all_samples.extend(samples)

    if "midv2020" in args.datasets:
        logger.info("Parsing MIDV-2020...")
        samples = list(parse_midv2020(MIDV2020_ROOT))
        logger.info("MIDV-2020: %d samples", len(samples))
        all_samples.extend(samples)

    if not all_samples:
        logger.error("No samples found!")
        sys.exit(1)

    # ── Apply per-type limit ──
    if args.max_per_type > 0:
        by_type: dict[str, list[FrameSample]] = {}
        for s in all_samples:
            by_type.setdefault(s.doc_type, []).append(s)

        limited: list[FrameSample] = []
        for dt in sorted(by_type):
            limited.extend(by_type[dt][: args.max_per_type])

        logger.info(
            "Limited %d → %d samples (max %d per type)",
            len(all_samples), len(limited), args.max_per_type,
        )
        all_samples = limited

    logger.info("Total: %d samples", len(all_samples))

    # ── Split & write ──
    train_samples, val_samples = split_by_doctype(
        all_samples, args.val_ratio, args.seed,
    )
    logger.info("Train: %d, Val: %d", len(train_samples), len(val_samples))

    args.output.mkdir(parents=True, exist_ok=True)
    write_dataset(train_samples, args.output, "train", args.workers)
    write_dataset(val_samples, args.output, "val", args.workers)
    write_yaml(args.output)

    logger.info("Done! Dataset at %s", args.output)
    logger.info("  Train: %d images", len(train_samples))
    logger.info("  Val:   %d images", len(val_samples))


if __name__ == "__main__":
    main()
