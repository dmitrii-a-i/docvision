#!/usr/bin/env python3
"""Prepare a compact VLM test set from MIDV datasets.

Samples frames, dewarps them using GT doc_quad, saves dewarped images + gt.json.
Prioritizes MIDV-2020 (semantic field names) over MIDV-500/2019 (opaque names).

Usage:
    python scripts/prepare_vlm_testset.py --output /tmp/vlm_testset/
    python scripts/prepare_vlm_testset.py --output /tmp/vlm_testset/ --frames-per-type 5 --max-frames 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dewarp import compute_output_size, dewarp, order_corners
from eval_ocr import (
    MIDV500_ROOT,
    MIDV2019_ROOT,
    MIDV2020_ROOT,
    SKIP_FIELDS,
    TextFieldSample,
    parse_text_fields_midv2020,
    parse_text_fields_midv_legacy,
)

log = logging.getLogger("prepare_vlm_testset")


def group_by_frame(
    samples: list[TextFieldSample],
) -> dict[Path, list[TextFieldSample]]:
    """Group field samples by source image (= one frame)."""
    by_frame: dict[Path, list[TextFieldSample]] = defaultdict(list)
    for s in samples:
        by_frame[s.source_image].append(s)
    return by_frame


def stratified_sample(
    samples: list[TextFieldSample],
    frames_per_type: int,
    max_frames: int,
    seed: int,
) -> list[Path]:
    """Stratified sampling: up to frames_per_type frames per doc_type, capped at max_frames.

    Prioritizes MIDV-2020 by sorting doc_types so midv2020 types come first.
    """
    rng = random.Random(seed)

    # Group frames by (source_dataset, doc_type)
    by_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
    by_frame = group_by_frame(samples)
    for frame_path, frame_samples in by_frame.items():
        s0 = frame_samples[0]
        key = (s0.source_dataset, s0.doc_type)
        by_key[key].append(frame_path)

    # Sort keys: midv2020 first, then midv2019, then midv500
    dataset_priority = {"midv2020": 0, "midv2019": 1, "midv500": 2}
    sorted_keys = sorted(by_key.keys(), key=lambda k: (dataset_priority.get(k[0], 9), k[1]))

    selected: list[Path] = []
    for key in sorted_keys:
        if len(selected) >= max_frames:
            break
        frames = by_key[key]
        rng.shuffle(frames)
        n = min(frames_per_type, len(frames), max_frames - len(selected))
        selected.extend(frames[:n])

    selected.sort()
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a compact VLM test set from MIDV datasets",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/vlm_testset"),
        help="Output directory (default: /tmp/vlm_testset)",
    )
    parser.add_argument(
        "--frames-per-type", type=int, default=3,
        help="Max frames to sample per doc_type (default: 3)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=50,
        help="Max total frames (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Collect all text field samples
    all_samples: list[TextFieldSample] = []

    if MIDV2020_ROOT.exists():
        log.info("Parsing MIDV-2020...")
        all_samples.extend(parse_text_fields_midv2020(MIDV2020_ROOT))

    if MIDV2019_ROOT.exists():
        log.info("Parsing MIDV-2019...")
        all_samples.extend(
            parse_text_fields_midv_legacy(MIDV2019_ROOT, "midv2019", double_nested=False)
        )

    if MIDV500_ROOT.exists():
        log.info("Parsing MIDV-500...")
        all_samples.extend(
            parse_text_fields_midv_legacy(MIDV500_ROOT, "midv500", double_nested=True)
        )

    log.info("Total text field samples: %d", len(all_samples))
    if not all_samples:
        log.error("No samples found. Check dataset paths.")
        sys.exit(1)

    # Stratified sampling
    by_frame = group_by_frame(all_samples)
    selected_frames = stratified_sample(
        all_samples, args.frames_per_type, args.max_frames, args.seed,
    )
    log.info("Selected %d frames", len(selected_frames))

    # Prepare output dirs
    out_dir = args.output
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    gt: dict[str, dict] = {}
    saved = 0

    for frame_path in selected_frames:
        frame_samples = by_frame[frame_path]
        s0 = frame_samples[0]

        # Read source image
        image = cv2.imread(str(frame_path))
        if image is None:
            log.warning("Cannot read %s, skipping", frame_path)
            continue

        # Dewarp using GT doc_quad
        quad = np.array(s0.doc_quad, dtype=np.float32)
        ordered = order_corners(quad)
        dw, dh = compute_output_size(ordered)
        if dw < 32 or dh < 32:
            log.warning("Dewarped size too small for %s, skipping", frame_path)
            continue
        dewarped = dewarp(image, ordered, (dw, dh))

        # Generate a unique stem
        stem = f"{s0.source_dataset}_{s0.doc_type}_{frame_path.stem}"
        out_path = img_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), dewarped)
        saved += 1

        # Collect GT fields for this frame
        fields: dict[str, str] = {}
        for s in frame_samples:
            fields[s.field_name] = s.gt_value

        gt[stem] = {
            "doc_type": s0.doc_type,
            "source_dataset": s0.source_dataset,
            "fields": fields,
        }

    # Save gt.json
    gt_path = out_dir / "gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    log.info("Saved %d dewarped images to %s", saved, img_dir)
    log.info("Saved GT for %d frames to %s", len(gt), gt_path)

    # Summary
    datasets = defaultdict(int)
    doc_types = defaultdict(int)
    for entry in gt.values():
        datasets[entry["source_dataset"]] += 1
        doc_types[entry["doc_type"]] += 1

    print(f"\nTest set summary: {len(gt)} frames, {sum(len(e['fields']) for e in gt.values())} fields")
    print(f"  Datasets: {dict(datasets)}")
    print(f"  Doc types ({len(doc_types)}): {dict(doc_types)}")


if __name__ == "__main__":
    main()
