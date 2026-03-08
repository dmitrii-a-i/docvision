#!/usr/bin/env python3
"""Compare EasyOCR vs PaddleOCR on MIDV GT field crops.

Usage:
    python scripts/eval_ocr_engines.py --max-samples 200
    python scripts/eval_ocr_engines.py --max-samples 500 --output data/ocr_comparison/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dewarp import compute_output_size, dewarp, order_corners
from eval_ocr import (
    MIDV2019_ROOT,
    MIDV2020_ROOT,
    MIDV500_ROOT,
    MIN_CROP_PX,
    FieldResult,
    TextFieldSample,
    compute_cer,
    parse_text_fields_midv2020,
    parse_text_fields_midv_legacy,
    save_viz,
)

log = logging.getLogger("eval_ocr_engines")


# ── Engine factories ─────────────────────────────────────────────────────────


def make_easyocr_reader(lang: list[str], gpu: bool):
    """Return a function: crop (np.ndarray) -> str."""
    import easyocr

    reader = easyocr.Reader(lang, gpu=gpu)

    def read_crop(crop: np.ndarray) -> str:
        return " ".join(reader.readtext(crop, detail=0))

    return read_crop


def make_paddleocr_reader(lang: str, use_gpu: bool):
    """Return a function: crop (np.ndarray) -> str."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def read_crop(crop: np.ndarray) -> str:
        result = ocr.ocr(crop, cls=True)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for line in result[0] if line[1]]
        return " ".join(lines)

    return read_crop


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_engine(
    samples: list[TextFieldSample],
    read_fn,
    max_samples: int,
    seed: int,
    collect_crops: bool = False,
) -> tuple[list[FieldResult], list[np.ndarray], int, float]:
    """Evaluate OCR engine on sampled frames. Returns (results, crops, n_frames, total_time)."""
    from tqdm import tqdm

    # Group by frame
    by_frame: dict[Path, list[TextFieldSample]] = defaultdict(list)
    for s in samples:
        by_frame[s.source_image].append(s)

    frames = list(by_frame.keys())
    if len(frames) > max_samples:
        rng = random.Random(seed)
        frames = rng.sample(frames, max_samples)
    frames.sort()

    results: list[FieldResult] = []
    crops: list[np.ndarray] = []
    total_ocr_time = 0.0

    for frame_path in tqdm(frames, desc="OCR eval"):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        frame_samples = by_frame[frame_path]
        sample0 = frame_samples[0]

        quad = np.array(sample0.doc_quad, dtype=np.float32)
        ordered = order_corners(quad)
        dw, dh = compute_output_size(ordered)
        if dw < 32 or dh < 32:
            continue
        dewarped = dewarp(image, ordered, (dw, dh))

        sx = dw / sample0.template_width
        sy = dh / sample0.template_height

        for sample in frame_samples:
            pts = np.array(sample.field_quad, dtype=np.float32)
            pts[:, 0] *= sx
            pts[:, 1] *= sy

            x_min = max(0, int(np.floor(pts[:, 0].min())))
            y_min = max(0, int(np.floor(pts[:, 1].min())))
            x_max = min(dw, int(np.ceil(pts[:, 0].max())))
            y_max = min(dh, int(np.ceil(pts[:, 1].max())))

            if (x_max - x_min) < MIN_CROP_PX or (y_max - y_min) < MIN_CROP_PX:
                continue

            crop = dewarped[y_min:y_max, x_min:x_max]

            t0 = time.perf_counter()
            ocr_text = read_fn(crop)
            total_ocr_time += time.perf_counter() - t0

            gt_norm = sample.gt_value.strip()
            ocr_norm = ocr_text.strip()
            cer = compute_cer(gt_norm, ocr_norm)
            exact = gt_norm.lower() == ocr_norm.lower()

            results.append(FieldResult(
                field_name=sample.field_name,
                doc_type=sample.doc_type,
                gt_value=gt_norm,
                ocr_text=ocr_norm,
                cer=cer,
                exact_match=exact,
            ))
            if collect_crops:
                crops.append(crop.copy())

    return results, crops, len(frames), total_ocr_time


# ── Reporting ────────────────────────────────────────────────────────────────


def engine_summary(name: str, results: list[FieldResult], n_frames: int, ocr_time: float) -> dict:
    """Compute summary metrics for one engine."""
    if not results:
        return {"engine": name, "n_fields": 0}

    total_cer = sum(r.cer for r in results) / len(results)
    total_em = sum(r.exact_match for r in results) / len(results)
    avg_time = (ocr_time / len(results)) * 1000  # ms per field

    return {
        "engine": name,
        "n_frames": n_frames,
        "n_fields": len(results),
        "avg_cer": round(total_cer, 4),
        "exact_match_rate": round(total_em, 4),
        "avg_ms_per_field": round(avg_time, 1),
        "total_ocr_time_s": round(ocr_time, 1),
    }


def print_comparison(summaries: list[dict]) -> None:
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 70}")
    print("  OCR Engine Comparison")
    print(f"{'=' * 70}")
    print(f"  {'Engine':<15s} {'Fields':>7s} {'CER':>8s} {'Exact':>8s} {'ms/field':>10s}")
    print(f"  {'-' * 50}")
    for s in summaries:
        if s["n_fields"] == 0:
            continue
        print(f"  {s['engine']:<15s} {s['n_fields']:>7d} {s['avg_cer']:>8.4f} "
              f"{s['exact_match_rate']:>7.1%} {s['avg_ms_per_field']:>9.1f}")
    print(f"{'=' * 70}")

    # Winner
    valid = [s for s in summaries if s["n_fields"] > 0]
    if len(valid) >= 2:
        best = min(valid, key=lambda s: s["avg_cer"])
        print(f"\n  Winner by CER: {best['engine']} (CER={best['avg_cer']:.4f})")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR engines on MIDV GT crops")
    parser.add_argument("--engines", nargs="+", default=["easyocr", "paddleocr"],
                        choices=["easyocr", "paddleocr"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for results")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Collect samples
    all_samples: list[TextFieldSample] = []

    if MIDV500_ROOT.exists():
        log.info("Parsing MIDV-500...")
        all_samples.extend(parse_text_fields_midv_legacy(
            MIDV500_ROOT, "midv500", double_nested=True,
        ))

    if MIDV2019_ROOT.exists():
        log.info("Parsing MIDV-2019...")
        all_samples.extend(parse_text_fields_midv_legacy(
            MIDV2019_ROOT, "midv2019", double_nested=False,
        ))

    if MIDV2020_ROOT.exists():
        log.info("Parsing MIDV-2020...")
        all_samples.extend(parse_text_fields_midv2020(MIDV2020_ROOT))

    log.info("Total text field samples: %d", len(all_samples))
    if not all_samples:
        log.error("No samples found. Check dataset paths.")
        return

    use_gpu = not args.no_gpu
    save_crops = args.output is not None
    summaries = []

    for engine_name in args.engines:
        print(f"\n--- {engine_name.upper()} ---")

        if engine_name == "easyocr":
            read_fn = make_easyocr_reader([args.lang], gpu=use_gpu)
        else:
            read_fn = make_paddleocr_reader(args.lang, use_gpu=use_gpu)

        results, crops, n_frames, ocr_time = evaluate_engine(
            all_samples, read_fn, args.max_samples, args.seed,
            collect_crops=save_crops,
        )

        summary = engine_summary(engine_name, results, n_frames, ocr_time)
        summaries.append(summary)

        # Per-doctype breakdown
        by_doc: dict[str, list[FieldResult]] = defaultdict(list)
        for r in results:
            by_doc[r.doc_type].append(r)

        print(f"\n  {'doc_type':<25s} {'count':>6s} {'CER':>8s} {'exact':>8s}")
        for dtype in sorted(by_doc):
            group = by_doc[dtype]
            cer = sum(r.cer for r in group) / len(group)
            em = sum(r.exact_match for r in group) / len(group)
            print(f"  {dtype:<25s} {len(group):>6d} {cer:>8.3f} {em:>7.1%}")

        # Save viz
        if args.output and results and crops:
            args.output.mkdir(parents=True, exist_ok=True)
            viz_path = args.output / f"{engine_name}_viz.jpg"
            save_viz(results, crops, viz_path, max_panels=80)

    # Summary comparison
    print_comparison(summaries)

    # Save JSON
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / "comparison.json"
        with open(report_path, "w") as f:
            json.dump(summaries, f, indent=2)
        log.info("Results saved to %s", report_path)


if __name__ == "__main__":
    main()
