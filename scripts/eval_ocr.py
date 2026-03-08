#!/usr/bin/env python3
"""Evaluate OCR quality on GT-cropped document fields from MIDV datasets.

Pipeline per frame:
    MIDV annotations → parse (field_name, value, quad, doc_quad)
    Image + doc_quad → dewarp → dewarped image
    field_quad × (dw/tw, dh/th) → crop → EasyOCR → compare with GT value

Usage:
    python scripts/eval_ocr.py --max-samples 10           # smoke test
    python scripts/eval_ocr.py --max-samples 500 --output eval_ocr_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dewarp import compute_output_size, dewarp, order_corners

log = logging.getLogger("eval_ocr")

# ── Dataset roots ─────────────────────────────────────────────────────────────

MIDV500_ROOT = Path("/mnt/B/Data/docvision/midv-500/dataset")
MIDV2019_ROOT = Path("/mnt/B/Data/docvision/midv-2019/dataset")
MIDV2020_ROOT = Path("/mnt/B/Data/docvision/midv-2020/dataset")

SKIP_FIELDS = {"photo", "signature", "doc_quad", "face"}
MIN_CROP_PX = 10


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class TextFieldSample:
    source_image: Path
    doc_quad: list[tuple[float, float]]
    template_width: int
    template_height: int
    field_name: str
    field_quad: list[tuple[float, float]]
    gt_value: str
    doc_type: str
    source_dataset: str


@dataclass
class FieldResult:
    field_name: str
    doc_type: str
    gt_value: str
    ocr_text: str
    cer: float
    exact_match: bool


# ── Metrics ───────────────────────────────────────────────────────────────────


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate = edit_distance / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


# ── Parsers ───────────────────────────────────────────────────────────────────


def _is_valid_value(value: str) -> bool:
    """Check if a field value is usable for OCR evaluation."""
    return bool(value) and "*" not in value


def _canonical_doctype(dirname: str) -> str:
    """'01_alb_id' → 'alb_id'."""
    parts = dirname.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return dirname


def parse_text_fields_midv_legacy(
    root: Path, dataset_name: str, *, double_nested: bool,
) -> list[TextFieldSample]:
    """Parse MIDV-500 / MIDV-2019 text fields with GT values."""
    if not root.is_dir():
        log.warning("%s root not found: %s", dataset_name, root)
        return []

    samples: list[TextFieldSample] = []

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.endswith(".zip"):
            continue

        doc_type = _canonical_doctype(doc_dir.name)
        base = doc_dir / doc_dir.name if double_nested else doc_dir
        gt_dir = base / "ground_truth"
        img_dir = base / "images"

        # Read template: field quads + values
        template_json = gt_dir / f"{doc_dir.name}.json"
        if not template_json.is_file():
            continue

        with open(template_json) as f:
            tpl = json.load(f)

        fields: list[tuple[str, list[tuple[float, float]], str]] = []
        for fname, fdata in tpl.items():
            if fname.lower() in SKIP_FIELDS:
                continue
            value = fdata.get("value", "")
            quad = fdata.get("quad")
            if not quad or len(quad) != 4 or not _is_valid_value(value):
                continue
            fields.append((
                fname,
                [(float(p[0]), float(p[1])) for p in quad],
                value,
            ))

        if not fields:
            continue

        # Template image dimensions
        template_img = img_dir / f"{doc_dir.name}.tif"
        if not template_img.is_file():
            continue
        tpl_img = cv2.imread(str(template_img))
        if tpl_img is None:
            continue
        th, tw = tpl_img.shape[:2]

        # Per-frame annotations
        for gt_file in sorted(gt_dir.rglob("*.json")):
            rel = gt_file.relative_to(gt_dir)
            if len(rel.parts) < 2:  # skip template-level JSON
                continue

            frame_img_path = img_dir / rel.with_suffix(".tif")
            if not frame_img_path.is_file():
                continue

            try:
                with open(gt_file) as f:
                    frame_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            raw_quad = frame_data.get("quad")
            if not raw_quad or len(raw_quad) != 4:
                continue

            doc_quad = [(float(p[0]), float(p[1])) for p in raw_quad]

            for fname, fquad, fvalue in fields:
                samples.append(TextFieldSample(
                    source_image=frame_img_path,
                    doc_quad=doc_quad,
                    template_width=tw,
                    template_height=th,
                    field_name=fname,
                    field_quad=fquad,
                    gt_value=fvalue,
                    doc_type=doc_type,
                    source_dataset=dataset_name,
                ))

    return samples


def parse_text_fields_midv2020(root: Path) -> list[TextFieldSample]:
    """Parse MIDV-2020 text fields with GT values (VIA format)."""
    if not root.is_dir():
        log.warning("MIDV-2020 root not found: %s", root)
        return []

    samples: list[TextFieldSample] = []
    ann_dir = root / "annotations"
    img_dir = root / "images"

    for ann_json in sorted(ann_dir.glob("*.json")):
        doc_type = ann_json.stem

        with open(ann_json) as f:
            via = json.load(f)

        # Per-variant (template) field annotations
        VariantInfo = tuple[int, int, list[tuple[str, list[tuple[float, float]], str]]]
        variants: dict[str, VariantInfo] = {}

        for entry in via.get("_via_img_metadata", {}).values():
            filename = entry.get("filename", "")
            if not filename.endswith(".jpg"):
                continue

            variant_id = Path(filename).stem  # "00", "01", ...

            # Template image dimensions
            tpl_path = img_dir / doc_type / filename
            if not tpl_path.is_file():
                continue
            tpl_img = cv2.imread(str(tpl_path))
            if tpl_img is None:
                continue
            th, tw = tpl_img.shape[:2]

            fields: list[tuple[str, list[tuple[float, float]], str]] = []
            for region in entry.get("regions", []):
                ra = region.get("region_attributes", {})
                fname = ra.get("field_name", "")
                if fname.lower() in SKIP_FIELDS:
                    continue
                value = ra.get("value", "")
                if not _is_valid_value(value):
                    continue

                sa = region["shape_attributes"]
                if sa.get("name") == "rect":
                    x, y = float(sa["x"]), float(sa["y"])
                    w, h = float(sa["width"]), float(sa["height"])
                    quad = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                elif sa.get("name") == "polygon":
                    xs, ys = sa["all_points_x"], sa["all_points_y"]
                    if len(xs) != 4 or len(ys) != 4:
                        continue
                    quad = [(float(xs[i]), float(ys[i])) for i in range(4)]
                else:
                    continue

                fields.append((fname, quad, value))

            if fields:
                variants[variant_id] = (tw, th, fields)

        # Per-clip frame annotations
        clip_dir = ann_dir / doc_type
        if not clip_dir.is_dir():
            continue

        for clip_json in sorted(clip_dir.glob("*.json")):
            variant_id = clip_json.stem
            if variant_id not in variants:
                continue

            tw, th, fields = variants[variant_id]

            with open(clip_json) as f:
                clip_data = json.load(f)

            clip_img_dir = img_dir / doc_type / variant_id

            for entry in clip_data.get("_via_img_metadata", {}).values():
                frame_filename = entry.get("filename", "")
                if not frame_filename.endswith(".jpg"):
                    continue

                # Extract doc_quad
                doc_quad = None
                for region in entry.get("regions", []):
                    ra = region.get("region_attributes", {})
                    if ra.get("field_name") == "doc_quad":
                        sa = region["shape_attributes"]
                        xs = sa.get("all_points_x", [])
                        ys = sa.get("all_points_y", [])
                        if len(xs) == 4 and len(ys) == 4:
                            doc_quad = [
                                (float(xs[i]), float(ys[i])) for i in range(4)
                            ]
                        break

                if doc_quad is None:
                    continue

                frame_path = clip_img_dir / frame_filename
                if not frame_path.is_file():
                    continue

                for fname, fquad, fvalue in fields:
                    samples.append(TextFieldSample(
                        source_image=frame_path,
                        doc_quad=doc_quad,
                        template_width=tw,
                        template_height=th,
                        field_name=fname,
                        field_quad=fquad,
                        gt_value=fvalue,
                        doc_type=doc_type,
                        source_dataset="midv2020",
                    ))

    return samples


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    samples: list[TextFieldSample],
    reader,
    max_samples: int,
    seed: int,
    collect_crops: bool = False,
) -> tuple[list[FieldResult], list[np.ndarray], int]:
    """Evaluate OCR on sampled frames. Returns (results, crops, n_frames)."""
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
    skipped = 0

    for frame_path in tqdm(frames, desc="OCR evaluation"):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        frame_samples = by_frame[frame_path]
        sample0 = frame_samples[0]

        # Dewarp using GT doc_quad
        quad = np.array(sample0.doc_quad, dtype=np.float32)
        ordered = order_corners(quad)
        dw, dh = compute_output_size(ordered)
        if dw < 32 or dh < 32:
            continue
        dewarped = dewarp(image, ordered, (dw, dh))

        sx = dw / sample0.template_width
        sy = dh / sample0.template_height

        for sample in frame_samples:
            # Scale field quad to dewarped coordinates
            pts = np.array(sample.field_quad, dtype=np.float32)
            pts[:, 0] *= sx
            pts[:, 1] *= sy

            x_min = max(0, int(np.floor(pts[:, 0].min())))
            y_min = max(0, int(np.floor(pts[:, 1].min())))
            x_max = min(dw, int(np.ceil(pts[:, 0].max())))
            y_max = min(dh, int(np.ceil(pts[:, 1].max())))

            if (x_max - x_min) < MIN_CROP_PX or (y_max - y_min) < MIN_CROP_PX:
                skipped += 1
                continue

            crop = dewarped[y_min:y_max, x_min:x_max]
            ocr_text = " ".join(reader.readtext(crop, detail=0))

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

    if skipped:
        log.info("Skipped %d crops (too small)", skipped)

    return results, crops, len(frames)


# ── Visualization ─────────────────────────────────────────────────────────────

PANEL_W = 320
CROP_MAX_H = 120
TEXT_H = 54
BORDER_H = 4


def make_viz_panel(crop: np.ndarray, result: FieldResult) -> Image.Image:
    """Build a fixed-width panel showing crop image with GT/OCR text overlay."""
    panel_h = BORDER_H + CROP_MAX_H + TEXT_H
    panel = Image.new("RGB", (PANEL_W, panel_h), (40, 40, 40))
    draw = ImageDraw.Draw(panel)

    # Top border colored by CER
    if result.cer == 0:
        border_color = (0, 200, 0)
    elif result.cer <= 0.5:
        border_color = (220, 200, 0)
    else:
        border_color = (220, 40, 40)
    draw.rectangle([0, 0, PANEL_W, BORDER_H], fill=border_color)

    # Scale crop to fill width, cap height
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_img = Image.fromarray(crop_rgb)
    scale = PANEL_W / crop_img.width
    new_h = min(CROP_MAX_H, int(crop_img.height * scale))
    crop_img = crop_img.resize((PANEL_W, new_h), Image.LANCZOS)
    y_off = BORDER_H + (CROP_MAX_H - new_h) // 2
    panel.paste(crop_img, (0, y_off))

    # Text lines
    text_y = BORDER_H + CROP_MAX_H + 4
    draw.text((6, text_y), f"GT:  {result.gt_value}", fill=(100, 220, 100))
    ocr_color = (100, 220, 100) if result.exact_match else (220, 80, 80)
    draw.text((6, text_y + 16), f"OCR: {result.ocr_text}", fill=ocr_color)
    draw.text(
        (6, text_y + 32),
        f"CER: {result.cer:.3f}  {result.field_name}  {result.doc_type}",
        fill=(160, 160, 160),
    )
    return panel


def make_grid(panels: list[Image.Image], cols: int) -> Image.Image:
    """Arrange panels in a grid on a dark background."""
    if not panels:
        return Image.new("RGB", (1, 1))
    pw, ph = panels[0].size
    rows = math.ceil(len(panels) / cols)
    grid = Image.new("RGB", (cols * pw, rows * ph), (30, 30, 30))
    for idx, p in enumerate(panels):
        r, c = divmod(idx, cols)
        grid.paste(p, (c * pw, r * ph))
    return grid


def save_viz(
    results: list[FieldResult],
    crops: list[np.ndarray],
    path: Path,
    max_panels: int = 80,
) -> None:
    """Save a grid visualization of OCR results sorted by worst CER first."""
    paired = sorted(zip(results, crops), key=lambda rc: rc[0].cer, reverse=True)
    paired = paired[:max_panels]
    panels = [make_viz_panel(crop, res) for res, crop in paired]
    grid = make_grid(panels, cols=5)
    grid.save(str(path), quality=92)
    log.info("Saved OCR viz (%d panels) to %s", len(panels), path)


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_report(results: list[FieldResult], n_frames: int) -> dict:
    """Print and return evaluation report."""
    if not results:
        print("No results.")
        return {}

    total_cer = sum(r.cer for r in results) / len(results)
    total_em = sum(r.exact_match for r in results) / len(results)

    print(f"\n{'=' * 50}")
    print("  OCR Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Samples: {n_frames} frames, {len(results)} fields")
    print(f"Overall CER: {total_cer:.4f}")
    print(f"Exact match: {total_em:.1%}")

    # Per field type
    by_field: dict[str, list[FieldResult]] = defaultdict(list)
    for r in results:
        by_field[r.field_name].append(r)

    print(f"\n{'Per field type:'}")
    print(f"  {'field_name':<20s} {'count':>6s} {'CER':>8s} {'exact_match':>12s}")
    for fname in sorted(by_field):
        group = by_field[fname]
        cer = sum(r.cer for r in group) / len(group)
        em = sum(r.exact_match for r in group) / len(group)
        print(f"  {fname:<20s} {len(group):>6d} {cer:>8.3f} {em:>11.1%}")

    # Per doc type
    by_doc: dict[str, list[FieldResult]] = defaultdict(list)
    for r in results:
        by_doc[r.doc_type].append(r)

    print(f"\n{'Per doc type:'}")
    print(f"  {'doc_type':<25s} {'count':>6s} {'CER':>8s} {'exact_match':>12s}")
    for dtype in sorted(by_doc):
        group = by_doc[dtype]
        cer = sum(r.cer for r in group) / len(group)
        em = sum(r.exact_match for r in group) / len(group)
        print(f"  {dtype:<25s} {len(group):>6d} {cer:>8.3f} {em:>11.1%}")

    report = {
        "n_frames": n_frames,
        "n_fields": len(results),
        "overall_cer": total_cer,
        "overall_exact_match": total_em,
        "per_field": {
            fname: {
                "count": len(group),
                "cer": sum(r.cer for r in group) / len(group),
                "exact_match": sum(r.exact_match for r in group) / len(group),
            }
            for fname, group in sorted(by_field.items())
        },
        "per_doc_type": {
            dtype: {
                "count": len(group),
                "cer": sum(r.cer for r in group) / len(group),
                "exact_match": sum(r.exact_match for r in group) / len(group),
            }
            for dtype, group in sorted(by_doc.items())
        },
        "results": [asdict(r) for r in results],
    }
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR on MIDV GT field crops",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["midv500", "midv2019", "midv2020"],
        choices=["midv500", "midv2019", "midv2020"],
    )
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Max frames to sample (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", nargs="+", default=["en"],
        help="EasyOCR languages (default: en)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save JSON results to file",
    )
    parser.add_argument(
        "--viz", type=Path, default=None,
        help="Save OCR visualization grid to file (e.g. ocr_viz.jpg)",
    )
    parser.add_argument(
        "--viz-max", type=int, default=80,
        help="Max panels in visualization (default: 80)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Collect samples
    all_samples: list[TextFieldSample] = []

    if "midv500" in args.datasets and MIDV500_ROOT.exists():
        log.info("Parsing MIDV-500...")
        all_samples.extend(parse_text_fields_midv_legacy(
            MIDV500_ROOT, "midv500", double_nested=True,
        ))

    if "midv2019" in args.datasets and MIDV2019_ROOT.exists():
        log.info("Parsing MIDV-2019...")
        all_samples.extend(parse_text_fields_midv_legacy(
            MIDV2019_ROOT, "midv2019", double_nested=False,
        ))

    if "midv2020" in args.datasets and MIDV2020_ROOT.exists():
        log.info("Parsing MIDV-2020...")
        all_samples.extend(parse_text_fields_midv2020(MIDV2020_ROOT))

    log.info("Total text field samples: %d", len(all_samples))
    if not all_samples:
        log.error("No samples found. Check dataset paths.")
        return

    # Init EasyOCR (lazy import to keep startup fast when just checking --help)
    import easyocr

    log.info("Initializing EasyOCR (lang=%s)...", args.lang)
    reader = easyocr.Reader(args.lang, gpu=True)

    # Evaluate
    results, crops, n_frames = evaluate(
        all_samples, reader, args.max_samples, args.seed,
        collect_crops=args.viz is not None,
    )

    # Report
    report = print_report(results, n_frames)

    # Visualization
    if args.viz and results:
        args.viz.parent.mkdir(parents=True, exist_ok=True)
        save_viz(results, crops, args.viz, args.viz_max)

    if args.output and report:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
