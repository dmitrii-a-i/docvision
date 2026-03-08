#!/usr/bin/env python3
"""Experiment: use Qwen2.5-VL-3B as a document field extractor.

Takes dewarped document images, prompts the VLM to extract all text fields
as JSON, and compares with ground truth.

Usage:
    python scripts/eval_ocr_vlm.py --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dewarp import compute_output_size, dewarp, order_corners

log = logging.getLogger("eval_ocr_vlm")

# ── Dataset roots ─────────────────────────────────────────────────────────────

MIDV500_ROOT = Path("/mnt/B/Data/docvision/midv-500/dataset")
MIDV2019_ROOT = Path("/mnt/B/Data/docvision/midv-2019/dataset")
MIDV2020_ROOT = Path("/mnt/B/Data/docvision/midv-2020/dataset")

SKIP_FIELDS = {"photo", "signature", "doc_quad", "face"}
MIN_CROP_PX = 10

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


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


# ── Metrics ───────────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
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
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


# ── Parsers (reuse from eval_ocr) ────────────────────────────────────────────

def _is_valid_value(value: str) -> bool:
    return bool(value) and "*" not in value


def _canonical_doctype(dirname: str) -> str:
    parts = dirname.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return dirname


def parse_text_fields_midv_legacy(
    root: Path, dataset_name: str, *, double_nested: bool,
) -> list[TextFieldSample]:
    if not root.is_dir():
        return []

    samples: list[TextFieldSample] = []

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.endswith(".zip"):
            continue

        doc_type = _canonical_doctype(doc_dir.name)
        base = doc_dir / doc_dir.name if double_nested else doc_dir
        gt_dir = base / "ground_truth"
        img_dir = base / "images"

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

        template_img = img_dir / f"{doc_dir.name}.tif"
        if not template_img.is_file():
            continue
        tpl_img = cv2.imread(str(template_img))
        if tpl_img is None:
            continue
        th, tw = tpl_img.shape[:2]

        for gt_file in sorted(gt_dir.rglob("*.json")):
            rel = gt_file.relative_to(gt_dir)
            if len(rel.parts) < 2:
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
    if not root.is_dir():
        return []

    samples: list[TextFieldSample] = []
    ann_dir = root / "annotations"
    img_dir = root / "images"

    for ann_json in sorted(ann_dir.glob("*.json")):
        doc_type = ann_json.stem

        with open(ann_json) as f:
            via = json.load(f)

        VariantInfo = tuple[int, int, list[tuple[str, list[tuple[float, float]], str]]]
        variants: dict[str, VariantInfo] = {}

        for entry in via.get("_via_img_metadata", {}).values():
            filename = entry.get("filename", "")
            if not filename.endswith(".jpg"):
                continue

            variant_id = Path(filename).stem

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


# ── VLM inference ─────────────────────────────────────────────────────────────


def load_model(model_id: str, device: str):
    """Load Qwen2.5-VL model and processor."""
    log.info("Loading model %s...", model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def extract_fields_vlm(
    model,
    processor,
    pil_image: Image.Image,
    field_names: list[str],
) -> dict[str, str]:
    """Ask VLM to extract field values from dewarped document image."""
    fields_str = ", ".join(field_names)
    prompt = (
        f"This is a dewarped scan of an identity document. "
        f"Extract the text values of these fields: {fields_str}. "
        f"Return ONLY a JSON object with field names as keys and extracted text as values. "
        f"If a field is not visible, use an empty string."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
    )[0]

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(output_text[start:end])
    except json.JSONDecodeError:
        pass

    log.warning("Failed to parse VLM output: %s", output_text[:200])
    return {}


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_vlm(
    samples: list[TextFieldSample],
    model,
    processor,
    max_samples: int,
    seed: int,
) -> list[dict]:
    """Evaluate VLM on dewarped document frames."""
    # Group by frame
    by_frame: dict[Path, list[TextFieldSample]] = defaultdict(list)
    for s in samples:
        by_frame[s.source_image].append(s)

    frames = list(by_frame.keys())
    if len(frames) > max_samples:
        rng = random.Random(seed)
        frames = rng.sample(frames, max_samples)
    frames.sort()

    all_results = []

    for i, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        frame_samples = by_frame[frame_path]
        sample0 = frame_samples[0]

        # Dewarp
        quad = np.array(sample0.doc_quad, dtype=np.float32)
        ordered = order_corners(quad)
        dw, dh = compute_output_size(ordered)
        if dw < 32 or dh < 32:
            continue
        dewarped = dewarp(image, ordered, (dw, dh))

        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB))

        # Get unique field names for this frame
        field_names = sorted(set(s.field_name for s in frame_samples))

        # Build GT dict
        gt_dict = {s.field_name: s.gt_value.strip() for s in frame_samples}

        log.info(
            "[%d/%d] %s  doc_type=%s  fields=%d",
            i + 1, len(frames), frame_path.name,
            sample0.doc_type, len(field_names),
        )

        t0 = time.time()
        predicted = extract_fields_vlm(model, processor, pil_img, field_names)
        elapsed = time.time() - t0

        # Compare
        frame_results = []
        for fname in field_names:
            gt_val = gt_dict.get(fname, "")
            pred_val = predicted.get(fname, "").strip()
            cer = compute_cer(gt_val, pred_val)
            exact = gt_val.lower() == pred_val.lower()
            frame_results.append({
                "field_name": fname,
                "doc_type": sample0.doc_type,
                "gt_value": gt_val,
                "vlm_text": pred_val,
                "cer": cer,
                "exact_match": exact,
            })
            status = "OK" if exact else f"CER={cer:.3f}"
            log.info("  %-20s GT=%-20s VLM=%-20s %s", fname, gt_val, pred_val, status)

        log.info("  (%.2fs, %d fields)", elapsed, len(field_names))
        all_results.extend(frame_results)

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL on MIDV document fields",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["midv500", "midv2019", "midv2020"],
        choices=["midv500", "midv2019", "midv2020"],
    )
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:1", help="GPU device")
    parser.add_argument("--output", type=Path, default=None)
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
        log.error("No samples found.")
        return

    # Load model
    model, processor = load_model(MODEL_ID, args.device)

    # Evaluate
    results = evaluate_vlm(
        all_samples, model, processor, args.max_samples, args.seed,
    )

    # Summary
    if results:
        avg_cer = sum(r["cer"] for r in results) / len(results)
        avg_em = sum(r["exact_match"] for r in results) / len(results)
        print(f"\n{'=' * 50}")
        print(f"  VLM Evaluation: {MODEL_ID}")
        print(f"{'=' * 50}")
        print(f"Fields: {len(results)}")
        print(f"Overall CER: {avg_cer:.4f}")
        print(f"Exact match: {avg_em:.1%}")

    if args.output and results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
