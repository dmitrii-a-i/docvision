#!/usr/bin/env python3
"""Quick eval of the full pipeline (corner→dewarp→VLM API) against MIDV-500 GT.

Usage:
    VLM_API_KEY=... VLM_BASE_URL=https://api.openai.com/v1 VLM_MODEL=gpt-5.4 \
        python scripts/eval_api.py --max-docs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.corner import CornerDetector
from app.models.vlm import APIClient, _parse_json_output, _make_few_shot_prompt

log = logging.getLogger("eval_api")

MIDV500_ROOT = Path("/mnt/B/Data/docvision/midv-500/dataset")
SKIP_FIELDS = {"photo", "signature", "doc_quad", "face"}


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


def normalize_key(key: str) -> str:
    import re
    return re.sub(r"[\s_\-]+", "_", key.lower().strip())


def match_fields(gt: dict[str, str], pred: dict[str, str]) -> list[dict]:
    """Match predicted fields to GT fields by VALUE similarity (GT keys are field01..N)."""
    gt_values = [(k, v.strip()) for k, v in gt.items()]
    pred_values = [(k, v.strip()) for k, v in pred.items()]
    used_pred = set()
    results = []

    # For each GT field, find best matching pred value
    for gk, gv in gt_values:
        best_cer = float("inf")
        best_idx = -1
        gv_lower = gv.lower()

        for i, (pk, pv) in enumerate(pred_values):
            if i in used_pred:
                continue
            # Try exact match first (case-insensitive)
            if pv.lower() == gv_lower:
                best_cer = 0.0
                best_idx = i
                break
            # Then CER-based matching — only consider if value is somewhat similar
            cer = compute_cer(gv, pv)
            if cer < best_cer:
                best_cer = cer
                best_idx = i

        if best_idx >= 0 and best_cer < 0.8:
            pk, pv = pred_values[best_idx]
            used_pred.add(best_idx)
            exact = gv.lower() == pv.lower()
            results.append({"gt_key": gk, "gt_val": gv, "pred_key": pk, "pred_val": pv,
                            "cer": best_cer, "exact": exact})
        else:
            results.append({"gt_key": gk, "gt_val": gv, "pred_key": None, "pred_val": "",
                            "cer": 1.0, "exact": False})

    return results


def collect_midv500_samples(root: Path, max_docs: int, seed: int) -> list[dict]:
    """Collect (image_path, gt_fields, doc_type) from MIDV-500."""
    samples = []

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.endswith(".zip"):
            continue

        doc_type = doc_dir.name.split("_", 1)[1] if doc_dir.name.split("_")[0].isdigit() else doc_dir.name
        base = doc_dir / doc_dir.name
        gt_dir = base / "ground_truth"
        img_dir = base / "images"

        template_json = gt_dir / f"{doc_dir.name}.json"
        if not template_json.is_file():
            continue

        with open(template_json) as f:
            tpl = json.load(f)

        gt_fields = {}
        for fname, fdata in tpl.items():
            if fname.lower() in SKIP_FIELDS:
                continue
            value = fdata.get("value", "")
            if value and "*" not in value:
                gt_fields[fname] = value

        if not gt_fields:
            continue

        # Pick one frame per doc type
        frame_images = sorted(img_dir.rglob("*.tif"))
        frame_images = [f for f in frame_images if f.stem != doc_dir.name]  # skip template
        if not frame_images:
            continue

        samples.append({
            "image_path": frame_images[0],
            "gt_fields": gt_fields,
            "doc_type": doc_type,
        })

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:max_docs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not settings.use_external_vlm:
        log.error("Set VLM_API_KEY, VLM_BASE_URL, VLM_MODEL env vars")
        sys.exit(1)

    log.info("Model: %s via %s", settings.VLM_MODEL, settings.VLM_BASE_URL)

    # Init models
    corner = CornerDetector(settings.CORNER_MODEL, settings.device, settings.CORNER_CONF)
    vlm = APIClient(settings.VLM_API_KEY, settings.VLM_BASE_URL, settings.VLM_MODEL)

    # Collect samples
    samples = collect_midv500_samples(MIDV500_ROOT, args.max_docs, args.seed)
    log.info("Evaluating %d documents", len(samples))

    all_field_results = []
    doc_results = []

    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample["image_path"]))
        if img is None:
            continue

        # Dewarp
        det = corner.detect_and_dewarp(img)
        if det is None:
            log.warning("  %s: no document detected, skip", sample["doc_type"])
            continue
        dewarped, _ = det

        # VLM OCR
        t0 = time.time()
        try:
            pred_fields = vlm.extract_fields(dewarped)
        except Exception as e:
            log.warning("  %s: VLM error: %s", sample["doc_type"], e)
            continue
        elapsed = time.time() - t0

        # Match & score
        field_results = match_fields(sample["gt_fields"], pred_fields)
        all_field_results.extend(field_results)

        doc_cer = sum(r["cer"] for r in field_results) / len(field_results) if field_results else 1.0
        doc_exact = sum(r["exact"] for r in field_results) / len(field_results) if field_results else 0.0

        doc_results.append({
            "doc_type": sample["doc_type"],
            "n_fields": len(field_results),
            "avg_cer": doc_cer,
            "exact_rate": doc_exact,
            "elapsed": elapsed,
        })

        print(f"  [{i+1}/{len(samples)}] {sample['doc_type']:<25s} "
              f"CER={doc_cer:.3f}  exact={doc_exact:.0%}  "
              f"fields={len(field_results)}  ({elapsed:.1f}s)")

        # Show mismatches
        for r in field_results:
            if not r["exact"]:
                print(f"         {r['gt_key']}: {r['gt_val']!r} → {r['pred_val']!r}  (CER={r['cer']:.2f})")

    # Summary
    if not all_field_results:
        print("No results.")
        return

    total_cer = sum(r["cer"] for r in all_field_results) / len(all_field_results)
    total_exact = sum(r["exact"] for r in all_field_results) / len(all_field_results)
    total_time = sum(d["elapsed"] for d in doc_results)

    print(f"\n{'=' * 60}")
    print(f"  Model: {settings.VLM_MODEL}")
    print(f"  Documents: {len(doc_results)}, Fields: {len(all_field_results)}")
    print(f"  Avg CER: {total_cer:.4f}")
    print(f"  Exact match: {total_exact:.1%}")
    print(f"  Total time: {total_time:.1f}s ({total_time/len(doc_results):.1f}s/doc)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
