#!/usr/bin/env python3
"""Evaluate VLM prompt strategies for document OCR.

Loads a test set (images + gt.json from prepare_vlm_testset.py), runs 5 prompt
strategies through Qwen2.5-VL, computes metrics, and generates visualizations.

Usage:
    python scripts/eval_vlm_prompts.py --testset /data/docvision/vlm_eval/testset/ \
        --output /data/docvision/vlm_eval/ --device cuda:1

    # Run only specific strategies
    python scripts/eval_vlm_prompts.py --testset ... --output ... \
        --strategies generic field_list ocr_only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("eval_vlm_prompts")

# ── Prompt strategies ────────────────────────────────────────────────────────

STRATEGY_GENERIC = "generic"
STRATEGY_FIELD_LIST = "field_list"
STRATEGY_OCR_ONLY = "ocr_only"
STRATEGY_STRUCTURED = "structured"
STRATEGY_FEW_SHOT = "few_shot"
STRATEGY_TWO_STAGE = "two_stage"

ALL_STRATEGIES = [
    STRATEGY_GENERIC,
    STRATEGY_FIELD_LIST,
    STRATEGY_OCR_ONLY,
    STRATEGY_STRUCTURED,
    STRATEGY_FEW_SHOT,
    STRATEGY_TWO_STAGE,
]

PROMPT_GENERIC = (
    "This is a dewarped scan of an identity document (passport, ID card, or driving license). "
    "Extract ALL visible text fields from this document. "
    "Return ONLY a JSON object where keys are field names (e.g. name, surname, birth_date, "
    "expiry_date, id_number, nationality, gender, etc.) and values are the extracted text. "
    "Include every readable text field you can find."
)

PROMPT_OCR_ONLY = (
    "This is a dewarped scan of an identity document. "
    "Read ALL text exactly as printed on the document, character by character. "
    "Do not interpret, translate, or reformat any text. "
    "Return ONLY a JSON object where keys are field labels visible on the document "
    "and values are the exact text printed next to them."
)

PROMPT_STRUCTURED = (
    "This is a dewarped scan of an identity document. "
    "Extract all text fields from this document. "
    "Return one field per line in the format:\n"
    "FIELD_NAME: value\n\n"
    "For example:\n"
    "surname: SMITH\n"
    "given_name: JOHN\n"
    "birth_date: 01.01.1990\n\n"
    "List every readable field."
)


def make_field_list_prompt(field_names: list[str]) -> str:
    fields_str = ", ".join(field_names)
    return (
        "This is a dewarped scan of an identity document. "
        f"Extract the following fields: {fields_str}. "
        "Return ONLY a JSON object with these exact field names as keys "
        "and the extracted text as values. "
        "If a field is not visible, use an empty string."
    )


def make_few_shot_prompt(example_fields: dict[str, str]) -> str:
    example_json = json.dumps(example_fields, ensure_ascii=False, indent=2)
    return (
        "This is a dewarped scan of an identity document. "
        "Extract all text fields and return them as a JSON object.\n\n"
        "Example output for a similar document:\n"
        f"```json\n{example_json}\n```\n\n"
        "Now extract the fields from this document. Return ONLY a JSON object."
    )


def make_classify_prompt(known_types: list[str]) -> str:
    types_str = ", ".join(known_types)
    return (
        "This is a dewarped scan of an identity document. "
        f"Classify this document as one of the following types: {types_str}. "
        "Reply with ONLY the type name, nothing else."
    )


def parse_classify_output(text: str, known_types: list[str]) -> str | None:
    """Match VLM classification output to a known doc type."""
    text_clean = text.strip().lower().replace("-", "_").replace(" ", "_")
    # Exact match
    for t in known_types:
        if t.lower() == text_clean:
            return t
    # Substring match — VLM might output "aze_passport" inside a sentence
    for t in known_types:
        if t.lower() in text_clean:
            return t
    # Fuzzy fallback
    best_score = 0.0
    best_type = None
    for t in known_types:
        score = SequenceMatcher(None, text_clean, t.lower()).ratio()
        if score > best_score:
            best_score = score
            best_type = t
    if best_type and best_score >= 0.5:
        return best_type
    return None


# ── Parsing VLM output ──────────────────────────────────────────────────────


def parse_json_output(text: str) -> dict[str, str] | None:
    """Try to extract a JSON object from VLM output."""
    # Try to find JSON block in markdown
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1)

    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            pass
    return None


def parse_structured_output(text: str) -> dict[str, str] | None:
    """Parse 'FIELD: value' line format."""
    fields: dict[str, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        # Match "field_name: value" or "field_name : value"
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*?)\s*:\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip()
            fields[key] = val
    return fields if fields else None


def parse_vlm_output(text: str, strategy: str) -> dict[str, str] | None:
    """Parse VLM output according to strategy."""
    if strategy == STRATEGY_STRUCTURED:
        result = parse_structured_output(text)
        if result:
            return result
        # Fall back to JSON parsing
        return parse_json_output(text)
    return parse_json_output(text)


# ── Field matching ───────────────────────────────────────────────────────────


def normalize_key(key: str) -> str:
    """Normalize a field key for matching: lowercase, strip, collapse separators."""
    k = key.lower().strip()
    k = re.sub(r"[\s_\-]+", "_", k)
    return k


def fuzzy_match_keys(
    gt_keys: list[str], pred_keys: list[str], threshold: float = 0.6,
) -> dict[str, str | None]:
    """Match predicted keys to GT keys using fuzzy matching.

    Returns {gt_key: matched_pred_key or None}.
    """
    gt_norm = {normalize_key(k): k for k in gt_keys}
    pred_norm = {normalize_key(k): k for k in pred_keys}

    matched: dict[str, str | None] = {}
    used_pred: set[str] = set()

    # Exact matches first
    for gn, gk in gt_norm.items():
        if gn in pred_norm and gn not in used_pred:
            matched[gk] = pred_norm[gn]
            used_pred.add(gn)

    # Fuzzy matches for remaining
    for gn, gk in gt_norm.items():
        if gk in matched:
            continue
        best_score = 0.0
        best_pn = None
        for pn, pk in pred_norm.items():
            if pn in used_pred:
                continue
            score = SequenceMatcher(None, gn, pn).ratio()
            if score > best_score:
                best_score = score
                best_pn = pn
        if best_pn and best_score >= threshold:
            matched[gk] = pred_norm[best_pn]
            used_pred.add(best_pn)
        else:
            matched[gk] = None

    return matched


# ── Metrics ──────────────────────────────────────────────────────────────────


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


# ── VLM runner ───────────────────────────────────────────────────────────────


_MODEL_FAMILY = None  # "qwen" or "paddleocr"


def load_model(model_id: str, device: str):
    """Load VLM model and processor. Supports Qwen2.5-VL and PaddleOCR-VL."""
    global _MODEL_FAMILY
    import torch

    log.info("Loading %s on %s...", model_id, device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log.info("Using dtype: %s", dtype)

    model_id_lower = model_id.lower()
    if "deepseek-ocr" in model_id_lower or "deepseek_ocr" in model_id_lower:
        _MODEL_FAMILY = "deepseek_ocr"
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, use_safetensors=True,
        ).eval().to(device).to(dtype)
        processor = tokenizer  # store tokenizer as "processor" for unified API
    elif "paddleocr" in model_id_lower or "paddle" in model_id_lower:
        _MODEL_FAMILY = "paddleocr"
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype,
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id)
    elif "qwen3" in model_id_lower:
        _MODEL_FAMILY = "qwen"
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        _MODEL_FAMILY = "qwen"
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    log.info("Model loaded (family=%s).", _MODEL_FAMILY)
    return model, processor


def run_vlm(model, processor, pil_image: Image.Image, prompt: str) -> str:
    """Run VLM inference on a single image with a prompt."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if _MODEL_FAMILY == "deepseek_ocr":
        import tempfile
        # DeepSeek-OCR-2 uses model.infer() with image file path
        tokenizer = processor  # processor is actually tokenizer for this model
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp, format="PNG")
            tmp_path = tmp.name
        try:
            # Wrap prompt in DeepSeek-OCR format
            ds_prompt = f"<image>\n{prompt}"
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = model.infer(
                    tokenizer, prompt=ds_prompt, image_file=tmp_path,
                    output_path=tmp_dir, base_size=1024, image_size=768,
                    crop_mode=False, save_results=False,
                )
            return str(res) if res else ""
        finally:
            os.unlink(tmp_path)
    elif _MODEL_FAMILY == "paddleocr":
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    else:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
        )[0]


# ── Evaluation core ──────────────────────────────────────────────────────────


def evaluate_strategy(
    strategy: str,
    gt_data: dict[str, dict],
    images_dir: Path,
    model,
    processor,
    templates: dict[str, dict] | None = None,
) -> dict:
    """Run one strategy on all test images and compute metrics."""
    from tqdm import tqdm

    results: list[dict] = []
    total_cer = 0.0
    total_fields = 0
    exact_matches = 0
    parse_failures = 0
    hallucination_count = 0
    classify_correct = 0
    classify_total = 0

    # For few_shot: group by doc_type, use first frame's GT as example for others
    frames_by_doctype: dict[str, list[str]] = defaultdict(list)
    for stem, entry in gt_data.items():
        frames_by_doctype[entry["doc_type"]].append(stem)
    # Sort so frame 0 is deterministic
    for dt in frames_by_doctype:
        frames_by_doctype[dt].sort()

    # Build few-shot examples: for each doc_type, frame 0's GT is the example
    few_shot_examples: dict[str, dict[str, str]] = {}
    for dt, stems in frames_by_doctype.items():
        if stems:
            few_shot_examples[dt] = gt_data[stems[0]]["fields"]

    # For two_stage: prepare classification prompt
    if strategy == STRATEGY_TWO_STAGE and templates:
        known_types = sorted(templates.keys())
        classify_prompt = make_classify_prompt(known_types)

    stems = sorted(gt_data.keys())

    for stem in tqdm(stems, desc=f"  {strategy}"):
        entry = gt_data[stem]
        gt_fields = entry["fields"]
        doc_type = entry["doc_type"]

        img_path = images_dir / f"{stem}.png"
        if not img_path.is_file():
            log.warning("Image not found: %s", img_path)
            continue

        pil_img = Image.open(img_path).convert("RGB")

        # Build prompt
        if strategy == STRATEGY_GENERIC:
            prompt = PROMPT_GENERIC
        elif strategy == STRATEGY_FIELD_LIST:
            prompt = make_field_list_prompt(list(gt_fields.keys()))
        elif strategy == STRATEGY_OCR_ONLY:
            prompt = PROMPT_OCR_ONLY
        elif strategy == STRATEGY_STRUCTURED:
            prompt = PROMPT_STRUCTURED
        elif strategy == STRATEGY_FEW_SHOT:
            # Use example from a different frame of the same doc_type
            dt_stems = frames_by_doctype[doc_type]
            if len(dt_stems) > 1 and stem != dt_stems[0]:
                example = few_shot_examples[doc_type]
            elif len(dt_stems) > 1:
                # This IS frame 0 — use frame 1's GT as example
                example = gt_data[dt_stems[1]]["fields"]
            else:
                # Only one frame for this doc_type — use own GT (not ideal but needed)
                example = gt_fields
            prompt = make_few_shot_prompt(example)
        elif strategy == STRATEGY_TWO_STAGE:
            # Stage 1: classify document type
            try:
                classify_output = run_vlm(model, processor, pil_img, classify_prompt)
            except RuntimeError as e:
                log.warning("  %s: classify failed (%s), using generic", stem, e)
                classify_output = "ERROR"
            predicted_type = parse_classify_output(classify_output, known_types)
            classify_total += 1
            if predicted_type == doc_type:
                classify_correct += 1

            # Stage 2: use template for predicted type (or fallback to generic)
            if predicted_type and predicted_type in templates:
                template_fields = templates[predicted_type]["fields"]
                prompt = make_few_shot_prompt(template_fields)
            else:
                log.warning(
                    "  %s: classify failed (%r), falling back to generic",
                    stem, classify_output.strip()[:60],
                )
                prompt = PROMPT_GENERIC
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        t0 = time.time()
        try:
            raw_output = run_vlm(model, processor, pil_img, prompt)
        except RuntimeError as e:
            log.warning("  %s: VLM inference failed (%s), skipping", stem, e)
            parse_failures += 1
            results.append({
                "stem": stem, "doc_type": doc_type,
                "raw_output": f"ERROR: {e}", "elapsed": round(time.time() - t0, 2),
                "parse_ok": False, "field_results": [],
            })
            continue
        elapsed = time.time() - t0

        # Parse output
        pred_fields = parse_vlm_output(raw_output, strategy)

        frame_result = {
            "stem": stem,
            "doc_type": doc_type,
            "raw_output": raw_output,
            "elapsed": round(elapsed, 2),
            "parse_ok": pred_fields is not None,
            "field_results": [],
        }
        if strategy == STRATEGY_TWO_STAGE:
            frame_result["classify_output"] = classify_output.strip()[:100]
            frame_result["predicted_type"] = predicted_type
            frame_result["classify_correct"] = (predicted_type == doc_type)

        if pred_fields is None:
            parse_failures += 1
            results.append(frame_result)
            continue

        # Match GT keys to predicted keys
        key_matches = fuzzy_match_keys(
            list(gt_fields.keys()), list(pred_fields.keys()),
        )

        # Count hallucinations (predicted keys not matched to any GT key)
        matched_pred_keys = {v for v in key_matches.values() if v is not None}
        extra_keys = set(pred_fields.keys()) - matched_pred_keys
        hallucination_count += len(extra_keys)

        # Per-field CER and exact match
        for gt_key, pred_key in key_matches.items():
            gt_val = gt_fields[gt_key].strip()
            if pred_key is not None:
                pred_val = pred_fields[pred_key].strip()
            else:
                pred_val = ""

            cer = compute_cer(gt_val, pred_val)
            exact = gt_val.lower() == pred_val.lower()

            total_cer += cer
            total_fields += 1
            if exact:
                exact_matches += 1

            frame_result["field_results"].append({
                "gt_key": gt_key,
                "pred_key": pred_key,
                "gt_value": gt_val,
                "pred_value": pred_val,
                "cer": round(cer, 4),
                "exact_match": exact,
            })

        frame_result["extra_keys"] = list(extra_keys)
        results.append(frame_result)

    # Aggregate metrics
    avg_cer = total_cer / total_fields if total_fields else 0.0
    exact_rate = exact_matches / total_fields if total_fields else 0.0

    # Per doc_type breakdown
    per_doctype: dict[str, dict] = defaultdict(
        lambda: {"cer_sum": 0.0, "n_fields": 0, "exact": 0}
    )
    for r in results:
        dt = r["doc_type"]
        for fr in r["field_results"]:
            per_doctype[dt]["cer_sum"] += fr["cer"]
            per_doctype[dt]["n_fields"] += 1
            if fr["exact_match"]:
                per_doctype[dt]["exact"] += 1

    per_doctype_summary = {}
    for dt, d in sorted(per_doctype.items()):
        per_doctype_summary[dt] = {
            "n_fields": d["n_fields"],
            "avg_cer": round(d["cer_sum"] / d["n_fields"], 4) if d["n_fields"] else 0,
            "exact_match_rate": round(d["exact"] / d["n_fields"], 4) if d["n_fields"] else 0,
        }

    summary = {
        "strategy": strategy,
        "n_frames": len(stems),
        "n_fields": total_fields,
        "avg_cer": round(avg_cer, 4),
        "exact_match_rate": round(exact_rate, 4),
        "parse_failures": parse_failures,
        "hallucinations": hallucination_count,
        "per_doc_type": per_doctype_summary,
        "results": results,
    }
    if strategy == STRATEGY_TWO_STAGE and classify_total:
        summary["classify_accuracy"] = round(classify_correct / classify_total, 4)
        summary["classify_correct"] = classify_correct
        summary["classify_total"] = classify_total
    return summary


# ── Visualization ────────────────────────────────────────────────────────────

THUMB_W = 240
FIELD_COL_W = 400
PANEL_H_PER_FIELD = 20
PANEL_PADDING = 16
HEADER_H = 28


def _get_font(size: int = 13):
    """Try to load a monospace font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def make_frame_panel(
    img_path: Path,
    frame_result: dict,
) -> Image.Image | None:
    """Create a panel: doc thumbnail (left) + field comparison (right)."""
    if not img_path.is_file():
        return None

    field_results = frame_result.get("field_results", [])
    n_fields = max(len(field_results), 1)

    panel_h = max(
        HEADER_H + n_fields * PANEL_H_PER_FIELD + PANEL_PADDING * 2,
        THUMB_W * 2 // 3,  # minimum height based on thumbnail aspect
    )

    panel = Image.new("RGB", (THUMB_W + FIELD_COL_W, panel_h), (35, 35, 40))
    draw = ImageDraw.Draw(panel)
    font = _get_font()

    # Thumbnail
    thumb = Image.open(img_path).convert("RGB")
    scale = THUMB_W / thumb.width
    thumb_h = int(thumb.height * scale)
    if thumb_h > panel_h:
        scale = panel_h / thumb.height
        thumb = thumb.resize((int(thumb.width * scale), panel_h), Image.LANCZOS)
    else:
        thumb = thumb.resize((THUMB_W, thumb_h), Image.LANCZOS)
    panel.paste(thumb, (0, 0))

    # Header
    stem = frame_result.get("stem", "?")
    parse_ok = frame_result.get("parse_ok", False)
    status = "OK" if parse_ok else "PARSE FAIL"
    status_color = (100, 220, 100) if parse_ok else (220, 60, 60)
    draw.text(
        (THUMB_W + 8, 4), f"{stem}  [{status}]", fill=status_color, font=font,
    )

    # Field comparisons
    y = HEADER_H + 4
    for fr in field_results:
        gt_key = fr["gt_key"]
        gt_val = fr["gt_value"]
        pred_val = fr["pred_value"]
        cer = fr["cer"]
        exact = fr["exact_match"]

        if exact:
            color = (80, 220, 80)   # green
        elif cer <= 0.3:
            color = (220, 200, 60)  # yellow
        else:
            color = (220, 60, 60)   # red

        line = f"{gt_key}: {gt_val}  ->  {pred_val}"
        if len(line) > 55:
            line = line[:55] + "..."
        draw.text((THUMB_W + 8, y), line, fill=color, font=font)
        y += PANEL_H_PER_FIELD

    # Extra keys (hallucinations)
    for ek in frame_result.get("extra_keys", []):
        draw.text(
            (THUMB_W + 8, y), f"[EXTRA] {ek}", fill=(180, 80, 220), font=font,
        )
        y += PANEL_H_PER_FIELD

    return panel


def make_strategy_viz(
    strategy_result: dict,
    images_dir: Path,
    output_path: Path,
) -> None:
    """Generate a grid visualization for one strategy."""
    panels: list[Image.Image] = []
    for frame_result in strategy_result["results"]:
        stem = frame_result["stem"]
        img_path = images_dir / f"{stem}.png"
        panel = make_frame_panel(img_path, frame_result)
        if panel:
            panels.append(panel)

    if not panels:
        return

    # Stack panels vertically in columns
    cols = 2
    col_panels: list[list[Image.Image]] = [[] for _ in range(cols)]
    for i, p in enumerate(panels):
        col_panels[i % cols].append(p)

    # Compute column heights
    col_widths = [THUMB_W + FIELD_COL_W] * cols
    col_heights = [sum(p.height for p in col) for col in col_panels]
    total_h = max(col_heights) if col_heights else 1
    total_w = sum(col_widths)

    grid = Image.new("RGB", (total_w, total_h), (25, 25, 30))
    for c in range(cols):
        x = c * col_widths[0]
        y = 0
        for p in col_panels[c]:
            grid.paste(p, (x, y))
            y += p.height

    grid.save(str(output_path), quality=92)
    log.info("Saved %s viz (%d panels) to %s", strategy_result["strategy"], len(panels), output_path)


def make_comparison_viz(
    all_results: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate a comparison summary image across all strategies."""
    font = _get_font(14)
    row_h = 28
    col_w = 180
    header_h = 40

    strategies = sorted(all_results.keys())
    n_rows = len(strategies)
    metrics = ["avg_cer", "exact_match_rate", "parse_failures", "hallucinations"]
    n_cols = 1 + len(metrics)  # strategy name + metrics

    img_w = col_w * n_cols
    img_h = header_h + row_h * (n_rows + 1)

    img = Image.new("RGB", (img_w, img_h), (30, 30, 35))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((10, 6), "VLM Prompt Strategy Comparison", fill=(200, 200, 200), font=font)

    # Column headers
    y = header_h
    headers = ["Strategy", "Avg CER", "Exact Match", "Parse Fails", "Hallucinations"]
    for i, h in enumerate(headers):
        draw.text((i * col_w + 10, y), h, fill=(160, 160, 160), font=font)
    y += row_h

    # Draw separator
    draw.line([(0, y - 4), (img_w, y - 4)], fill=(80, 80, 80), width=1)

    # Data rows
    for strategy in strategies:
        r = all_results[strategy]
        cer = r["avg_cer"]
        em = r["exact_match_rate"]
        pf = r["parse_failures"]
        hal = r["hallucinations"]

        # Color code CER
        if cer <= 0.1:
            cer_color = (80, 220, 80)
        elif cer <= 0.3:
            cer_color = (220, 200, 60)
        else:
            cer_color = (220, 60, 60)

        draw.text((0 * col_w + 10, y), strategy, fill=(200, 200, 200), font=font)
        draw.text((1 * col_w + 10, y), f"{cer:.4f}", fill=cer_color, font=font)
        draw.text((2 * col_w + 10, y), f"{em:.1%}", fill=(200, 200, 200), font=font)
        draw.text((3 * col_w + 10, y), str(pf), fill=(200, 200, 200), font=font)
        draw.text((4 * col_w + 10, y), str(hal), fill=(200, 200, 200), font=font)
        y += row_h

    img.save(str(output_path), quality=95)
    log.info("Saved comparison viz to %s", output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM prompt strategies on document OCR",
    )
    parser.add_argument(
        "--testset", type=Path, required=True,
        help="Path to testset dir (images/ + gt.json)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for results and viz",
    )
    parser.add_argument(
        "--device", default="cuda:1",
        help="Device for VLM inference (default: cuda:1)",
    )
    parser.add_argument(
        "--model", default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=ALL_STRATEGIES,
        choices=ALL_STRATEGIES,
        help="Strategies to evaluate (default: all)",
    )
    parser.add_argument(
        "--templates", type=Path, default=None,
        help="Path to templates.json (required for two_stage strategy)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set HF_HOME to avoid filling root FS
    os.environ.setdefault("HF_HOME", "/data/docvision/hf_cache")

    # Load test set
    gt_path = args.testset / "gt.json"
    images_dir = args.testset / "images"
    if not gt_path.is_file():
        log.error("gt.json not found at %s", gt_path)
        sys.exit(1)
    if not images_dir.is_dir():
        log.error("images/ not found at %s", images_dir)
        sys.exit(1)

    with open(gt_path) as f:
        gt_data = json.load(f)
    log.info("Loaded test set: %d frames", len(gt_data))

    # Load templates if provided
    templates = None
    if args.templates and args.templates.is_file():
        with open(args.templates) as f:
            templates = json.load(f)
        log.info("Loaded %d doc type templates", len(templates))
    elif STRATEGY_TWO_STAGE in args.strategies:
        # Try to find templates.json next to testset or output
        for candidate in [args.testset / "templates.json", args.output / "templates.json"]:
            if candidate.is_file():
                with open(candidate) as f:
                    templates = json.load(f)
                log.info("Auto-found templates at %s (%d types)", candidate, len(templates))
                break
        if templates is None:
            log.error("two_stage strategy requires --templates or templates.json next to testset/output")
            sys.exit(1)

    # Prepare output dirs
    results_dir = args.output / "results"
    viz_dir = args.output / "viz"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_model(args.model, args.device)

    # Run strategies
    all_results: dict[str, dict] = {}

    for strategy in args.strategies:
        log.info("=== Strategy: %s ===", strategy)
        t0 = time.time()
        result = evaluate_strategy(strategy, gt_data, images_dir, model, processor, templates)
        elapsed = time.time() - t0

        classify_str = ""
        if "classify_accuracy" in result:
            classify_str = f"  classify={result['classify_accuracy']:.1%}"
        log.info(
            "  %s: CER=%.4f  exact=%.1f%%  parse_fails=%d  hallucinations=%d%s  (%.1fs)",
            strategy,
            result["avg_cer"],
            result["exact_match_rate"] * 100,
            result["parse_failures"],
            result["hallucinations"],
            classify_str,
            elapsed,
        )

        # Save per-strategy results
        result_path = results_dir / f"{strategy}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Generate viz
        viz_path = viz_dir / f"{strategy}_viz.jpg"
        make_strategy_viz(result, images_dir, viz_path)

        # Store summary (without per-frame results for the report)
        all_results[strategy] = {
            k: v for k, v in result.items() if k != "results"
        }

    # Comparison viz
    if len(all_results) > 1:
        make_comparison_viz(all_results, viz_dir / "comparison.jpg")

    # Save report
    report_path = args.output / "report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("Saved report to %s", report_path)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("  VLM Prompt Strategy Comparison")
    print(f"{'=' * 80}")
    print(f"  {'Strategy':<15s} {'CER':>8s} {'Exact':>8s} {'ParseFail':>10s} {'Halluc':>8s}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")
    for strategy in args.strategies:
        r = all_results.get(strategy, {})
        print(
            f"  {strategy:<15s} {r.get('avg_cer', 0):>8.4f} "
            f"{r.get('exact_match_rate', 0):>7.1%} "
            f"{r.get('parse_failures', 0):>10d} "
            f"{r.get('hallucinations', 0):>8d}"
        )
    print()


if __name__ == "__main__":
    main()
