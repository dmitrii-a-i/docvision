#!/usr/bin/env python3
"""Quick smoke test: Qwen2.5-VL-7B on dewarped document images.

Usage:
    python scripts/smoke_vlm_ocr.py --images data/dewarp_examples/ --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

log = logging.getLogger("test_vlm_ocr")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "This is a dewarped scan of an identity document (passport, ID card, or driving license). "
    "Extract ALL visible text fields from this document. "
    "Return ONLY a JSON object where keys are field names (e.g. name, surname, birth_date, "
    "expiry_date, id_number, nationality, gender, etc.) and values are the extracted text. "
    "Include every readable text field you can find."
)


def load_model(model_id: str, device: str):
    log.info("Loading %s on %s...", model_id, device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    log.info("Model loaded.")
    return model, processor


def run_vlm(model, processor, pil_image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model, processor = load_model(MODEL_ID, args.device)

    # Collect images (skip *_viz.png)
    if args.images.is_dir():
        paths = sorted(
            p for p in args.images.glob("*.png")
            if "_viz" not in p.stem
        )
    else:
        paths = [args.images]

    for img_path in paths:
        pil_img = Image.open(img_path).convert("RGB")
        log.info("Processing %s (%dx%d)...", img_path.name, *pil_img.size)

        t0 = time.time()
        output = run_vlm(model, processor, pil_img, PROMPT)
        elapsed = time.time() - t0

        print(f"\n{'=' * 60}")
        print(f"  {img_path.name}  ({elapsed:.1f}s)")
        print(f"{'=' * 60}")

        # Try to parse JSON
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                fields = json.loads(output[start:end])
                for k, v in fields.items():
                    print(f"  {k:<25s} {v}")
            else:
                print(f"  [raw] {output}")
        except json.JSONDecodeError:
            print(f"  [raw] {output}")


if __name__ == "__main__":
    main()
