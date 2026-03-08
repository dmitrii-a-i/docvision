"""VLM-based OCR: local Qwen2.5-VL or external OpenAI-compatible API."""

from __future__ import annotations

import base64
import io
import json
import logging
import re

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _make_few_shot_prompt(example_fields: dict[str, str] | None = None) -> str:
    """Build a few-shot prompt for document field extraction."""
    base = (
        "This is a dewarped scan of an identity document. "
        "Extract all text fields and return them as a JSON object.\n\n"
    )
    if example_fields:
        example_json = json.dumps(example_fields, ensure_ascii=False, indent=2)
        base += (
            "Example output for a similar document:\n"
            f"```json\n{example_json}\n```\n\n"
        )
    base += "Now extract the fields from this document. Return ONLY a JSON object."
    return base


def _parse_json_output(text: str) -> dict[str, str]:
    """Extract JSON object from VLM output."""
    # Try markdown code block first
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            pass
    return {}


def _image_to_base64(image: np.ndarray) -> str:
    """Encode BGR numpy image to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


class LocalVLM:
    """Local Qwen2.5-VL inference."""

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        log.info("Loading local VLM: %s on %s ...", model_id, device)
        dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu":
            dtype = torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        log.info("Local VLM loaded.")

    def extract_fields(self, image: np.ndarray) -> dict[str, str]:
        """Extract document fields from a dewarped image."""
        import torch
        from qwen_vl_utils import process_vision_info

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        prompt = _make_few_shot_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
        )[0]

        return _parse_json_output(raw_output)


class APIClient:
    """External OpenAI-compatible VLM API client."""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        log.info("VLM API client configured: model=%s, base_url=%s", model, base_url)

    def extract_fields(self, image: np.ndarray) -> dict[str, str]:
        """Extract document fields via external API."""
        from openai import BadRequestError

        b64 = _image_to_base64(image)
        prompt = _make_few_shot_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=512,
            )
        except BadRequestError:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
            )

        raw_output = response.choices[0].message.content or ""
        return _parse_json_output(raw_output)
