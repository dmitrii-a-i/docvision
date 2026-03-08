from __future__ import annotations

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Device: "auto" picks cuda if available, else cpu
    DEVICE: str = "auto"

    # Pipeline mode: lite / standard / api
    PIPELINE_MODE: str = "standard"

    # Model paths (relative to project root)
    CORNER_MODEL: str = "models/corner_detect.pt"
    FIELD_MODEL: str = "models/field_detect.pt"

    # Local VLM
    VLM_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # External VLM API (overrides local if VLM_API_KEY is set)
    VLM_API_KEY: str = ""
    VLM_BASE_URL: str = ""
    VLM_MODEL: str = ""

    # OCR settings (lite mode)
    OCR_LANG: str = "en"

    # HuggingFace cache
    HF_HOME: str = "/root/.cache/huggingface"

    # YOLO confidence
    CORNER_CONF: float = 0.25
    FIELD_CONF: float = 0.25

    @property
    def device(self) -> str:
        if self.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.DEVICE

    @property
    def use_external_vlm(self) -> bool:
        return bool(self.VLM_API_KEY)

    @property
    def ocr_languages(self) -> list[str]:
        return [lang.strip() for lang in self.OCR_LANG.split(",")]

    model_config = {"env_prefix": ""}


settings = Settings()
