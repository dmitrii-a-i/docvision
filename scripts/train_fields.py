#!/usr/bin/env python3
"""Train YOLO11n for document field detection.

Usage:
    # Test run (1 epoch, local)
    python scripts/train_fields.py --epochs 1

    # Full training (on server)
    python scripts/train_fields.py --epochs 150 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = Path("/data/docvision/yolo_fields/dataset.yaml")
RUNS_DIR = PROJECT_DIR / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11n for document field detection"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--data", type=Path, default=DATASET_YAML,
        help=f"Path to dataset.yaml (default: {DATASET_YAML})",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Base model")
    parser.add_argument("--name", default="doc_fields", help="Run name")
    parser.add_argument("--no-clearml", action="store_true", help="Disable ClearML logging")
    args = parser.parse_args()

    # Init ClearML
    if CLEARML_AVAILABLE and not args.no_clearml:
        Task.init(
            project_name="docvision",
            task_name=args.name,
            auto_connect_frameworks={"matplotlib": True, "tensorboard": True},
        )

    if args.resume:
        # Find last checkpoint
        last_ckpt = RUNS_DIR / "detect" / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        model = YOLO(args.model)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR / "detect"),
        name=args.name,
        exist_ok=True,
        # Augmentation — tailored for dewarped documents
        mosaic=0,
        mixup=0,
        fliplr=0,
        flipud=0,
        perspective=0,
        degrees=3.0,
        scale=0.3,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        erasing=0.2,
        # Training params
        patience=20,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
