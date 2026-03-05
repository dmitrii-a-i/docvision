#!/usr/bin/env python3
"""Train YOLO11n-pose for document corner detection.

Usage:
    # Test run (1 epoch, local)
    python scripts/train.py --epochs 1

    # Full training (on server)
    python scripts/train.py --epochs 100 --batch 32 --device 0
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
DATASET_YAML = Path("/mnt/B/Data/docvision/yolo_corners/dataset.yaml")
RUNS_DIR = PROJECT_DIR / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11n-pose for document corner detection"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--data", type=Path, default=DATASET_YAML,
        help=f"Path to dataset.yaml (default: {DATASET_YAML})",
    )
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Base model")
    parser.add_argument("--name", default="doc_corners", help="Run name")
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
        last_ckpt = RUNS_DIR / "pose" / args.name / "weights" / "last.pt"
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
        project=str(RUNS_DIR / "pose"),
        name=args.name,
        exist_ok=True,
        # Augmentation
        degrees=15.0,       # rotation
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        # Training params
        patience=20,         # early stopping
        save_period=10,      # save checkpoint every N epochs
        val=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
