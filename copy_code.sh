#!/bin/bash
OUT="/tmp/all_code_docviz.txt"
ROOT="$(cd "$(dirname "$0")" && pwd)"

{
    echo "=== PROJECT STRUCTURE ==="
    tree -I '__pycache__|.git|.idea|runs|*.pt|*.jpg|*.pdf' "$ROOT"
    echo ""

    for f in \
        .gitignore \
        README.md \
        scripts/dewarp.py \
        scripts/prepare_yolo_dataset.py \
        scripts/prepare_field_dataset.py \
        scripts/train.py \
        scripts/train_fields.py \
        scripts/predict_viz.py \
        scripts/visualize_yolo_keypoints.py \
        docs/datasets.md
    do
        if [ -f "$ROOT/$f" ]; then
            echo "================================================================"
            echo "=== $f ==="
            echo "================================================================"
            cat "$ROOT/$f"
            echo ""
        fi
    done
} > "$OUT"

echo "Written to $OUT"
subl "$OUT"
