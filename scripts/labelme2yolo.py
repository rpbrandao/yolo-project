"""
labelme2yolo.py
Converts LabelMe JSON annotations to YOLO format (.txt files).

Usage:
    python scripts/labelme2yolo.py \
        --input  dataset/images/train/ \
        --output dataset/labels/train/ \
        --classes dog cat bicycle

YOLO label format (per line):
    <class_id> <x_center> <y_center> <width> <height>
    (all values normalized 0–1 relative to image dimensions)
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def convert_polygon_to_bbox(points):
    """Convert a list of (x, y) polygon points to a bounding box."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def labelme_to_yolo(json_path: Path, output_dir: Path, class_map: dict):
    """Convert a single LabelMe JSON file to YOLO .txt format."""
    with open(json_path) as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_map:
            print(f"  [SKIP] Unknown label '{label}' in {json_path.name}")
            continue

        class_id = class_map[label]
        pts = shape["points"]

        if shape["shape_type"] == "rectangle":
            x1, y1 = pts[0]
            x2, y2 = pts[1]
        elif shape["shape_type"] == "polygon":
            x1, y1, x2, y2 = convert_polygon_to_bbox(pts)
        else:
            print(f"  [SKIP] Unsupported shape_type '{shape['shape_type']}'")
            continue

        # Normalize
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width    = abs(x2 - x1) / img_w
        height   = abs(y2 - y1) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width    = max(0.0, min(1.0, width))
        height   = max(0.0, min(1.0, height))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if lines:
        out_file = output_dir / (json_path.stem + ".txt")
        out_file.write_text("\n".join(lines) + "\n")
        print(f"  [OK] {json_path.name} → {out_file.name}  ({len(lines)} objects)")
    else:
        print(f"  [WARN] No valid annotations in {json_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON → YOLO TXT")
    parser.add_argument("--input",   required=True, help="Dir with LabelMe JSON files")
    parser.add_argument("--output",  required=True, help="Dir to save YOLO .txt files")
    parser.add_argument("--classes", nargs="+", required=True,
                        help="Class names in order (index = class_id)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = {name: idx for idx, name in enumerate(args.classes)}
    print(f"Class map: {class_map}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in input directory.")
        return

    print(f"\nConverting {len(json_files)} files...\n")
    for jf in json_files:
        labelme_to_yolo(jf, output_dir, class_map)

    print(f"\n✅ Done! Labels saved to: {output_dir}")


if __name__ == "__main__":
    main()
