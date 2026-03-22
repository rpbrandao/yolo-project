"""
split_dataset.py
Splits images (and matching YOLO labels) into train / val / test sets.

Usage:
    python scripts/split_dataset.py \
        --images_dir  dataset/images/all/ \
        --labels_dir  dataset/labels/all/ \
        --output_dir  dataset/ \
        --train_ratio 0.7 \
        --val_ratio   0.2 \
        --test_ratio  0.1 \
        --seed        42
"""

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_files(pairs, img_dst: Path, lbl_dst: Path):
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dst / img_path.name)
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
        else:
            # Create empty label for images with no annotations
            (lbl_dst / (img_path.stem + ".txt")).touch()


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--images_dir",  required=True)
    parser.add_argument("--labels_dir",  default=None,
                        help="Dir with YOLO .txt labels (default: auto-detect)")
    parser.add_argument("--output_dir",  default="dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.2)
    parser.add_argument("--test_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else images_dir.parent / "labels" / images_dir.name
    output_dir = Path(args.output_dir)

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"Found {len(images)} images in {images_dir}")

    random.seed(args.seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, imgs in splits.items():
        pairs = [(img, labels_dir / (img.stem + ".txt")) for img in imgs]
        copy_files(
            pairs,
            output_dir / "images" / split,
            output_dir / "labels" / split,
        )
        print(f"  {split:5s}: {len(imgs)} images")

    print(f"\n✅ Dataset split saved to: {output_dir}")


if __name__ == "__main__":
    main()
