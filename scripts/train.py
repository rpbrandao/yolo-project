"""
train.py
Wrapper script for YOLOv5/YOLOv8 training with transfer learning.

Usage (YOLOv5):
    python scripts/train.py \
        --data    dataset/data.yaml \
        --weights yolov5s.pt \
        --epochs  100 \
        --batch   16 \
        --img     640 \
        --project results \
        --name    exp1

Usage (YOLOv8):
    python scripts/train.py --yolo_version 8 \
        --data    dataset/data.yaml \
        --weights yolov8s.pt \
        --epochs  100 \
        --batch   16 \
        --img     640 \
        --project results \
        --name    exp1
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_yolov5(args):
    """Launch YOLOv5 training via its train.py CLI."""
    # Clone YOLOv5 if not present
    yolov5_dir = Path("yolov5")
    if not yolov5_dir.exists():
        print("Cloning YOLOv5...")
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"], check=True)

    cmd = [
        sys.executable, "yolov5/train.py",
        "--data",    str(args.data),
        "--weights", args.weights,
        "--epochs",  str(args.epochs),
        "--batch-size", str(args.batch),
        "--img",     str(args.img),
        "--project", args.project,
        "--name",    args.name,
        "--cache",
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_yolov8(args):
    """Launch YOLOv8 training via the Ultralytics Python API."""
    try:
        from ultralytics import YOLO
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        project=args.project,
        name=args.name,
        device=args.device or "cpu",
        pretrained=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train YOLO with transfer learning")
    parser.add_argument("--yolo_version", type=int, default=5, choices=[5, 8])
    parser.add_argument("--data",    required=True, help="Path to data.yaml")
    parser.add_argument("--weights", default="yolov5s.pt",
                        help="Pre-trained weights (e.g. yolov5s.pt, yolov8s.pt)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--img",     type=int, default=640)
    parser.add_argument("--project", default="results")
    parser.add_argument("--name",    default="exp")
    parser.add_argument("--device",  default=None, help="cuda device or 'cpu'")
    args = parser.parse_args()

    print(f"🚀 Starting YOLOv{args.yolo_version} training...")
    print(f"   Data   : {args.data}")
    print(f"   Weights: {args.weights}")
    print(f"   Epochs : {args.epochs}")
    print(f"   Batch  : {args.batch}")
    print(f"   ImgSz  : {args.img}")
    print()

    if args.yolo_version == 5:
        train_yolov5(args)
    else:
        train_yolov8(args)

    print(f"\n✅ Training complete! Results saved to: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
