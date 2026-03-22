"""
detect.py
Run inference with a trained YOLO model and save annotated results.

Usage:
    python scripts/detect.py \
        --weights results/exp/weights/best.pt \
        --source  dataset/images/test/ \
        --conf    0.4 \
        --iou     0.45 \
        --img     640 \
        --output  results/detections/
"""

import argparse
import sys
from pathlib import Path


def detect_yolov5(args):
    yolov5_dir = Path("yolov5")
    if not yolov5_dir.exists():
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)

    import subprocess
    cmd = [
        sys.executable, "yolov5/detect.py",
        "--weights", args.weights,
        "--source",  args.source,
        "--conf-thres", str(args.conf),
        "--iou-thres",  str(args.iou),
        "--img",        str(args.img),
        "--project",    str(Path(args.output).parent),
        "--name",       Path(args.output).name,
        "--save-txt",
        "--save-conf",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def detect_yolov8(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.img,
        save=True,
        save_txt=True,
        project=str(Path(args.output).parent),
        name=Path(args.output).name,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO inference / detection")
    parser.add_argument("--weights",      required=True, help="Path to best.pt")
    parser.add_argument("--source",       required=True, help="Image/dir/video path")
    parser.add_argument("--yolo_version", type=int, default=5, choices=[5, 8])
    parser.add_argument("--conf",         type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou",          type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img",          type=int,   default=640)
    parser.add_argument("--output",       default="results/detections")
    args = parser.parse_args()

    print(f"🔍 Running detection with YOLOv{args.yolo_version}...")
    print(f"   Weights: {args.weights}")
    print(f"   Source : {args.source}")
    print(f"   Conf   : {args.conf}")
    print()

    if args.yolo_version == 5:
        detect_yolov5(args)
    else:
        detect_yolov8(args)

    print(f"\n✅ Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
