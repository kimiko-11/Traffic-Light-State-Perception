"""
Train YOLOv8 on the traffic light dataset.

Usage:
    python train.py --data Traffic-Light-3/data.yaml

After training, best weights are saved to:
    models/best.pt
"""

import argparse
import shutil
import platform
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml: str, model_size: str = "s", epochs: int = 20, batch: int = 4):
    print(f"\n[INFO] Training YOLOv8{model_size} for {epochs} epochs...")
    print(f"[INFO] Dataset : {data_yaml}")
    print(f"[INFO] Batch   : {batch}")
    print(f"[INFO] Epochs  : {epochs}\n")

    # Windows needs workers=0 to avoid multiprocessing freeze
    workers = 0 if platform.system() == "Windows" else 4

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data          = data_yaml,
        epochs        = epochs,
        imgsz         = 416,        # smaller = faster on CPU
        batch         = batch,      # keep low on CPU (4-8)
        workers       = workers,    # 0 on Windows
        lr0           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        flipud        = 0.0,
        fliplr        = 0.5,
        mosaic        = 1.0,
        patience      = 10,
        project       = "runs/traffic_light",
        name          = "exp",
        exist_ok      = True,
        plots         = True,
        verbose       = True,
        amp           = False,      # disable AMP — causes issues on CPU
    )

    # Copy best weights to models/
    best = Path("runs/traffic_light/exp/weights/best.pt")
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy(best, "models/best.pt")
        print(f"\n✅ Training complete!")
        print(f"   Best weights → models/best.pt")
        print(f"   Run dashboard: python app.py")
    else:
        print("[WARN] best.pt not found — check runs/traffic_light/exp/weights/")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True,       help="Path to data.yaml")
    p.add_argument("--model",  default="s",          help="YOLOv8 size: n/s/m/l/x")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs (20 = quick test)")
    p.add_argument("--batch",  type=int, default=4,  help="Batch size (4-8 for CPU)")
    args = p.parse_args()

    train(args.data, args.model, args.epochs, args.batch)
