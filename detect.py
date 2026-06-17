"""
╔══════════════════════════════════════════════════════════╗
║         TRAFFIC LIGHT DETECTION SYSTEM                   ║
║         YOLOv8 + PyTorch                                 ║
║                                                          ║
║  Detects: Red / Yellow / Green traffic lights            ║
║  Output:  Real-time STOP / WAIT / GO advisory            ║
║  Input:   Webcam, video file, or image folder            ║
╚══════════════════════════════════════════════════════════╝

Author : [Your Name]
Date   : 2025

How it works:
  1. YOLOv8 detects traffic light bounding boxes in each frame
  2. HSV colour classifier confirms Red / Yellow / Green state
  3. A priority engine picks the most critical light (Red > Yellow > Green)
  4. A HUD overlay tells the driver: STOP / WAIT / GO
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import argparse
import os

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONF_THRESH = 0.40      # minimum detection confidence
IOU_THRESH  = 0.45      # NMS IoU threshold
IMG_SIZE    = 640       # YOLOv8 input size

# If you have a custom trained model, set the path here.
# Otherwise leave as None — the system will use YOLOv8 pretrained on COCO
# which includes "traffic light" as a class.
CUSTOM_MODEL_PATH = "models/best.pt"

# COCO class ID for traffic light (used when no custom model)
COCO_TRAFFIC_LIGHT_CLASS = 9

# Class mapping for custom-trained model
CUSTOM_CLASS_NAMES = {
    0: "green",
    1: "red",
    2: "yellow",
}

# Driver advisory: state → (label, BGR colour, message)
ADVISORY = {
    "red":    ("STOP",     (0,   0, 220), "Do NOT move"),
    "yellow": ("WAIT",     (0, 165, 255), "Prepare to stop or go"),
    "green":  ("GO",       (0, 200,   0), "Safe to drive"),
    "off":    ("CAUTION",  (180,180,180), "Light unclear — drive carefully"),
    "none":   ("SCANNING", (80,  80,  80),"No traffic light detected"),
}

# Light priority (most critical first)
PRIORITY = ["red", "yellow", "green", "off", "none"]


# ─────────────────────────────────────────────────────────────
# HSV COLOUR CLASSIFIER
# Fallback when model doesn't sub-classify light colour.
# Analyses the pixel hue inside the detected bounding box.
# ─────────────────────────────────────────────────────────────
def classify_color(roi: np.ndarray) -> str:
    if roi is None or roi.size == 0:
        return "off"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = roi.shape[:2]
    total_pixels = h * w

    masks = {
        "red":    cv2.inRange(hsv, (0,  120, 70), (10, 255,255))
                + cv2.inRange(hsv, (170,120, 70), (180,255,255)),
        "yellow": cv2.inRange(hsv, (15, 100,100), (35, 255,255)),
        "green":  cv2.inRange(hsv, (40,  80, 80), (90, 255,255)),
    }

    scores = {c: cv2.countNonZero(m) for c, m in masks.items()}
    best   = max(scores, key=scores.get)

    # Require at least 3% of the ROI to be the colour
    return best if scores[best] > (total_pixels * 0.03) else "off"


# ─────────────────────────────────────────────────────────────
# HUD OVERLAY
# Draws bounding boxes and the advisory banner on the frame.
# ─────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, state: str, detections: list,
             fps: float, mode: str) -> np.ndarray:
    h, w = frame.shape[:2]
    label, color, advice = ADVISORY[state]

    # Bounding boxes
    for (x1, y1, x2, y2, conf, cls_name) in detections:
        _, box_color, _ = ADVISORY.get(cls_name, ADVISORY["off"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        tag = f"{cls_name.upper()}  {conf:.0%}"
        tw, th = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.rectangle(frame, (x1, y1-24), (x1+tw+8, y1), box_color, -1)
        cv2.putText(frame, tag, (x1+4, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Semi-transparent bottom banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

    # Advisory text
    cv2.putText(frame, label, (20, h-52),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
    cv2.putText(frame, advice, (22, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200,200,200), 1)

    # Top-right info bar
    cv2.putText(frame, f"FPS: {fps:5.1f}", (w-130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 1)
    cv2.putText(frame, f"Mode: {mode}", (w-160, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

    # Top-left title
    cv2.putText(frame, "Traffic Light Detection", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

    return frame


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device      : {device.upper()}")

    # Use custom trained model if available
    if Path(CUSTOM_MODEL_PATH).exists():
        print(f"[INFO] Model       : Custom trained ({CUSTOM_MODEL_PATH})")
        model = YOLO(CUSTOM_MODEL_PATH)
        mode  = "custom"
    else:
        print("[INFO] Model       : YOLOv8n pretrained (COCO)")
        print("[INFO] Tip         : Place trained weights at models/best.pt for better accuracy")
        model = YOLO("yolov8n.pt")   # auto-downloads ~6MB
        mode  = "pretrained"

    model.to(device)
    return model, mode, device


# ─────────────────────────────────────────────────────────────
# PROCESS A SINGLE FRAME
# ─────────────────────────────────────────────────────────────
def process_frame(frame, model, mode):
    results = model.predict(
        frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMG_SIZE,
        verbose=False,
    )[0]

    detections   = []
    best_priority = len(PRIORITY) - 1
    state         = "none"

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])

        # Filter: pretrained COCO model only keeps traffic light class
        if mode == "pretrained" and cls_id != COCO_TRAFFIC_LIGHT_CLASS:
            continue

        # Get colour
        if mode == "custom":
            cls_name = CUSTOM_CLASS_NAMES.get(cls_id, "off")
        else:
            roi      = frame[max(0,y1):y2, max(0,x1):x2]
            cls_name = classify_color(roi)

        detections.append((x1, y1, x2, y2, conf, cls_name))

        # Track highest-priority light
        if cls_name in PRIORITY:
            p = PRIORITY.index(cls_name)
            if p < best_priority:
                best_priority = p
                state = cls_name

    return detections, state


# ─────────────────────────────────────────────────────────────
# RUN — WEBCAM or VIDEO
# ─────────────────────────────────────────────────────────────
def run_video(source, model, mode):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    print(f"[INFO] Source      : {source}")
    print("[INFO] Press Q to quit\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        detections, state = process_frame(frame, model, mode)

        fps   = 1.0 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()

        frame = draw_hud(frame, state, detections, fps, mode)
        cv2.imshow("Traffic Light Detection  [Q = quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# RUN — IMAGE or IMAGE FOLDER
# ─────────────────────────────────────────────────────────────
def run_images(source, model, mode):
    source = Path(source)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source.is_dir():
        images = [f for f in source.iterdir() if f.suffix.lower() in exts]
    else:
        images = [source]

    if not images:
        print(f"[ERROR] No images found in: {source}")
        return

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print(f"[INFO] Processing {len(images)} image(s)...")
    print(f"[INFO] Results saved to: results/\n")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        detections, state = process_frame(frame, model, mode)
        _, color, advice  = ADVISORY[state]

        frame = draw_hud(frame, state, detections, fps=0.0, mode=mode)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), frame)
        print(f"  {img_path.name:40s}  →  {state.upper():8s}  ({advice})")

    print(f"\n[INFO] Done. Open the 'results/' folder to see annotated images.")

    # Show the last image
    cv2.imshow("Last Result  [any key = close]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Light Detection — YOLOv8 + PyTorch"
    )
    parser.add_argument(
        "--source", default="0",
        help=(
            "Input source:\n"
            "  0          → default webcam\n"
            "  1, 2       → other webcams\n"
            "  video.mp4  → video file\n"
            "  image.jpg  → single image\n"
            "  images/    → folder of images"
        )
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════╗
║   TRAFFIC LIGHT DETECTION SYSTEM         ║
║   YOLOv8 + PyTorch                       ║
╚══════════════════════════════════════════╝""")

    model, mode, device = load_model()

    # Determine source type
    src = args.source
    try:
        src = int(src)   # webcam index
        is_image = False
    except ValueError:
        path = Path(src)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        is_image = path.is_dir() or path.suffix.lower() in image_exts

    if is_image:
        run_images(src, model, mode)
    else:
        run_video(src, model, mode)
