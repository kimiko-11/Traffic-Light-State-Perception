"""
Traffic Light Detection — Web Dashboard
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import torch
from pathlib import Path
from io import BytesIO
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ── Load model once at startup ──────────────────────────────
CUSTOM_MODEL = "models/best.pt"
COCO_TL_CLASS = 9

device = "cuda" if torch.cuda.is_available() else "cpu"
if Path(CUSTOM_MODEL).exists():
    model = YOLO(CUSTOM_MODEL)
    mode  = "custom"
    print(f"[INFO] Loaded custom model: {CUSTOM_MODEL}")
else:
    model = YOLO("yolov8n.pt")
    mode  = "pretrained"
    print("[INFO] Loaded YOLOv8n pretrained (COCO)")
model.to(device)

CUSTOM_CLASS_NAMES = {0: "green", 1: "red", 2: "yellow"}

ADVISORY = {
    "red":    {"label": "STOP",     "message": "Do NOT move",              "color": "#e53e3e", "emoji": "🔴"},
    "yellow": {"label": "WAIT",     "message": "Prepare to stop or go",    "color": "#d97706", "emoji": "🟡"},
    "green":  {"label": "GO",       "message": "Safe to drive",            "color": "#16a34a", "emoji": "🟢"},
    "off":    {"label": "CAUTION",  "message": "Light unclear",            "color": "#6b7280", "emoji": "❓"},
    "none":   {"label": "NO LIGHT", "message": "No traffic light detected","color": "#4b5563", "emoji": "🔍"},
}

PRIORITY = ["red", "yellow", "green", "off", "none"]

BOX_COLORS = {
    "red":    (0,   0, 220),
    "yellow": (0, 165, 255),
    "green":  (0, 200,   0),
    "off":    (180,180,180),
}


def classify_color(roi):
    if roi is None or roi.size == 0:
        return "off"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = roi.shape[:2]
    masks = {
        "red":    cv2.inRange(hsv,(0,120,70),(10,255,255)) + cv2.inRange(hsv,(170,120,70),(180,255,255)),
        "yellow": cv2.inRange(hsv,(15,100,100),(35,255,255)),
        "green":  cv2.inRange(hsv,(40,80,80),(90,255,255)),
    }
    scores = {c: cv2.countNonZero(m) for c, m in masks.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > (h * w * 0.03) else "off"


def run_detection(img_bytes):
    # Decode image (supports jpg, png, webp, bmp)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    results = model.predict(frame, conf=0.40, iou=0.45, imgsz=640, verbose=False)[0]

    detections    = []
    best_priority = len(PRIORITY) - 1
    state         = "none"

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])

        if mode == "pretrained" and cls_id != COCO_TL_CLASS:
            continue

        if mode == "custom":
            cls_name = CUSTOM_CLASS_NAMES.get(cls_id, "off")
        else:
            roi      = frame[max(0,y1):y2, max(0,x1):x2]
            cls_name = classify_color(roi)

        detections.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":round(conf,2),"label":cls_name})

        if cls_name in PRIORITY:
            p = PRIORITY.index(cls_name)
            if p < best_priority:
                best_priority = p
                state = cls_name

    # Draw boxes on image
    for d in detections:
        c = BOX_COLORS.get(d["label"], (180,180,180))
        cv2.rectangle(frame, (d["x1"],d["y1"]), (d["x2"],d["y2"]), c, 3)
        tag = f"{d['label'].upper()}  {d['conf']:.0%}"
        tw  = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        cv2.rectangle(frame, (d["x1"], d["y1"]-26), (d["x1"]+tw+10, d["y1"]), c, -1)
        cv2.putText(frame, tag, (d["x1"]+5, d["y1"]-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Encode result image to base64
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "state":      state,
        "advisory":   ADVISORY[state],
        "detections": detections,
        "image_b64":  img_b64,
        "count":      len(detections),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        result = run_detection(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║  Traffic Light Detection Dashboard   ║")
    print("║  Open: http://localhost:5000          ║")
    print("╚══════════════════════════════════════╝\n")
    app.run(debug=False, port=5000)
