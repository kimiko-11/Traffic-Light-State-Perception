# 🚦 Traffic Light Detection System
**YOLOv8 + PyTorch** — Real-time vehicle advisory (STOP / WAIT / GO)

---

## 📁 Project Structure

```
traffic_light_detection/
├── detect.py               ← Main detection script (run this)
├── train.py                ← Train the model on your dataset
├── download_dataset.py     ← Download dataset from Roboflow
├── requirements.txt        ← Python dependencies
└── models/
    └── best.pt             ← Place trained weights here
```

---

## ⚙️ Setup (run once)

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Quick test (no training needed)
Uses YOLOv8 pretrained on COCO with HSV colour fallback.
Works immediately with no setup beyond pip install.

```bash
# Webcam
python detect.py --source 0

# Video file
python detect.py --source video.mp4

# Single image
python detect.py --source image.jpg

# Folder of images  →  results saved to results/
python detect.py --source images/
```

### Option B — Full pipeline (download → train → detect)

```bash
# Step 1: Download dataset (2,740 labelled images)
python download_dataset.py

# Step 2: Train YOLOv8 (~30-60 mins on GPU)
python train.py --data traffic-light-v9orl-3/data.yaml

# Step 3: Run detection (weights auto-loaded from models/best.pt)
python detect.py --source 0
```

---

## 🎮 Controls
Press **Q** to quit the detection window.

---

## 📺 Output

| Traffic Light | Advisory Banner |
|---------------|----------------|
| 🔴 Red | **STOP** — Do NOT move |
| 🟡 Yellow | **WAIT** — Prepare to stop or go |
| 🟢 Green | **GO** — Safe to drive |
| ❓ Unclear | **CAUTION** — Drive carefully |

---

## 🏗️ System Design

```
Camera Frame
     │
     ▼
┌─────────────────┐
│ YOLOv8 Detector │  detects traffic light bounding boxes
│   (PyTorch)     │  classifies: red / yellow / green
└────────┬────────┘
         │  if colour not sub-classified:
         ▼
┌─────────────────┐
│  HSV Colour     │  pixel-level hue analysis inside box
│  Classifier     │  (fallback for pretrained COCO model)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Priority Engine │  RED > YELLOW > GREEN (safety-first)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Driver HUD    │  STOP / WAIT / GO overlay on video
└─────────────────┘
```

---

## 📊 Dataset & Model Performance

- **Dataset:** Traffic Light Computer Vision Model (Roboflow Universe, by Student)
- **Images:** 2,740 | **Classes:** Red, Green, Yellow
- **mAP@50:** 96.9% | **Precision:** 94.6% | **Recall:** 91.3%
- **Model:** YOLOv8s (can swap to n/m/l/x for speed/accuracy tradeoff)
