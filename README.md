#  Traffic Light Detection System
**YOLOv8 + PyTorch** — Real-time vehicle advisory (STOP / WAIT / GO)

---

## Project Structure

```
traffic_light_detection/
├── detect.py               ← Main detection script (run this)
├── train.py                ← Train the model on your dataset
├── download_dataset.py     ← Download dataset from Roboflow
├── requirements.txt        ← Python dependencies
└── models/
    └── best.pt             ← Place trained weights here
```
after forking and cloning the repo
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py

This project uses YOLOv8s specialized  for traffic light state recognition, it was fine-tuned on a custom Roboflow dataset containing labeled red, yellow, and green traffic lights.  transfer learning was used to adapt YOLOv8's existing knowledge, allowing the model to learn traffic light color classification efficiently. After 20 training epochs, the best-performing weights were saved as best.pt, resulting in a model capable of accurately detecting and classifying traffic light states for real-time driver advisory applications.
