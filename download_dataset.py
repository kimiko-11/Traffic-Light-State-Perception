"""
Download the traffic light dataset from Roboflow.
Run this once before training.

Usage:
    python download_dataset.py
"""

from roboflow import Roboflow

API_KEY   = "Aaakaq080qWHm4Kx70zB"
WORKSPACE = "student-esos5"
PROJECT   = "traffic-light-v9orl"
VERSION   = 3

print("[INFO] Downloading Traffic Light Dataset from Roboflow...")
print(f"       Project : {PROJECT}  |  Version : {VERSION}\n")

rf      = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8")

print(f"\n✅ Dataset downloaded to: {dataset.location}")
print(f"   Now run:  python train.py --data {dataset.location}/data.yaml")
