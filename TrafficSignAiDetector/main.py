import os
import cv2
import yaml
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from process_data import download_data, process_dataset

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(os.getcwd())
DATASET_DIR = BASE_DIR / "datasets"
PROJECT_DIR = BASE_DIR / "traffic_sign_project"
IMGSZ = 640
SEED = 42

DATASET_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ==========================================
# YAML CONFIG
# ==========================================
def create_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 43,
        "names": [str(i) for i in range(43)],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path

# ==========================================
# TRAINING (YOLOv8s)
# ==========================================
def train_model(yaml_path):
    print("\n--- Starting Training (YOLOv8s) ---")
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=20,
        workers=8,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,
        patience=20,
        amp=True,
        cache='disk',
    )
    return model, results

# ==========================================
# QUICK TEST
# ==========================================
def predict_sample(model):
    val_dir = DATASET_DIR / "images" / "val"
    img_path = next(val_dir.glob("*.jpg"), None)
    if img_path is None:
        print("No validation images found")
        return

    results = model.predict(source=str(img_path), conf=0.7)
    for r in results:
        out = r.plot()
        cv2.imwrite("prediction_preview.jpg", out)
        print("Saved prediction_preview.jpg")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)

    # Download and process from process_data.py
    raw_path = download_data(force_download=False)
    process_dataset(raw_path, DATASET_DIR, IMGSZ, SEED, force_rebuild=True)
    
    # Create YAML, train, and test
    yaml_path = create_yaml()
    model, _ = train_model(yaml_path)
    predict_sample(model)
