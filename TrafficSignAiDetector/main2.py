import os
import torch
from ultralytics import YOLO

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_YAML = os.path.join(ROOT_DIR, "dataset_yolo", "data.yaml")


def train_yolo():
    print("DATA_YAML:", DATA_YAML)
    print("Exists:", os.path.exists(DATA_YAML))

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU not found. Training will run on CPU.")

    model = YOLO("yolo26s.pt")

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        name="traffic_sign_yolo26s"
    )


if __name__ == "__main__":
    train_yolo()