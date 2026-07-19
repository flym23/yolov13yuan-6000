"""Train the original YOLOv13 baseline on the configured URPC2020 dataset."""

import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":
    model = YOLO(str(ROOT / "ultralytics/cfg/models/v13/yolov13.yaml"))
    model.load(str(ROOT / "yolov13n.pt"))
    model.train(
        data="/home/room305/ZZF/URPC2020/data.yaml",
        epochs=300,
        patience=40,
        batch=16,
        workers=2,
        amp=False,
        deterministic=True,
        plots=False,
        project=str(ROOT / "runs/train"),
        name="baseline",
    )
