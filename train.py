"""Train the original YOLOv13 baseline on the configured URPC2020 dataset."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
# Avoid DataLoader pin-memory cleanup races when several experiments share one GPU.
os.environ["PIN_MEMORY"] = "false"

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO(str(ROOT / "ultralytics/cfg/models/v13/yolov13.yaml"))
    model.load(str(ROOT / "yolov13n.pt"))
    model.train(
        data="/home/room305/ZZF/URPC2020half/data.yaml",
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
