import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO
import os
import json
import argparse
from pathlib import Path
import torch

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ap_per_class


SCALE_AREA_RANGES = {
    "APS": (0.0, 32.0**2),
    "APM": (32.0**2, 96.0**2),
    "APL": (96.0**2, float("inf")),
}


class ScaleAwareDetectionValidator(DetectionValidator):
    """Add COCO-style APS/APM/APL to the normal YOLO validation pass."""

    scale_area_ranges = SCALE_AREA_RANGES

    @staticmethod
    def _box_area_xyxy(boxes):
        if len(boxes) == 0:
            return boxes.new_zeros((0,))
        wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp(min=0)
        return wh[:, 0] * wh[:, 1]

    @staticmethod
    def _area_mask(areas, min_area, max_area):
        mask = areas >= min_area
        if max_area != float("inf"):
            mask = mask & (areas < max_area)
        return mask

    def init_metrics(self, model):
        super().init_metrics(model)
        self.scale_stats = {
            name: {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
            for name in self.scale_area_ranges
        }
        self.scale_maps = {name: 0.0 for name in self.scale_area_ranges}

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)

        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            target_area = self._box_area_xyxy(bbox)

            if len(pred):
                if self.args.single_cls:
                    pred[:, 5] = 0
                predn = self._prepare_pred(pred, pbatch)
                pred_area = self._box_area_xyxy(predn[:, :4])
            else:
                predn = torch.zeros((0, 6), device=self.device)
                pred_area = torch.zeros(0, device=self.device)

            for name, (min_area, max_area) in self.scale_area_ranges.items():
                target_mask = self._area_mask(target_area, min_area, max_area)
                pred_mask = self._area_mask(pred_area, min_area, max_area)
                target_cls = cls[target_mask]
                target_bbox = bbox[target_mask]
                scale_pred = predn[pred_mask]

                stat = {
                    "tp": torch.zeros(len(scale_pred), self.niou, dtype=torch.bool, device=self.device),
                    "conf": scale_pred[:, 4] if len(scale_pred) else torch.zeros(0, device=self.device),
                    "pred_cls": scale_pred[:, 5] if len(scale_pred) else torch.zeros(0, device=self.device),
                    "target_cls": target_cls,
                }
                if len(target_cls) and len(scale_pred):
                    stat["tp"] = self._process_batch(scale_pred, target_bbox, target_cls)

                for key, value in stat.items():
                    self.scale_stats[name][key].append(value)

    def _compute_scale_map(self, scale_stats):
        stats = {key: torch.cat(value, 0).cpu().numpy() for key, value in scale_stats.items()}
        if len(stats["target_cls"]) == 0:
            return 0.0
        ap = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            names=self.names,
        )[5]
        return float(ap.mean()) if len(ap) else 0.0

    def get_stats(self):
        stats = super().get_stats()
        self.scale_maps = {
            name: self._compute_scale_map(scale_stats)
            for name, scale_stats in self.scale_stats.items()
        }
        for name, value in self.scale_maps.items():
            stats[f"metrics/{name}(B)"] = value
        self.metrics.scale_maps = self.scale_maps
        self.metrics.scale_area_ranges = self.scale_area_ranges
        return stats

    def print_results(self):
        super().print_results()
        if hasattr(self, "scale_maps"):
            LOGGER.info(
                ("%22s" + "%11.3g" * 3)
                % ("scale AP", self.scale_maps["APS"], self.scale_maps["APM"], self.scale_maps["APL"])
            )

# ---------------- 1. Runtime settings ----------------
os.environ['WANDB_DISABLED'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description="Validate URPC YOLO weights with scale-aware AP metrics.")
    parser.add_argument(
        "--weights",
        default="/home/rtx6000/ZZF/yolov13yuan-6000/runs/train/exp3/weights/best.pt",
        help="Path to best.pt or another trained checkpoint.",
    )
    parser.add_argument("--name", default="exp", help="Name under runs/test for this validation run.")
    parser.add_argument("--device", default="0", help="CUDA device id used for validation.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    return parser.parse_args()


def to_float_dict(values):
    out = {}
    for key, value in values.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = str(value)
    return out


args = parse_args()
best_weights_path = args.weights

if not os.path.exists(best_weights_path):
    print(f"Weights file does not exist: {best_weights_path}")
    print("Finish training first, or pass a checkpoint path with --weights.")
    exit(1)

print(f"Loading weights: {best_weights_path}")
model = YOLO(best_weights_path)
print("Starting validation...")
data_yaml = '/home/rtx6000/ZZF/yolov13yuan-6000/data.yaml'

results = model.val(
    validator=ScaleAwareDetectionValidator,
    data=data_yaml,
    split='val',
    imgsz=args.imgsz,
    batch=args.batch,
    conf=0.001,
    iou=0.5,
    device=args.device,
    plots=True,
    save_json=True,
    project='/home/rtx6000/ZZF/yolov13yuan-6000/runs/test',
    name=args.name,
)

save_dir = Path(results.save_dir)
metrics = to_float_dict(getattr(results, "results_dict", {}))
scale_maps = getattr(results, "scale_maps", {})
summary = {
    "weights": best_weights_path,
    "metrics": metrics,
    "scale_metrics_percent": {},
}

if scale_maps:
    print("\nScale-aware AP metrics (COCO area ranges, AP@0.50:0.95):")
    for name in ("APS", "APM", "APL"):
        print(f"{name}: {scale_maps[name] * 100:.2f}%")

    metrics_path = save_dir / "scale_ap_metrics.json"
    summary["scale_metrics_percent"] = {name: value * 100 for name, value in scale_maps.items()}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "area_ranges_px2": results.scale_area_ranges,
                "metrics": summary["scale_metrics_percent"],
            },
            f,
            indent=2,
        )
    print(f"Scale-aware AP metrics saved to: {metrics_path}")

summary_path = save_dir / "summary_metrics.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Summary metrics saved to: {summary_path}")
