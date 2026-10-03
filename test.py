import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["WANDB_DISABLED"] = "true"
# Avoid the asynchronous DataLoader pin-memory teardown failure on the shared GPU runtime.
os.environ["PIN_MEMORY"] = "false"
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
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
        self.class_scale_maps = {str(name): {scale: None for scale in self.scale_area_ranges} for name in self.names.values()}

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

    def _compute_scale_maps(self, scale_stats):
        stats = {key: torch.cat(value, 0).cpu().numpy() for key, value in scale_stats.items()}
        if len(stats["target_cls"]) == 0:
            return 0.0, {}
        result = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            names=self.names,
        )
        ap, class_indices = result[5], result[6]
        per_class = {
            str(self.names[int(class_index)]): float(ap[index].mean()) * 100.0
            for index, class_index in enumerate(class_indices)
        }
        return (float(ap.mean()) if len(ap) else 0.0), per_class

    def get_stats(self):
        stats = super().get_stats()
        self.scale_maps = {}
        self.class_scale_maps = {str(name): {} for name in self.names.values()}
        for scale_name, scale_stats in self.scale_stats.items():
            mean_ap, per_class = self._compute_scale_maps(scale_stats)
            self.scale_maps[scale_name] = mean_ap
            for class_name in self.class_scale_maps:
                self.class_scale_maps[class_name][scale_name] = per_class.get(class_name)
        for name, value in self.scale_maps.items():
            stats[f"metrics/{name}(B)"] = value
        self.metrics.scale_maps = self.scale_maps
        self.metrics.scale_area_ranges = self.scale_area_ranges
        # ``model.val()`` returns ``self.metrics`` rather than the validator itself. Persist
        # the per-class map there so callers can write class_scale_ap.json instead of silently
        # emitting an empty ``classes`` object.
        self.metrics.class_scale_maps = self.class_scale_maps
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
        default="/home/room305/ZZF/yolov13yuan-6000/runs/baseline_d1_urpc2020_20260930_r1/train/seed1/A0/weights/best.pt",
        help="Path to best.pt or another trained checkpoint.",
    )
    parser.add_argument("--name", default="A0", help="Name under runs/test for this validation run.")
    parser.add_argument("--device", default="0", help="CUDA device id used for validation.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Validation dataloader workers.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data.yaml",
        help="Dataset YAML used for validation.",
    )
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "test", help="Validation output directory.")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False, help="Save validation plots.")
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
package_path = Path(ultralytics.__file__).resolve()
if ROOT not in package_path.parents:
    raise RuntimeError(f"Imported ultralytics outside project root: {package_path}")
best_weights_path = args.weights

if not os.path.exists(best_weights_path):
    print(f"Weights file does not exist: {best_weights_path}")
    print("Finish training first, or pass a checkpoint path with --weights.")
    exit(1)

print(f"Loading weights: {best_weights_path}")
model = YOLO(best_weights_path)
print("Starting validation...")
data_yaml = args.data.resolve()
if not data_yaml.is_file():
    raise FileNotFoundError(data_yaml)

results = model.val(
    validator=ScaleAwareDetectionValidator,
    data=str(data_yaml),
    split='val',
    imgsz=args.imgsz,
    batch=args.batch,
    workers=args.workers,
    conf=0.001,
    iou=0.5,
    device=args.device,
    plots=args.plots,
    save_json=True,
    project=str(args.project.resolve()),
    name=args.name,
)

save_dir = Path(results.save_dir)
metrics = to_float_dict(getattr(results, "results_dict", {}))
scale_maps = getattr(results, "scale_maps", {})
summary = {
    "weights": best_weights_path,
    "metrics": metrics,
    "scale_metrics_percent": {},
    "per_class_metrics_percent": {},
}

box_metrics = results.box
names = getattr(results, "names", {})
for metric_index, class_index in enumerate(box_metrics.ap_class_index):
    class_name = str(names[int(class_index)])
    precision, recall, map50, map75, map50_95 = box_metrics.class_result(metric_index)
    summary["per_class_metrics_percent"][class_name] = {
        "P": float(precision) * 100.0,
        "R": float(recall) * 100.0,
        "mAP50": float(map50) * 100.0,
        "mAP75": float(map75) * 100.0,
        "mAP50-95": float(map50_95) * 100.0,
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

    class_scale_path = save_dir / "class_scale_ap.json"
    class_scale_payload = {
        "area_ranges_px2": results.scale_area_ranges,
        "classes": getattr(results, "class_scale_maps", {}),
    }
    with open(class_scale_path, "w", encoding="utf-8") as f:
        json.dump(class_scale_payload, f, indent=2)
    summary["class_scale_ap"] = class_scale_payload["classes"]
    print(f"Class-scale AP metrics saved to: {class_scale_path}")

summary_path = save_dir / "summary_metrics.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Summary metrics saved to: {summary_path}")
