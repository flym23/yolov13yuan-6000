"""Validate DURC YOLOv13 weights with normal and scale-aware AP metrics."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["WANDB_DISABLED"] = "true"

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ap_per_class


ROOT_DIR = Path(__file__).resolve().parent
SCALE_AREA_RANGES = {
    "APS": (0.0, 32.0**2),
    "APM": (32.0**2, 96.0**2),
    "APL": (96.0**2, float("inf")),
}


def metric_names_from_data(data: dict, expected_nc: int) -> dict[int, str]:
    """Return complete zero-based metric names, rejecting malformed dataset metadata."""
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        raise TypeError("dataset YAML must define class names as a list or mapping")
    expected_ids = set(range(expected_nc))
    if set(names) != expected_ids:
        raise ValueError(
            f"dataset class names must use zero-based IDs {sorted(expected_ids)}, got {sorted(names)}"
        )
    return names


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
        model_nc = len(model.names)
        self.names = metric_names_from_data(self.data, model_nc)
        self.nc = model_nc
        self.metrics.names = self.names
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
                    "tp": torch.zeros(
                        len(scale_pred), self.niou, dtype=torch.bool, device=self.device
                    ),
                    "conf": scale_pred[:, 4]
                    if len(scale_pred)
                    else torch.zeros(0, device=self.device),
                    "pred_cls": scale_pred[:, 5]
                    if len(scale_pred)
                    else torch.zeros(0, device=self.device),
                    "target_cls": target_cls,
                }
                if len(target_cls) and len(scale_pred):
                    stat["tp"] = self._process_batch(
                        scale_pred, target_bbox, target_cls
                    )

                for key, value in stat.items():
                    self.scale_stats[name][key].append(value)

    def _compute_scale_map(self, scale_stats):
        stats = {
            key: torch.cat(value, 0).cpu().numpy() for key, value in scale_stats.items()
        }
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
                % (
                    "scale AP",
                    self.scale_maps["APS"],
                    self.scale_maps["APM"],
                    self.scale_maps["APL"],
                )
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to a trained best.pt checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT_DIR / "data_durc.yaml",
        help="Zero-based DURC dataset YAML.",
    )
    parser.add_argument(
        "--name", required=True, help="Name under runs/test for this validation run."
    )
    parser.add_argument(
        "--device", default="0", help="CUDA device id used for validation."
    )
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Validation dataloader workers.",
    )
    return parser.parse_args()


def to_float_dict(values):
    out = {}
    for key, value in values.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = str(value)
    return out


def main():
    args = parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = YOLO(str(args.weights))
    results = model.val(
        validator=ScaleAwareDetectionValidator,
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=0.001,
        iou=0.5,
        device=args.device,
        plots=True,
        save_json=True,
        project=str(ROOT_DIR / "runs/test"),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    scale_maps = getattr(results, "scale_maps", {})
    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "amp": False,
        "metrics": to_float_dict(getattr(results, "results_dict", {})),
        "scale_metrics_percent": {
            name: value * 100 for name, value in scale_maps.items()
        },
    }
    if scale_maps:
        with (save_dir / "scale_ap_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "area_ranges_px2": SCALE_AREA_RANGES,
                    "metrics": summary["scale_metrics_percent"],
                },
                file,
                indent=2,
            )
    with (save_dir / "summary_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
