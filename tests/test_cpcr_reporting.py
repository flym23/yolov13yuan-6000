"""Regression tests for CPCR diagnostic reporting and conditional L5 gate evaluation."""

from __future__ import annotations

import torch

from test import ScaleAwareDetectionValidator
from tools.collect_cpcr_ablation import evaluate_l4


def test_scale_validator_reports_per_class_tp_fp_fn_and_size_recall():
    validator = object.__new__(ScaleAwareDetectionValidator)
    validator.nc = 2
    validator.stats = {
        "tp": [torch.tensor([[True, False, False, False, False, False], [False, False, False, False, False, False]], dtype=torch.bool)],
        "conf": [torch.tensor([0.9, 0.8])],
        "pred_cls": [torch.tensor([0.0, 1.0])],
        "target_cls": [torch.tensor([0.0, 1.0])],
        "target_img": [torch.tensor([0.0, 1.0])],
    }
    validator.scale_stats = {
        name: {"tp": [torch.tensor([[True, False, False, False, False, False]], dtype=torch.bool)], "conf": [torch.tensor([0.9])], "pred_cls": [torch.tensor([0.0])], "target_cls": [torch.tensor([0.0])]} for name in ("APS", "APM", "APL")
    }
    report = validator._class_diagnostics()
    assert report["0"]["TP_iou50"] == 1 and report["0"]["FP_iou50"] == 0 and report["0"]["FN_iou50"] == 0
    assert report["1"]["TP_iou50"] == 0 and report["1"]["FP_iou50"] == 1 and report["1"]["FN_iou50"] == 1
    assert report["0"]["APS_recall_iou50"] == 1.0 and report["1"]["APS_recall_iou50"] == 0.0


def test_l4_gate_requires_all_documented_statistics():
    rows = [{"seed": seed, "P": 80.0, "R": 74.0, "mAP50": 79.2, "mAP75": 47.8, "mAP50-95": 45.9, "APS": 13.8, "APM": 38.5, "APL": 47.1, "GFLOPs": 6.65} for seed in (0, 1, 2)]
    assert evaluate_l4(rows)["passed"]
    rows[0]["mAP50-95"] = 45.0
    assert not evaluate_l4(rows)["passed"]
