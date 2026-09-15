#!/usr/bin/env python3
"""Derive repaired DCRQ B3/B4 YAMLs from the verified B2 parent without layer insertion."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml


PARENT_NAME = "yolov13n-dcrq-b2.yaml"
OUTPUTS = {"B3_DAWRB_D_RPQC": "yolov13n-dcrq-b3.yaml", "B4_DCRQ_FULL": "yolov13n-dcrq-b4.yaml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True, help="Absolute audit location for the pre-repair YAMLs.")
    return parser.parse_args()


def layers(config: dict) -> list:
    return list(config.get("backbone", [])) + list(config.get("head", []))


def validate_b2(config: dict) -> None:
    graph = layers(config)
    if len(graph) != 33 or graph[4][2] != "DSC3k2" or graph[20][2] != "Concat":
        raise ValueError("B2 parent is not the expected 33-layer DCRQ graph")
    if graph[-1][2] != "RPQUDQDetect" or list(graph[-1][0]) != [23, 27, 31]:
        raise ValueError("B2 Detect topology changed")
    if float(config.get("quality_gain", 0.0)) != 0.25:
        raise ValueError("B2 quality_gain must remain 0.25")


def repaired(parent: dict, stage: str) -> dict:
    result = copy.deepcopy(parent)
    graph = layers(result)
    graph[4][2] = "DAWRBDSC3k2"
    graph[4][3] = [512, False, 0.25, 4, 0.05, True]
    if stage == "B4_DCRQ_FULL":
        graph[20][2] = "RCFConcat"
        graph[20][3] = [4, 0.04, True]
    result["backbone"], result["head"] = graph[:9], graph[9:]
    result["dcrq_repair"] = {
        "stage": stage,
        "parent": PARENT_NAME,
        "same_index_layer4": "DAWRBDSC3k2",
        "layer_count": 33,
        "pretrained_coverage_requirement": "B2_subset",
    }
    final_graph = layers(result)
    if len(final_graph) != 33 or list(final_graph[-1][0]) != [23, 27, 31]:
        raise AssertionError("Repaired graph changed layer count or Detect inputs")
    return result


def main() -> None:
    args = parse_args()
    root, archive_dir = args.root.resolve(), args.archive_dir.resolve()
    if not root.is_absolute() or not archive_dir.is_absolute():
        raise ValueError("root and archive-dir must be absolute")
    model_dir = root / "ultralytics" / "cfg" / "models" / "v13"
    parent_path = model_dir / PARENT_NAME
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    validate_b2(parent)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = []
    for stage, name in OUTPUTS.items():
        output = model_dir / name
        if output.is_file():
            backup = archive_dir / name
            if not backup.is_file():
                shutil.copy2(output, backup)
            archive.append({"stage": stage, "old_yaml": str(output), "backup": str(backup), "sha256": sha256(backup)})
        output.write_text(
            "# REPAIRED: derived from DCRQ B2; same-index layer4 DAWRBDSC3k2, never an inserted layer.\n"
            + yaml.safe_dump(repaired(parent, stage), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"generated {output}")
    (archive_dir / "old_yaml_manifest.json").write_text(
        json.dumps({"parent": str(parent_path), "parent_sha256": sha256(parent_path), "old_yamls": archive}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
