#!/usr/bin/env python3
"""Archive an exact manifest and delete only the validated old DCRQ B3/B4 stage directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("B3_DAWRB_D_RPQC", "B4_DCRQ_FULL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-root", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--old-yaml-manifest", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def stage_record(path: Path) -> dict:
    files, metrics = [], {}
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        item = {"relative_path": str(file.relative_to(path)), "size": file.stat().st_size}
        if file.suffix.lower() in {".yaml", ".yml", ".json", ".csv", ".txt"}:
            item["sha256"] = sha256(file)
        files.append(item)
        if file.name == "all_seed_summary.json":
            try:
                metrics = json.loads(file.read_text(encoding="utf-8")).get("mean", {})
            except (OSError, ValueError):
                metrics = {"unreadable_summary": str(file)}
    return {"path": str(path), "files": files, "mean_metrics": metrics}


def main() -> None:
    args = parse_args()
    old_root = args.old_run_root.resolve()
    coverage_path, yaml_manifest = args.coverage_report.resolve(), args.old_yaml_manifest.resolve()
    manifest_path = args.manifest_out.resolve()
    if not old_root.is_dir() or old_root.is_symlink():
        raise RuntimeError(f"unsafe old run root: {old_root}")
    if not coverage_path.is_file() or not yaml_manifest.is_file():
        raise FileNotFoundError("Validated coverage report or archived old-YAML manifest is missing")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not coverage.get("B3_DAWRB_D_RPQC", {}).get("equal_to_b2") or not coverage.get("B4_DCRQ_FULL", {}).get("equal_to_b2"):
        raise RuntimeError("Refusing cleanup because repaired pretrained coverage did not equal B2")
    targets = []
    for category in ("train", "test"):
        for stage in STAGES:
            target = old_root / category / stage
            if not target.is_dir() or target.is_symlink() or target.parent != old_root / category:
                raise RuntimeError(f"Exact old B3/B4 stage directory is absent or unsafe: {target}")
            targets.append(target)
    manifest = {
        "old_run_root": str(old_root),
        "deleted_after_validation": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pretrained_coverage": coverage,
        "old_yaml_manifest": json.loads(yaml_manifest.read_text(encoding="utf-8")),
        "targets": [stage_record(target) for target in targets],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    for target in targets:
        shutil.rmtree(target)
        print(f"deleted {target}")
    print(f"repair manifest {manifest_path}")


if __name__ == "__main__":
    main()
