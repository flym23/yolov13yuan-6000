#!/usr/bin/env python3
"""Create an isolated, zero-based-label URPC view for DURC experiments."""

import argparse
import os
from pathlib import Path

import yaml


SOURCE_CLASS_IDS = frozenset({1, 2, 3, 4})


def normalize_label_line(line: str, source: Path, line_number: int) -> str:
    """Map the source URPC label IDs 1..4 to YOLO's required 0..3 IDs."""
    fields = line.split()
    if not fields:
        return ""
    try:
        class_id = int(fields[0])
    except ValueError as error:
        raise ValueError(
            f"{source}:{line_number}: invalid class id {fields[0]!r}"
        ) from error
    if class_id not in SOURCE_CLASS_IDS:
        expected = ", ".join(map(str, sorted(SOURCE_CLASS_IDS)))
        raise ValueError(
            f"{source}:{line_number}: class id {class_id} is outside {{{expected}}}"
        )
    fields[0] = str(class_id - 1)
    return " ".join(fields)


def convert_label_file(source: Path, destination: Path) -> None:
    """Copy one label file while applying the one-based to zero-based mapping."""
    normalized = [
        normalize_label_line(line, source, line_number)
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        )
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        "\n".join(normalized) + ("\n" if normalized else ""), encoding="utf-8"
    )
    os.replace(temporary, destination)


def resolve_split_path(dataset: dict, split: str) -> Path | None:
    value = dataset.get(split)
    if not value:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"{split} must be a single image directory, got {type(value).__name__}"
        )
    path = Path(value)
    return path if path.is_absolute() else Path(dataset["path"]) / path


def labels_path(images_path: Path) -> Path:
    parts = list(images_path.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise ValueError(
            f"image path must contain an 'images' component: {images_path}"
        ) from error
    parts[index] = "labels"
    return Path(*parts)


def make_image_link(source: Path, destination: Path) -> None:
    """Link source images without touching the shared dataset or duplicating files."""
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(
                f"refusing to replace unrelated image link: {destination}"
            )
        return
    if destination.exists():
        raise RuntimeError(
            f"refusing to replace non-link image directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


def build_durc_dataset(source_data: Path, target_root: Path, target_data: Path) -> dict:
    """Build the DURC-private dataset view and return its generated YAML mapping."""
    source_data = source_data.resolve()
    source = yaml.safe_load(source_data.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or "path" not in source or "names" not in source:
        raise ValueError(f"invalid dataset YAML: {source_data}")

    generated = {"path": str(target_root.resolve()), "names": source["names"]}
    for split in ("train", "val", "test"):
        images = resolve_split_path(source, split)
        if images is None:
            continue
        label_root = labels_path(images)
        if not images.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(
                f"missing {split} images or labels: {images}, {label_root}"
            )
        make_image_link(images, target_root / "images" / split)
        destination_root = target_root / "labels" / split
        for label_file in sorted(label_root.rglob("*.txt")):
            convert_label_file(
                label_file, destination_root / label_file.relative_to(label_root)
            )
        generated[split] = f"images/{split}"

    target_data.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_data.with_suffix(f"{target_data.suffix}.tmp")
    temporary.write_text(
        yaml.safe_dump(generated, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, target_data)
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = build_durc_dataset(args.source_data, args.target_root, args.target_data)
    print(f"DURC dataset prepared: {args.target_data} ({', '.join(generated)})")


if __name__ == "__main__":
    main()
