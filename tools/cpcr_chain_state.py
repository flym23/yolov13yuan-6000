#!/usr/bin/env python3
"""Small atomic state writer for the CPCR server-side training chain."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--detail")
    args = parser.parse_args()
    path = args.path.resolve()
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update({"run_id": args.run_id, "status": args.status, "updated_at": datetime.now(timezone.utc).isoformat()})
    if args.stage is not None:
        current["stage"] = args.stage
    if args.detail is not None:
        current["detail"] = args.detail
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(current, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
