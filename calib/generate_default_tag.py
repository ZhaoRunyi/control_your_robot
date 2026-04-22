#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calib.utils import (
    create_a4_tag_png,
    ensure_dir,
    get_default_tag_config,
    save_json,
)


def main() -> int:
    calib_root = ensure_dir(PROJECT_ROOT / "calib")
    tag_config = get_default_tag_config()

    png_path = calib_root / tag_config["png_name"]
    metadata_path = calib_root / tag_config["metadata_name"]

    metadata = create_a4_tag_png(
        output_path=png_path,
        dictionary_name=tag_config["dictionary_name"],
        marker_id=tag_config["marker_id"],
        tag_size_mm=tag_config["tag_size_mm"],
        dpi=tag_config["dpi"],
    )
    save_json(metadata_path, metadata)

    print(f"tag_png: {png_path}")
    print(f"tag_metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
