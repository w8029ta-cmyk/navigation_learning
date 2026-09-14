#!/usr/bin/env python
"""下载并解压本实验 GNSS 数据。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ladder.io.download import fetch_experiment_data
from ladder.util.config import load_config


def main() -> None:
    cfg = load_config()
    print("=== gnss-pos-ladder fetch ===")
    print(f"experiment: {cfg['experiment']['name']}")
    files = fetch_experiment_data(cfg)
    print("=== done ===")
    for k, v in files.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
