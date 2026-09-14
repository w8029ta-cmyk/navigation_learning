from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    # ladder/util/config.py → parents[2] = 仓库根目录
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "config" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve relative paths against repo root
    root = repo_root()
    paths = cfg.setdefault("paths", {})
    for key in ("data_root", "results_root"):
        p = Path(paths[key])
        if not p.is_absolute():
            paths[key] = str((root / p).resolve())
    return cfg


def gps_week_dow(year: int, doy: int) -> tuple[int, int]:
    """年积日 → GPS 周 / 周内日（与 IGS 产品目录一致）。"""
    from datetime import date, timedelta

    d = date(year, 1, 1) + timedelta(days=doy - 1)
    gps0 = date(1980, 1, 6)
    delta = (d - gps0).days
    return delta // 7, delta % 7


def yyyy_ddd(year: int, doy: int) -> tuple[str, str, str]:
    return f"{year:04d}", f"{doy:03d}", f"{year % 100:02d}"
