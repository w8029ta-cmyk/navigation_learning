"""结果指标与简单作图。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_enu_series(csv_path: str | Path, out_png: str | Path, title: str) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        return
    e, n, u = data["e"], data["n"], data["u"]
    t = np.arange(len(e))
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, e, lw=0.8)
    ax[0].set_ylabel("E (m)")
    ax[1].plot(t, n, lw=0.8)
    ax[1].set_ylabel("N (m)")
    ax[2].plot(t, u, lw=0.8)
    ax[2].set_ylabel("U (m)")
    ax[2].set_xlabel("epoch index")
    fig.suptitle(title)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
