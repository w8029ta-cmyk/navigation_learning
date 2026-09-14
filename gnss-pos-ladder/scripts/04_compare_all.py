#!/usr/bin/env python
"""汇总三阶结果，写出 compare 摘要。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ladder.util.config import load_config
from ladder.verify.metrics import save_json


def main() -> None:
    cfg = load_config()
    res = Path(cfg["paths"]["results_root"])
    out = {"experiment": cfg["experiment"]["name"], "stages": {}}
    for stage, name in [("spp", "A_SPP"), ("rtk", "B_RTK"), ("ppp", "C_PPP")]:
        p = res / stage / f"{stage}_self_summary.json"
        if p.exists():
            out["stages"][name] = json.loads(p.read_text(encoding="utf-8"))
        else:
            out["stages"][name] = {"missing": True, "path": str(p)}
    save_json(out, res / "compare" / "ladder_summary.json")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
