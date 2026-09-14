#!/usr/bin/env python
"""跑短基线相对定位。RTKLIB 对照见 scripts/05_run_rtklib_compare.py。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ladder.rinex.obs import read_rinex_obs
from ladder.rtk.engine import run_rtk_float, summarize_rtk, write_rtk_csv
from ladder.util.config import load_config, yyyy_ddd
from ladder.verify.metrics import plot_enu_series, save_json


def main() -> None:
    cfg = load_config()
    yyyy, ddd, _ = yyyy_ddd(cfg["experiment"]["year"], cfg["experiment"]["doy"])
    rnx_dir = Path(cfg["paths"]["data_root"]) / "rinex" / f"{yyyy}{ddd}"
    rover = next(rnx_dir.glob(f"{cfg['stations']['rover']}*.rnx"))
    base = next(rnx_dir.glob(f"{cfg['stations']['base']}*.rnx"))
    nav = next(rnx_dir.glob("BRDM*.rnx"))
    out_dir = Path(cfg["paths"]["results_root"]) / "rtk"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = read_rinex_obs(rover, max_epochs=1).approx_xyz
    base_xyz = read_rinex_obs(base, max_epochs=1).approx_xyz
    print("[B] self RTK (session-fixed) ...")
    results = run_rtk_float(
        rover,
        base,
        nav,
        elev_mask_deg=cfg["rtk"]["elev_mask_deg"],
        max_epochs=cfg["rtk"]["max_epochs"],
        sample_every_n=cfg["rtk"].get("sample_every_n", 1),
        session_epochs=cfg["rtk"].get("session_epochs", 60),
        use_ekf=bool(cfg["rtk"].get("use_ekf", False)),
    )
    csv = out_dir / "rtk_self.csv"
    write_rtk_csv(results, csv, ref)
    summary = summarize_rtk(results, ref, base_xyz)
    save_json(summary, out_dir / "rtk_self_summary.json")
    plot_enu_series(csv, out_dir / "rtk_self_enu.png", "Stage B self DD ENU vs rover approx")
    print("self summary:", summary)


if __name__ == "__main__":
    main()
