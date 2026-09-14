#!/usr/bin/env python
"""跑 SPP。RTKLIB 对照见 scripts/05_run_rtklib_compare.py。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ladder.rinex.obs import read_rinex_obs
from ladder.spp.engine import run_spp, summarize_spp, write_spp_csv
from ladder.util.config import load_config, yyyy_ddd
from ladder.verify.metrics import plot_enu_series, save_json


def main() -> None:
    cfg = load_config()
    yyyy, ddd, _ = yyyy_ddd(cfg["experiment"]["year"], cfg["experiment"]["doy"])
    data = Path(cfg["paths"]["data_root"])
    rnx_dir = data / "rinex" / f"{yyyy}{ddd}"
    sta = cfg["stations"]["spp_station"]
    obs = next(rnx_dir.glob(f"{sta}*.rnx"))
    nav = next(rnx_dir.glob("BRDM*.rnx"))
    out_dir = Path(cfg["paths"]["results_root"]) / "spp"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[A] self SPP ...")
    header = read_rinex_obs(obs, max_epochs=1)
    ref = header.approx_xyz
    results = run_spp(
        obs,
        nav,
        elev_mask_deg=cfg["spp"]["elev_mask_deg"],
        max_epochs=cfg["spp"]["max_epochs"],
        sample_every_n=cfg["spp"]["sample_every_n"],
        use_if_combination=bool(cfg["spp"].get("use_if_combination", False)),
    )
    csv = out_dir / "spp_self.csv"
    write_spp_csv(results, csv, ref)
    summary = summarize_spp(results, ref)
    save_json(summary, out_dir / "spp_self_summary.json")
    plot_enu_series(csv, out_dir / "spp_self_enu.png", "Stage A self SPP ENU vs approx")
    print("self summary:", summary)


if __name__ == "__main__":
    main()
