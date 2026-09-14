#!/usr/bin/env python
"""跑 PPP。RTKLIB 对照见 scripts/05_run_rtklib_compare.py。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ladder.ppp.engine import run_ppp, summarize_ppp, write_ppp_csv
from ladder.rinex.obs import read_rinex_obs
from ladder.util.config import gps_week_dow, load_config, yyyy_ddd
from ladder.verify.metrics import plot_enu_series, save_json


def main() -> None:
    cfg = load_config()
    year, doy = cfg["experiment"]["year"], cfg["experiment"]["doy"]
    yyyy, ddd, _ = yyyy_ddd(year, doy)
    week, _ = gps_week_dow(year, doy)
    rnx_dir = Path(cfg["paths"]["data_root"]) / "rinex" / f"{yyyy}{ddd}"
    prod_dir = Path(cfg["paths"]["data_root"]) / "products" / f"{week}"
    obs = next(rnx_dir.glob(f"{cfg['stations']['spp_station']}*.rnx"))
    sp3 = next(prod_dir.glob("*.SP3"))
    clk = next(prod_dir.glob("*.CLK"))
    bia_files = sorted(prod_dir.glob("*OSB.BIA")) or sorted(prod_dir.glob("*.BIA"))
    bia = bia_files[0] if bia_files else None
    if bia:
        print("using BIA:", bia.name)
    atx_cands = sorted(Path(cfg["paths"]["data_root"]).glob("**/*igs20*.atx")) + sorted(
        prod_dir.glob("*.atx")
    )
    atx = atx_cands[0] if atx_cands else None
    if atx:
        print("using ATX:", atx.name)
    out_dir = Path(cfg["paths"]["results_root"]) / "ppp"
    out_dir.mkdir(parents=True, exist_ok=True)

    hdr = read_rinex_obs(obs, max_epochs=1)
    ref = hdr.approx_xyz
    print(f"antenna: type={hdr.ant_type!r} delta_HEN={hdr.ant_delta_hen.tolist()}")
    print("[C] self PPP ...")
    results = run_ppp(
        obs,
        sp3,
        clk,
        bia_path=bia,
        atx_path=atx,
        elev_mask_deg=cfg["ppp"]["elev_mask_deg"],
        max_epochs=cfg["ppp"]["max_epochs"],
        sample_every_n=cfg["ppp"].get("sample_every_n", 5),
    )
    csv = out_dir / "ppp_self.csv"
    write_ppp_csv(results, csv, ref)
    summary = summarize_ppp(results, ref, skip_minutes=cfg["ppp"]["converge_minutes"])
    save_json(summary, out_dir / "ppp_self_summary.json")
    plot_enu_series(csv, out_dir / "ppp_self_enu.png", "Stage C self PPP ENU vs approx")
    print("self summary:", summary)


if __name__ == "__main__":
    main()
