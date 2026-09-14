#!/usr/bin/env python
"""PPP 残差诊断（固定 approx）。

看码/相 IF、高度角/方位、有无 OSB、相对 RTKLIB .pos。不改引擎，只出诊断结果。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ladder.ephemeris.precise import read_clk, read_sp3
from ladder.models.atmosphere import (
    earth_rotation_corr,
    elevation_azimuth,
    geometric_range,
    saastamoinen_zhd_zwd,
    trop_delay_est_ztd,
)
from ladder.ppp.engine import _doy, _pick_dual_code_phase
from ladder.products.bias import read_bias_sinex
from ladder.rinex.obs import read_rinex_obs
from ladder.util.config import gps_week_dow, load_config, yyyy_ddd
from ladder.util.constants import CLIGHT
from ladder.util.coord import ecef_to_enu, xyz_to_enu_rotation


def _parse_rtklib_pos(path: Path, ref: np.ndarray) -> List[dict]:
    rows = []
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("%") or line.startswith("@"):
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                # GPST yyyy/mm/dd hh:mm:ss.sss x y z ...
                if "/" in p[0]:
                    t = datetime.fromisoformat(
                        f"{p[0].replace('/', '-')}T{p[1]}+00:00"
                    )
                    xyz = np.array([float(p[2]), float(p[3]), float(p[4])])
                else:
                    continue
            except ValueError:
                continue
            enu = ecef_to_enu(xyz, ref)
            rows.append({"time": t, "xyz": xyz, "enu": enu})
    return rows


def _sat_geom(
    store, sat: str, t_rx: datetime, rec: np.ndarray, p_guess: float
) -> Optional[Tuple[np.ndarray, float, float, float, float]]:
    travel = p_guess / CLIGHT
    sat_xyz = sat_clk = None
    for _ in range(3):
        t_tx = datetime.fromtimestamp(t_rx.timestamp() - travel, tz=t_rx.tzinfo)
        sat_xyz = store.interpolate_pos(sat, t_tx)
        sat_clk = store.interpolate_clk(sat, t_tx)
        if sat_xyz is None or sat_clk is None:
            return None
        sat_xyz_rot = earth_rotation_corr(sat_xyz, travel)
        travel = np.linalg.norm(sat_xyz_rot - rec) / CLIGHT
    if sat_xyz is None or sat_clk is None:
        return None
    sat_xyz = earth_rotation_corr(sat_xyz, travel)
    rho = float(np.linalg.norm(sat_xyz - rec))
    # 对照 geodist：同一发射时刻 ECEF 上的 Sagnac 残差量级
    rho_sag, _ = geometric_range(
        store.interpolate_pos(sat, datetime.fromtimestamp(t_rx.timestamp() - travel, tz=t_rx.tzinfo)),
        rec,
    )
    el, az = elevation_azimuth(rec, sat_xyz)
    return sat_xyz, float(sat_clk), rho, el, az


def main() -> None:
    cfg = load_config()
    year, doy = cfg["experiment"]["year"], cfg["experiment"]["doy"]
    yyyy, ddd, _ = yyyy_ddd(year, doy)
    week, _ = gps_week_dow(year, doy)
    rnx_dir = Path(cfg["paths"]["data_root"]) / "rinex" / f"{yyyy}{ddd}"
    prod_dir = Path(cfg["paths"]["data_root"]) / "products" / f"{week}"
    obs_path = next(rnx_dir.glob(f"{cfg['stations']['spp_station']}*.rnx"))
    sp3 = next(prod_dir.glob("*.SP3"))
    clk = next(prod_dir.glob("*.CLK"))
    bia_files = sorted(prod_dir.glob("*OSB.BIA")) or sorted(prod_dir.glob("*.BIA"))
    bia = bia_files[0] if bia_files else None

    obs = read_rinex_obs(obs_path, systems=("G", "C"))
    store = read_clk(clk, read_sp3(sp3))
    osb = read_bias_sinex(bia) if bia else None
    ref = obs.approx_xyz.copy()
    zhd0, zwd0 = saastamoinen_zhd_zwd(ref, rel_humi=0.7)
    ztd0 = zhd0 + zwd0

    sample = max(1, int(cfg["ppp"].get("sample_every_n", 10)))
    elev_mask = np.deg2rad(cfg["ppp"]["elev_mask_deg"])
    epochs = obs.epochs[::sample]

    # --- 1) 固定 approx：每历元估 dt+ZTD，收码残差；可选每星相位常数模糊度 ---
    out_rows = []
    by_sat_osb: Dict[str, List[float]] = defaultdict(list)
    by_sat_no: Dict[str, List[float]] = defaultdict(list)
    el_bins = defaultdict(list)  # (bin, kind) -> residuals
    az_bins = defaultdict(list)

    # code-only LS 位置（无相位）— 有/无 OSB
    code_enu_osb, code_enu_no = [], []

    for ep in epochs:
        doy_f = _doy(ep.time)
        meas_osb, meas_no = {}, {}
        for sat, vals in ep.data.items():
            if sat[0] != "G":
                continue
            po = _pick_dual_code_phase(sat, vals, osb)
            pn = _pick_dual_code_phase(sat, vals, None)
            if po is None:
                continue
            p_if, l_if, _sys = po
            if po:
                meas_osb[sat] = (po[0], po[1])
            if pn:
                meas_no[sat] = (pn[0], pn[1])
        if len(meas_osb) < 5:
            continue

        # 固定位置，估 dt, ztd（码 only）
        def fit_clock_ztd(meas):
            xx = np.array([0.0, ztd0])  # dt(s), ztd
            last = None
            for _ in range(5):
                rows, zs = [], []
                for sat, (p_if, _) in meas.items():
                    g = _sat_geom(store, sat, ep.time, ref, p_if)
                    if g is None:
                        continue
                    _, sat_clk, rho, el, az = g
                    if el < elev_mask:
                        continue
                    trop, dmw = trop_delay_est_ztd(ref, el, xx[1], doy=doy_f)
                    pred = rho + CLIGHT * (xx[0] - sat_clk) + trop
                    w = np.sin(max(el, np.deg2rad(5))) ** 2
                    sig = 0.9 / np.sqrt(w)
                    H = np.array([CLIGHT, dmw]) / sig
                    rows.append(H)
                    zs.append((p_if - pred) / sig)
                if len(rows) < 4:
                    return None, None, []
                A = np.vstack(rows)
                b = np.asarray(zs)
                try:
                    dx = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return None, None, []
                xx = xx + dx
                last = (A, b, meas)
                if np.linalg.norm(dx) < 1e-6:
                    break
            # 再算未加权残差
            res_list = []
            for sat, (p_if, l_if) in meas.items():
                g = _sat_geom(store, sat, ep.time, ref, p_if)
                if g is None:
                    continue
                _, sat_clk, rho, el, az = g
                if el < elev_mask:
                    continue
                trop, _ = trop_delay_est_ztd(ref, el, xx[1], doy=doy_f)
                pred_p = rho + CLIGHT * (xx[0] - sat_clk) + trop
                rp = p_if - pred_p
                # 相位：用 L-P 作模糊度，残差 ≈ (l-p) 常数后的相位几何残差
                amb0 = l_if - p_if
                rl = (l_if - (pred_p + amb0))
                res_list.append(
                    {
                        "sat": sat,
                        "el_deg": np.rad2deg(el),
                        "az_deg": np.rad2deg(az) % 360,
                        "code_m": rp,
                        "phase_m": rl,
                        "p_if": p_if,
                        "l_if": l_if,
                    }
                )
            return xx, float(np.sqrt(np.mean([r["code_m"] ** 2 for r in res_list]))), res_list

        xx_o, rms_o, res_o = fit_clock_ztd(meas_osb)
        xx_n, rms_n, res_n = fit_clock_ztd(meas_no)
        if res_o:
            for r in res_o:
                by_sat_osb[r["sat"]].append(r["code_m"])
                el_bins[int(r["el_deg"] // 10) * 10].append(r["code_m"])
                az_bins[int(r["az_deg"] // 45) * 45].append(r["code_m"])
                out_rows.append(
                    {
                        "time": ep.time.isoformat(),
                        "sat": r["sat"],
                        "el_deg": r["el_deg"],
                        "az_deg": r["az_deg"],
                        "code_osb_m": r["code_m"],
                        "phase_osb_m": r["phase_m"],
                    }
                )
        if res_n:
            for r in res_n:
                by_sat_no[r["sat"]].append(r["code_m"])

        # code-only 自由位置 LS
        def code_only_pos(meas, use_osb_tag):
            xyz = ref.copy()
            dt = 0.0
            ztd = ztd0
            for _ in range(6):
                rows, zs = [], []
                for sat, (p_if, _) in meas.items():
                    g = _sat_geom(store, sat, ep.time, xyz, p_if)
                    if g is None:
                        continue
                    sat_xyz, sat_clk, rho, el, az = g
                    if el < elev_mask:
                        continue
                    trop, dmw = trop_delay_est_ztd(xyz, el, ztd, doy=doy_f)
                    u = (xyz - sat_xyz) / rho
                    pred = rho + CLIGHT * (dt - sat_clk) + trop
                    w = np.sin(max(el, np.deg2rad(5))) ** 2
                    sig = 0.9 / np.sqrt(w)
                    H = np.zeros(5)
                    H[:3] = u
                    H[3] = CLIGHT
                    H[4] = dmw
                    rows.append(H / sig)
                    zs.append((p_if - pred) / sig)
                if len(rows) < 5:
                    return None
                A = np.vstack(rows)
                b = np.asarray(zs)
                # 轻阻尼到 approx，避免单历元飞掉，但仍允许数米移动
                damp = np.array([1 / 25.0, 1 / 25.0, 1 / 25.0, 1e-12, 1 / 0.05**2])
                try:
                    dx = np.linalg.solve(A.T @ A + np.diag(damp), A.T @ b)
                except np.linalg.LinAlgError:
                    return None
                xyz = xyz + dx[:3]
                dt += dx[3]
                ztd += dx[4]
                if np.linalg.norm(dx[:3]) < 1e-3:
                    break
            return ecef_to_enu(xyz, ref)

        eo = code_only_pos(meas_osb, True)
        en = code_only_pos(meas_no, False)
        if eo is not None:
            code_enu_osb.append(eo)
        if en is not None:
            code_enu_no.append(en)

    def enu_stats(arr):
        a = np.asarray(arr)
        if len(a) == 0:
            return {}
        return {
            "n": int(len(a)),
            "mean_enu_m": a.mean(axis=0).tolist(),
            "rms_enu_m": np.sqrt((a**2).mean(axis=0)).tolist(),
            "rms_3d_m": float(np.sqrt((a**2).sum(axis=1).mean())),
        }

    sat_mean_osb = {
        s: {"mean_code_m": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for s, v in sorted(by_sat_osb.items())
    }
    sat_mean_no = {
        s: {"mean_code_m": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for s, v in sorted(by_sat_no.items())
    }

    # elevation correlation
    el_corr = {}
    if out_rows:
        els = np.array([r["el_deg"] for r in out_rows])
        cds = np.array([r["code_osb_m"] for r in out_rows])
        if len(els) > 10 and np.std(els) > 1:
            el_corr["corr_el_code"] = float(np.corrcoef(els, cds)[0, 1])
            azs = np.array([r["az_deg"] for r in out_rows])
            # 用 cos/sin az 看南北投影
            el_corr["corr_sin_az_code"] = float(np.corrcoef(np.sin(np.deg2rad(azs)), cds)[0, 1])
            el_corr["corr_cos_az_code"] = float(np.corrcoef(np.cos(np.deg2rad(azs)), cds)[0, 1])
            el_corr["code_mean_m"] = float(np.mean(cds))
            el_corr["code_rms_m"] = float(np.sqrt(np.mean(cds**2)))
            el_corr["phase_mean_m"] = float(np.mean([r["phase_osb_m"] for r in out_rows]))
            el_corr["phase_rms_m"] = float(
                np.sqrt(np.mean([r["phase_osb_m"] ** 2 for r in out_rows]))
            )

    el_bin_mean = {
        str(k): float(np.mean(v)) for k, v in sorted(el_bins.items()) if v
    }
    az_bin_mean = {
        str(k): float(np.mean(v)) for k, v in sorted(az_bins.items()) if v
    }

    # vs RTKLIB
    rtk_path = Path(cfg["paths"]["results_root"]) / "rtklib_ref" / "ppp.pos"
    self_csv = Path(cfg["paths"]["results_root"]) / "ppp" / "ppp_self.csv"
    diff_summary = {}
    if rtk_path.exists() and self_csv.exists():
        rtk = _parse_rtklib_pos(rtk_path, ref)
        self_rows = []
        with open(self_csv, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.strip().split(",")
                t = datetime.fromisoformat(p[0])
                xyz = np.array([float(p[1]), float(p[2]), float(p[3])])
                self_rows.append((t, xyz, ecef_to_enu(xyz, ref)))
        # 最近邻 30s 匹配
        diffs = []
        for tr, xyzr, enur in [(r["time"], r["xyz"], r["enu"]) for r in rtk]:
            best = None
            for ts, xyzs, enus in self_rows:
                dt = abs((ts - tr).total_seconds())
                if best is None or dt < best[0]:
                    best = (dt, enus - enur, xyzs - xyzr)
            if best and best[0] <= 150:
                diffs.append(best[1])
        if diffs:
            d = np.asarray(diffs)
            diff_summary = {
                "n_matched": int(len(d)),
                "self_minus_rtklib_mean_enu_m": d.mean(axis=0).tolist(),
                "self_minus_rtklib_rms_enu_m": np.sqrt((d**2).mean(axis=0)).tolist(),
                "self_minus_rtklib_rms_3d_m": float(np.sqrt((d**2).sum(axis=1).mean())),
            }

    report = {
        "ref_approx_xyz": ref.tolist(),
        "ztd_apriori_m": ztd0,
        "n_epochs_sampled": len(epochs),
        "n_residual_rows": len(out_rows),
        "fixed_approx_code_stats": el_corr,
        "code_mean_by_elev_bin_deg": el_bin_mean,
        "code_mean_by_az_bin_deg": az_bin_mean,
        "sat_code_mean_with_osb": sat_mean_osb,
        "sat_code_mean_without_osb": sat_mean_no,
        "osb_effect_on_sat_means": {
            s: {
                "with": sat_mean_osb.get(s, {}).get("mean_code_m"),
                "without": sat_mean_no.get(s, {}).get("mean_code_m"),
                "delta": (
                    sat_mean_osb[s]["mean_code_m"] - sat_mean_no[s]["mean_code_m"]
                    if s in sat_mean_osb and s in sat_mean_no
                    else None
                ),
            }
            for s in sorted(set(sat_mean_osb) | set(sat_mean_no))
        },
        "code_only_pos_with_osb": enu_stats(code_enu_osb),
        "code_only_pos_without_osb": enu_stats(code_enu_no),
        "self_vs_rtklib_pos": diff_summary,
        "interpretation_hints": [
            "若 code_only_pos mean N≈-4m：偏差在码几何/改正，不是模糊度锁死独有。",
            "若 with/without OSB 位置差很小：OSB 不是主因。",
            "若残差与 az/el 强相关：缺 PCO/ARP/映射或南天几何投影。",
            "self-rtklib ENU 差指向自研相对对照的系统差。",
        ],
    }

    out_dir = Path(cfg["paths"]["results_root"]) / "ppp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ppp_residual_diag.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # 采样残差 CSV（限量）
    csv_path = out_dir / "ppp_residual_diag.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("time,sat,el_deg,az_deg,code_osb_m,phase_osb_m\n")
        for r in out_rows[:: max(1, len(out_rows) // 5000)]:
            f.write(
                f"{r['time']},{r['sat']},{r['el_deg']:.2f},{r['az_deg']:.2f},"
                f"{r['code_osb_m']:.4f},{r['phase_osb_m']:.4f}\n"
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
