"""Stage C：静态 IF-PPP（默认 GPS-only）。

精密钟差要加相对论（缺过会偏数米）。ZTD / OSB / ARP / ATX / 缠绕 / 固体潮都加了。
估计用阻尼 LS；朴素 EKF 这批数据更差，已回退。BDS 试混过约 4 m，默认关。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ladder.ephemeris.precise import read_clk, read_sp3
from ladder.models.antenna import (
    ant_range_corr,
    hen_to_enu,
    if_windup_meters,
    sat_ant_offset_ecef,
    windup_cycles,
)
from ladder.models.atmosphere import (
    elevation_azimuth,
    geometric_range,
    saastamoinen_zhd_zwd,
    trop_delay_est_ztd,
)
from ladder.models.tides import solid_earth_tide
from ladder.products.antex import AntexStore, read_antex
from ladder.products.bias import OsbStore, read_bias_sinex
from ladder.rinex.obs import read_rinex_obs
from ladder.spp.engine import iono_free
from ladder.util.constants import CLIGHT, FREQ
from ladder.util.coord import ecef_to_enu


@dataclass
class PppEpochResult:
    time: datetime
    xyz: np.ndarray
    ztd: float
    nsat: int
    residual_rms: float
    success: bool


def _doy(t: datetime) -> float:
    return float(t.timetuple().tm_yday) + (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0


def _pick_dual_code_phase(
    sat: str, vals: dict, osb: Optional[OsbStore]
) -> Optional[Tuple[float, float, str]]:
    """返回 (P_IF, L_IF_m, sys)；BDS 排除 GEO C01–C05。"""
    if sat[0] == "G":
        c1 = next((k for k in ("C1W", "C1C", "C1P", "C1X") if k in vals), None)
        c2 = next((k for k in ("C2W", "C2P", "C2X", "C2S", "C2L") if k in vals), None)
        if c1 and str(c1).startswith("C1W"):
            l1 = next((k for k in ("L1W", "L1C", "L1X") if k in vals), None)
        else:
            l1 = next((k for k in ("L1C", "L1W", "L1X") if k in vals), None)
        l2 = next((k for k in ("L2W", "L2X", "L2P") if k in vals), None)
        if not all([c1, c2, l1, l2]):
            return None
        f1, f2 = FREQ[("G", 1)], FREQ[("G", 2)]
        p1, p2 = float(vals[c1]), float(vals[c2])
        if osb is not None:
            p1 = osb.correct(sat, c1, p1)
            p2 = osb.correct(sat, c2, p2)
        p_if = iono_free(p1, p2, f1, f2)
        lam1, lam2 = CLIGHT / f1, CLIGHT / f2
        l_if = (f1 * f1 * vals[l1] * lam1 - f2 * f2 * vals[l2] * lam2) / (f1 * f1 - f2 * f2)
        return p_if, l_if, "G"
    if sat[0] == "C":
        if int(sat[1:]) <= 5:
            return None
        c1 = next((k for k in ("C2I", "C1X", "C1P") if k in vals), None)
        c2 = next((k for k in ("C6I", "C7I") if k in vals), None)
        l1 = next((k for k in ("L2I", "L1X", "L1P") if k in vals), None)
        if c2 == "C6I":
            l2 = next((k for k in ("L6I",) if k in vals), None)
            f2 = FREQ[("C", 6)]
        elif c2 == "C7I":
            l2 = next((k for k in ("L7I",) if k in vals), None)
            f2 = FREQ[("C", 7)]
        else:
            return None
        if not all([c1, c2, l1, l2]):
            return None
        f1 = FREQ[("C", 2)]
        p1, p2 = float(vals[c1]), float(vals[c2])
        if osb is not None:
            p1 = osb.correct(sat, c1, p1)
            p2 = osb.correct(sat, c2, p2)
        p_if = iono_free(p1, p2, f1, f2)
        lam1, lam2 = CLIGHT / f1, CLIGHT / f2
        l_if = (f1 * f1 * vals[l1] * lam1 - f2 * f2 * vals[l2] * lam2) / (f1 * f1 - f2 * f2)
        return p_if, l_if, "C"
    return None


def run_ppp(
    obs_path: str | Path,
    sp3_path: str | Path,
    clk_path: str | Path,
    bia_path: str | Path | None = None,
    atx_path: str | Path | None = None,
    elev_mask_deg: float = 10.0,
    max_epochs: int = 0,
    sample_every_n: int = 1,
    apply_arp: bool = True,
    apply_windup: bool = True,
    apply_sat_pco: bool = True,
    apply_rcv_pco: bool = True,
    apply_tide: bool = True,
    use_bds: bool = False,
) -> List[PppEpochResult]:
    obs = read_rinex_obs(obs_path, systems=("G", "C"))
    store = read_sp3(sp3_path)
    store = read_clk(clk_path, store)
    osb = read_bias_sinex(bia_path) if bia_path else None

    del_enu = hen_to_enu(*obs.ant_delta_hen) if apply_arp else np.zeros(3)
    antex: Optional[AntexStore] = None
    if atx_path and Path(atx_path).is_file():
        antex = read_antex(
            atx_path,
            want_rcv_types={obs.ant_type} if obs.ant_type else None,
        )
        if apply_rcv_pco and obs.ant_type:
            pco = antex.rcv_pco_enu(obs.ant_type)
            if pco is not None:
                del_enu = del_enu + np.asarray(pco, dtype=float)

    elev_mask = np.deg2rad(elev_mask_deg)
    xyz = obs.approx_xyz.copy()
    dt = 0.0
    isb_c = 0.0
    zhd0, zwd0 = saastamoinen_zhd_zwd(xyz, rel_humi=0.7)
    ztd = zhd0 + zwd0
    ztd_apriori = ztd
    amb: Dict[str, float] = {}
    prev_lif: Dict[str, float] = {}
    phw: Dict[str, float] = {}
    results: List[PppEpochResult] = []

    epochs = obs.epochs[:: max(1, sample_every_n)]
    if max_epochs:
        epochs = epochs[:max_epochs]

    for ep in epochs:
        doy = _doy(ep.time)
        sats, meas, sys_of = [], {}, {}
        for sat, vals in ep.data.items():
            if sat[0] == "C" and not use_bds:
                continue
            if sat[0] not in ("G", "C"):
                continue
            picked = _pick_dual_code_phase(sat, vals, osb)
            if picked is None:
                continue
            p_if, l_if, sys = picked
            if sat in prev_lif and abs(l_if - prev_lif[sat]) > 5.0:
                amb.pop(sat, None)
                phw.pop(sat, None)
            prev_lif[sat] = l_if
            sats.append(sat)
            meas[sat] = (p_if, l_if)
            sys_of[sat] = sys
        if len(sats) < 5:
            continue
        has_bds = any(sys_of[s] == "C" for s in sats)

        for s in sats:
            if s not in amb:
                amb[s] = float(meas[s][1] - meas[s][0])

        n_base = 6 if has_bds else 5
        sat_index = {s: i for i, s in enumerate(sats)}
        n = n_base + len(sats)
        xx = np.zeros(n)
        xx[:3] = xyz
        xx[3] = dt
        xx[4] = ztd
        if has_bds:
            xx[5] = isb_c
        for s, i in sat_index.items():
            xx[n_base + i] = amb[s]

        last_b = None
        t0 = results[0].time if results else ep.time
        age_s = (ep.time - t0).total_seconds()
        tide = solid_earth_tide(xyz, ep.time) if apply_tide else np.zeros(3)

        for _iter in range(6):
            rows, zs = [], []
            rr = xx[:3] + tide
            for sat in sats:
                p_if, l_if = meas[sat]
                travel = p_if / CLIGHT
                sat_xyz = sat_clk = None
                for _k in range(3):
                    t_tx = datetime.fromtimestamp(ep.time.timestamp() - travel, tz=ep.time.tzinfo)
                    peph = store.peph2pos(sat, t_tx, apply_relativity=True)
                    if peph is None:
                        sat_xyz = sat_clk = None
                        break
                    sat_xyz, sat_clk = peph
                    if apply_sat_pco and antex is not None:
                        pco = antex.sat_pco_l1(sat, t_tx)
                        if pco is not None:
                            n_, e_, u_ = pco
                            sat_xyz = sat_xyz + sat_ant_offset_ecef(
                                sat_xyz, t_tx, np.array([n_, e_, u_], dtype=float)
                            )
                    rho_g, _e = geometric_range(sat_xyz, rr)
                    if rho_g <= 0.0:
                        sat_xyz = sat_clk = None
                        break
                    travel = rho_g / CLIGHT
                if sat_xyz is None or sat_clk is None:
                    continue
                rho, e_los = geometric_range(sat_xyz, rr)
                if rho <= 0.0:
                    continue
                el, _ = elevation_azimuth(rr, sat_xyz)
                if el < elev_mask:
                    continue
                trop, dtrp_dztd = trop_delay_est_ztd(rr, el, xx[4], doy=doy)
                dant = ant_range_corr(del_enu, e_los, rr)
                isb = float(xx[5]) if (has_bds and sys_of[sat] == "C") else 0.0
                pred_p = rho + CLIGHT * (xx[3] - sat_clk) + isb + trop + dant
                pred_l = pred_p + xx[n_base + sat_index[sat]]
                if apply_windup:
                    phw[sat] = windup_cycles(ep.time, sat_xyz, rr, phw.get(sat, 0.0))
                    pred_l = pred_l + if_windup_meters(phw[sat], sys_of[sat])
                el_c = max(el, np.deg2rad(5.0))
                w_el = np.sin(el_c) ** 2
                sig_p = 0.9 / np.sqrt(w_el)
                sig_l = 0.006 / np.sqrt(w_el)
                Hp = np.zeros(n)
                Hp[:3] = -e_los
                Hp[3] = CLIGHT
                Hp[4] = dtrp_dztd
                if has_bds and sys_of[sat] == "C":
                    Hp[5] = 1.0
                Hl = Hp.copy()
                Hl[n_base + sat_index[sat]] = 1.0
                rows.append(Hp / sig_p)
                zs.append((p_if - pred_p) / sig_p)
                rows.append(Hl / sig_l)
                zs.append((l_if - pred_l) / sig_l)

            n_meas = len(rows)
            Hz = np.zeros(n)
            Hz[4] = 1.0
            sig_ztd = 0.25 if age_s < 3600 else 0.15
            rows.append(Hz / sig_ztd)
            zs.append((ztd_apriori - xx[4]) / sig_ztd)
            if n_meas < 10:
                break
            A = np.vstack(rows)
            b = np.asarray(zs)
            damp = np.ones(n) * 1e-12
            if age_s < 1800:
                damp[:3] = 1.0 / (2.0**2)
            else:
                damp[:3] = 1.0 / (0.02**2)
            if not results:
                damp[:3] = 1e-8
            damp[3] = 1e-12
            damp[4] = 1.0 / (0.05**2)
            if has_bds:
                damp[5] = 1.0 / (10.0**2)
            damp[n_base:] = 1.0 / (30.0**2)
            try:
                dx = np.linalg.solve(A.T @ A + np.diag(damp), A.T @ b)
            except np.linalg.LinAlgError:
                break
            xx = xx + dx
            last_b = b[:n_meas]
            if np.linalg.norm(dx[:3]) < 5e-4 and abs(dx[4]) < 5e-4:
                break

        if last_b is None:
            continue
        if not (1.5 < xx[4] < 3.0):
            xx[4] = float(np.clip(0.7 * xx[4] + 0.3 * ztd_apriori, 1.8, 2.8))
        xyz = xx[:3].copy()
        dt = float(xx[3])
        ztd = float(xx[4])
        if has_bds:
            isb_c = float(xx[5])
        ztd_apriori = 0.99 * ztd_apriori + 0.01 * ztd
        for s, i in sat_index.items():
            amb[s] = float(xx[n_base + i])
        results.append(
            PppEpochResult(
                time=ep.time,
                xyz=xyz.copy(),
                ztd=ztd,
                nsat=len(sats),
                residual_rms=float(np.sqrt(np.mean(last_b**2))),
                success=True,
            )
        )
    return results


def summarize_ppp(results: List[PppEpochResult], ref_xyz: np.ndarray, skip_minutes: float = 60.0) -> dict:
    if not results:
        return {"n": 0}
    t0 = results[0].time.timestamp()
    use = [r for r in results if (r.time.timestamp() - t0) >= skip_minutes * 60.0] or results
    enus = np.array([ecef_to_enu(r.xyz, ref_xyz) for r in use])
    return {
        "n": len(use),
        "n_all": len(results),
        "mean_enu_m": enus.mean(axis=0).tolist(),
        "rms_enu_m": np.sqrt((enus**2).mean(axis=0)).tolist(),
        "rms_3d_m": float(np.sqrt((enus**2).sum(axis=1).mean())),
        "horizontal_rms_m": float(np.sqrt(np.mean(enus[:, 0] ** 2 + enus[:, 1] ** 2))),
        "mean_ztd_m": float(np.mean([r.ztd for r in use])),
        "skip_minutes": skip_minutes,
        "mode": "IF PPP LS (GPS, tide+ARP/ATX/windup, float amb)",
    }


def write_ppp_csv(results: List[PppEpochResult], path: str | Path, ref_xyz: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,x,y,z,e,n,u,ztd,nsat,rms\n")
        for r in results:
            e, n, u = ecef_to_enu(r.xyz, ref_xyz)
            f.write(
                f"{r.time.isoformat()},{r.xyz[0]:.4f},{r.xyz[1]:.4f},{r.xyz[2]:.4f},"
                f"{e:.4f},{n:.4f},{u:.4f},{r.ztd:.4f},{r.nsat},{r.residual_rms:.4f}\n"
            )
