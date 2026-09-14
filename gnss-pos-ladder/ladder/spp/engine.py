"""Stage A：GPS+BDS 伪距单点。

默认单频 + Klobuchar + Saastamoinen + TGD（和对照 conf 的 brdc 电离层一路）。
`use_if_combination` 可切双频 IF，这天数据通常更噪。历元内剔最大粗差星。
GEO C01–C05 不进解（算星会发散）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ladder.ephemeris.broadcast import sat_pos_at
from ladder.models.atmosphere import (
    earth_rotation_corr,
    elevation_azimuth,
    klobuchar_iono,
    saastamoinen_tropo,
)
from ladder.rinex.nav import NavStore, read_rinex_nav
from ladder.rinex.obs import ObsEpoch, read_rinex_obs
from ladder.util.constants import CLIGHT, FREQ, datetime_to_gtime
from ladder.util.coord import ecef_to_enu


@dataclass
class SppEpochResult:
    time: datetime
    xyz: np.ndarray
    dt_gps: float
    dt_bds: float
    nsat: int
    residual_rms: float
    success: bool


def _pick_dual_pseudorange(sat: str, vals: Dict[str, float]) -> Optional[Tuple[float, float, float, float]]:
    """返回 (P1, P2, f1, f2) 或 None。"""
    sys = sat[0]
    if sys == "G":
        p1 = next((vals[k] for k in ("C1C", "C1P", "C1W", "C1X") if k in vals), None)
        p2 = next((vals[k] for k in ("C2W", "C2P", "C2X", "C2S", "C2L") if k in vals), None)
        if p1 is None or p2 is None:
            return None
        return p1, p2, FREQ[("G", 1)], FREQ[("G", 2)]
    if sys == "C":
        p1_code = next((k for k in ("C2I", "C1I", "C1X", "C1P") if k in vals), None)
        p2_code = next((k for k in ("C6I", "C7I", "C5P", "C5X") if k in vals), None)
        if p1_code is None or p2_code is None:
            return None
        f1 = FREQ[("C", 2)]
        if p2_code.startswith("C6"):
            f2 = FREQ[("C", 6)]
        else:
            f2 = FREQ[("C", 7)]
        return vals[p1_code], vals[p2_code], f1, f2
    return None


def _pick_l1_pseudorange(sat: str, vals: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """返回 (P1, f1)。"""
    if sat[0] == "G":
        p1 = next((vals[k] for k in ("C1C", "C1P", "C1W", "C1X") if k in vals), None)
        if p1 is None:
            return None
        return p1, FREQ[("G", 1)]
    if sat[0] == "C":
        p1 = next((vals[k] for k in ("C2I", "C1I", "C1X", "C1P") if k in vals), None)
        if p1 is None:
            return None
        return p1, FREQ[("C", 2)]
    return None


def iono_free(p1: float, p2: float, f1: float, f2: float) -> float:
    return (f1 * f1 * p1 - f2 * f2 * p2) / (f1 * f1 - f2 * f2)


def _obs_variance(el: float, use_if: bool) -> float:
    """高度角方差权：a^2*(b^2 + c^2/sin^2(el))；IF 再 ×9。"""
    el = max(el, np.deg2rad(5.0))
    varr = 0.3**2 * (0.3**2 + 0.3**2 / np.sin(el) ** 2)
    if use_if:
        varr *= 9.0
    return float(varr)


def _build_obs(
    epoch: ObsEpoch,
    nav: NavStore,
    x: np.ndarray,
    elev_mask: float,
    use_if: bool,
    exclude: Set[str],
) -> Tuple[List[np.ndarray], List[float], List[str], List[float]]:
    """组装加权设计阵行；状态 [x,y,z, dt_gps, dt_bds]。"""
    rows: List[np.ndarray] = []
    zs: List[float] = []
    used: List[str] = []
    raw_res: List[float] = []
    sow = datetime_to_gtime(epoch.time).sow

    for sat, vals in epoch.data.items():
        if sat in exclude:
            continue
        if sat[0] == "C" and int(sat[1:]) <= 5:
            continue

        eph_tgd = 0.0
        if use_if:
            dual = _pick_dual_pseudorange(sat, vals)
            if dual is None:
                continue
            p1, p2, f1, f2 = dual
            pr = iono_free(p1, p2, f1, f2)
            ion = 0.0
        else:
            single = _pick_l1_pseudorange(sat, vals)
            if single is None:
                continue
            pr, f1 = single
            # 广播钟差参考 IF；单频需扣 TGD（RTKLIB prange 单频分支）
            got0 = sat_pos_at(nav, sat, epoch.time)
            if got0 is not None:
                eph_tgd = got0[3].tgd
            pr = pr - CLIGHT * eph_tgd

        travel = pr / CLIGHT
        sat_xyz = None
        sat_clk = 0.0
        for _ in range(3):
            t_tx = datetime.fromtimestamp(epoch.time.timestamp() - travel, tz=epoch.time.tzinfo)
            got = sat_pos_at(nav, sat, t_tx)
            if got is None:
                sat_xyz = None
                break
            sat_xyz, sat_clk, _vel, eph = got
            eph_tgd = eph.tgd
            sat_xyz = earth_rotation_corr(sat_xyz, travel)
            rho = np.linalg.norm(sat_xyz - x[:3])
            travel = rho / CLIGHT
        if sat_xyz is None:
            continue

        el, az = elevation_azimuth(x[:3], sat_xyz)
        if el < elev_mask:
            continue

        trop = saastamoinen_tropo(x[:3], el)
        if not use_if:
            ion_par = nav.ion_params(sat)
            ion = klobuchar_iono(x[:3], el, az, sow, ion_par)
            # 若实际用了非 L1 频率（BDS B1I≈1561），按 f^2 缩放
            f_l1 = FREQ[("G", 1)]
            if abs(f1 - f_l1) > 1e6:
                ion *= (f_l1 / f1) ** 2
        else:
            ion = 0.0

        rho = np.linalg.norm(sat_xyz - x[:3])
        dt_r = x[3] if sat[0] == "G" else x[4]
        pred = rho + CLIGHT * (dt_r - sat_clk) + ion + trop
        residual = pr - pred
        u = (x[:3] - sat_xyz) / rho
        H = np.zeros(5)
        H[0:3] = u
        if sat[0] == "G":
            H[3] = CLIGHT
        else:
            H[4] = CLIGHT
        sig = np.sqrt(_obs_variance(el, use_if))
        rows.append(H / sig)
        zs.append(residual / sig)
        used.append(sat)
        raw_res.append(residual)

    return rows, zs, used, raw_res


def _solve_epoch(
    epoch: ObsEpoch,
    nav: NavStore,
    x0: np.ndarray,
    elev_mask_deg: float,
    use_if: bool = False,
) -> SppEpochResult:
    """状态： [x,y,z, dt_gps, dt_bds]"""
    x = x0.copy()
    elev_mask = np.deg2rad(elev_mask_deg)
    exclude: Set[str] = set()
    success = False
    rms = 1e9
    n_used = 0

    # 最多剔除 3 颗粗差星（简化 RAIM）
    for _raim in range(4):
        x = x0.copy()
        last_used: List[str] = []
        last_raw: List[float] = []
        for _iter in range(10):
            rows, zs, used, raw_res = _build_obs(epoch, nav, x, elev_mask, use_if, exclude)
            n_used = len(rows)
            last_used, last_raw = used, raw_res
            if n_used < 5:
                break
            A = np.vstack(rows)
            b = np.asarray(zs)
            try:
                dx, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                break
            x = x + dx
            if np.linalg.norm(dx[:3]) < 1e-3 and abs(dx[3]) * CLIGHT < 1e-3 and abs(dx[4]) * CLIGHT < 1e-3:
                success = True
                break
        else:
            success = n_used >= 5

        if n_used < 5 or not last_raw:
            success = False
            break

        rms = float(np.sqrt(np.mean(np.asarray(last_raw) ** 2)))
        # 残差检验：最大 |v| 过大则剔除该星重算
        abs_res = np.abs(np.asarray(last_raw))
        worst = int(np.argmax(abs_res))
        thr = max(30.0, 4.0 * rms)
        if abs_res[worst] > thr and n_used > 6:
            exclude.add(last_used[worst])
            success = False
            continue
        break

    return SppEpochResult(
        time=epoch.time,
        xyz=x[:3].copy(),
        dt_gps=float(x[3]),
        dt_bds=float(x[4]),
        nsat=n_used,
        residual_rms=rms,
        success=success and n_used >= 5,
    )


def run_spp(
    obs_path: str | Path,
    nav_path: str | Path,
    elev_mask_deg: float = 15.0,
    max_epochs: int = 0,
    sample_every_n: int = 1,
    use_if_combination: bool = False,
) -> List[SppEpochResult]:
    obs = read_rinex_obs(obs_path, systems=("G", "C"), max_epochs=0)
    nav = read_rinex_nav(nav_path, systems=("G", "C"))
    x = np.zeros(5)
    x[:3] = obs.approx_xyz if np.linalg.norm(obs.approx_xyz) > 1 else np.array([-2279828.0, 5004705.0, 3215040.0])
    results: List[SppEpochResult] = []
    epochs = obs.epochs[:: max(1, sample_every_n)]
    if max_epochs:
        epochs = epochs[:max_epochs]
    for ep in epochs:
        r = _solve_epoch(ep, nav, x, elev_mask_deg, use_if=use_if_combination)
        if r.success:
            x[:3] = r.xyz
            x[3] = r.dt_gps
            x[4] = r.dt_bds
            results.append(r)
    return results


def summarize_spp(results: List[SppEpochResult], ref_xyz: np.ndarray) -> dict:
    if not results:
        return {"n": 0}
    enus = np.array([ecef_to_enu(r.xyz, ref_xyz) for r in results])
    return {
        "n": len(results),
        "mean_enu_m": enus.mean(axis=0).tolist(),
        "rms_enu_m": np.sqrt((enus**2).mean(axis=0)).tolist(),
        "rms_3d_m": float(np.sqrt((enus**2).sum(axis=1).mean())),
        "horizontal_rms_m": float(np.sqrt(np.mean(enus[:, 0] ** 2 + enus[:, 1] ** 2))),
        "mean_nsat": float(np.mean([r.nsat for r in results])),
        "mode": "L1+Klobuchar+TGD (RTKLIB-like)",
    }


def write_spp_csv(results: List[SppEpochResult], path: str | Path, ref_xyz: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,x,y,z,e,n,u,nsat,rms\n")
        for r in results:
            e, n, u = ecef_to_enu(r.xyz, ref_xyz)
            f.write(
                f"{r.time.isoformat()},{r.xyz[0]:.4f},{r.xyz[1]:.4f},{r.xyz[2]:.4f},"
                f"{e:.4f},{n:.4f},{u:.4f},{r.nsat},{r.residual_rms:.4f}\n"
            )
