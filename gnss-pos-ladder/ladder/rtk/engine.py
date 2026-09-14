"""Stage B：超短基线静态相对定位。

默认分会话取整固定（`_solve_session`），这批数据上更稳。
`use_ekf=True` 才走连续浮点 EKF + LAMBDA；易误固定，配置里默认关。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ladder.ephemeris.broadcast import sat_pos_at
from ladder.models.atmosphere import earth_rotation_corr, elevation_azimuth
from ladder.rinex.nav import read_rinex_nav
from ladder.rinex.obs import read_rinex_obs
from ladder.rtk.lambda_ils import lambda_fix
from ladder.util.constants import CLIGHT, FREQ
from ladder.util.coord import ecef_to_enu


@dataclass
class RtkEpochResult:
    time: datetime
    rover_xyz: np.ndarray
    nsat: int
    residual_rms: float
    success: bool
    ratio: float = 0.0
    fixed: bool = False
    mode: str = "float"


def _code_phase(sat: str, vals: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
    if sat[0] == "G":
        P = next((vals[k] for k in ("C1C", "C1X", "C1W") if k in vals), None)
        L = next((vals[k] for k in ("L1C", "L1X", "L1W") if k in vals), None)
        if P is None or L is None:
            return None
        return P, L, CLIGHT / FREQ[("G", 1)]
    if sat[0] == "C":
        if int(sat[1:]) <= 5:
            return None
        P = next((vals[k] for k in ("C2I", "C1X", "C1P") if k in vals), None)
        L = next((vals[k] for k in ("L2I", "L1X", "L1P") if k in vals), None)
        if P is None or L is None:
            return None
        return P, L, CLIGHT / FREQ[("C", 2)]
    return None


def _sat_xyz(nav, sat: str, t: datetime, pr: float):
    travel = pr / CLIGHT
    t_tx = datetime.fromtimestamp(t.timestamp() - travel, tz=t.tzinfo)
    got = sat_pos_at(nav, sat, t_tx)
    if got is None:
        return None
    return earth_rotation_corr(got[0], travel)


def _epoch_gps(ep_r, ep_b, nav, x_rov, elev_mask):
    out = []
    for sat in sorted(set(ep_r.data) & set(ep_b.data)):
        if sat[0] != "G":
            continue
        rr = _code_phase(sat, ep_r.data[sat])
        bb = _code_phase(sat, ep_b.data[sat])
        if rr is None or bb is None:
            continue
        pr, lr, lam = rr
        pb, lb, _ = bb
        xyz = _sat_xyz(nav, sat, ep_r.time, pr)
        if xyz is None:
            continue
        el, _ = elevation_azimuth(x_rov, xyz)
        if el < elev_mask:
            continue
        out.append(dict(sat=sat, el=el, xyz=xyz, pr=pr, pb=pb, lr=lr, lb=lb, lam=lam, time=ep_r.time))
    return out


def _dd_rho(c, k, x_rov, x_base):
    return (np.linalg.norm(c["xyz"] - x_rov) - np.linalg.norm(c["xyz"] - x_base)) - (
        np.linalg.norm(k["xyz"] - x_rov) - np.linalg.norm(k["xyz"] - x_base)
    )


def _los_dd(c, k, x_rov):
    """∂(DD ρ)/∂x_rov ≈ u_c − u_k，u = (x−sat)/|x−sat|。"""
    return (x_rov - c["xyz"]) / np.linalg.norm(c["xyz"] - x_rov) - (x_rov - k["xyz"]) / np.linalg.norm(
        k["xyz"] - x_rov
    )


def _code_xyz(gs, ref, x0, x_base):
    x = x0.copy()
    for _ in range(6):
        rows, zs = [], []
        for m in gs:
            if m["sat"] == ref["sat"]:
                continue
            dd_p = (m["pr"] - m["pb"]) - (ref["pr"] - ref["pb"])
            dd_rho = _dd_rho(m, ref, x, x_base)
            u = _los_dd(m, ref, x)
            w = np.sin(m["el"]) ** 2
            rows.append(np.sqrt(w) * u)
            zs.append(np.sqrt(w) * (dd_p - dd_rho))
        if len(rows) < 4:
            return x0
        dx, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(zs), rcond=None)
        x = x + dx
        if np.linalg.norm(dx) < 1e-4:
            break
    return x


def _solve_session(frames, x_base, x_prior):
    """旧路径：分时段 round(N) 保持。返回 (xyz, fixed, mode, namb, phase_rms, ...) 或 None。"""
    if len(frames) < 8:
        return None
    cnt = Counter()
    for gs in frames:
        for m in gs:
            cnt[m["sat"]] += 1
    ref_sat = cnt.most_common(1)[0][0]

    common = None
    usable = []
    for gs in frames:
        by = {m["sat"]: m for m in gs}
        if ref_sat not in by:
            continue
        usable.append(by)
        s = set(by) - {ref_sat}
        common = s if common is None else (common & s)
    if not usable or not common or len(common) < 4:
        return None
    amb_sats = sorted(common)
    ref0 = usable[0][ref_sat]

    x_code = _code_xyz(list(usable[0].values()), ref0, x_prior, x_base)
    Nfix = {}
    fracs = []
    for sat in amb_sats:
        m = usable[0][sat]
        dd_l = (m["lr"] - m["lb"]) - (ref0["lr"] - ref0["lb"])
        dd_rho = _dd_rho(m, ref0, x_code, x_base)
        Nf = (dd_l * m["lam"] - dd_rho) / m["lam"]
        fracs.append(abs(Nf - round(Nf)))
        Nfix[sat] = int(round(Nf))
    mean_frac = float(np.mean(fracs))

    x = x_code.copy()
    b = np.array([0.0])
    for _ in range(8):
        rows, zs = [], []
        for by in usable:
            ref = by[ref_sat]
            for sat in amb_sats:
                m = by[sat]
                dd_rho = _dd_rho(m, ref, x, x_base)
                u = _los_dd(m, ref, x)
                w = np.sin(m["el"]) ** 2
                dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * m["lam"]
                rows.append(np.sqrt(w) * u)
                zs.append(np.sqrt(w) * (dd_l_m - dd_rho - m["lam"] * Nfix[sat]))
        A, b = np.vstack(rows), np.asarray(zs)
        dx, *_ = np.linalg.lstsq(A, b, rcond=None)
        x = x + dx
        if np.linalg.norm(dx) < 1e-5:
            break
    phase_rms = float(np.sqrt(np.mean(b**2)))
    fixed = phase_rms < 0.25 and mean_frac < 0.35 and len(amb_sats) >= 4
    mode = f"{'fixed' if fixed else 'float'}|ref={ref_sat}|namb={len(amb_sats)}|rms={phase_rms:.3f}|frac={mean_frac:.3f}"
    return x, fixed, mode, len(amb_sats), phase_rms, usable, ref_sat, amb_sats, Nfix


def run_rtk_float(
    rover_obs: str | Path,
    base_obs: str | Path,
    nav_path: str | Path,
    elev_mask_deg: float = 15.0,
    max_epochs: int = 0,
    sample_every_n: int = 1,
    ratio_th: float = 2.0,
    session_epochs: int = 60,
    use_ekf: bool = False,
) -> List[RtkEpochResult]:
    """默认会话固定（稳）；`use_ekf=True` 启用连续 EKF+LAMBDA（仍实验中）。"""
    if use_ekf:
        return _run_rtk_ekf(
            rover_obs,
            base_obs,
            nav_path,
            elev_mask_deg=elev_mask_deg,
            max_epochs=max_epochs,
            sample_every_n=sample_every_n,
            ratio_th=ratio_th,
        )
    return _run_rtk_session(
        rover_obs,
        base_obs,
        nav_path,
        elev_mask_deg=elev_mask_deg,
        max_epochs=max_epochs,
        sample_every_n=sample_every_n,
        session_epochs=session_epochs,
    )


def _run_rtk_session(
    rover_obs,
    base_obs,
    nav_path,
    elev_mask_deg=15.0,
    max_epochs=0,
    sample_every_n=1,
    session_epochs=60,
) -> List[RtkEpochResult]:
    rover = read_rinex_obs(rover_obs, systems=("G", "C"))
    base = read_rinex_obs(base_obs, systems=("G", "C"))
    nav = read_rinex_nav(nav_path, systems=("G", "C"))
    base_map = {e.time: e for e in base.epochs}
    x_base = base.approx_xyz.copy()
    x_prior = rover.approx_xyz.copy()
    elev_mask = np.deg2rad(elev_mask_deg)

    epochs = rover.epochs[:: max(1, sample_every_n)]
    if max_epochs:
        epochs = epochs[:max_epochs]

    all_frames = []
    for ep_r in epochs:
        ep_b = base_map.get(ep_r.time)
        if ep_b is None:
            continue
        gs = _epoch_gps(ep_r, ep_b, nav, x_prior, elev_mask)
        if len(gs) >= 5:
            all_frames.append(gs)
    if not all_frames:
        return []

    results: List[RtkEpochResult] = []
    for i0 in range(0, len(all_frames), session_epochs):
        session = all_frames[i0 : i0 + session_epochs]
        sol = _solve_session(session, x_base, x_prior)
        if sol is None:
            continue
        x, fixed, mode, namb, phase_rms, usable, ref_sat, amb_sats, Nfix = sol
        if fixed:
            x_prior = x
        for by in usable:
            ref = by[ref_sat]
            zs = []
            nused = 0
            t = None
            for sat in amb_sats:
                if sat not in by:
                    continue
                m = by[sat]
                t = m["time"]
                dd_rho = _dd_rho(m, ref, x, x_base)
                dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * m["lam"]
                zs.append(dd_l_m - dd_rho - m["lam"] * Nfix[sat])
                nused += 1
            if t is None or nused < 4:
                continue
            results.append(
                RtkEpochResult(
                    time=t,
                    rover_xyz=x.copy(),
                    nsat=nused,
                    residual_rms=float(np.sqrt(np.mean(np.square(zs)))),
                    success=True,
                    ratio=float(namb),
                    fixed=fixed,
                    mode=mode,
                )
            )
    return results


def _run_rtk_ekf(
    rover_obs,
    base_obs,
    nav_path,
    elev_mask_deg=15.0,
    max_epochs=0,
    sample_every_n=1,
    ratio_th=2.0,
) -> List[RtkEpochResult]:
    """连续静态 EKF：状态 [xyz | N_dd(周)]，周期 LAMBDA。"""
    rover = read_rinex_obs(rover_obs, systems=("G", "C"))
    base = read_rinex_obs(base_obs, systems=("G", "C"))
    nav = read_rinex_nav(nav_path, systems=("G", "C"))
    base_map = {e.time: e for e in base.epochs}
    x_base = base.approx_xyz.copy()
    elev_mask = np.deg2rad(elev_mask_deg)

    epochs = rover.epochs[:: max(1, sample_every_n)]
    if max_epochs:
        epochs = epochs[:max_epochs]

    # 状态
    xyz = rover.approx_xyz.copy()
    amb_sats: List[str] = []
    x_amb = np.zeros(0)
    P = np.eye(3) * 25.0  # xyz 初方差
    ref_sat: Optional[str] = None
    fixed_hold: Dict[str, int] = {}
    hold_ok = False
    results: List[RtkEpochResult] = []
    epoch_i = 0
    sig_code, sig_phase = 0.8, 0.01  # m，再乘 elevation 权重
    q_pos = (1e-4) ** 2  # 静态过程噪声 (m^2 / epoch)
    q_amb = (1e-6) ** 2

    for ep_r in epochs:
        ep_b = base_map.get(ep_r.time)
        if ep_b is None:
            continue
        gs = _epoch_gps(ep_r, ep_b, nav, xyz, elev_mask)
        if len(gs) < 5:
            continue
        by = {m["sat"]: m for m in gs}
        # 参考星：最高高度角
        ref = max(gs, key=lambda m: m["el"])
        new_ref = ref["sat"]
        sats = sorted(s for s in by if s != new_ref)
        if len(sats) < 4:
            continue

        # 参考星或卫星集变化 → 重建模糊度状态
        if new_ref != ref_sat or set(sats) != set(amb_sats):
            old = {s: float(x_amb[i]) for i, s in enumerate(amb_sats)} if amb_sats else {}
            amb_sats = sats
            ref_sat = new_ref
            n = 3 + len(amb_sats)
            x_full = np.zeros(n)
            x_full[:3] = xyz
            P_new = np.eye(n) * 1e2
            P_new[:3, :3] = P[:3, :3] if P.shape[0] >= 3 else np.eye(3) * 25.0
            for i, s in enumerate(amb_sats):
                m = by[s]
                dd_l = (m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])
                dd_rho = _dd_rho(m, ref, xyz, x_base)
                if s in fixed_hold and hold_ok:
                    x_full[3 + i] = float(fixed_hold[s])
                    P_new[3 + i, 3 + i] = 1e-8
                elif s in old:
                    x_full[3 + i] = old[s]
                    P_new[3 + i, 3 + i] = 1.0
                else:
                    x_full[3 + i] = (dd_l * m["lam"] - dd_rho) / m["lam"]
                    P_new[3 + i, 3 + i] = 100.0
            x_amb = x_full[3:].copy()
            P = P_new
            xyz = x_full[:3].copy()
            hold_ok = hold_ok and all(s in fixed_hold for s in amb_sats)

        n = 3 + len(amb_sats)
        x = np.zeros(n)
        x[:3] = xyz
        x[3:] = x_amb

        # 预测（静态）
        Q = np.zeros((n, n))
        Q[:3, :3] = np.eye(3) * q_pos
        Q[3:, 3:] = np.eye(len(amb_sats)) * q_amb
        P = P + Q

        # 量测更新：码 + 相双差
        for sat in amb_sats:
            m = by[sat]
            i_amb = amb_sats.index(sat)
            lam = m["lam"]
            dd_rho = _dd_rho(m, ref, x[:3], x_base)
            u = _los_dd(m, ref, x[:3])
            w = max(np.sin(m["el"]) ** 2, 0.05)

            # 码
            dd_p = (m["pr"] - m["pb"]) - (ref["pr"] - ref["pb"])
            H = np.zeros(n)
            H[:3] = u
            R = (sig_code**2) / w
            innov = dd_p - dd_rho
            S = float(H @ P @ H + R)
            K = (P @ H) / S
            x = x + K * innov
            P = (np.eye(n) - np.outer(K, H)) @ P

            # 相
            dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * lam
            H = np.zeros(n)
            H[:3] = u
            H[3 + i_amb] = lam
            R = (sig_phase**2) / w
            innov = dd_l_m - (dd_rho + lam * x[3 + i_amb])
            S = float(H @ P @ H + R)
            if S < 1e-18:
                continue
            K = (P @ H) / S
            # 粗差门限
            if abs(innov) > 0.5 and not hold_ok:
                # 可能周跳：放大该模糊度方差并重初始化
                x[3 + i_amb] = (dd_l_m - dd_rho) / lam
                P[3 + i_amb, :] = 0.0
                P[:, 3 + i_amb] = 0.0
                P[3 + i_amb, 3 + i_amb] = 100.0
                hold_ok = False
                fixed_hold.pop(sat, None)
                continue
            x = x + K * innov
            P = (np.eye(n) - np.outer(K, H)) @ P

        xyz = x[:3].copy()
        x_amb = x[3:].copy()
        P = 0.5 * (P + P.T)

        # 周期尝试 LAMBDA（需浮点已收敛）
        ratio = 0.0
        fixed = False
        mode = "float"
        if (
            epoch_i > 60
            and len(amb_sats) >= 4
            and not hold_ok
            and float(np.mean(np.diag(P[3:, 3:]))) < 4.0
        ):
            Qa = P[3:, 3:].copy()
            Qa = 0.5 * (Qa + Qa.T) + np.eye(len(amb_sats)) * 1e-8
            try:
                afix, ratio, ok = lambda_fix(x_amb, Qa, ratio_th=ratio_th)
            except Exception:
                afix, ratio, ok = x_amb, 0.0, False
            if ok:
                # 固定后用相位残差验收，防止误固定
                trial = {s: int(round(float(afix[i]))) for i, s in enumerate(amb_sats)}
                zs_try = []
                for sat in amb_sats:
                    m = by[sat]
                    dd_rho = _dd_rho(m, ref, xyz, x_base)
                    dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * m["lam"]
                    zs_try.append(dd_l_m - dd_rho - m["lam"] * trial[sat])
                try_rms = float(np.sqrt(np.mean(np.square(zs_try))))
                if try_rms < 0.05:
                    fixed_hold = trial
                    for i, s in enumerate(amb_sats):
                        x_amb[i] = float(fixed_hold[s])
                        P[3 + i, :] = 0.0
                        P[:, 3 + i] = 0.0
                        P[3 + i, 3 + i] = 1e-10
                    hold_ok = True
                    fixed = True
                    mode = f"fixed|ref={ref_sat}|ratio={ratio:.2f}|rms={try_rms:.3f}"
                else:
                    mode = f"float|reject_fix|ratio={ratio:.2f}|rms={try_rms:.3f}"
            else:
                mode = f"float|ref={ref_sat}|ratio={ratio:.2f}|namb={len(amb_sats)}"
        elif hold_ok:
            fixed = True
            rows, zs = [], []
            for sat in amb_sats:
                m = by[sat]
                dd_rho = _dd_rho(m, ref, xyz, x_base)
                u = _los_dd(m, ref, xyz)
                w = np.sin(m["el"]) ** 2
                dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * m["lam"]
                N = fixed_hold.get(sat, int(round(x_amb[amb_sats.index(sat)])))
                rows.append(np.sqrt(w) * u)
                zs.append(np.sqrt(w) * (dd_l_m - dd_rho - m["lam"] * N))
            if len(rows) >= 3:
                dx, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(zs), rcond=None)
                xyz = xyz + dx
                x[:3] = xyz
            # 固定保持期间若残差变差则解锁
            raw = []
            for sat in amb_sats:
                m = by[sat]
                dd_rho = _dd_rho(m, ref, xyz, x_base)
                dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * m["lam"]
                N = fixed_hold.get(sat, 0)
                raw.append(dd_l_m - dd_rho - m["lam"] * N)
            hold_rms = float(np.sqrt(np.mean(np.square(raw)))) if raw else 9.0
            if hold_rms > 0.15:
                hold_ok = False
                fixed_hold = {}
                fixed = False
                mode = f"unlock|rms={hold_rms:.3f}"
                for i in range(len(amb_sats)):
                    P[3 + i, 3 + i] = max(P[3 + i, 3 + i], 25.0)
            else:
                mode = f"hold|ref={ref_sat}|namb={len(amb_sats)}|rms={hold_rms:.3f}"
                ratio = 99.0
        else:
            mode = f"float|ref={ref_sat}|namb={len(amb_sats)}"

        # 残差
        zs = []
        for sat in amb_sats:
            m = by[sat]
            dd_rho = _dd_rho(m, ref, xyz, x_base)
            dd_l_m = ((m["lr"] - m["lb"]) - (ref["lr"] - ref["lb"])) * m["lam"]
            Ni = fixed_hold.get(sat, x_amb[amb_sats.index(sat)]) if fixed else x_amb[amb_sats.index(sat)]
            zs.append(dd_l_m - dd_rho - m["lam"] * Ni)
        results.append(
            RtkEpochResult(
                time=ep_r.time,
                rover_xyz=xyz.copy(),
                nsat=len(amb_sats),
                residual_rms=float(np.sqrt(np.mean(np.square(zs)))) if zs else 9.0,
                success=True,
                ratio=float(ratio),
                fixed=fixed,
                mode=mode,
            )
        )
        epoch_i += 1
    return results


def summarize_rtk(results: List[RtkEpochResult], ref_xyz: np.ndarray, base_xyz: np.ndarray | None = None) -> dict:
    if not results:
        return {"n": 0}
    fixed = [r for r in results if r.fixed]
    use = fixed or results
    enus = np.array([ecef_to_enu(r.rover_xyz, ref_xyz) for r in use])
    # 对 hold/会话：按唯一坐标聚一下；连续 EKF 用全部历元
    uniq = {}
    for r in use:
        key = tuple(np.round(r.rover_xyz, 4))
        uniq[key] = r
    if len(uniq) <= max(3, len(use) // 20):
        # 少数离散解（旧会话模式）
        stats_enus = np.array([ecef_to_enu(np.array(k), ref_xyz) for k in uniq])
        n_sessions = len(uniq)
    else:
        stats_enus = enus
        n_sessions = 1
    out = {
        "n": len(results),
        "n_fixed": len(fixed),
        "fix_rate": float(len(fixed) / len(results)),
        "n_sessions": n_sessions,
        "mean_enu_m": stats_enus.mean(axis=0).tolist(),
        "rms_enu_m": np.sqrt((stats_enus**2).mean(axis=0)).tolist(),
        "rms_3d_m": float(np.sqrt((stats_enus**2).sum(axis=1).mean())),
        "horizontal_rms_m": float(np.sqrt(np.mean(stats_enus[:, 0] ** 2 + stats_enus[:, 1] ** 2))),
        "phase_residual_rms_m": float(np.mean([r.residual_rms for r in use])),
        "mode": use[-1].mode if use else "",
        "static_session": n_sessions > 1,
        "static_ekf": "hold" in (use[-1].mode if use else "") or "fixed|ref" in (use[-1].mode if use else "") and "frac" not in (use[-1].mode if use else ""),
    }
    if base_xyz is not None and len(stats_enus):
        if n_sessions > 1:
            bls = [float(np.linalg.norm(np.array(k) - base_xyz)) for k in uniq]
        else:
            bls = [float(np.linalg.norm(r.rover_xyz - base_xyz)) for r in use]
        out["baseline_mean_m"] = float(np.mean(bls))
        out["approx_baseline_m"] = float(np.linalg.norm(ref_xyz - base_xyz))
        out["baseline_err_mean_m"] = float(np.mean(bls) - out["approx_baseline_m"])
    return out


def write_rtk_csv(results: List[RtkEpochResult], path: str | Path, ref_xyz: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,x,y,z,e,n,u,nsat,rms,ratio,fixed,mode\n")
        for r in results:
            e, n, u = ecef_to_enu(r.rover_xyz, ref_xyz)
            f.write(
                f"{r.time.isoformat()},{r.rover_xyz[0]:.4f},{r.rover_xyz[1]:.4f},{r.rover_xyz[2]:.4f},"
                f"{e:.4f},{n:.4f},{u:.4f},{r.nsat},{r.residual_rms:.4f},{r.ratio:.3f},{int(r.fixed)},{r.mode}\n"
            )
