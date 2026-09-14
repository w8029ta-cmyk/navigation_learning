"""接收机/卫星天线改正与相位缠绕。

ARP→APC、星端 PCO、相位缠绕。写的时候对照过 RTKLIB 的 antmodel / satantoff / windupcorr。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

from ladder.util.constants import CLIGHT, FREQ
from ladder.util.coord import xyz_to_enu_rotation, xyz_to_llh


def ant_range_corr(del_enu: np.ndarray, e_los_ecef: np.ndarray, rec_xyz: np.ndarray) -> float:
    """接收机天线改正 (m)：dant = −off·e，off 为 ENU。

    观测侧做 meas − dant（等价 pred += dant）。
    del_enu: ARP→APC 的 ENU；e_los_ecef: 接收机→卫星单位向量。
    """
    R = xyz_to_enu_rotation(rec_xyz)
    e_enu = R @ e_los_ecef
    return float(-np.dot(del_enu, e_enu))


def hen_to_enu(h: float, e: float, n: float) -> np.ndarray:
    """RINEX ANTENNA: DELTA H/E/N → ENU {e,n,u}（RTKLIB sta.del 约定）。"""
    return np.array([e, n, h], dtype=float)


def sun_ecef_approx(t: datetime) -> np.ndarray:
    """太阳 ECEF 近似（忽略 ERP，UT≈GPST）。"""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    # J2000.0
    t0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    tt = (t.timestamp() - t0.timestamp()) / 86400.0 / 36525.0
    Ms = np.deg2rad(357.5277233 + 35999.05034 * tt)
    ls = np.deg2rad(
        280.460
        + 36000.770 * tt
        + 1.914666471 * np.sin(Ms)
        + 0.019994643 * np.sin(2.0 * Ms)
    )
    eps = np.deg2rad(23.439291 - 0.0130042 * tt)
    rs = 149597870000.0 * (
        1.000140612 - 0.016708617 * np.cos(Ms) - 0.000139589 * np.cos(2.0 * Ms)
    )
    # 近似：忽略岁差章动，把平赤道惯性系当 ECEF（PPP 缠绕/星固系足够用）
    sine, cose = np.sin(eps), np.cos(eps)
    sinl, cosl = np.sin(ls), np.cos(ls)
    return np.array([rs * cosl, rs * cose * sinl, rs * sine * sinl], dtype=float)


def _norm(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def sat_ant_offset_ecef(
    sat_xyz: np.ndarray,
    t: datetime,
    pco_xyz: np.ndarray,
) -> np.ndarray:
    """卫星天线 PCO：星固系 → ECEF 改正向量。"""
    rsun = sun_ecef_approx(t)
    ez = _norm(-sat_xyz)
    if ez is None:
        return np.zeros(3)
    es = _norm(rsun - sat_xyz)
    if es is None:
        return np.zeros(3)
    ey = _norm(np.cross(ez, es))
    if ey is None:
        return np.zeros(3)
    ex = np.cross(ey, ez)
    return pco_xyz[0] * ex + pco_xyz[1] * ey + pco_xyz[2] * ez


def windup_cycles(
    t: datetime,
    sat_xyz: np.ndarray,
    rec_xyz: np.ndarray,
    phw_prev: float,
) -> float:
    """相位缠绕（周）；连续去跳。"""
    rsun = sun_ecef_approx(t)
    ek = _norm(rec_xyz - sat_xyz)
    if ek is None:
        return phw_prev
    ezs = _norm(-sat_xyz)
    if ezs is None:
        return phw_prev
    ess = _norm(rsun - sat_xyz)
    if ess is None:
        return phw_prev
    eys = _norm(np.cross(ezs, ess))
    if eys is None:
        return phw_prev
    exs = np.cross(eys, ezs)

    # 接收机：x=北, y=西（RTKLIB）
    R = xyz_to_enu_rotation(rec_xyz)  # rows: e,n,u in ECEF cols
    # E = enu_from_ecef; exr = north = R[1], eyr = west = -east = -R[0]
    exr = R[1]
    eyr = -R[0]

    eks = np.cross(ek, eys)
    ekr = np.cross(ek, eyr)
    ds = exs - ek * np.dot(ek, exs) - eks
    dr = exr - ek * np.dot(ek, exr) + ekr
    nds = float(np.linalg.norm(ds))
    ndr = float(np.linalg.norm(dr))
    if nds < 1e-12 or ndr < 1e-12:
        return phw_prev
    cosp = float(np.dot(ds, dr) / (nds * ndr))
    cosp = max(-1.0, min(1.0, cosp))
    ph = float(np.arccos(cosp) / (2.0 * np.pi))
    if np.dot(ek, np.cross(ds, dr)) < 0.0:
        ph = -ph
    return ph + np.floor(phw_prev - ph + 0.5)


def if_windup_meters(phw_cycles: float, sys: str = "G") -> float:
    """无电离层组合相位缠绕改正量 (m)。"""
    if sys == "G":
        f1, f2 = FREQ[("G", 1)], FREQ[("G", 2)]
    elif sys == "C":
        f1, f2 = FREQ[("C", 2)], FREQ[("C", 6)]
    else:
        f1, f2 = FREQ[("G", 1)], FREQ[("G", 2)]
    return float(phw_cycles * CLIGHT / (f1 + f2))
