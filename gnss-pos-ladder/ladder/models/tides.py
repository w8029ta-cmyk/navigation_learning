"""固体潮（仅固体潮，ERP=0）。

几何距离用 xyz + 潮汐位移。加上以后 PPP 3D 几乎不动（cm 级），U 偏不在这里。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

import numpy as np

from ladder.models.antenna import sun_ecef_approx
from ladder.util.constants import WGS84_A
from ladder.util.coord import xyz_to_enu_rotation, xyz_to_llh

GME = 3.986004415e14
GMS = 1.327124e20
GMM = 4.902801e12


def _ast_args(t: float) -> np.ndarray:
    """章动用天文幅角（rad）。"""
    f = np.zeros(5)
    f[0] = np.deg2rad(134.96340251 + 1717915923.2178e-5 * t)  # l
    f[1] = np.deg2rad(357.52910918 + 129596581.0481e-5 * t)  # l'
    f[2] = np.deg2rad(93.27209062 + 1739527262.8478e-5 * t)  # F
    f[3] = np.deg2rad(297.85019547 + 1602961601.2090e-5 * t)  # D
    f[4] = np.deg2rad(125.04455501 - 6962890.5431e-5 * t)  # Ω
    return f


def moon_ecef_approx(t: datetime) -> np.ndarray:
    """月球 ECEF 近似（忽略 ERP，与 `sun_ecef_approx` 同级）。"""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    t0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    tt = (t.timestamp() - t0.timestamp()) / 86400.0 / 36525.0
    f = _ast_args(tt)
    eps = np.deg2rad(23.439291 - 0.0130042 * tt)
    sine, cose = np.sin(eps), np.cos(eps)
    lm = np.deg2rad(
        218.32
        + 481267.883 * tt
        + 6.29 * np.sin(f[0])
        - 1.27 * np.sin(f[0] - 2.0 * f[3])
        + 0.66 * np.sin(2.0 * f[3])
        + 0.21 * np.sin(2.0 * f[0])
        - 0.19 * np.sin(f[1])
        - 0.11 * np.sin(2.0 * f[2])
    )
    pm = np.deg2rad(
        5.13 * np.sin(f[2])
        + 0.28 * np.sin(f[0] + f[2])
        - 0.28 * np.sin(f[2] - f[0])
        - 0.17 * np.sin(f[2] - 2.0 * f[3])
    )
    rm = WGS84_A / np.sin(
        np.deg2rad(
            0.9508
            + 0.0518 * np.cos(f[0])
            + 0.0095 * np.cos(f[0] - 2.0 * f[3])
            + 0.0078 * np.cos(2.0 * f[3])
            + 0.0028 * np.cos(2.0 * f[0])
        )
    )
    sinl, cosl = np.sin(lm), np.cos(lm)
    sinp, cosp = np.sin(pm), np.cos(pm)
    return np.array(
        [
            rm * cosp * cosl,
            rm * (cose * cosp * sinl - sine * sinp),
            rm * (sine * cosp * sinl + cose * sinp),
        ],
        dtype=float,
    )


def _gmst_approx(t: datetime) -> float:
    """平格林尼治恒星时近似 (rad)。"""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    # 简化：从 J2000 起算
    t0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = (t.timestamp() - t0.timestamp()) / 86400.0
    return float(np.deg2rad(280.46061837 + 360.98564736629 * d) % (2 * np.pi))


def _tide_pl(eu: np.ndarray, rp: np.ndarray, gmp: float, pos: np.ndarray) -> np.ndarray:
    """单天体固体潮。pos=[lat,lon,h]。"""
    r = float(np.linalg.norm(rp))
    if r <= 0:
        return np.zeros(3)
    ep = rp / r
    k2 = gmp / GME * (WGS84_A**4) / (r**3)
    k3 = k2 * WGS84_A / r
    latp = np.arcsin(ep[2])
    lonp = np.arctan2(ep[1], ep[0])
    cosp = np.cos(latp)
    sinl = np.sin(pos[0])
    cosl = np.cos(pos[0])
    p = (3.0 * sinl * sinl - 1.0) / 2.0
    h2 = 0.6078 - 0.0006 * p
    l2 = 0.0847 + 0.0002 * p
    a = float(np.dot(ep, eu))
    dp = k2 * 3.0 * l2 * a
    du = k2 * (h2 * (1.5 * a * a - 0.5) - 3.0 * l2 * a * a)
    h3, l3 = 0.292, 0.015
    dp += k3 * l3 * (7.5 * a * a - 1.5)
    du += k3 * (h3 * (2.5 * a * a * a - 1.5 * a) - l3 * (7.5 * a * a - 1.5) * a)
    du += 0.75 * 0.0025 * k2 * np.sin(2.0 * latp) * np.sin(2.0 * pos[0]) * np.sin(pos[1] - lonp)
    du += 0.75 * 0.0022 * k2 * cosp * cosp * cosl * cosl * np.sin(2.0 * (pos[1] - lonp))
    return dp * ep + du * eu


def solid_earth_tide(rec_xyz: np.ndarray, t: datetime) -> np.ndarray:
    """返回 ECEF 潮汐位移 (m)；不含永久形变消除项。"""
    pos = xyz_to_llh(rec_xyz)  # lat, lon, h
    R = xyz_to_enu_rotation(rec_xyz)  # rows e,n,u
    # E 矩阵：ECEF→ENU 的转置关系；eu = 天顶方向在 ECEF = R 第三行
    eu = R[2]
    E_cols = R.T  # columns are e,n,u in ECEF — 与 RTKLIB xyz2enu 的 E 一致用法
    # RTKLIB: eu[0]=E[2], eu[1]=E[5], eu[2]=E[8] → E 按列主序存 enu 基向量
    # 即 E = [[e0,n0,u0],[e1,n1,u1],[e2,n2,u2]] = R.T
    rsun = sun_ecef_approx(t)
    rmoon = moon_ecef_approx(t)
    dr1 = _tide_pl(eu, rsun, GMS, pos)
    dr2 = _tide_pl(eu, rmoon, GMM, pos)
    gmst = _gmst_approx(t)
    sin2l = np.sin(2.0 * pos[0])
    du = -0.012 * sin2l * np.sin(gmst + pos[1])
    dr = dr1 + dr2 + du * E_cols[:, 2]
    return dr
