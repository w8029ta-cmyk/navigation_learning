"""广播星历 → 卫星 ECEF 位置/钟差。

GPS / BDS 注意点：
- μ、ωe 略有不同（BDS 用 CGCS2000 常用 μ）
- BDS toe 要注意北斗时与 GPS 时；RINEX3 已统一到 GPST，这里按 GPST 用
- GEO（C01–C05）要额外旋转
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from ladder.rinex.nav import BroadcastEphemeris, NavStore
from ladder.util.constants import CLIGHT, MU_BDS, MU_GPS, OMEGA_E, datetime_to_gtime


def _kepler(e: float, Mk: float) -> float:
    Ek = Mk
    for _ in range(20):
        Ek_new = Mk + e * np.sin(Ek)
        if abs(Ek_new - Ek) < 1e-13:
            break
        Ek = Ek_new
    return Ek


def sat_pos_broadcast(
    eph: BroadcastEphemeris,
    t: datetime,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """返回 (xyz ECEF [m], clock [s], vel approx [m/s])."""
    sys = eph.sat[0]
    mu = MU_BDS if sys == "C" else MU_GPS
    omega_e = OMEGA_E

    gt = datetime_to_gtime(t)
    # RINEX 北斗 week 相对 GPS week +1356。
    # MEO/IGSO：toe 属 BDT，观测为 GPST，需 BDT=GPST-14s。
    # GEO：实测若再减 14s 会把残差从米级拉到公里级，故 GEO 不减。
    eph_week = int(eph.week)
    sow = gt.sow
    prn = int(eph.sat[1:])
    if sys == "C":
        eph_week += 1356
        if prn > 5:
            sow -= 14.0
    tk = (gt.week - eph_week) * 604800.0 + (sow - eph.toe_sow)
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800


    A = eph.sqrt_a * eph.sqrt_a
    n0 = np.sqrt(mu / A**3)
    n = n0 + eph.delta_n
    Mk = eph.m0 + n * tk
    Ek = _kepler(eph.e, Mk)
    sk = np.sin(Ek)
    ck = np.cos(Ek)
    vk = np.arctan2(np.sqrt(1 - eph.e**2) * sk, ck - eph.e)
    phi = vk + eph.omega
    du = eph.cuc * np.cos(2 * phi) + eph.cus * np.sin(2 * phi)
    dr = eph.crc * np.cos(2 * phi) + eph.crs * np.sin(2 * phi)
    di = eph.cic * np.cos(2 * phi) + eph.cis * np.sin(2 * phi)
    u = phi + du
    r = A * (1 - eph.e * ck) + dr
    i = eph.i0 + di + eph.idot * tk
    x_orb = r * np.cos(u)
    y_orb = r * np.sin(u)

    omega = eph.omega0 + (eph.omega_dot - omega_e) * tk - omega_e * eph.toe_sow
    cos_o, sin_o = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    x = x_orb * cos_o - y_orb * cos_i * sin_o
    y = x_orb * sin_o + y_orb * cos_i * cos_o
    z = y_orb * sin_i
    xyz = np.array([x, y, z], dtype=float)

    # BDS GEO (PRN 1-5): additional rotation to CGCS2000/ECEF
    if sys == "C" and prn <= 5:
        # simplified GEO transformation (ICD-BDS)
        theta = omega_e * tk
        Rz = np.array(
            [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
            dtype=float,
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(-5 * np.pi / 180), np.sin(-5 * np.pi / 180)],
                [0, -np.sin(-5 * np.pi / 180), np.cos(-5 * np.pi / 180)],
            ],
            dtype=float,
        )
        xyz = Rz @ Rx @ xyz

    # clock
    dt_toc = (t - eph.toc).total_seconds()
    clk = eph.af0 + eph.af1 * dt_toc + eph.af2 * dt_toc * dt_toc
    # relativistic
    clk += -2.0 * np.sqrt(mu * A) * eph.e * sk / (CLIGHT**2)

    # rough velocity by dt
    dt = 0.001
    tk2 = tk + dt
    Mk2 = eph.m0 + n * tk2
    Ek2 = _kepler(eph.e, Mk2)
    vk2 = np.arctan2(np.sqrt(1 - eph.e**2) * np.sin(Ek2), np.cos(Ek2) - eph.e)
    phi2 = vk2 + eph.omega
    u2 = phi2 + eph.cuc * np.cos(2 * phi2) + eph.cus * np.sin(2 * phi2)
    r2 = A * (1 - eph.e * np.cos(Ek2)) + eph.crc * np.cos(2 * phi2) + eph.crs * np.sin(2 * phi2)
    i2 = eph.i0 + eph.cic * np.cos(2 * phi2) + eph.cis * np.sin(2 * phi2) + eph.idot * tk2
    x_orb2, y_orb2 = r2 * np.cos(u2), r2 * np.sin(u2)
    omega2 = eph.omega0 + (eph.omega_dot - omega_e) * tk2 - omega_e * eph.toe_sow
    xyz2 = np.array(
        [
            x_orb2 * np.cos(omega2) - y_orb2 * np.cos(i2) * np.sin(omega2),
            x_orb2 * np.sin(omega2) + y_orb2 * np.cos(i2) * np.cos(omega2),
            y_orb2 * np.sin(i2),
        ]
    )
    if sys == "C" and prn <= 5:
        theta = omega_e * tk2
        Rz = np.array(
            [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
            dtype=float,
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(-5 * np.pi / 180), np.sin(-5 * np.pi / 180)],
                [0, -np.sin(-5 * np.pi / 180), np.cos(-5 * np.pi / 180)],
            ],
            dtype=float,
        )
        xyz2 = Rz @ Rx @ xyz2
    vel = (xyz2 - xyz) / dt
    return xyz, clk, vel

def sat_pos_at(
    store: NavStore,
    sat: str,
    t: datetime,
) -> Optional[Tuple[np.ndarray, float, np.ndarray, BroadcastEphemeris]]:
    eph = store.select(sat, t)
    if eph is None:
        return None
    # signal travel iteration done outside; here just toe-based
    xyz, clk, vel = sat_pos_broadcast(eph, t)
    return xyz, clk, vel, eph
