"""对流层、Klobuchar 电离层、几何距离（含 Sagnac）。"""

from __future__ import annotations

import numpy as np

from ladder.util.constants import CLIGHT, OMEGA_E
from ladder.util.coord import xyz_to_llh

# RTKLIB ionmodel 缺省参数（2004/1/1）
_ION_DEFAULT = np.array(
    [
        0.1118e-7,
        -0.7451e-8,
        -0.5961e-7,
        0.1192e-6,
        0.1167e6,
        -0.2294e6,
        -0.1311e6,
        0.1049e7,
    ],
    dtype=float,
)


def elevation_azimuth(rec_xyz: np.ndarray, sat_xyz: np.ndarray) -> tuple[float, float]:
    from ladder.util.coord import xyz_to_enu_rotation

    enu = xyz_to_enu_rotation(rec_xyz) @ (sat_xyz - rec_xyz)
    e, n, u = enu
    horiz = np.hypot(e, n)
    el = np.arctan2(u, horiz)
    az = np.arctan2(e, n)
    return float(el), float(az)


def saastamoinen_zhd_zwd(rec_xyz: np.ndarray, rel_humi: float = 0.7) -> tuple[float, float]:
    """Saastamoinen 天顶干/湿延迟 (m)。rel_humi=0 → 仅干延迟（PPP ZTD 估计用）。"""
    lat, _, h = xyz_to_llh(rec_xyz)
    if h < -100 or h > 1e4:
        h = max(0.0, min(h, 5000.0))
    p = 1013.25 * (1 - 2.2557e-5 * h) ** 5.2568
    t = 15.0 - 6.5e-3 * h + 273.15
    e = 6.108 * rel_humi * np.exp((17.15 * t - 4684.0) / (t - 38.45)) if rel_humi > 0 else 0.0
    zhd = 0.0022768 * p / (1 - 0.00266 * np.cos(2 * lat) - 0.00028 * h / 1000.0)
    zwd = 0.002277 * (1255.0 / t + 0.05) * e
    return float(zhd), float(zwd)


def trop_mapping(elev: float) -> tuple[float, float]:
    """简化映射：干/湿均用 1/sin(el)。"""
    el = max(elev, np.deg2rad(3.0))
    m = 1.0 / np.sin(el)
    return float(m), float(m)


def _interpc(coef: list[float], lat_deg: float) -> float:
    i = int(lat_deg / 15.0)
    if i < 1:
        return coef[0]
    if i > 4:
        return coef[4]
    return coef[i - 1] * (1.0 - lat_deg / 15.0 + i) + coef[i] * (lat_deg / 15.0 - i)


def _mapf(el: float, a: float, b: float, c: float) -> float:
    s = np.sin(el)
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (s + (a / (s + b / (s + c))))


def niell_mapping(rec_xyz: np.ndarray, elev: float, doy: float) -> tuple[float, float]:
    """Niell 映射函数，返回 (m_h, m_w)。"""
    # hydro-ave, hydro-amp, wet at lat 15,30,45,60,75
    coef = [
        [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3],
        [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3],
        [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3],
        [0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5],
        [0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5],
        [0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5],
        [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641694e-4],
        [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3],
        [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2],
    ]
    aht = [2.53e-5, 5.49e-3, 1.14e-3]
    lat, _, hgt = xyz_to_llh(rec_xyz)
    el = max(elev, np.deg2rad(3.0))
    lat_d = abs(lat * 180.0 / np.pi)
    y = (doy - 28.0) / 365.25 + (0.5 if lat < 0 else 0.0)
    cosy = np.cos(2.0 * np.pi * y)
    ah = [_interpc(coef[i], lat_d) - _interpc(coef[i + 3], lat_d) * cosy for i in range(3)]
    aw = [_interpc(coef[i + 6], lat_d) for i in range(3)]
    dm = (1.0 / np.sin(el) - _mapf(el, aht[0], aht[1], aht[2])) * hgt / 1e3
    mh = _mapf(el, ah[0], ah[1], ah[2]) + dm
    mw = _mapf(el, aw[0], aw[1], aw[2])
    return float(mh), float(mw)


def saastamoinen_tropo(rec_xyz: np.ndarray, elev: float, rel_humi: float = 0.7) -> float:
    """Saastamoinen 斜路径干+湿，单位 m。"""
    zhd, zwd = saastamoinen_zhd_zwd(rec_xyz, rel_humi=rel_humi)
    mh, mw = trop_mapping(elev)
    return float(mh * zhd + mw * zwd)


def trop_delay_est_ztd(
    rec_xyz: np.ndarray,
    elev: float,
    ztd: float,
    doy: float | None = None,
) -> tuple[float, float]:
    """对流层延迟：m_h*ZHD + m_w*(ZTD-ZHD)；返回 (delay, ∂delay/∂ZTD)。"""
    zhd, _ = saastamoinen_zhd_zwd(rec_xyz, rel_humi=0.0)
    if doy is not None:
        mh, mw = niell_mapping(rec_xyz, elev, doy)
    else:
        mh, mw = trop_mapping(elev)
    return float(mh * zhd + mw * (ztd - zhd)), float(mw)


def earth_rotation_corr(sat_xyz: np.ndarray, travel_time: float) -> np.ndarray:
    """信号传播期间地球自转：把发射时刻位置转到接收时刻 ECEF。"""
    theta = OMEGA_E * travel_time
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)
    return R @ sat_xyz


def geometric_range(sat_xyz: np.ndarray, rec_xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """几何距离 + Sagnac；e 为接收机→卫星单位向量。"""
    e = sat_xyz - rec_xyz
    r = float(np.linalg.norm(e))
    if r < 1.0:
        return -1.0, e
    e = e / r
    r += OMEGA_E * (sat_xyz[0] * rec_xyz[1] - sat_xyz[1] * rec_xyz[0]) / CLIGHT
    return r, e


def klobuchar_iono(
    rec_xyz: np.ndarray,
    elev: float,
    azim: float,
    sow: float,
    alpha_beta: np.ndarray | None = None,
) -> float:
    """Klobuchar L1 斜路径延迟 (m)。"""
    ion = alpha_beta if alpha_beta is not None and np.linalg.norm(alpha_beta) > 0 else _ION_DEFAULT
    lat, lon, h = xyz_to_llh(rec_xyz)
    if h < -1e3 or elev <= 0:
        return 0.0
    # earth-centered angle / pierce point (semi-circles)
    psi = 0.0137 / (elev / np.pi + 0.11) - 0.022
    phi = lat / np.pi + psi * np.cos(azim)
    phi = max(-0.416, min(0.416, phi))
    lam = lon / np.pi + psi * np.sin(azim) / np.cos(phi * np.pi)
    phi_m = phi + 0.064 * np.cos((lam - 1.617) * np.pi)
    t = 43200.0 * lam + sow
    t = t - np.floor(t / 86400.0) * 86400.0
    f = 1.0 + 16.0 * (0.53 - elev / np.pi) ** 3
    amp = ion[0] + phi_m * (ion[1] + phi_m * (ion[2] + phi_m * ion[3]))
    per = ion[4] + phi_m * (ion[5] + phi_m * (ion[6] + phi_m * ion[7]))
    amp = max(0.0, amp)
    per = max(72000.0, per)
    x = 2.0 * np.pi * (t - 50400.0) / per
    if abs(x) < 1.57:
        delay = CLIGHT * f * (5e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0)))
    else:
        delay = CLIGHT * f * 5e-9
    return float(delay)
