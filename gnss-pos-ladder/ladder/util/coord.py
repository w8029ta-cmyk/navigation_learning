"""坐标转换：ECEF ↔ 大地坐标，以及 ENU 用于误差展示。"""

from __future__ import annotations

import numpy as np

from ladder.util.constants import WGS84_A, WGS84_E2


def xyz_to_llh(xyz: np.ndarray) -> np.ndarray:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = np.arctan2(z + WGS84_E2 * n * sin_lat, p)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / np.cos(lat) - n
    return np.array([lat, lon, h], dtype=float)


def llh_to_xyz(llh: np.ndarray) -> np.ndarray:
    lat, lon, h = float(llh[0]), float(llh[1]), float(llh[2])
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * cos_lon
    y = (n + h) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)


def xyz_to_enu_rotation(ref_xyz: np.ndarray) -> np.ndarray:
    lat, lon, _ = xyz_to_llh(ref_xyz)
    sL, cL = np.sin(lon), np.cos(lon)
    sB, cB = np.sin(lat), np.cos(lat)
    return np.array(
        [
            [-sL, cL, 0.0],
            [-sB * cL, -sB * sL, cB],
            [cB * cL, cB * sL, sB],
        ],
        dtype=float,
    )


def ecef_to_enu(xyz: np.ndarray, ref_xyz: np.ndarray) -> np.ndarray:
    R = xyz_to_enu_rotation(ref_xyz)
    return R @ (np.asarray(xyz, dtype=float) - np.asarray(ref_xyz, dtype=float))
