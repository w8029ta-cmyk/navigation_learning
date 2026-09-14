"""与 KF-GINS 兼容的导航 / IMU 误差文件读写。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .attitude import quat_to_euler

# Leador-A15 数据集 GNSS 周（与 truth.nav 一致）
DEFAULT_GNSS_WEEK = 2017


def nav_row_to_line(week: int, time_s: float, nav_state) -> str:
    """格式化单个导航历元（11 列，格式同 truth.nav / KF-GINS NavResult）。"""
    lat_deg = math.degrees(nav_state.lat)
    lon_deg = math.degrees(nav_state.lon)
    vn, ve, vd = nav_state.v_n
    roll, pitch, yaw = np.degrees(quat_to_euler(nav_state.q_b_n))
    return (
        f"{week:4d} {time_s:12.3f} "
        f"{lat_deg:16.10f} {lon_deg:16.10f} {nav_state.h:9.3f} "
        f"{vn:9.3f} {ve:9.3f} {vd:9.3f} "
        f"{roll:11.8f} {pitch:11.8f} {yaw:11.8f}"
    )


def imu_error_row_to_line(time_s: float, gyro_bias, acc_bias, gyro_scale, acc_scale) -> str:
    """13 列：时间、陀螺零偏(deg/h)、加计零偏(mGal)、陀螺比例因子(ppm)、加计比例因子(ppm)。"""
    bg = np.asarray(gyro_bias, dtype=float) * 180.0 / math.pi * 3600.0
    ba = np.asarray(acc_bias, dtype=float) * 1e5
    sg = np.asarray(gyro_scale, dtype=float) * 1e6
    sa = np.asarray(acc_scale, dtype=float) * 1e6
    return (
        f"{time_s:12.3f} "
        f"{bg[0]:+.6f} {bg[1]:+.6f} {bg[2]:+.6f} "
        f"{ba[0]:+.6f} {ba[1]:+.6f} {ba[2]:+.6f} "
        f"{sg[0]:+.6f} {sg[1]:+.6f} {sg[2]:+.6f} "
        f"{sa[0]:+.6f} {sa[1]:+.6f} {sa[2]:+.6f}"
    )


def write_nav_file(path: Path, records, week: int = DEFAULT_GNSS_WEEK):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for time_s, nav in records:
            f.write(nav_row_to_line(week, time_s, nav) + "\n")


def write_imu_error_file(path: Path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for time_s, ins in records:
            f.write(imu_error_row_to_line(
                time_s, ins.gyro_bias, ins.acc_bias, ins.gyro_scale, ins.acc_scale,
            ) + "\n")


def load_nav_file(path: Path) -> np.ndarray:
    """读取 11 列导航文件 -> [week, time, lat_deg, lon_deg, h, vn, ve, vd, roll, pitch, yaw]。"""
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = line.split()
            rows.append([float(x) for x in c[:11]])
    return np.asarray(rows, dtype=float)


def interpolate_nav(nav: np.ndarray, time_s: float) -> np.ndarray:
    times = nav[:, 1]
    if time_s <= times[0]:
        return nav[0]
    if time_s >= times[-1]:
        return nav[-1]
    i = np.searchsorted(times, time_s)
    a, b = nav[i - 1], nav[i]
    r = (time_s - a[1]) / (b[1] - a[1])
    row = a + r * (b - a)
    row[1] = time_s
    return row
