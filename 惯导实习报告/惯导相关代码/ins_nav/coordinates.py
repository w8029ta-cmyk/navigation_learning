"""大地坐标和 ECEF 直角坐标之间的转换。"""

import math

from .constants import WGS84_A, WGS84_B, first_eccentricity


def prime_vertical_radius(lat):
    """卯酉圈曲率半径 R_N，单位 m。

    公式：R_N = a / sqrt(1 - e^2 * sin^2(lat))
    """
    e = first_eccentricity()
    return WGS84_A / math.sqrt(1.0 - e * e * math.sin(lat) ** 2)


def meridian_radius(lat):
    """子午圈曲率半径 R_M，单位 m。

    公式：R_M = a * (1 - e^2) / (1 - e^2 * sin^2(lat))^(3/2)
    """
    e = first_eccentricity()
    return WGS84_A * (1.0 - e * e) / (1.0 - e * e * math.sin(lat) ** 2) ** 1.5


def geodetic_to_ecef(lat, lon, h):
    """大地坐标 (lat, lon, h) -> ECEF 直角坐标 (x, y, z)。

    lat/lon 单位是弧度，h 单位 m。
    """
    rn = prime_vertical_radius(lat)
    e_sq = first_eccentricity() ** 2
    x = (rn + h) * math.cos(lat) * math.cos(lon)
    y = (rn + h) * math.cos(lat) * math.sin(lon)
    z = ((1.0 - e_sq) * rn + h) * math.sin(lat)
    return x, y, z


def ecef_to_geodetic(x, y, z):
    """ECEF 直角坐标 (x, y, z) -> 大地坐标 (lat, lon, h)。

    这里用经典的收敛迭代法，最多迭代 100 次。
    """
    lon = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    lat = math.atan2(z, horizontal)
    e_sq = first_eccentricity() ** 2

    for _ in range(100):
        rn = prime_vertical_radius(lat)
        h = horizontal / math.cos(lat) - rn
        next_lat = math.atan2(
            z * (rn + h),
            horizontal * ((1.0 - e_sq) * rn + h),
        )
        if abs(next_lat - lat) < 1e-10:
            lat = next_lat
            break
        lat = next_lat

    rn = prime_vertical_radius(lat)
    h = horizontal / math.cos(lat) - rn
    return lat, lon, h
