"""时间和常量，集中放这里。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# WGS84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
OMEGA_E = 7.2921151467e-5
MU_GPS = 3.986005e14
MU_BDS = 3.986004418e14
CLIGHT = 299792458.0

# 频率 (Hz) —— Stage A/C 无电离层组合用
FREQ = {
    ("G", 1): 1575.42e6,
    ("G", 2): 1227.60e6,
    ("C", 2): 1561.098e6,  # B1I
    ("C", 6): 1268.520e6,  # B3I
    ("C", 7): 1207.140e6,  # B2I
}

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


@dataclass(frozen=True)
class GTime:
    """周内秒表示，便于和广播星历 toe/toc 对齐。"""

    week: int
    sow: float

    def to_datetime(self) -> datetime:
        return GPS_EPOCH + timedelta(weeks=self.week, seconds=self.sow)

    def __sub__(self, other: "GTime") -> float:
        return (self.week - other.week) * 604800.0 + (self.sow - other.sow)


def datetime_to_gtime(dt: datetime) -> GTime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - GPS_EPOCH
    secs = delta.total_seconds()
    week = int(secs // 604800)
    sow = secs - week * 604800
    return GTime(week, sow)


def yydoy_to_datetime(year: int, doy: int, sod: float = 0.0) -> datetime:
    from datetime import date, timedelta

    d = date(year, 1, 1) + timedelta(days=doy - 1)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(seconds=sod)
