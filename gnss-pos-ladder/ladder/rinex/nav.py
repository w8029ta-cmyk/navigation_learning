"""RINEX 3 导航文件（广播星历 / 导航电文）解析：GPS LNAV + BDS。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ladder.util.constants import GTime, datetime_to_gtime


@dataclass
class BroadcastEphemeris:
    sat: str
    toc: datetime
    # clock
    af0: float
    af1: float
    af2: float
    # Kepler / orbit
    iode: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    e: float
    cus: float
    sqrt_a: float
    toe_sow: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    codes_l2: float
    week: float
    tgd: float
    # BDS may use TGD1/TGD2; store extra
    tgd2: float = 0.0
    iodc: float = 0.0

    def toe(self) -> GTime:
        return GTime(int(self.week), float(self.toe_sow))


@dataclass
class NavStore:
    eph: Dict[str, List[BroadcastEphemeris]] = field(default_factory=dict)
    # Klobuchar α0..α3, β0..β3（RINEX3 IONOSPHERIC CORR）
    ion_gps: Optional[np.ndarray] = None
    ion_bds: Optional[np.ndarray] = None

    def add(self, e: BroadcastEphemeris) -> None:
        self.eph.setdefault(e.sat, []).append(e)

    def select(self, sat: str, t: datetime) -> Optional[BroadcastEphemeris]:
        cands = self.eph.get(sat, [])
        if not cands:
            return None
        # choose nearest toc
        best = min(cands, key=lambda e: abs((e.toc - t).total_seconds()))
        if abs((best.toc - t).total_seconds()) > 7200:
            return None
        return best

    def ion_params(self, sat: str) -> Optional[np.ndarray]:
        """返回 8 参数 [α|β]；BDS 优先 BDSA/BDSB，否则回退 GPS。"""
        if sat.startswith("C") and self.ion_bds is not None:
            return self.ion_bds
        return self.ion_gps


def _f(vals: List[str], idx: int) -> float:
    if idx >= len(vals):
        return 0.0
    s = vals[idx].replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_epoch(head: str) -> tuple[str, datetime]:
    sat = head[0:3].strip()
    year = int(head[4:8])
    month = int(head[9:11])
    day = int(head[12:14])
    hour = int(head[15:17])
    minute = int(head[18:20])
    sec = int(float(head[21:23]))
    toc = datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)
    return sat, toc


def _parse_iono_line(line: str) -> Optional[np.ndarray]:
    """解析 RINEX3 `GPSA/GPSB/... IONOSPHERIC CORR` 一行 4 个系数。"""
    try:
        vals = [float(line[5 + i * 12 : 5 + (i + 1) * 12].replace("D", "E").replace("d", "e")) for i in range(4)]
        return np.asarray(vals, dtype=float)
    except ValueError:
        return None


def read_rinex_nav(path: str | Path, systems: tuple[str, ...] = ("G", "C")) -> NavStore:
    path = Path(path)
    store = NavStore()
    gps_a = gps_b = bds_a = bds_b = None
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no END OF HEADER")
            if "IONOSPHERIC CORR" in line:
                tag = line[0:4].strip()
                coeffs = _parse_iono_line(line)
                if coeffs is not None:
                    if tag == "GPSA":
                        gps_a = coeffs
                    elif tag == "GPSB":
                        gps_b = coeffs
                    elif tag == "BDSA":
                        bds_a = coeffs
                    elif tag == "BDSB":
                        bds_b = coeffs
            if "END OF HEADER" in line:
                break
        if gps_a is not None and gps_b is not None:
            store.ion_gps = np.concatenate([gps_a, gps_b])
        if bds_a is not None and bds_b is not None:
            store.ion_bds = np.concatenate([bds_a, bds_b])
        while True:
            line = f.readline()
            if not line:
                break
            if len(line) < 23:
                continue
            sat = line[0:3].strip()
            if not sat or sat[0] not in systems:
                # consume possible continuation blindly if it looks like eph — skip block by reading until next sat line is hard;
                # simpler: if not our system, read 7 more lines typical for GPS
                # Detect by first char
                if line[0] in "GREJCIS":
                    for _ in range(7):
                        nxt = f.readline()
                        if not nxt:
                            break
                continue
            sat, toc = _parse_epoch(line)
            nums = [line[23:42], line[42:61], line[61:80]]
            block = [line]
            for _ in range(7):
                cont = f.readline()
                if not cont:
                    break
                block.append(cont)
                nums.extend([cont[4:23], cont[23:42], cont[42:61], cont[61:80]])
            # GPS/BDS broadcast has 8 lines → up to 29 fields after epoch clock
            # nums currently: af0 af1 af2 + 7*4 = 31 slots possibly
            e = BroadcastEphemeris(
                sat=sat,
                toc=toc,
                af0=_f(nums, 0),
                af1=_f(nums, 1),
                af2=_f(nums, 2),
                iode=_f(nums, 3),
                crs=_f(nums, 4),
                delta_n=_f(nums, 5),
                m0=_f(nums, 6),
                cuc=_f(nums, 7),
                e=_f(nums, 8),
                cus=_f(nums, 9),
                sqrt_a=_f(nums, 10),
                toe_sow=_f(nums, 11),
                cic=_f(nums, 12),
                omega0=_f(nums, 13),
                cis=_f(nums, 14),
                i0=_f(nums, 15),
                crc=_f(nums, 16),
                omega=_f(nums, 17),
                omega_dot=_f(nums, 18),
                idot=_f(nums, 19),
                codes_l2=_f(nums, 20),
                week=_f(nums, 21),
                tgd=_f(nums, 25) if sat[0] == "G" else _f(nums, 25),
                tgd2=_f(nums, 26) if sat[0] == "C" else 0.0,
                iodc=_f(nums, 26) if sat[0] == "G" else _f(nums, 27),
            )
            # BDS week in RINEX is continuous GPS week in RINEX3 usually
            store.add(e)
    return store
