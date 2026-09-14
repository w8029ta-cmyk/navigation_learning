"""RINEX 3 观测文件解析（GPS/BDS）。

只够本实验用的 mixed OBS（MO）；未知观测码直接跳过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ObsEpoch:
    time: datetime
    # sat -> {obs_code: value}
    data: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class RinexObs:
    approx_xyz: np.ndarray
    marker: str
    epochs: List[ObsEpoch]
    sys_obs_types: Dict[str, List[str]]
    # ANTENNA: DELTA H/E/N（m）；天线类型（含罩，20 字符）
    ant_delta_hen: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ant_type: str = ""


def _parse_xyz(line: str) -> np.ndarray | None:
    if "APPROX POSITION XYZ" not in line:
        return None
    try:
        x = float(line[0:14])
        y = float(line[14:28])
        z = float(line[28:42])
        return np.array([x, y, z], dtype=float)
    except ValueError:
        return None


def _parse_ant_delta(line: str) -> np.ndarray | None:
    if "ANTENNA: DELTA H/E/N" not in line:
        return None
    try:
        h = float(line[0:14])
        e = float(line[14:28])
        n = float(line[28:42])
        return np.array([h, e, n], dtype=float)
    except ValueError:
        return None


def read_rinex_obs(
    path: str | Path,
    systems: Tuple[str, ...] = ("G", "C"),
    max_epochs: int = 0,
) -> RinexObs:
    path = Path(path)
    approx = np.zeros(3)
    ant_delta = np.zeros(3)
    ant_type = ""
    marker = path.stem[:4].upper()
    sys_types: Dict[str, List[str]] = {}
    epochs: List[ObsEpoch] = []

    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        # header
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"无 END OF HEADER: {path}")
            if "MARKER NAME" in line:
                marker = line[:60].strip() or marker
            if "ANT # / TYPE" in line:
                ant_type = line[20:40].rstrip()
            xyz = _parse_xyz(line)
            if xyz is not None:
                approx = xyz
            hen = _parse_ant_delta(line)
            if hen is not None:
                ant_delta = hen
            if "SYS / # / OBS TYPES" in line:
                sys = line[0].strip()
                n = int(line[3:6])
                types: List[str] = []
                # first line has up to 13 types
                chunk = line[7:58]
                types.extend([chunk[i : i + 4].strip() for i in range(0, len(chunk), 4) if chunk[i : i + 4].strip()])
                while len(types) < n:
                    cont = f.readline()
                    chunk = cont[7:58]
                    types.extend([chunk[i : i + 4].strip() for i in range(0, len(chunk), 4) if chunk[i : i + 4].strip()])
                if sys in systems:
                    sys_types[sys] = types[:n]
            if "END OF HEADER" in line:
                break

        # body
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(">"):
                continue
            # > YYYY MM DD HH MM SS.sssssss  epoch_flag  num_sats
            year = int(line[2:6])
            month = int(line[7:9])
            day = int(line[10:12])
            hour = int(line[13:15])
            minute = int(line[16:18])
            sec = float(line[19:29])
            flag = int(line[31:32] or 0)
            nsat = int(line[32:35])
            if flag > 1:
                # skip special epochs but consume sat lines
                for _ in range(nsat):
                    f.readline()
                continue
            whole = int(sec)
            micro = int(round((sec - whole) * 1e6))
            t = datetime(year, month, day, hour, minute, whole, micro, tzinfo=timezone.utc)
            ep = ObsEpoch(time=t, data={})
            for _ in range(nsat):
                sline = f.readline()
                if not sline:
                    break
                sat = sline[0:3].strip()
                if not sat or sat[0] not in systems:
                    continue
                sys = sat[0]
                types = sys_types.get(sys, [])
                vals: Dict[str, float] = {}
                # RINEX3: each obs 16 chars (14 value + 1 LLI + 1 SSI)
                for i, code in enumerate(types):
                    start = 3 + i * 16
                    field = sline[start : start + 14]
                    if not field.strip():
                        continue
                    try:
                        vals[code] = float(field)
                    except ValueError:
                        continue
                if vals:
                    ep.data[sat] = vals
            if ep.data:
                epochs.append(ep)
            if max_epochs and len(epochs) >= max_epochs:
                break

    return RinexObs(
        approx_xyz=approx,
        marker=marker,
        epochs=epochs,
        sys_obs_types=sys_types,
        ant_delta_hen=ant_delta,
        ant_type=ant_type,
    )
