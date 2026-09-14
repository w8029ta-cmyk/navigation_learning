"""精密星历 SP3 + 钟差 CLK。

位置用 Neville 插值。钟差要自己加 −2 r·v/c²（IGS/CODE CLK 不含这项，漏了 PPP 会偏数米）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ladder.util.constants import CLIGHT, OMEGA_E

# RTKLIB NMAX=10 → 11 点 Neville；5 min SP3 取附近最多 11 点
_SP3_NEVILLE_N = 10


@dataclass
class PreciseStore:
    epochs: List[datetime] = field(default_factory=list)
    # sat -> list aligned with epochs, None if missing
    pos: Dict[str, List[Optional[np.ndarray]]] = field(default_factory=dict)
    clk_epochs: List[datetime] = field(default_factory=list)
    clk: Dict[str, List[Optional[float]]] = field(default_factory=dict)

    def interpolate_pos(self, sat: str, t: datetime) -> Optional[np.ndarray]:
        """SP3 位置 Neville 插值（发射时刻 ECEF）。"""
        return _interp_sp3_nevil(self.epochs, self.pos.get(sat), t)

    def interpolate_clk(self, sat: str, t: datetime) -> Optional[float]:
        series = self.clk.get(sat)
        if not series or len(self.clk_epochs) < 2:
            return None
        return _interp_scalar(self.clk_epochs, series, t)

    def peph2pos(
        self, sat: str, t: datetime, apply_relativity: bool = True
    ) -> Optional[Tuple[np.ndarray, float]]:
        """位置 + 钟差（可选相对论）。返回 (rs_ecef_at_tx, dts_s)；Sagnac 在 geodist。"""
        rs = self.interpolate_pos(sat, t)
        dts = self.interpolate_clk(sat, t)
        if rs is None or dts is None:
            return None
        if apply_relativity:
            t2 = t + timedelta(milliseconds=1)
            rs2 = self.interpolate_pos(sat, t2)
            if rs2 is not None:
                vel = (rs2 - rs) / 1.0e-3
                dts = dts - 2.0 * float(np.dot(rs, vel)) / (CLIGHT * CLIGHT)
        return rs, float(dts)


def _parse_epoch_line(line: str) -> datetime:
    # *  2026  7 19  0  0  0.00000000
    year = int(line[3:7])
    month = int(line[8:10])
    day = int(line[11:13])
    hour = int(line[14:16])
    minute = int(line[17:19])
    sec = float(line[20:31])
    whole = int(sec)
    micro = int(round((sec - whole) * 1e6))
    return datetime(year, month, day, hour, minute, whole, micro, tzinfo=timezone.utc)


def read_sp3(path: str | Path) -> PreciseStore:
    path = Path(path)
    store = PreciseStore()
    cur_t: datetime | None = None
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if line.startswith("*"):
                cur_t = _parse_epoch_line(line)
                store.epochs.append(cur_t)
                for sat in store.pos:
                    store.pos[sat].append(None)
            elif line.startswith("P") and cur_t is not None:
                sat = line[1:4].strip()
                # SP3 often 'PG01' / 'PC11'
                if len(sat) == 3 and sat[0] in "GREJC":
                    pass
                else:
                    sat = line[1:4]
                x = float(line[4:18]) * 1000.0
                y = float(line[18:32]) * 1000.0
                z = float(line[32:46]) * 1000.0
                if sat not in store.pos:
                    store.pos[sat] = [None] * (len(store.epochs) - 1)
                    store.pos[sat].append(np.array([x, y, z]))
                else:
                    # pad
                    while len(store.pos[sat]) < len(store.epochs) - 1:
                        store.pos[sat].append(None)
                    if len(store.pos[sat]) == len(store.epochs) - 1:
                        store.pos[sat].append(np.array([x, y, z]))
                    else:
                        store.pos[sat][-1] = np.array([x, y, z])
    # pad all to same length
    n = len(store.epochs)
    for sat in store.pos:
        while len(store.pos[sat]) < n:
            store.pos[sat].append(None)
    return store


def read_clk(path: str | Path, store: PreciseStore | None = None) -> PreciseStore:
    store = store or PreciseStore()
    path = Path(path)
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        in_data = False
        for line in f:
            if "END OF HEADER" in line:
                in_data = True
                continue
            if not in_data:
                continue
            if not line.startswith("AS "):
                continue
            parts = line.split()
            # AS sat yyyy mm dd hh mm ss.s nval clock ...
            if len(parts) < 10:
                continue
            sat = parts[1]
            try:
                year = int(parts[2])
                month = int(parts[3])
                day = int(parts[4])
                hour = int(parts[5])
                minute = int(parts[6])
                sec = float(parts[7])
                clock = float(parts[9])
            except ValueError:
                continue
            whole = int(sec)
            micro = int(round((sec - whole) * 1e6))
            t = datetime(year, month, day, hour, minute, whole, micro, tzinfo=timezone.utc)
            if not store.clk_epochs or store.clk_epochs[-1] != t:
                store.clk_epochs.append(t)
                for s in store.clk:
                    store.clk[s].append(None)
            if sat not in store.clk:
                store.clk[sat] = [None] * (len(store.clk_epochs) - 1)
                store.clk[sat].append(clock)
            else:
                while len(store.clk[sat]) < len(store.clk_epochs) - 1:
                    store.clk[sat].append(None)
                if len(store.clk[sat]) == len(store.clk_epochs) - 1:
                    store.clk[sat].append(clock)
                else:
                    store.clk[sat][-1] = clock
    n = len(store.clk_epochs)
    for sat in store.clk:
        while len(store.clk[sat]) < n:
            store.clk[sat].append(None)
    return store


def _neville(t: np.ndarray, p: np.ndarray) -> float:
    """Neville 多项式插值（t 为相对秒，求 t=0）。"""
    n = len(t)
    y = p.astype(float).copy()
    for j in range(1, n):
        for i in range(n - j):
            y[i] = (t[i + j] * y[i] - t[i] * y[i + 1]) / (t[i + j] - t[i])
    return float(y[0])


def _interp_sp3_nevil(
    epochs: List[datetime],
    series: Optional[List[Optional[np.ndarray]]],
    t: datetime,
) -> Optional[np.ndarray]:
    """附近最多 11 点 Neville，并按 OMGE*(t−t_k) 归算到插值时刻。"""
    if not series or len(epochs) < 2:
        return None
    ts = np.array([e.timestamp() for e in epochs], dtype=float)
    tt = t.timestamp()
    if tt < ts[0] - 900 or tt > ts[-1] + 900:
        return None
    tt_c = float(np.clip(tt, ts[0], ts[-1]))
    idx = int(np.searchsorted(ts, tt_c) - 1)
    idx = max(0, min(idx, len(ts) - 2))

    # 选以 idx 为中心、有数据的最多 NMAX+1 点
    order = list(range(len(epochs)))
    order.sort(key=lambda i: abs(ts[i] - tt_c))
    chosen: List[int] = []
    for i in order:
        if series[i] is None:
            continue
        chosen.append(i)
        if len(chosen) >= _SP3_NEVILLE_N + 1:
            break
    if len(chosen) < 2:
        for j, p in enumerate(series):
            if p is not None:
                return p.copy()
        return None
    chosen.sort()
    # RTKLIB: t[j]=timediff(time, peph[j].time)=t_interp−t_sample；Neville 求 t=0
    t_rel = np.array([tt_c - ts[i] for i in chosen], dtype=float)
    xyz = np.zeros(3)
    for k in range(3):
        if k < 2:
            # 地球自转：把各 SP3 历元位置转到插值时刻 ECEF
            comps = []
            for i, tr in zip(chosen, t_rel):
                p = series[i]
                assert p is not None
                s, c = np.sin(OMEGA_E * tr), np.cos(OMEGA_E * tr)
                if k == 0:
                    comps.append(c * p[0] - s * p[1])
                else:
                    comps.append(s * p[0] + c * p[1])
            xyz[k] = _neville(t_rel, np.asarray(comps, dtype=float))
        else:
            comps = [float(series[i][2]) for i in chosen]  # type: ignore[index]
            xyz[k] = _neville(t_rel, np.asarray(comps, dtype=float))
    return xyz


def _interp_scalar(epochs: List[datetime], series: List[Optional[float]], t: datetime) -> Optional[float]:
    ts = np.array([e.timestamp() for e in epochs], dtype=float)
    tt = t.timestamp()
    if tt < ts[0]:
        tt = ts[0]
    if tt > ts[-1]:
        tt = ts[-1]
    idx = int(np.searchsorted(ts, tt) - 1)
    idx = max(0, min(idx, len(ts) - 2))
    i0 = idx
    while i0 >= 0 and series[i0] is None:
        i0 -= 1
    i1 = idx + 1
    while i1 < len(series) and series[i1] is None:
        i1 += 1
    if i0 < 0 or i1 >= len(series):
        for j in range(len(series)):
            if series[j] is not None:
                return float(series[j])
        return None
    if ts[i1] == ts[i0]:
        return float(series[i0])
    w = (tt - ts[i0]) / (ts[i1] - ts[i0])
    return float((1 - w) * series[i0] + w * series[i1])
