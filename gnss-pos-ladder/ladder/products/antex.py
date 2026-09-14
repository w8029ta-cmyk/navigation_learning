"""ANTEX：读卫星/接收机 PCO（NORTH/EAST/UP 或 X/Y/Z）。

完整 PCV 网格没解析，静态 PPP 先用 PCO。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PcoEntry:
    """off: 卫星为星固系 XYZ (m)；接收机为 ENU=E/N/U (m)，按频率存 L1/L2。"""

    type_name: str
    kind: str  # "sat" | "rcv"
    sat: str = ""
    # freq code -> (e/n/u or x/y/z)
    off: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None


@dataclass
class AntexStore:
    sats: Dict[str, List[PcoEntry]] = field(default_factory=dict)  # G01 -> entries
    rcvs: Dict[str, PcoEntry] = field(default_factory=dict)  # antenna type -> entry

    def sat_pco_l1(self, sat: str, t: datetime | None = None) -> Optional[Tuple[float, float, float]]:
        entries = self.sats.get(sat) or []
        for e in reversed(entries):  # 较新的在后
            if t is not None and e.valid_from and t < e.valid_from:
                continue
            if t is not None and e.valid_until and t >= e.valid_until:
                continue
            for key in ("G01", "L1", "C01", "E01", "1"):
                if key in e.off:
                    return e.off[key]
            if e.off:
                return next(iter(e.off.values()))
        return None

    def rcv_pco_enu(self, ant_type: str) -> Optional[Tuple[float, float, float]]:
        """返回接收机 L1 PCO 的 (E, N, U)。缺则试 L2；IF 近似用 L1。"""
        key = ant_type.strip().upper()
        e = self.rcvs.get(key)
        if e is None:
            for k, v in self.rcvs.items():
                if k.startswith(key[:16]):
                    e = v
                    break
        if e is None:
            return None
        for fk in ("G01", "L1", "C01", "1"):
            if fk in e.off:
                n, e_, u = e.off[fk]  # ANTEX 顺序 NORTH EAST UP
                return (e_, n, u)
        if e.off:
            n, e_, u = next(iter(e.off.values()))
            return (e_, n, u)
        return None


def _parse_date(s: str) -> Optional[datetime]:
    s = s.strip()
    if not s or s.startswith(":"):
        return None
    parts = s.split()
    if len(parts) < 3:
        return None
    try:
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        hh = int(parts[3]) if len(parts) > 3 else 0
        mm = int(parts[4]) if len(parts) > 4 else 0
        ss = float(parts[5]) if len(parts) > 5 else 0.0
        whole = int(ss)
        micro = int(round((ss - whole) * 1e6))
        return datetime(y, m, d, hh, mm, whole, micro, tzinfo=timezone.utc)
    except ValueError:
        return None


def read_antex(
    path: str | Path,
    want_sats: Optional[set[str]] = None,
    want_rcv_types: Optional[set[str]] = None,
) -> AntexStore:
    """读 ANTEX；可只保留关心的卫星/天线类型以省内存。"""
    path = Path(path)
    store = AntexStore()
    want_rcv = {t.strip().upper() for t in (want_rcv_types or set())}
    cur: Optional[PcoEntry] = None
    freq = ""

    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            label = line[60:].strip() if len(line) >= 60 else ""
            if label == "START OF ANTENNA":
                cur = PcoEntry(type_name="", kind="")
                freq = ""
            elif cur is None:
                continue
            elif label == "TYPE / SERIAL NO":
                typ = line[0:20].rstrip()
                serial = line[20:40].strip()
                cur.type_name = typ
                if serial and len(serial) <= 3 and serial[0] in "GREJCSI":
                    cur.kind = "sat"
                    cur.sat = serial[:3]
                else:
                    cur.kind = "rcv"
            elif label == "VALID FROM":
                cur.valid_from = _parse_date(line[:60])
            elif label == "VALID UNTIL":
                cur.valid_until = _parse_date(line[:60])
            elif label == "START OF FREQUENCY":
                freq = line[3:6].strip()
            elif label == "NORTH / EAST / UP" and freq:
                try:
                    n = float(line[0:10]) / 1000.0
                    e = float(line[10:20]) / 1000.0
                    u = float(line[20:30]) / 1000.0
                    cur.off[freq] = (n, e, u)
                except ValueError:
                    pass
            elif label == "END OF ANTENNA":
                if cur.kind == "sat":
                    if want_sats is None or cur.sat in want_sats:
                        store.sats.setdefault(cur.sat, []).append(cur)
                elif cur.kind == "rcv":
                    key = cur.type_name.strip().upper()
                    if not want_rcv or key in want_rcv or any(key.startswith(w[:16]) for w in want_rcv):
                        store.rcvs[key] = cur
                cur = None
    return store
