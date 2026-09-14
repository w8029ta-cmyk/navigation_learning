"""CODE Bias-SINEX OSB（*.BIA）读取：卫星伪距/相位绝对偏差。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from ladder.util.constants import CLIGHT


@dataclass
class OsbStore:
    """key=(prn, obs_code) → bias in meters（已从 ns 转换）。"""

    bias_m: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def get(self, prn: str, code: str, default: float = 0.0) -> float:
        return self.bias_m.get((prn, code), default)

    def correct(self, prn: str, code: str, obs_m: float) -> float:
        """观测值减去 OSB，使与精密钟差参考一致。"""
        return obs_m - self.get(prn, code, 0.0)


def read_bias_sinex(path: str | Path) -> OsbStore:
    """解析 Bias-SINEX `+BIAS/SOLUTION` 段的 OSB 行。"""
    path = Path(path)
    store = OsbStore()
    in_sol = False
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if line.startswith("+BIAS/SOLUTION"):
                in_sol = True
                continue
            if line.startswith("-BIAS/SOLUTION"):
                break
            if not in_sol or not line.startswith(" OSB"):
                continue
            # OSB  SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT EST...
            # 固定宽度大致：类型(1:5) SVN(6:10) PRN(11:14) station(15:24) OBS1(25:29) ...
            parts = line.split()
            if len(parts) < 9:
                continue
            # parts: OSB, SVN, PRN, [STATION?], OBS1, ...
            # 无测站时：OSB G080 G01 C1C 2026:200:00000 2026:201:00000 ns -6.2942 0.0042
            if parts[0] != "OSB":
                continue
            prn = parts[2]
            # station 字段可能为空；OBS1 跟在 PRN 后或测站后
            # 用列切片更稳（SINEX 固定格式）
            obs1 = line[25:29].strip()
            unit = line[65:69].strip() if len(line) > 69 else "ns"
            try:
                val = float(line[70:91].strip())
            except ValueError:
                # fallback split
                try:
                    # find unit token
                    idx_ns = parts.index("ns") if "ns" in parts else -1
                    if idx_ns < 0 or idx_ns + 1 >= len(parts):
                        continue
                    obs1 = parts[3] if parts[3][0] in "CLSD" else parts[4]
                    val = float(parts[idx_ns + 1])
                    unit = "ns"
                except (ValueError, IndexError):
                    continue
            if not obs1 or not prn:
                continue
            if unit == "ns":
                store.bias_m[(prn, obs1)] = val * CLIGHT * 1e-9
            else:
                store.bias_m[(prn, obs1)] = val
    return store
