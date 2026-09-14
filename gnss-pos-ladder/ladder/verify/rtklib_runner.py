"""调本机 RTKLIB 可执行文件做对照（源码不进仓库）。

RTKLIB 2.4.2 的 rnx2rtkp 没有 `-sys`；传了会直接打印 usage 然后失败。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def run_rnx2rtkp(
    rtklib_bin: str | Path,
    obs: Path,
    nav: Path,
    out_pos: Path,
    mode: str = "single",
    base_obs: Optional[Path] = None,
    sp3: Optional[Path] = None,
    clk: Optional[Path] = None,
    elev_mask: float = 15.0,
    interval_sec: float = 0.0,
    ecef: bool = True,
    fix_and_hold: bool = False,
) -> Path:
    """
    mode: single | static | ppp-static | kinematic | dgps
    """
    bin_dir = Path(rtklib_bin)
    exe = bin_dir / "rnx2rtkp.exe"
    if not exe.exists():
        raise FileNotFoundError(exe)
    out_pos = Path(out_pos)
    out_pos.parent.mkdir(parents=True, exist_ok=True)

    mode_map = {
        "single": "0",
        "dgps": "1",
        "kinematic": "2",
        "static": "3",
        "ppp-kinematic": "6",
        "ppp-static": "7",
    }
    pos_mode = mode_map.get(mode, "0")

    # 用绝对路径；不要传 -sys（2.4.2 不支持）
    cmd = [
        str(exe.resolve()),
        "-o",
        str(out_pos.resolve()),
        "-p",
        pos_mode,
        "-m",
        str(elev_mask),
    ]
    if ecef:
        cmd.append("-e")
    if interval_sec and interval_sec > 0:
        cmd.extend(["-ti", str(interval_sec)])
    if fix_and_hold and mode in {"static", "kinematic"}:
        cmd.append("-h")

    cmd.append(str(Path(obs).resolve()))
    if base_obs is not None:
        cmd.append(str(Path(base_obs).resolve()))
    cmd.append(str(Path(nav).resolve()))
    if sp3 is not None:
        # 2.4.2 要求扩展名 .sp3 / .eph；若是 .SP3 先复制小写扩展名副本
        sp3 = Path(sp3)
        if sp3.suffix.lower() == ".sp3" and sp3.suffix != ".sp3":
            link = sp3.with_suffix(".sp3")
            if not link.exists():
                link.write_bytes(sp3.read_bytes())
            sp3 = link
        cmd.append(str(sp3.resolve()))
    if clk is not None:
        cmd.append(str(Path(clk).resolve()))

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(bin_dir))
    log = out_pos.with_suffix(".rtklib.log")
    log.write_text(
        "CMD: "
        + " ".join(cmd)
        + f"\n\nreturncode={proc.returncode}\n\nSTDOUT:\n"
        + (proc.stdout or "")
        + "\nSTDERR:\n"
        + (proc.stderr or ""),
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0 and not out_pos.exists():
        raise RuntimeError(f"rnx2rtkp failed ({proc.returncode}), see {log}")
    if not out_pos.exists() or out_pos.stat().st_size < 50:
        raise RuntimeError(f"rnx2rtkp produced no/empty pos, see {log}")
    return out_pos
