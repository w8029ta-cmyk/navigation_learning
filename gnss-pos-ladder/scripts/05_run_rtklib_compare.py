#!/usr/bin/env python
"""用本机 RTKLIB 2.4.2 跑 SPP / static RTK / PPP，和我这边结果对比。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ladder.rinex.obs import read_rinex_obs
from ladder.util.config import gps_week_dow, load_config, yyyy_ddd
from ladder.util.coord import ecef_to_enu
from ladder.verify.metrics import save_json


def run_rtklib(exe: Path, conf: Path, out_pos: Path, files: list[Path], ti: float = 300.0) -> Path:
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    # SP3 扩展名小写副本（2.4.2 要求 .sp3/.eph）
    fixed_files = []
    for f in files:
        f = Path(f)
        if f.suffix.upper() == ".SP3" and f.suffix != ".sp3":
            sp3 = f.with_suffix(".sp3")
            if not sp3.exists():
                sp3.write_bytes(f.read_bytes())
            fixed_files.append(sp3)
        else:
            fixed_files.append(f)

    cmd = [
        str(exe),
        "-k",
        str(conf.resolve()),
        "-o",
        str(out_pos.resolve()),
        "-ti",
        str(ti),
        "-e",
        "-t",
    ]
    cmd.extend(str(p.resolve()) for p in fixed_files)
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(exe.parent))
    log = out_pos.with_suffix(".rtklib.log")
    log.write_text(
        "CMD: " + " ".join(cmd) + f"\n\nrc={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        encoding="utf-8",
        errors="ignore",
    )
    if not out_pos.exists() or out_pos.stat().st_size < 80:
        raise RuntimeError(f"RTKLIB failed or empty output: {log}")
    print(f"  -> {out_pos} ({out_pos.stat().st_size} bytes)")
    return out_pos


def parse_xyz_pos(path: Path) -> np.ndarray:
    """返回 Nx3 ECEF。兼容 % 注释头。"""
    rows = []
    for line in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if not line.strip() or line.startswith("%") or line.startswith("#"):
            continue
        parts = line.split()
        # 常见：time... x y z ... 或 GPST ymd hms x y z
        # 找三个连续大数字当 ECEF
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        # ECEF 量级 1e6
        xyz = None
        for i in range(len(nums) - 2):
            if abs(nums[i]) > 1e5 and abs(nums[i + 1]) > 1e5 and abs(nums[i + 2]) > 1e5:
                xyz = nums[i : i + 3]
                break
        if xyz is not None:
            rows.append(xyz)
    if not rows:
        raise RuntimeError(f"no XYZ parsed from {path}")
    return np.asarray(rows, dtype=float)


def summarize_vs_ref(xyz: np.ndarray, ref: np.ndarray) -> dict:
    enus = np.array([ecef_to_enu(p, ref) for p in xyz])
    return {
        "n": int(len(xyz)),
        "mean_enu_m": enus.mean(axis=0).tolist(),
        "rms_enu_m": np.sqrt((enus**2).mean(axis=0)).tolist(),
        "rms_3d_m": float(np.sqrt((enus**2).sum(axis=1).mean())),
        "horizontal_rms_m": float(np.sqrt(np.mean(enus[:, 0] ** 2 + enus[:, 1] ** 2))),
    }


def main() -> None:
    cfg = load_config()
    year, doy = cfg["experiment"]["year"], cfg["experiment"]["doy"]
    yyyy, ddd, _ = yyyy_ddd(year, doy)
    week, _ = gps_week_dow(year, doy)
    data = Path(cfg["paths"]["data_root"])
    rnx = data / "rinex" / f"{yyyy}{ddd}"
    prod = data / "products" / f"{week}"
    exe = Path(cfg["paths"]["rtklib_bin"]) / "rnx2rtkp.exe"
    out_root = Path(cfg["paths"]["results_root"]) / "rtklib_ref"
    out_root.mkdir(parents=True, exist_ok=True)

    rover = next(rnx.glob(f"{cfg['stations']['rover']}*.rnx"))
    base = next(rnx.glob(f"{cfg['stations']['base']}*.rnx"))
    nav = next(rnx.glob("BRDM*.rnx"))
    sp3 = next(prod.glob("*.SP3"))
    clk = next(prod.glob("*.CLK"))
    ref = read_rinex_obs(rover, max_epochs=1).approx_xyz
    base_xyz = read_rinex_obs(base, max_epochs=1).approx_xyz

    conf_spp = ROOT / "config" / "rtklib_spp.conf"
    conf_rtk = ROOT / "config" / "rtklib_rtk.conf"
    conf_ppp = ROOT / "config" / "rtklib_ppp.conf"

    # 间隔 300s，全日可在合理时间内跑完
    spp_pos = run_rtklib(exe, conf_spp, out_root / "spp.pos", [rover, nav], ti=300)
    rtk_pos = run_rtklib(exe, conf_rtk, out_root / "rtk_static.pos", [rover, base, nav], ti=300)
    ppp_pos = run_rtklib(exe, conf_ppp, out_root / "ppp.pos", [rover, nav, sp3, clk], ti=300)

    report = {
        "note": "RTKLIB 2.4.2 对照；navsys=GPS+BDS；参考=RINEX approx",
        "rover_approx_xyz": ref.tolist(),
        "base_approx_xyz": base_xyz.tolist(),
        "approx_baseline_m": float(np.linalg.norm(ref - base_xyz)),
        "rtklib": {},
        "self": {},
    }
    for name, path in [("A_SPP", spp_pos), ("B_RTK", rtk_pos), ("C_PPP", ppp_pos)]:
        xyz = parse_xyz_pos(path)
        report["rtklib"][name] = summarize_vs_ref(xyz, ref)
        if name == "B_RTK":
            bl = np.array([np.linalg.norm(p - base_xyz) for p in xyz])
            report["rtklib"][name]["baseline_mean_m"] = float(bl.mean())
            report["rtklib"][name]["baseline_std_m"] = float(bl.std())
            report["rtklib"][name]["baseline_err_mean_m"] = float(bl.mean() - report["approx_baseline_m"])

    self_sum = Path(cfg["paths"]["results_root"]) / "compare" / "ladder_summary.json"
    if self_sum.exists():
        report["self"] = json.loads(self_sum.read_text(encoding="utf-8")).get("stages", {})

    # 并排差值
    compare = {}
    for k in ("A_SPP", "B_RTK", "C_PPP"):
        if k in report["rtklib"] and k in report["self"]:
            compare[k] = {
                "self_3d_m": report["self"][k].get("rms_3d_m"),
                "rtklib_3d_m": report["rtklib"][k].get("rms_3d_m"),
                "self_horizontal_m": report["self"][k].get("horizontal_rms_m"),
                "rtklib_horizontal_m": report["rtklib"][k].get("horizontal_rms_m"),
            }
    report["side_by_side"] = compare

    out_json = out_root / "rtklib_vs_self.json"
    save_json(report, out_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved", out_json)


if __name__ == "__main__":
    main()
