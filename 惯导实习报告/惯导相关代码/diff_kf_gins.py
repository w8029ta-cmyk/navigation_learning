"""将 Python 实验三输出的 NavResult_py.nav 与数据集真值 truth.nav 逐列对比。

默认输入：
  - ours: dataset/NavResult_py.nav（本仓库 run_dataset_experiment 生成）
  - ref:  kf-gins.yaml 中的 truthpath，即 dataset/truth.nav

用法:
  python diff_kf_gins.py
  python diff_kf_gins.py --run
  python diff_kf_gins.py --ours dataset/NavResult_py.nav --ref dataset/truth.nav
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ins_nav.nav_output import interpolate_nav, load_nav_file
from ins_nav.simulation import DEFAULT_CONFIG, load_config, run_dataset_experiment


# .nav 文件第 3–11 列对应的导航量名称
COL_NAMES = [
    "lat(deg)", "lon(deg)", "h(m)",
    "vN(m/s)", "vE(m/s)", "vD(m/s)",
    "roll(deg)", "pitch(deg)", "yaw(deg)",
]


def diff_nav_files(ours_path: Path, ref_path: Path, t_start=None, t_end=None):
    """在重叠时段内对齐两份 .nav，返回各列差值统计及 NED 位置 RMS。"""
    ours = load_nav_file(ours_path)
    ref = load_nav_file(ref_path)

    # 取两文件时间交集；历元时间戳在 .nav 第 2 列（GPS 周内秒）
    t0 = max(ours[0, 1], ref[0, 1]) if t_start is None else t_start
    t1 = min(ours[-1, 1], ref[-1, 1]) if t_end is None else t_end
    times = ours[(ours[:, 1] >= t0) & (ours[:, 1] <= t1), 1]

    diffs = {name: [] for name in COL_NAMES}
    pos_ned = []

    for t in times:
        row_o = interpolate_nav(ours, t)
        row_r = interpolate_nav(ref, t)
        delta = row_o[2:11] - row_r[2:11]
        for name, val in zip(COL_NAMES, delta):
            diffs[name].append(val)

        # 经纬高差转为北-东-地位置误差（m）
        lat_r = math.radians(row_r[2])
        rm = 6378137.0
        dn = (row_o[2] - row_r[2]) * rm
        de = (row_o[3] - row_r[3]) * rm * math.cos(lat_r)
        dd = row_o[4] - row_r[4]
        pos_ned.append([dn, de, dd])

    pos_ned = np.asarray(pos_ned)
    report = {
        "t0": t0,
        "t1": t1,
        "n": len(times),
        "pos_rms_m": float(np.linalg.norm(np.sqrt(np.mean(pos_ned ** 2, axis=0)))),
        "columns": {},
    }
    for name, vals in diffs.items():
        arr = np.asarray(vals)
        report["columns"][name] = {
            "mean": float(np.mean(arr)),
            "max_abs": float(np.max(np.abs(arr))),
            "rms": float(np.sqrt(np.mean(arr ** 2))),
        }
    return report


def print_report(report, ours_label, ref_label):
    """打印逐列均值 / 最大绝对值 / 均方根及合成位置 RMS。"""
    print(f"\n【导航结果对比】{ours_label} 与 {ref_label}")
    print(f"重叠时段：{report['t0']:.1f} – {report['t1']:.1f} s，"
          f"共 {report['n']} 个历元")
    print(f"位置均方根误差（北-东-地近似）：{report['pos_rms_m']:.4f} m")
    print(f"  {'导航量':<14} {'均值':>12} {'最大绝对值':>12} {'均方根':>12}")
    print(f"  {'-' * 54}")
    col_labels = {
        "lat(deg)": "纬度(deg)", "lon(deg)": "经度(deg)", "h(m)": "高度(m)",
        "vN(m/s)": "北向速度", "vE(m/s)": "东向速度", "vD(m/s)": "垂向速度",
        "roll(deg)": "横滚(deg)", "pitch(deg)": "俯仰(deg)", "yaw(deg)": "航向(deg)",
    }
    for name, stats in report["columns"].items():
        label = col_labels.get(name, name)
        print(f"  {label:<14} {stats['mean']:>+12.6f} "
              f"{stats['max_abs']:>12.6f} {stats['rms']:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Python 导航结果与 truth.nav 逐列 diff")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--ours", default=None, help="Python 输出 NavResult_py.nav")
    parser.add_argument("--ref", default=None, help="参考真值 truth.nav（默认读配置文件）")
    parser.add_argument("--run", action="store_true", help="先跑实验三再 diff")
    parser.add_argument("--use-align", action="store_true",
                        help="--run 时：用实验二粗对准替代 yaml 初始姿态")
    parser.add_argument("--use-calib", action="store_true",
                        help="--run 时：用静止段零偏估计替代 yaml IMU 初值")
    parser.add_argument("--estimate-scale", action="store_true",
                        help="--run 时：启用 21 维 KF（额外估计比例因子）")
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG)
    out_dir = Path(cfg["outputpath"])
    ours_path = Path(args.ours) if args.ours else out_dir / "NavResult_py.nav"
    ref_path = Path(args.ref) if args.ref else Path(cfg["truthpath"])

    # 缺少 Python 输出时，可选先跑实验三生成 NavResult_py.nav
    if args.run or not ours_path.exists():
        print("正在运行实验三，生成 NavResult_py.nav …")
        run_dataset_experiment(
            args.config, args.duration,
            use_align=args.use_align, use_calib=args.use_calib,
            estimate_scale=args.estimate_scale,
        )

    if not ours_path.exists():
        print(f"未找到 Python 解算结果文件：{ours_path}")
        return 1
    if not ref_path.exists():
        print(f"未找到参考真值文件：{ref_path}")
        return 1

    report = diff_nav_files(ours_path, ref_path)
    print_report(report, ours_path.name, ref_path.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
