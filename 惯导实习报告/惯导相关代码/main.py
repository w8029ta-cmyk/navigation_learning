"""命令行入口 —— 三个教学实验。

用法：
  python main.py              依次运行实验一、二、三
  python main.py --calib      实验一：IMU 六位置 + 速率法标定
  python main.py --align      实验二：解析粗对准
  python main.py --dataset    实验三：KF-GINS 数据集三层惯导解算

详见 README.md。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ins_nav.simulation import run_dataset_experiment
from ins_nav.calibration import run_calibration_demo
from ins_nav.alignment import run_alignment_demo


def main():
    parser = argparse.ArgumentParser(description="惯性导航课程设计")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--calib", action="store_true", help="实验一：IMU 标定")
    g.add_argument("--align", action="store_true", help="实验二：解析粗对准")
    g.add_argument("--dataset", action="store_true", help="实验三：KF-GINS 数据集")

    parser.add_argument("--config", default=None, help="kf-gins.yaml 路径")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="数据集时长 (s)，yaml endtime=-1 时生效")
    parser.add_argument("--use-align", action="store_true",
                        help="实验三：用静止段粗对准替代 yaml initatt")
    parser.add_argument("--use-calib", action="store_true",
                        help="实验三：用静止段 bias 作为 KF 初值")
    parser.add_argument("--estimate-scale", action="store_true",
                        help="实验三：启用 21 维 KF（估计比例因子 scale）")

    args = parser.parse_args()

    if args.calib:
        run_calibration_demo()
    elif args.align:
        run_alignment_demo()
    elif args.dataset:
        run_dataset_experiment(
            args.config, args.duration,
            use_align=args.use_align, use_calib=args.use_calib,
            estimate_scale=args.estimate_scale,
        )
    else:
        run_calibration_demo()
        print()
        run_alignment_demo()
        print()
        run_dataset_experiment(args.config, args.duration)


if __name__ == "__main__":
    main()
