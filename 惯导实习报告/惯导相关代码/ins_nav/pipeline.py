"""将实验一、实验二的输出接入实验三解算流程。

可选功能：
  --use-align  用实验二解析粗对准结果替代 yaml 中的 initatt
  --use-calib  用实验一静止段零偏估计替代 yaml 中的 initgyrbias / initaccbias
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .alignment import coarse_alignment
from .config_loader import KfGinsConfig

STATIC_SEC = 30.0


def extract_static_imu(samples, starttime: float, static_sec: float = STATIC_SEC):
    """从起始静止段提取陀螺角速率与加表比力序列。

    返回 (N, 3) 的 gyro_rates 和 acc_specific_forces，单位 rad/s 与 m/s²。
    """
    t_end = starttime + static_sec
    gyro_rows, acc_rows = [], []
    for t, imu in samples:
        if t > t_end:
            break
        gyro_rows.append(imu.dtheta / imu.dt)
        acc_rows.append(imu.dvel / imu.dt)
    if not gyro_rows:
        raise ValueError("静止段无 IMU 样本")
    return np.asarray(gyro_rows), np.asarray(acc_rows)


def align_initial_attitude(samples, lat_deg: float, starttime: float, n_avg: int = 20):
    """实验二：对静止段 IMU 做解析粗对准，返回欧拉角 [roll, pitch, yaw]（度）。"""
    gyro, acc = extract_static_imu(samples, starttime)
    lat = math.radians(lat_deg)
    euler, _, _ = coarse_alignment(gyro, acc, lat, N_avg=n_avg)
    return np.degrees(euler)


def calibrate_static_bias(samples, cfg_dict):
    """实验一风格：利用静止段与初始姿态估计陀螺/加表零偏。"""
    from .simulation import estimate_bias

    return estimate_bias(cfg_dict, samples)


def apply_pipeline_options(
    kf_cfg: KfGinsConfig,
    cfg_dict: dict,
    imu_samples,
    use_align: bool = False,
    use_calib: bool = False,
):
    """根据命令行选项，覆盖 yaml 中的初始姿态或 IMU 误差初值。"""
    kf_cfg = copy.deepcopy(kf_cfg)
    cfg_dict = dict(cfg_dict)

    if use_align:
        lat_deg = cfg_dict["initpos"][0]
        att_deg = align_initial_attitude(imu_samples, lat_deg, cfg_dict["starttime"])
        cfg_dict["initatt"] = att_deg.tolist()
        kf_cfg.initatt = att_deg.tolist()

    if use_calib:
        bg, ba = calibrate_static_bias(imu_samples, cfg_dict)
        kf_cfg.init_gyro_bias = bg
        kf_cfg.init_acc_bias = ba

    cfg_dict["_kf_cfg"] = kf_cfg
    return kf_cfg, cfg_dict
