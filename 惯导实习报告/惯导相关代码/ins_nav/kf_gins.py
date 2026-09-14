"""15/21 维 INS/GNSS 松组合扩展卡尔曼滤波（KF-GINS 风格）。

状态向量（15 维）：
  δr(3)   — 位置误差（NED 米制）
  δv(3)   — 速度误差
  φ(3)    — 姿态误差（小角度）
  δε_g(3) — 陀螺仪零偏误差
  δ∇_a(3) — 加速度计零偏误差

可选扩展至 21 维（cfg.estimate_scale=True）：
  δsg(3)  — 陀螺仪比例因子误差
  δsa(3)  — 加速度计比例因子误差

每个 IMU 历元：先在线补偿 IMU → 机械编排推导航状态 → 时间更新 P →
GNSS 量测到达时做位置更新并反馈修正导航状态与 IMU 误差估计。
"""

from __future__ import annotations

import math

import numpy as np

from .attitude import quat_multiply, quat_to_dcm, rotvec_to_quat
from .config_loader import KfGinsConfig
from .coordinates import meridian_radius, prime_vertical_radius
from .mechanization import IMUSample, NavState, earth_rates, mechanization_step, skew


class InsGnssState:
    """导航状态 + IMU 误差估计（机体系零偏/比例因子）。"""

    def __init__(self, nav: NavState, gyro_bias, acc_bias, gyro_scale=None, acc_scale=None):
        self.nav = nav
        self.gyro_bias = np.asarray(gyro_bias, dtype=float).reshape(3)
        self.acc_bias = np.asarray(acc_bias, dtype=float).reshape(3)
        self.gyro_scale = np.zeros(3) if gyro_scale is None else np.asarray(gyro_scale, dtype=float).reshape(3)
        self.acc_scale = np.zeros(3) if acc_scale is None else np.asarray(acc_scale, dtype=float).reshape(3)

    def copy(self):
        return InsGnssState(
            NavState(
                self.nav.lat, self.nav.lon, self.nav.h,
                self.nav.v_n.copy(), self.nav.q_b_n.copy(),
            ),
            self.gyro_bias.copy(),
            self.acc_bias.copy(),
            self.gyro_scale.copy(),
            self.acc_scale.copy(),
        )


def compensate_imu_sample(
    imu: IMUSample, gyro_bias, acc_bias, gyro_scale=None, acc_scale=None,
) -> IMUSample:
    """用当前滤波估计的零偏/比例因子对原始 IMU 增量做在线补偿。

    补偿公式（与 KF-GINS 一致）：
      Δθ' = (Δθ - ε_g·dt) / (1 + sg)
      Δv' = (Δv - ∇_a·dt) / (1 + sa)
    """
    bg = np.asarray(gyro_bias, dtype=float).reshape(3)
    ba = np.asarray(acc_bias, dtype=float).reshape(3)
    sg = np.zeros(3) if gyro_scale is None else np.asarray(gyro_scale, dtype=float).reshape(3)
    sa = np.zeros(3) if acc_scale is None else np.asarray(acc_scale, dtype=float).reshape(3)
    dt = imu.dt
    return IMUSample(
        dt,
        (imu.dtheta - bg * dt) / (1.0 + sg),
        (imu.dvel - ba * dt) / (1.0 + sa),
    )


def initial_ins_gnss_state(cfg: KfGinsConfig) -> InsGnssState:
    """从 yaml 配置构造初始导航状态与 IMU 误差初值。"""
    from .attitude import euler_to_quat

    lat, lon, h = cfg.initpos
    roll, pitch, yaw = cfg.initatt
    nav = NavState(
        math.radians(lat), math.radians(lon), h,
        np.asarray(cfg.initvel, dtype=float),
        euler_to_quat(math.radians(roll), math.radians(pitch), math.radians(yaw)),
    )
    return InsGnssState(
        nav, cfg.init_gyro_bias, cfg.init_acc_bias,
        cfg.init_gyro_scale, cfg.init_acc_scale,
    )


def _initial_covariance(cfg: KfGinsConfig) -> np.ndarray:
    """构造初始误差协方差 P0（对角阵，来自 yaml 中的标准差）。"""
    n = cfg.imu_noise
    parts = [
        cfg.initpos_std ** 2,
        cfg.initvel_std ** 2,
        cfg.initatt_std ** 2,
        n.gbstd ** 2,
        n.abstd ** 2,
    ]
    if cfg.estimate_scale:
        parts.extend([n.gsstd ** 2, n.asstd ** 2])
    return np.diag(np.concatenate(parts))


def _build_phi_q(state: InsGnssState, imu: IMUSample, cfg: KfGinsConfig):
    """构造离散时间状态转移矩阵 Φ 与过程噪声协方差 Qd。

    基于 15/21 维误差状态线性化模型，一阶近似 Φ ≈ I + F·dt。
    IMU 零偏/比例因子建模为一阶马尔可夫过程，相关时间由 yaml corrtime 给定。
    """
    dt = imu.dt
    nav = state.nav
    n_state = cfg.state_dim
    cbn = quat_to_dcm(nav.q_b_n)
    w_b = imu.dtheta / dt
    f_b = imu.dvel / dt
    f_n = cbn @ f_b
    wie, wen = earth_rates(nav.lat, nav.h, nav.v_n)
    tau = cfg.imu_noise.corrtime_s

    f_mat = np.zeros((n_state, n_state))
    f_mat[0:3, 3:6] = np.eye(3)
    f_mat[3:6, 3:6] = -skew(2.0 * wie + wen)
    f_mat[3:6, 6:9] = skew(f_n)
    f_mat[3:6, 12:15] = cbn
    f_mat[6:9, 6:9] = -skew(wie + wen)
    f_mat[6:9, 9:12] = -cbn
    f_mat[9:12, 9:12] = -np.eye(3) / tau
    f_mat[12:15, 12:15] = -np.eye(3) / tau

    if cfg.estimate_scale:
        f_mat[3:6, 18:21] = -cbn @ np.diag(f_b)
        f_mat[6:9, 15:18] = -cbn @ np.diag(w_b)
        f_mat[15:18, 15:18] = -np.eye(3) / tau
        f_mat[18:21, 18:21] = -np.eye(3) / tau

    phi = np.eye(n_state) + f_mat * dt

    n = cfg.imu_noise
    qd = np.zeros((n_state, n_state))
    qd[6:9, 6:9] = np.diag(n.arw ** 2 * dt)
    qd[3:6, 3:6] = np.diag(n.vrw ** 2 * dt)
    beta = 2.0 / tau
    qd[9:12, 9:12] = np.diag(n.gbstd ** 2 * beta * (1.0 - np.exp(-beta * dt)))
    qd[12:15, 12:15] = np.diag(n.abstd ** 2 * beta * (1.0 - np.exp(-beta * dt)))
    if cfg.estimate_scale:
        qd[15:18, 15:18] = np.diag(n.gsstd ** 2 * beta * (1.0 - np.exp(-beta * dt)))
        qd[18:21, 18:21] = np.diag(n.asstd ** 2 * beta * (1.0 - np.exp(-beta * dt)))
    return phi, qd


def _gnss_innovation(state: InsGnssState, gnss_row, antlever):
    """计算 GNSS 位置量测新息 z = dr_imu + C_b^n · l（含天线杆臂）。"""
    nav = state.nav
    g_lat, g_lon, g_h = gnss_row[1], gnss_row[2], gnss_row[3]
    rm = meridian_radius(nav.lat)
    rn = prime_vertical_radius(nav.lat)
    rm_h = rm + nav.h
    rn_h = rn + nav.h
    cos_lat = np.cos(nav.lat)

    dr_imu = np.array([
        (nav.lat - g_lat) * rm_h,
        (nav.lon - g_lon) * rn_h * cos_lat,
        g_h - nav.h,
    ])
    cbn = quat_to_dcm(nav.q_b_n)
    l_n = cbn @ np.asarray(antlever, dtype=float).reshape(3)
    return dr_imu + l_n, rm_h, rn_h, cos_lat, l_n


def _gnss_update(state: InsGnssState, p: np.ndarray, gnss_row, antlever, n_state: int):
    """GNSS 位置量测更新：计算卡尔曼增益并反馈修正导航状态。"""
    z, rm_h, rn_h, cos_lat, l_n = _gnss_innovation(state, gnss_row, antlever)
    h_mat = np.zeros((3, n_state))
    h_mat[0:3, 0:3] = np.eye(3)
    h_mat[0:3, 6:9] = skew(l_n)

    r_mat = np.diag(gnss_row[4:7] ** 2)
    s_mat = h_mat @ p @ h_mat.T + r_mat
    k_gain = p @ h_mat.T @ np.linalg.inv(s_mat)
    x = k_gain @ z

    i_kh = np.eye(n_state) - k_gain @ h_mat
    p_new = i_kh @ p @ i_kh.T + k_gain @ r_mat @ k_gain.T

    _feedback(state, x, rm_h, rn_h, cos_lat, estimate_scale=(n_state == 21))
    return p_new


def _feedback(state: InsGnssState, x: np.ndarray, rm_h: float, rn_h: float, cos_lat: float,
                estimate_scale: bool):
    """将滤波估计的误差状态 x 反馈到导航状态与 IMU 误差参数。"""
    nav = state.nav
    nav.lat -= x[0] / rm_h
    nav.lon -= x[1] / (rn_h * cos_lat)
    nav.h += x[2]
    nav.v_n -= x[3:6]
    nav.q_b_n = quat_multiply(rotvec_to_quat(x[6:9]), nav.q_b_n)
    state.gyro_bias += x[9:12]
    state.acc_bias += x[12:15]
    if estimate_scale:
        state.gyro_scale += x[15:18]
        state.acc_scale += x[18:21]


def run_kf_gins(
    samples, state0: InsGnssState, cfg: KfGinsConfig, gnss, datarate_hz,
    error_callback=None, record_callback=None,
):
    """主滤波循环：在线 IMU 补偿 + 机械编排 + EKF 时间/量测更新。

    参数
    ----
    samples : list of (time, IMUSample)
        原始 IMU 增量序列
    state0 : InsGnssState
        初始导航状态与 IMU 误差
    cfg : KfGinsConfig
        yaml 配置（噪声、杆臂、初值等）
    gnss : ndarray
        GNSS 位置量测 [time, lat, lon, h, std_n, std_e, std_d]
    datarate_hz : int
        IMU 采样率，用于按秒记录导航结果
    """
    state = state0.copy()
    n_state = cfg.state_dim
    p_mat = _initial_covariance(cfg)
    gnss_i = 0
    imu_pre = None
    t0 = tf = None
    n = 0
    nav_records = []
    imu_error_records = []

    for t, imu_raw in samples:
        if imu_pre is None:
            t0 = t
            imu_pre = imu_raw
            continue

        sg = state.gyro_scale if cfg.estimate_scale else None
        sa = state.acc_scale if cfg.estimate_scale else None
        imu = compensate_imu_sample(imu_raw, state.gyro_bias, state.acc_bias, sg, sa)
        state.nav = mechanization_step(state.nav, imu_pre, imu)
        n += 1
        tf = t

        phi, qd = _build_phi_q(state, imu, cfg)
        p_mat = phi @ p_mat @ phi.T + qd

        while gnss_i < len(gnss) and gnss[gnss_i, 0] <= t + 0.002:
            p_mat = _gnss_update(state, p_mat, gnss[gnss_i], cfg.antlever, n_state)
            gnss_i += 1

        if n % datarate_hz == 0:
            if error_callback:
                error_callback(t, state.nav)
            if record_callback:
                record_callback(t, state)
            nav_records.append((t, copy_nav(state.nav)))
            imu_error_records.append((t, state.copy()))

        imu_pre = imu_raw

    return state, p_mat, t0, tf, nav_records, imu_error_records


def copy_nav(nav: NavState) -> NavState:
    return NavState(nav.lat, nav.lon, nav.h, nav.v_n.copy(), nav.q_b_n.copy())
