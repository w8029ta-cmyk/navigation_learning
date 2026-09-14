"""IMU 确定性误差模型与标定演示（实验一）。

误差方程：y = K · x + b
  K = I + S，S 含比例因子误差与交轴耦合（非对角元）
  b 为零偏
"""

import math

import numpy as np

OMEGA_IE = 7.292115e-5  # rad/s，地球自转角速率


def six_position_acc_truth(g=9.794):
    """六位置加速度计理论比力（载体系，NED 约定，单位 m/s^2）。"""
    return np.array([
        [0.0, 0.0, g],       # 下
        [0.0, 0.0, -g],      # 上
        [0.0, g, 0.0],       # 右
        [0.0, -g, 0.0],      # 左
        [g, 0.0, 0.0],       # 前
        [-g, 0.0, 0.0],      # 后
    ])


def six_position_rotations():
    """与 six_position_acc_truth 对应的导航系→载体系旋转矩阵。"""
    def _rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def _ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    half_pi = 0.5 * math.pi
    return [
        np.eye(3),              # 下
        _rx(math.pi),           # 上
        _rx(-half_pi),          # 右
        _rx(half_pi),           # 左
        _ry(half_pi),           # 前
        _ry(-half_pi),          # 后
    ]


def earth_rate_ned(lat_deg=30.0):
    """地球自转角速度在导航系 NED 下的投影 (rad/s)。"""
    lat = math.radians(lat_deg)
    return np.array([
        OMEGA_IE * math.cos(lat),
        0.0,
        OMEGA_IE * math.sin(lat),
    ])


def six_position_gyro_truth(lat_deg=30.0):
    """六位置陀螺理论角速度：各姿态下地球自转在载体系的分量。"""
    w_n = earth_rate_ned(lat_deg)
    return np.array([R @ w_n for R in six_position_rotations()])


def build_calib_matrix(scale_ppm, cross_ppm):
    """由对角比例因子 (ppm) 与交轴耦合 (ppm) 构造标定矩阵 K。"""
    scale_ppm = np.asarray(scale_ppm, dtype=float).reshape(3)
    cross_ppm = np.asarray(cross_ppm, dtype=float).reshape(3, 3)
    K = np.eye(3)
    for i in range(3):
        K[i, i] = 1.0 + scale_ppm[i] * 1e-6
        for j in range(3):
            if i != j:
                K[i, j] = cross_ppm[i, j] * 1e-6
    return K


def apply_calib_model(true_values, K, bias, noise_std=None):
    """注入完整误差模型 y = K·x + b，可选加性高斯噪声。"""
    true_values = np.asarray(true_values, dtype=float).reshape(-1, 3)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    bias = np.asarray(bias, dtype=float).reshape(3)
    measured = (true_values @ K.T) + bias.reshape(1, 3)
    if noise_std is not None:
        noise_std = np.asarray(noise_std, dtype=float).reshape(3)
        measured = measured + np.random.normal(0.0, noise_std, size=measured.shape)
    return measured


def six_position_calibrate(truth, measured):
    """六位置最小二乘标定：y = K·x + b。

    truth / measured: (N, 3)，N 通常为 6
    返回 (K, bias, info)
    """
    truth = np.asarray(truth, dtype=float).reshape(-1, 3)
    measured = np.asarray(measured, dtype=float).reshape(-1, 3)
    A = np.hstack([truth, np.ones((truth.shape[0], 1))])
    Kb = np.zeros((3, 4))
    for axis in range(3):
        sol, *_ = np.linalg.lstsq(A, measured[:, axis], rcond=None)
        Kb[axis] = sol
    K_est = Kb[:, :3]
    b_est = Kb[:, 3]
    residual = measured - A @ Kb.T
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((measured - measured.mean(axis=0)) ** 2)
    R_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return K_est, b_est, {
        "K": K_est,
        "bias": b_est,
        "scale": np.diag(K_est) - 1.0,
        "cross": K_est - np.diag(np.diag(K_est)),
        "residual": residual,
        "R_sq": R_sq,
    }


def compensate_imu(measured, K, bias):
    """补偿算法：x = K^{-1} · (y - b)。"""
    measured = np.asarray(measured, dtype=float).reshape(-1, 3)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    bias = np.asarray(bias, dtype=float).reshape(3)
    return (measured - bias.reshape(1, 3)) @ np.linalg.inv(K).T


def rad_s_to_deg_h(rad_s):
    """rad/s -> deg/h（报告常用单位）。"""
    return np.asarray(rad_s, dtype=float) * (180.0 / math.pi) * 3600.0


def estimate_gyro_bias_static(truth, measured):
    """陀螺静态六位置法：在已知地球自转真值下求零偏。"""
    truth = np.asarray(truth, dtype=float).reshape(-1, 3)
    measured = np.asarray(measured, dtype=float).reshape(-1, 3)
    return np.mean(measured - truth, axis=0)


def build_gyro_rate_dataset(K_true, bias, rate_deg_s=100.0, n_samples=500,
                              noise_std=None):
    """陀螺速率法：三轴正反转，生成 (truth, measured) 配对。"""
    rate_truths, rate_meas = [], []
    for axis in range(3):
        for sign in (+1, -1):
            omega = np.zeros(3)
            omega[axis] = sign * rate_deg_s * math.pi / 180.0
            samples = np.tile(omega, (n_samples, 1))
            y = apply_calib_model(samples, K_true, bias, noise_std=noise_std)
            rate_truths.append(omega)
            rate_meas.append(y.mean(axis=0))
    return np.array(rate_truths), np.array(rate_meas)


def run_calibration_experiment(seed=42, g=9.794, lat_deg=30.0, N_per_pos=1000):
    """运行完整标定流程，返回报告与可视化所需的全部中间量。"""
    rng = np.random.default_rng(seed)

    true_acc_bias = np.array([30.0, -20.0, 10.0]) * 1e-5
    true_gyro_bias = np.array([0.5, -0.3, 0.2]) * 1e-4
    true_acc_scale = np.array([300.0, -150.0, 200.0])
    true_gyro_scale = np.array([-200.0, 100.0, 150.0])
    true_acc_cross = np.array([
        [0.0, 80.0, -50.0],
        [60.0, 0.0, 70.0],
        [-40.0, 90.0, 0.0],
    ])
    true_gyro_cross = np.array([
        [0.0, 120.0, -80.0],
        [100.0, 0.0, 150.0],
        [-60.0, 110.0, 0.0],
    ])

    K_acc_true = build_calib_matrix(true_acc_scale, true_acc_cross)
    K_gyro_true = build_calib_matrix(true_gyro_scale, true_gyro_cross)

    acc_truth = six_position_acc_truth(g)
    gyro_truth_static = six_position_gyro_truth(lat_deg)
    pos_labels = ["下(+D)", "上(-D)", "右(+E)", "左(-E)", "前(+N)", "后(-N)"]

    acc_raw_all, acc_truth_all = [], []
    gyro_raw_all, gyro_truth_all = [], []

    for a_true, w_true in zip(acc_truth, gyro_truth_static):
        a_samples = np.tile(a_true, (N_per_pos, 1))
        w_samples = np.tile(w_true, (N_per_pos, 1))
        a_noise = rng.normal(0.0, 1e-4, size=a_samples.shape)
        w_noise = rng.normal(0.0, 1e-7, size=w_samples.shape)
        a_raw = apply_calib_model(a_samples, K_acc_true, true_acc_bias) + a_noise
        w_raw = apply_calib_model(w_samples, K_gyro_true, true_gyro_bias) + w_noise
        acc_raw_all.append(a_raw)
        acc_truth_all.append(a_samples)
        gyro_raw_all.append(w_raw)
        gyro_truth_all.append(w_samples)

    acc_raw = np.vstack(acc_raw_all)
    acc_truth_flat = np.vstack(acc_truth_all)
    gyro_raw = np.vstack(gyro_raw_all)
    gyro_truth_flat = np.vstack(gyro_truth_all)

    acc_meas_pos = acc_raw.reshape(6, N_per_pos, 3).mean(axis=1)
    gyro_meas_pos = gyro_raw.reshape(6, N_per_pos, 3).mean(axis=1)

    # 加速度计：六位置法 -> 完整 K, b
    K_acc_est, b_acc_est, acc_info = six_position_calibrate(acc_truth, acc_meas_pos)

    # 陀螺：静态六位置 -> 零偏；速率法 -> K, b（比例因子 + 交轴耦合）
    b_gyro_static = estimate_gyro_bias_static(gyro_truth_static, gyro_meas_pos)
    rate_truth, rate_meas = build_gyro_rate_dataset(
        K_gyro_true, true_gyro_bias,
        rate_deg_s=100.0, n_samples=500,
        noise_std=np.array([1e-6, 1e-6, 1e-6]),
    )
    K_gyro_est, b_gyro_est, gyro_rate_info = six_position_calibrate(rate_truth, rate_meas)

    acc_comp = compensate_imu(acc_raw, K_acc_est, b_acc_est)
    gyro_comp = compensate_imu(gyro_raw, K_gyro_est, b_gyro_est)

    acc_cmp = compare_compensation(acc_truth_flat, acc_raw, acc_comp,
                                   unit_scale=1e5, unit_name="mGal")
    gyro_cmp = compare_compensation(gyro_truth_flat, gyro_raw, gyro_comp,
                                    unit_scale=1.0, unit_name="deg/h",
                                    extra_scale=rad_s_to_deg_h(1.0))

    pos_acc_err_before = []
    pos_acc_err_after = []
    for i in range(6):
        a_t = acc_truth[i]
        a_r = acc_meas_pos[i]
        a_c = compensate_imu(a_r.reshape(1, 3), K_acc_est, b_acc_est)[0]
        pos_acc_err_before.append(np.linalg.norm(a_r - a_t))
        pos_acc_err_after.append(np.linalg.norm(a_c - a_t))

    return {
        "g": g, "lat_deg": lat_deg, "pos_labels": pos_labels,
        "true_acc_bias": true_acc_bias, "true_gyro_bias": true_gyro_bias,
        "true_acc_scale": true_acc_scale, "true_gyro_scale": true_gyro_scale,
        "true_acc_cross": true_acc_cross, "true_gyro_cross": true_gyro_cross,
        "K_acc_true": K_acc_true, "K_gyro_true": K_gyro_true,
        "K_acc_est": K_acc_est, "K_gyro_est": K_gyro_est,
        "b_acc_est": b_acc_est, "b_gyro_est": b_gyro_est,
        "b_gyro_static": b_gyro_static,
        "acc_info": acc_info, "gyro_rate_info": gyro_rate_info,
        "acc_truth": acc_truth, "acc_meas_pos": acc_meas_pos,
        "gyro_truth_static": gyro_truth_static, "gyro_meas_pos": gyro_meas_pos,
        "rate_truth": rate_truth, "rate_meas": rate_meas,
        "acc_raw": acc_raw, "acc_comp": acc_comp, "acc_truth_flat": acc_truth_flat,
        "gyro_raw": gyro_raw, "gyro_comp": gyro_comp, "gyro_truth_flat": gyro_truth_flat,
        "acc_compare": acc_cmp, "gyro_compare": gyro_cmp,
        "pos_acc_err_before": np.array(pos_acc_err_before),
        "pos_acc_err_after": np.array(pos_acc_err_after),
    }


def compare_compensation(truth, raw, compensated, unit_scale=1.0, unit_name="",
                         extra_scale=1.0):
    """对比补偿前后相对真值的均方根误差。"""
    truth = np.asarray(truth, dtype=float).reshape(-1, 3)
    raw = np.asarray(raw, dtype=float).reshape(-1, 3)
    compensated = np.asarray(compensated, dtype=float).reshape(-1, 3)
    scale = unit_scale * extra_scale
    err_raw = (raw - truth) * scale
    err_comp = (compensated - truth) * scale
    rmse_raw = np.sqrt(np.mean(err_raw ** 2, axis=0))
    rmse_comp = np.sqrt(np.mean(err_comp ** 2, axis=0))
    return {
        "rmse_raw": rmse_raw,
        "rmse_comp": rmse_comp,
        "rmse_raw_total": np.sqrt(np.mean(np.sum(err_raw ** 2, axis=1))),
        "rmse_comp_total": np.sqrt(np.mean(np.sum(err_comp ** 2, axis=1))),
        "unit_name": unit_name,
    }


def _print_matrix(name, mat, unit=1.0, fmt=".2f"):
    print(f"  {name}:")
    for row in mat * unit:
        print(f"    [{', '.join(f'{v:{fmt}}' for v in row)}]")


def _print_K_as_ppm(name, K):
    """以 ppm 打印 K 相对单位阵的偏差（对角=比例因子，非对角=交轴耦合）。"""
    _print_matrix(name, K - np.eye(3), unit=1e6, fmt="+.2f")


def _print_compare_result(title, cmp_result, axis_names=("X", "Y", "Z")):
    unit = cmp_result["unit_name"]
    print(f"\n{title}")
    print(f"  {'轴':<4} {'补偿前 RMSE':>16} {'补偿后 RMSE':>16} {'改善比':>10}")
    for i, ax in enumerate(axis_names):
        before = cmp_result["rmse_raw"][i]
        after = cmp_result["rmse_comp"][i]
        ratio = before / after if after > 1e-15 else float("inf")
        print(f"  {ax:<4} {before:16.4f} {after:16.4f} {ratio:10.1f}x  ({unit})")
    b_tot = cmp_result["rmse_raw_total"]
    a_tot = cmp_result["rmse_comp_total"]
    ratio_tot = b_tot / a_tot if a_tot > 1e-15 else float("inf")
    print(f"  {'合':<4} {b_tot:16.4f} {a_tot:16.4f} {ratio_tot:10.1f}x  ({unit})")


def run_calibration_demo():
    """实验一：六位置标定 + 速率法 + 补偿（终端输出）。"""
    print("\n【实验一】IMU 标定与误差补偿（六位置法 + 速率法）")
    print("误差模型：y = K * x + b，K 含比例因子与交轴耦合")
    print("（与任务书一致：加表六位置法；陀螺静态六位置求零偏，速率法求比例因子/交轴耦合）")

    result = run_calibration_experiment(seed=42)

    acc_info = result["acc_info"]
    gyro_info = result["gyro_rate_info"]

    print("\n── 加速度计六位置标定 ──")
    print(f"  拟合优度 R^2 = {acc_info['R_sq']:.6f}")
    print(f"  零偏估计 (mGal)：  [{', '.join(f'{v*1e5:+.3f}' for v in result['b_acc_est'])}]")
    print(f"  零偏真值 (mGal)：  [{', '.join(f'{v*1e5:+.3f}' for v in result['true_acc_bias'])}]")
    print(f"  比例因子估计 (ppm)：[{', '.join(f'{v*1e6:+.2f}' for v in acc_info['scale'])}]")
    print(f"  比例因子真值 (ppm)：[{', '.join(f'{v:+.2f}' for v in result['true_acc_scale'])}]")
    _print_K_as_ppm("K-I 估计 (ppm)", result["K_acc_est"])
    _print_K_as_ppm("K-I 真值 (ppm)", result["K_acc_true"])

    print("\n── 陀螺仪静态六位置标定（零偏，deg/h）──")
    for i, ax in enumerate(["X", "Y", "Z"]):
        est = rad_s_to_deg_h(result["b_gyro_static"][i])
        tru = rad_s_to_deg_h(result["true_gyro_bias"][i])
        print(f"  {ax}轴  估计 {est:+.4f}  真值 {tru:+.4f}  误差 {est - tru:+.4f}")

    print("\n── 陀螺仪速率法标定（+/-100 deg/s，比例因子 + 交轴耦合）──")
    print(f"  拟合优度 R^2 = {gyro_info['R_sq']:.6f}")
    print(f"  零偏估计 (deg/h)：[{', '.join(f'{rad_s_to_deg_h(v):+.4f}' for v in result['b_gyro_est'])}]")
    print(f"  零偏真值 (deg/h)：[{', '.join(f'{rad_s_to_deg_h(v):+.4f}' for v in result['true_gyro_bias'])}]")
    print(f"  比例因子估计 (ppm)：[{', '.join(f'{v*1e6:+.2f}' for v in gyro_info['scale'])}]")
    print(f"  比例因子真值 (ppm)：[{', '.join(f'{v:+.2f}' for v in result['true_gyro_scale'])}]")
    _print_K_as_ppm("K-I 估计 (ppm)", result["K_gyro_est"])
    _print_K_as_ppm("K-I 真值 (ppm)", result["K_gyro_true"])

    _print_compare_result("── 加速度计补偿前后对比 ──", result["acc_compare"], ["N", "E", "D"])
    _print_compare_result("── 陀螺仪补偿前后对比 ──", result["gyro_compare"])

    print("\n── 各位置比力模长误差 (m/s^2) ──")
    print(f"  {'位置':<10} {'补偿前':>12} {'补偿后':>12}")
    for label, eb, ea in zip(result["pos_labels"],
                             result["pos_acc_err_before"],
                             result["pos_acc_err_after"]):
        print(f"  {label:<10} {eb:12.6f} {ea:12.6f}")

    return result
