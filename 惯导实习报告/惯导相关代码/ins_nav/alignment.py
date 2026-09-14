"""解析粗对准 + 多位置对准 + 姿态正交化。演示（实验二）

理论：
  地球重力 g 和地球自转角速度 omega_ie 这两个向量，
  在导航坐标系下的投影可以通过纬度解析计算；
  在机体系下的投影可以通过静止 IMU 数据取平均得到。
  两个已知投影的向量就可以唯一确定一个方向余弦矩阵 C_b^n。

  公式（讲义式写法）：
    f_b_avg  = mean( acc_meas )                 （机体系重力的反方向）
    w_ib_b_avg = mean( gyro_meas )              （机体系陀螺输出）
    g_n = [0, 0, g],  w_ie_n = [W cos(lat), 0, -W sin(lat)]

  由叉乘构造三个新向量：
    t1 = g_b            v1 = g_n
    t2 = g_b × w_ib_b   v2 = g_n × w_ie_n
    t3 = t1 × t2        v3 = v1 × v2
  归一化后构造矩阵：
    T = [t1^, t2^, t3^], V = [v1^, v2^, v3^]
    C_b^n = V * T^T
  最后再迭代做正交化。
"""

import math

import numpy as np

from .constants import WGS84_OMEGA
from .attitude import quat_to_euler, euler_to_quat, quat_to_dcm
from .gravity import normal_gravity


def _fmt_vec3(v, unit="", prec=6):
    """格式化三维向量为可读字符串。"""
    arr = np.asarray(v, dtype=float).reshape(3)
    body = ", ".join(f"{x:+.{prec}f}" for x in arr)
    suffix = f" {unit}" if unit else ""
    return f"[{body}]{suffix}"


def build_reference_vectors(lat):
    """构造导航系下的参考向量：正常重力 g^n 与地球自转角速度 ω_ie^n。

    正常重力由 Somigliana 公式计算（见 gravity.normal_gravity），
    在 NED 系中表示为 g^n = [0, 0, g]（D 轴向下为正）。

    返回 (g_n, w_ie_n)
    """
    _, g = normal_gravity(lat, 0.0)
    g_n = np.array([0.0, 0.0, g])
    w_ie_n = np.array([
        WGS84_OMEGA * math.cos(lat),
        0.0,
        -WGS84_OMEGA * math.sin(lat),
    ])
    return g_n, w_ie_n


def cross_align_two_vectors(vec1_body, vec2_body, vec1_nav, vec2_nav):
    """用两个向量的导航系/机体系投影构造 C_b^n。

    如果 vec2 很小（例如陀螺零偏被完全补偿），退化为"用 t1/g1 单独定水平 + 陀螺残余定航向"。
    """
    t1 = np.asarray(vec1_body, dtype=float).reshape(3)
    t2 = np.asarray(vec2_body, dtype=float).reshape(3)
    v1 = np.asarray(vec1_nav, dtype=float).reshape(3)
    v2 = np.asarray(vec2_nav, dtype=float).reshape(3)

    def _safe_norm(x):
        n = np.linalg.norm(x)
        return n if n > 1e-15 else 0.0

    t1_norm = _safe_norm(t1); t2_norm = _safe_norm(t2)
    v1_norm = _safe_norm(v1); v2_norm = _safe_norm(v2)

    # 只要两个向量都足够非零，标准解析粗对准
    if t2_norm > 1e-10 and v2_norm > 1e-10 and t1_norm > 1e-10:
        t3 = np.cross(t1, t2)
        v3 = np.cross(v1, v2)
        if np.linalg.norm(t3) > 1e-10 and np.linalg.norm(v3) > 1e-10:
            tmat = np.column_stack([t1 / t1_norm,
                                    t2 / t2_norm,
                                    t3 / np.linalg.norm(t3)])
            vmat = np.column_stack([v1 / v1_norm,
                                    v2 / v2_norm,
                                    v3 / np.linalg.norm(v3)])
            return vmat @ tmat.T

    # 退化情况：用 f_b 定 roll/pitch，yaw 暂时置 0
    g_b_unit = t1 / t1_norm
    # pitch = asin(-g_b_x), roll = atan2(g_b_y, g_b_z)
    pitch = math.asin(max(-1.0, min(1.0, -g_b_unit[0])))
    roll = math.atan2(g_b_unit[1], g_b_unit[2])
    yaw = 0.0
    return quat_to_dcm(euler_to_quat(roll, pitch, yaw))


def iteratively_orthogonalize(c_b_n, max_iter=100, tol=1e-12):
    """对有微小失配的 DCM 做迭代正交化。

    每一步用极分解得到最接近的正交矩阵；实现上直接用 SVD 把奇异值都替换为 1。
    """
    c = np.asarray(c_b_n, dtype=float)
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(c, full_matrices=False)
        c_new = U @ Vt
        if np.linalg.norm(c_new - c) < tol:
            return c_new
        c = c_new
    return c


def coarse_alignment(gyro_data, acc_data, lat, N_avg=1):
    """从静止 IMU 数据做解析粗对准。

    gyro_data / acc_data: (N, 3) IMU 增量样本（dt 不影响均值，只要给角增量除以 dt 得到角速率即可）
                          或者传 (N,3) 角速率 / 比力也可以，都按最后一维均值处理
    lat: 纬度 (rad)
    N_avg: 分成 N 段各自对准再平均，降低随机噪声
    返回 (欧拉角(roll,pitch,yaw rad), 四元数, C_b^n)
    """
    gyro_data = np.asarray(gyro_data, dtype=float).reshape(-1, 3)
    acc_data = np.asarray(acc_data, dtype=float).reshape(-1, 3)

    if N_avg <= 1:
        w_ib_b = gyro_data.mean(axis=0)
        f_b = acc_data.mean(axis=0)
    else:
        n = gyro_data.shape[0] // N_avg
        w_list, f_list = [], []
        for k in range(N_avg):
            w_list.append(gyro_data[k * n:(k + 1) * n].mean(axis=0))
            f_list.append(acc_data[k * n:(k + 1) * n].mean(axis=0))
        w_ib_b = np.mean(w_list, axis=0)
        f_b = np.mean(f_list, axis=0)

    # 重力在机体系的反方向 = -f_b
    g_b = -f_b
    g_n, w_ie_n = build_reference_vectors(lat)

    c_b_n = cross_align_two_vectors(g_b, w_ib_b, g_n, w_ie_n)
    c_b_n = iteratively_orthogonalize(c_b_n)

    # 由 C_b^n 提取欧拉角（3-2-1 顺序）
    pitch = math.asin(max(-1.0, min(1.0, -c_b_n[2, 0])))
    roll = math.atan2(c_b_n[2, 1], c_b_n[2, 2])
    yaw = math.atan2(c_b_n[1, 0], c_b_n[0, 0])

    q = euler_to_quat(roll, pitch, yaw)
    return np.array([roll, pitch, yaw]), q, c_b_n


def multi_position_alignment(position_observations, lat):
    """转台多位置对准：每个位置给一组 (acc_meas, gyro_meas)，分别粗对准再取平均。

    position_observations: list of dict, 每个 dict 有 'acc' (Nx3) 'gyro' (Nx3)
                          以及可选的 'roll/pitch/yaw_deg'（已知粗姿态）
    返回平均后的 (欧拉角, 四元数, C_b^n)
    """
    c_accum = np.zeros((3, 3))
    n = 0
    for obs in position_observations:
        euler, _, c_b_n = coarse_alignment(obs["gyro"], obs["acc"], lat)
        c_accum = c_accum + c_b_n
        n += 1
    c_avg = c_accum / n
    c_avg = iteratively_orthogonalize(c_avg)
    pitch = math.asin(max(-1.0, min(1.0, -c_avg[2, 0])))
    roll = math.atan2(c_avg[2, 1], c_avg[2, 2])
    yaw = math.atan2(c_avg[1, 0], c_avg[0, 0])
    q = euler_to_quat(roll, pitch, yaw)
    return np.array([roll, pitch, yaw]), q, c_avg


def run_alignment_demo(lat_deg=30.444787):
    lat = math.radians(lat_deg)
    g_n, w_ie_n = build_reference_vectors(lat)

    true_euler = np.array([math.radians(0.5), math.radians(-1.0), math.radians(45.0)])
    q_true = euler_to_quat(*true_euler)
    c_b_n_true = quat_to_dcm(q_true)
    f_b_true = -c_b_n_true.T @ g_n
    w_ib_b_true = c_b_n_true.T @ w_ie_n
    g_b_true = -f_b_true

    N = 20000
    n_avg = 20
    acc_sigma = 1e-4
    gyro_sigma = 1e-6
    np.random.seed(42)
    acc_data = np.tile(f_b_true, (N, 1)) + np.random.normal(0, acc_sigma, (N, 3))
    gyro_data = np.tile(w_ib_b_true, (N, 1)) + np.random.normal(0, gyro_sigma, (N, 3))

    f_b_mean = acc_data.mean(axis=0)
    w_ib_b_mean = gyro_data.mean(axis=0)
    g_b_meas = -f_b_mean

    euler, q, c_b_n_est = coarse_alignment(gyro_data, acc_data, lat, N_avg=n_avg)
    euler_err = euler - true_euler

    multi = [{"gyro": gyro_data, "acc": acc_data} for _ in range(6)]
    euler2, q2, c2 = multi_position_alignment(multi, lat)

    err_deg = float(np.max(np.abs(np.degrees(euler_err))))
    rad_s_to_deg_h = 180.0 / math.pi * 3600.0

    t2_body = np.cross(g_b_meas, w_ib_b_mean)
    t2_nav = np.cross(g_n, w_ie_n)

    print("\n【实验二】解析粗对准")
    print(f"测试纬度：{lat_deg:.6f} deg，IMU 样本数 {N}，分 {n_avg} 段取平均以降低噪声")
    print(f"加表噪声标准差：{acc_sigma} m/s^2；陀螺噪声标准差：{gyro_sigma} rad/s")
    print()
    print("【1】导航系参考矢量（由纬度解析，与姿态无关）")
    print(f"  g^n     = {_fmt_vec3(g_n, 'm/s^2')}")
    print(f"          |g^n| = {np.linalg.norm(g_n):.6f} m/s^2")
    print(f"  ω_ie^n  = {_fmt_vec3(w_ie_n, 'rad/s')}")
    print(f"          |ω_ie^n| = {np.linalg.norm(w_ie_n):.6e} rad/s "
          f"({np.linalg.norm(w_ie_n) * rad_s_to_deg_h:.4f} deg/h)")
    print()
    print("【2】假定真值姿态 → 反推理想 IMU（仿真 ground truth）")
    print(f"  真值欧拉角 (deg)：roll={math.degrees(true_euler[0]):+.4f}, "
          f"pitch={math.degrees(true_euler[1]):+.4f}, "
          f"yaw={math.degrees(true_euler[2]):+.4f}")
    print(f"  f^b_true = -(C_b^n)^T g^n = {_fmt_vec3(f_b_true, 'm/s^2')}")
    print(f"  g^b_true = (C_b^n)^T g^n   = {_fmt_vec3(g_b_true, 'm/s^2')}")
    print(f"  ω_ib^b   = (C_b^n)^T ω_ie^n = {_fmt_vec3(w_ib_b_true, 'rad/s')}")
    print(f"          ω_ib^b = {_fmt_vec3(w_ib_b_true * rad_s_to_deg_h, 'deg/h', prec=4)}")
    print()
    print("【3】加噪声后 IMU 均值（对准算法的实际输入）")
    print(f"  mean(acc)  = {_fmt_vec3(f_b_mean, 'm/s^2')}")
    print(f"  相对理想 f^b 残差 = {_fmt_vec3(f_b_mean - f_b_true, 'm/s^2', prec=8)}")
    print(f"  mean(gyro) = {_fmt_vec3(w_ib_b_mean, 'rad/s')}")
    print(f"  相对理想 ω 残差 = {_fmt_vec3(w_ib_b_mean - w_ib_b_true, 'rad/s', prec=10)}")
    print()
    print("【4】双矢量对准用的机体系向量")
    print(f"  g^b = -mean(acc) = {_fmt_vec3(g_b_meas, 'm/s^2')}")
    print(f"  |g^b| 误差 vs 真值：{np.linalg.norm(g_b_meas - g_b_true):.2e} m/s^2")
    print(f"  ω^b = mean(gyro) = {_fmt_vec3(w_ib_b_mean, 'rad/s')}")
    print(f"  |ω^b| 误差 vs 真值：{np.linalg.norm(w_ib_b_mean - w_ib_b_true):.2e} rad/s")
    print(f"  g^b × ω^b (机体系) = {_fmt_vec3(t2_body, prec=4)}  |·|={np.linalg.norm(t2_body):.4e}")
    print(f"  g^n × ω_ie^n (导航系) = {_fmt_vec3(t2_nav, prec=4)}  |·|={np.linalg.norm(t2_nav):.4e}")
    print()
    print("【5】姿态解算结果对比")
    print(f"  {'姿态角':<8} {'真值(deg)':>10} {'对准(deg)':>10} "
          f"{'误差(deg)':>10} {'多位置(deg)':>10}")
    print(f"  {'-' * 52}")
    for i, name in enumerate(["横滚", "俯仰", "航向"]):
        print(f"  {name:<8} {math.degrees(true_euler[i]):>+10.4f} "
              f"{math.degrees(euler[i]):>+10.4f} "
              f"{math.degrees(euler_err[i]):>+10.6f} "
              f"{math.degrees(euler2[i]):>+10.4f}")
    print(f"\n  最大姿态误差：{err_deg:.4f} deg")
    orth_err = np.linalg.norm(c_b_n_est.T @ c_b_n_est - np.eye(3))
    print(f"  估计 C_b^n 正交性 ||C^T C - I|| = {orth_err:.2e}")

    return euler, q, c_b_n_est
