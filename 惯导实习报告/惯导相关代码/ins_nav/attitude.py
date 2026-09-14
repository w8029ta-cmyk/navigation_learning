"""四元数、方向余弦矩阵、欧拉角之间的相互转换。

所有四元数都是 [w, x, y, z] 的 Hamilton 约定。
"""

import math

import numpy as np


def normalize_quat(q):
    """把四元数归一化，防止数值漂移。"""
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("四元数范数不能为 0")
    return q / norm


def quat_multiply(q1, q2):
    """Hamilton 乘积 q = q1 * q2。

    旋转合成顺序：先做 q2 的旋转，再做 q1 的旋转。
    """
    w1, x1, y1, z1 = normalize_quat(q1)
    w2, x2, y2, z2 = normalize_quat(q2)
    return normalize_quat(np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]))


def rotvec_to_quat(phi):
    """小角度旋转矢量 phi -> 四元数。

    phi 方向是旋转轴，模长是旋转角度 (rad)。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return normalize_quat(np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]]))
    axis = phi / angle
    return normalize_quat(np.r_[math.cos(0.5 * angle), math.sin(0.5 * angle) * axis])


def euler_to_quat(roll, pitch, yaw):
    """3-2-1 顺序欧拉角 -> 四元数 (C_b^n)。

    顺序：先绕 Z 轴转 yaw，再绕 Y 轴转 pitch，最后绕 X 轴转 roll。
    """
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return normalize_quat(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def quat_to_dcm(q):
    """四元数 -> 方向余弦矩阵 C_b^n。

    即把机体系向量投影到 NED 导航系的那个矩阵。
    """
    q0, q1, q2, q3 = normalize_quat(q)
    return np.array([
        [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
    ])


def quat_to_euler(q):
    """四元数 -> [roll, pitch, yaw]，单位 rad。"""
    c_b_n = quat_to_dcm(q)
    pitch = math.asin(max(-1.0, min(1.0, -c_b_n[2, 0])))
    roll = math.atan2(c_b_n[2, 1], c_b_n[2, 2])
    yaw = math.atan2(c_b_n[1, 0], c_b_n[0, 0])
    return np.array([roll, pitch, yaw])


def coarse_align_static(f_b, omega_b, g, earth_rate):
    """静基座粗对准：用加速度计和陀螺仪均值估计初始姿态。

    返回 (欧拉角, 四元数)。
    原理：水平方向加速度计测量就是重力反方向（调平），
          地球自转在水平面上的投影决定航向。
    """
    f_b = np.asarray(f_b, dtype=float).reshape(3)
    omega_b = np.asarray(omega_b, dtype=float).reshape(3)

    pitch = math.asin(max(-1.0, min(1.0, f_b[0] / g)))
    roll = math.atan2(-f_b[1], -f_b[2])

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    leveled_w_x = cp * omega_b[0] + sp * sr * omega_b[1] + sp * cr * omega_b[2]
    leveled_w_y = cr * omega_b[1] - sr * omega_b[2]
    if abs(earth_rate) > 0.0 and np.linalg.norm(omega_b[:2]) > 1e-12:
        yaw = math.atan2(-leveled_w_y, leveled_w_x)
    else:
        yaw = 0.0

    euler = np.array([roll, pitch, yaw])
    q = euler_to_quat(roll, pitch, yaw)
    return euler, q

