"""捷联惯导机械编排。

核心步骤：每个 IMU 采样周期依次做：
  1. 姿态更新 (四元数 + 二子样圆锥补偿)
  2. 速度更新 (机体系比力旋转 + 二子样划船补偿 + 科氏/重力项)
  3. 位置更新 (地理映射 dr_i)

实现参考 KF-GINS 的变量命名和公式约定，尽量用直白的变量名写出，
便于对照教材。
"""

import numpy as np

from .constants import WGS84_OMEGA
from .attitude import quat_multiply, quat_to_dcm, rotvec_to_quat
from .coordinates import meridian_radius, prime_vertical_radius
from .gravity import gravity_ned


class NavState:
    """导航状态：位置 + 速度 + 姿态。

    lat/lon 单位 rad，h 单位 m。
    v_n 是 NED 系下的速度 [v_n, v_e, v_d] (m/s)。
    q_b_n 是表示 C_b^n 的四元数 [w, x, y, z]。
    """

    def __init__(self, lat, lon, h, v_n, q_b_n):
        self.lat = float(lat)
        self.lon = float(lon)
        self.h = float(h)
        self.v_n = np.asarray(v_n, dtype=float).reshape(3)
        self.q_b_n = np.asarray(q_b_n, dtype=float).reshape(4)


class IMUSample:
    """单个 IMU 增量样本。

    dt: 采样周期 (s)
    dtheta: 陀螺仪在 dt 内的角增量 (rad)
    dvel: 加速度计在 dt 内的速度增量 (m/s)
    """

    def __init__(self, dt, dtheta, dvel):
        self.dt = float(dt)
        self.dtheta = np.asarray(dtheta, dtype=float).reshape(3)
        self.dvel = np.asarray(dvel, dtype=float).reshape(3)


def skew(v):
    """3D 向量对应的反对称矩阵 [v_x]x。"""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def dr_i(lat, h):
    """把 NED 系位移/速度增量映射为 [dlat, dlon, dh]。

    返回一个 3x3 对角矩阵，diag(1/(Rm+h), 1/((Rn+h)cos(lat)), -1)。
    """
    rn = prime_vertical_radius(lat)
    rm = meridian_radius(lat)
    return np.diag([
        1.0 / (rm + h),
        1.0 / ((rn + h) * np.cos(lat)),
        -1.0,
    ])


def earth_rates(lat, h, v_n):
    """计算 NED 系下的地球自转角速度 omega_ie_n 和输运速率 omega_en_n。"""
    rn = prime_vertical_radius(lat)
    rm = meridian_radius(lat)
    v_north, v_east, _ = np.asarray(v_n, dtype=float).reshape(3)

    omega_ie_n = np.array([
        WGS84_OMEGA * np.cos(lat),
        0.0,
        -WGS84_OMEGA * np.sin(lat),
    ])
    omega_en_n = np.array([
        v_east / (rn + h),
        -v_north / (rm + h),
        -v_east * np.tan(lat) / (rn + h),
    ])
    return omega_ie_n, omega_en_n


def velocity_update(state_prev, imu_prev, imu_cur):
    """速度更新：机体系比力 -> 导航系比力 + 科氏/重力。

    用二子样划船补偿 + 中点积分提升精度。
    """
    dt = imu_cur.dt
    vel_prev = state_prev.v_n.copy()
    cbn_prev = quat_to_dcm(state_prev.q_b_n)

    wie_n, wen_n = earth_rates(state_prev.lat, state_prev.h, vel_prev)

    # 机体系比力增量 + 二子样划船补偿
    d_vfb = (
        imu_cur.dvel
        + 0.5 * np.cross(imu_cur.dtheta, imu_cur.dvel)
        + np.cross(imu_prev.dtheta, imu_cur.dvel) / 12.0
        + np.cross(imu_prev.dvel, imu_cur.dtheta) / 12.0
    )

    # 等效导航系旋转（用前一时刻的 wie+wen 做一阶近似）
    cnn = np.eye(3) - skew((wie_n + wen_n) * dt / 2.0)
    d_vfn = cnn @ cbn_prev @ d_vfb
    # 重力/有害加速度项：正常重力 g^n（Somigliana + 高程修正）减去科氏/向心项
    d_vgn = (gravity_ned(state_prev.lat, state_prev.h)
             - np.cross(2.0 * wie_n + wen_n, vel_prev)) * dt

    # 中点积分，得到中间速度和中间位置
    midvel = vel_prev + 0.5 * (d_vfn + d_vgn)
    pos_prev = np.array([state_prev.lat, state_prev.lon, state_prev.h])
    midpos = pos_prev + dr_i(state_prev.lat, state_prev.h) @ midvel * dt / 2.0

    # 中点迭代：用中点纬度高和速度重算 wie/wen 及正常重力，提高数值精度
    wie_mid, wen_mid = earth_rates(midpos[0], midpos[2], midvel)
    cnn = np.eye(3) - skew((wie_mid + wen_mid) * dt / 2.0)
    d_vfn = cnn @ cbn_prev @ d_vfb
    d_vgn = (gravity_ned(midpos[0], midpos[2])
             - np.cross(2.0 * wie_mid + wen_mid, midvel)) * dt

    return vel_prev + d_vfn + d_vgn


def position_update(state_prev, vel_cur, imu_cur):
    """位置更新：中点速度 + 地理映射 dr_i。"""
    dt = imu_cur.dt
    pos_prev = np.array([state_prev.lat, state_prev.lon, state_prev.h])
    midvel = 0.5 * (state_prev.v_n + np.asarray(vel_cur, dtype=float).reshape(3))
    midpos = pos_prev + dr_i(state_prev.lat, state_prev.h) @ midvel * dt / 2.0
    return pos_prev + dr_i(midpos[0], midpos[2]) @ midvel * dt


def attitude_update(state_prev, pos_cur, vel_cur, imu_prev, imu_cur):
    """姿态更新：二子样圆锥补偿 + 中点导航系旋转。"""
    dt = imu_cur.dt
    pos_prev = np.array([state_prev.lat, state_prev.lon, state_prev.h])
    midpos = 0.5 * (pos_prev + np.asarray(pos_cur, dtype=float).reshape(3))
    midvel = 0.5 * (state_prev.v_n + np.asarray(vel_cur, dtype=float).reshape(3))

    wie_mid, wen_mid = earth_rates(midpos[0], midpos[2], midvel)

    # 导航系等效旋转四元数 (注意符号：C_b^n = C_nn^n * C_b^n * C_bb^b)
    q_nn = rotvec_to_quat(-(wie_mid + wen_mid) * dt)

    # 机体系等效旋转：二子样圆锥补偿
    q_bb = rotvec_to_quat(imu_cur.dtheta + np.cross(imu_prev.dtheta, imu_cur.dtheta) / 12.0)

    return quat_multiply(quat_multiply(q_nn, state_prev.q_b_n), q_bb)


def mechanization_step(state_prev, imu_prev, imu_cur):
    """机械编排一步：速度 -> 位置 -> 姿态，返回新的 NavState。"""
    vel_cur = velocity_update(state_prev, imu_prev, imu_cur)
    pos_cur = position_update(state_prev, vel_cur, imu_cur)
    q_cur = attitude_update(state_prev, pos_cur, vel_cur, imu_prev, imu_cur)

    return NavState(pos_cur[0], pos_cur[1], pos_cur[2], vel_cur, q_cur)
