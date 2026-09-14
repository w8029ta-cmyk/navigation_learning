"""近地惯导用的正常重力（Normal Gravity）模型。

正常重力是假定地球为参考椭球、内部质量均匀分布时，椭球面上各点的
理论重力值。惯导机械编排中的重力补偿项 g^n 即采用此模型，而非实测
重力或扰动重力。

本模块实现 GRS80 / WGS84 椭球上的 Somigliana 公式 + 高程二阶修正，
与 NIMA TR8350.2 及 KF-GINS 官方实现保持一致（椭球面公式闭式误差 < 1e-10）。

Somigliana 便捷形式（椭球面正常重力 g0）：

  g0(lat) = g_equator · (1 + k · sin²(lat)) / sqrt(1 - e² · sin²(lat))

  其中 k = (b/a) · (g_pole/g_equator) - 1 ≈ 0.00193185

  注意：部分文献用 0.005279·sin²(lat) 级数近似代替上式分母中的 sqrt 项，
  若与 Somigliana 分母混用，在 30° 纬度约有 8 mm/s² 偏差；
  KF-GINS 源码若采用该近似，与本模块结果会有可观测差异。

高程修正（大地高 h 处的正常重力）采用二阶 Taylor 展开：

  g(lat, h) = g0(lat) · (1 - 2h·(1+f+m-2f·sin²(lat))/a + 3h²/a²)

  其中
    e = 第一偏心率
    f = 扁率
    m = ω²·a²·b / GM  （离心力与引力之比相关的量纲一参数）

参考：GRS80 / NIMA TR8350.2 / 伍兹《惯性导航》正常重力章节
"""

import math

import numpy as np

from .constants import WGS84_A, WGS84_B, WGS84_OMEGA, WGS84_GM, WGS84_F

# ---------------------------------------------------------------------------
# 椭球面正常重力基准值（GRS80 标准，单位 m/s²）
# ---------------------------------------------------------------------------
G_EQUATOR = 9.7803267715   # 赤道处椭球面正常重力
G_POLE    = 9.8321863685   # 极点处椭球面正常重力

# 第一偏心率平方 e² = 1 - (b/a)²，Somigliana 公式分母用到
E_SQ = 1.0 - (WGS84_B ** 2) / (WGS84_A ** 2)

# Somigliana 公式中的系数 k，由赤道/极点重力比与椭球长短轴比确定
K_SOMIGLIANA = (G_POLE / G_EQUATOR) * (WGS84_B / WGS84_A) - 1.0

# 量纲一参数 m = ω²·a²·b / GM，出现在高程修正公式中
M_DIM = WGS84_OMEGA ** 2 * WGS84_A ** 2 * WGS84_B / WGS84_GM


def normal_gravity(lat, h=0.0):
    """计算指定纬度和大地高处的正常重力。

    参数
    ----
    lat : float
        大地纬度，弧度（rad）。北纬为正。
    h : float, 可选
        大地高，米（m）。椭球面以上为正，默认 0 表示椭球面。

    返回
    ----
    (surface_g, corrected_g) : tuple[float, float]
        surface_g   — 椭球面（h=0）处的正常重力模值，m/s²
        corrected_g — 考虑高程修正后的正常重力模值，m/s²
    """
    sin2 = math.sin(lat) ** 2

    # 第一步：Somigliana 公式，计算椭球面上的正常重力 g0(lat)
    surface_g = (
        G_EQUATOR * (1.0 + K_SOMIGLIANA * sin2)
        / math.sqrt(1.0 - E_SQ * sin2)
    )

    # 第二步：高程二阶修正，将 g0 外推到高度 h
    # 修正项随纬度变化：赤道附近 (sin²≈0) 与极点附近 (sin²≈1) 系数不同
    corrected_g = surface_g * (
        1.0
        - 2.0 * h / WGS84_A * (1.0 + WGS84_F + M_DIM - 2.0 * WGS84_F * sin2)
        + 3.0 * (h / WGS84_A) ** 2
    )
    return surface_g, corrected_g


def gravity_ned(lat, h):
    """返回 NED 导航坐标系下的正常重力向量 g^n。

    NED 系定义：x 北、y 东、z 地（Down 向下为正）。
    正常重力方向沿 plumb line 指向地心，在 NED 系中仅 D 轴分量非零：

        g^n = [0, 0, g(lat, h)]ᵀ

    该向量直接用于速度更新公式中的重力补偿项 g^n · dt。

    参数
    ----
    lat : float
        大地纬度（rad）
    h : float
        大地高（m）

    返回
    ----
    np.ndarray, shape (3,)
        NED 系正常重力向量，单位 m/s²
    """
    _, g = normal_gravity(lat, h)
    return np.array([0.0, 0.0, g])
