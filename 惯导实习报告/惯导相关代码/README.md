# 惯性导航课程设计（Python 实现）

基于 KF-GINS 公开数据集与算法框架，用 Python 实现捷联惯导（SINS）机械编排、IMU 标定、解析粗对准及 INS/GNSS 松组合扩展卡尔曼滤波，并完成三个教学实验。

## 环境要求

- Python 3.8+
- NumPy

```bash
pip install numpy
```

## 目录结构

```
ins/
├── main.py                 # 命令行入口
├── diff_kf_gins.py         # NavResult_py.nav 与 truth.nav 对比
├── README.md
├── ins_nav/                # 核心算法包
│   ├── attitude.py         # 四元数 / DCM / 欧拉角
│   ├── alignment.py        # 解析粗对准、多位置对准（实验二）
│   ├── calibration.py      # IMU 六位置法、速率法标定（实验一）
│   ├── constants.py        # WGS84 椭球常量
│   ├── coordinates.py      # 大地坐标与 ECEF 转换
│   ├── gravity.py          # Somigliana 正常重力模型
│   ├── mechanization.py    # 捷联惯导机械编排
│   ├── kf_gins.py          # 15/21 维 INS/GNSS EKF
│   ├── config_loader.py    # kf-gins.yaml 配置解析
│   ├── pipeline.py         # 实验一/二结果接入实验三，默认使用yaml给定的参数
│   ├── simulation.py       # 实验三仿真数据集解算与评估（实验三）
│   └── nav_output.py       # 导航结果文件读写
└── dataset/                # KF-GINS 示例数据
    ├── kf-gins.yaml        # 解算配置
    ├── Leador-A15.txt      # IMU 原始数据（200 Hz）
    ├── GNSS-RTK.txt        # GNSS 位置量测
    └── truth.nav           # 参考真值轨迹
```

运行实验三后会在 `dataset/` 下生成：

- `NavResult_py.nav` — Python 版导航解
- `ImuError_py.txt` — 滤波估计的 IMU 零偏与比例因子

## 快速开始

在项目根目录 `ins/` 下执行：

```bash
# 依次运行实验一、二、三（仅终端输出）
python main.py

# 单独运行
python main.py --calib      # 实验一：IMU 标定
python main.py --align      # 实验二：解析粗对准
python main.py --dataset    # 实验三：数据集惯导解算
```

### 实验三常用参数

```bash
# 指定解算时长（yaml 中 endtime=-1 时生效，默认 120 s）
python main.py --dataset --duration 300

# 用实验二粗对准结果替代 yaml 初始姿态
python main.py --dataset --use-align

# 用静止段零偏估计替代 yaml IMU 初值
python main.py --dataset --use-calib

# 启用 21 维滤波（额外估计陀螺/加表比例因子）
python main.py --dataset --estimate-scale

# 指定配置文件
python main.py --dataset --config dataset/kf-gins.yaml
```

### 与真值对比

```bash
# 先运行实验三，再与 truth.nav 逐列对比
python diff_kf_gins.py --run

# 指定 Python 输出与真值文件
python diff_kf_gins.py --ours dataset/NavResult_py.nav --ref dataset/truth.nav
```

## 三个实验说明

### 实验一：IMU 标定

对合成 IMU 数据演示两类标定方法：

1. **加速度计六位置法** — 六个静止姿态下利用 ±g 对消，求各轴零偏与比例因子
2. **陀螺仪静态零偏** — 静止段均值法
3. **陀螺仪速率法** — 已知 ±100 deg/s 正反转，最小二乘求零偏与比例因子

误差模型：`y = (I + S) · x + b`

### 实验二：解析粗对准

利用静止段 IMU 均值，通过地球重力 `g^n` 与地球自转 `ω_ie^n` 在导航系/机体系下的投影，构造方向余弦矩阵 `C_b^n`，并做迭代正交化。支持分块平均与多位置对准以降低噪声。

### 实验三：KF-GINS 数据集解算

在 Leador-A15 车载数据集上对比三条解算链路：

| 链路 | 说明 |
|------|------|
| 链路一 | 15/21 维 EKF 松组合 + 在线 IMU 误差补偿 + GNSS 天线杆臂 |
| 链路二 | 纯捷联惯导机械编排 + 静止段零偏补偿 |
| 链路三 | 纯捷联惯导机械编排 + 原始 IMU（无补偿） |

机械编排中的重力补偿采用 **Somigliana 正常重力模型**（GRS80/WGS84，含高程二阶修正），与 KF-GINS 官方实现一致。

终端输出包括：初始位置误差、静止段零偏估计、各链路位置/姿态均方根误差及汇总对比。

## 核心算法模块

| 模块 | 功能 |
|------|------|
| `gravity.py` | 椭球面正常重力 `g0(lat)` 与高程修正 `g(lat,h)`，输出 NED 系 `g^n = [0,0,g]` |
| `mechanization.py` | 姿态（圆锥补偿）→ 速度（划船补偿 + 正常重力/科氏项）→ 位置（地理映射） |
| `kf_gins.py` | 误差状态 EKF：时间更新 + GNSS 位置量测更新 + 反馈修正 |
| `alignment.py` | 双矢量解析粗对准（`g` 与 `ω_ie` 叉乘构造 `C_b^n`） |
| `calibration.py` | 六位置法、速率法、均值法标定 |

## 数据集说明

数据来自 [KF-GINS](https://github.com/i2Nav-WHU/KF-GINS) 公开的 Leador-A15 数据集片段，配置文件 `kf-gins.yaml` 格式与官方兼容。初始位置约武汉（纬度 30.44°，经度 114.47°），IMU 采样率 200 Hz。

## 参考

- KF-GINS 开源项目：武汉大学 i2Nav 实验室
- 正常重力：GRS80 / NIMA TR8350.2，Somigliana 公式
- 机械编排与 EKF：严恭敏《惯性导航》及相关讲义
