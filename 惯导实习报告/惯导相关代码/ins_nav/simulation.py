"""实验三：KF-GINS 数据集惯性导航解算。

在 Leador-A15 实测 IMU/GNSS 数据集上，对比三条解算链路：
  链路一 — 15/21 维 EKF 松组合 + 在线 IMU 零偏/比例因子补偿 + 天线杆臂
  链路二 — 纯捷联惯导机械编排 + 静止段估计零偏补偿 IMU
  链路三 — 纯捷联惯导机械编排 + 原始 IMU（无补偿）

机械编排中的重力项采用 Somigliana 正常重力模型（见 gravity.py）。
可选 --use-align / --use-calib 将实验二/一的结果接入初值。
解算结果写入 outputpath（NavResult_py.nav / ImuError_py.txt）。
"""

import math
from pathlib import Path

import numpy as np

from .attitude import euler_to_quat, quat_to_euler, quat_to_dcm
from .config_loader import KfGinsConfig, load_kf_gins_config
from .coordinates import meridian_radius, prime_vertical_radius
from .gravity import gravity_ned
from .kf_gins import compensate_imu_sample, initial_ins_gnss_state, run_kf_gins
from .mechanization import IMUSample, NavState, dr_i, earth_rates, mechanization_step
from .nav_output import write_imu_error_file, write_nav_file
from .pipeline import apply_pipeline_options

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "dataset" / "kf-gins.yaml"
STATIC_SEC = 30.0


def load_config(config_path=None) -> dict:
    cfg = load_kf_gins_config(config_path)
    return {
        "imupath": cfg.imupath,
        "gnsspath": cfg.gnsspath,
        "truthpath": cfg.truthpath,
        "outputpath": cfg.outputpath,
        "imudatarate": cfg.imudatarate,
        "starttime": cfg.starttime,
        "endtime": cfg.endtime,
        "initpos": cfg.initpos,
        "initvel": cfg.initvel,
        "initatt": cfg.initatt,
        "_kf_cfg": cfg,
    }


def load_imu(path, starttime, endtime, datarate_hz):
    samples = []
    last_time = None
    dt_nom = 1.0 / datarate_hz
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.split()
            t = float(cols[0])
            if t < starttime:
                last_time = t
                continue
            if endtime > 0 and t > endtime:
                break
            dt = (t - last_time) if last_time is not None else dt_nom
            last_time = t
            samples.append((t, IMUSample(dt, cols[1:4], cols[4:7])))
    return samples


def load_truth(path, starttime, endtime):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.split()
            t = float(cols[1])
            if t < starttime:
                continue
            if endtime > 0 and t > endtime:
                break
            rows.append([
                t,
                math.radians(float(cols[2])), math.radians(float(cols[3])), float(cols[4]),
                float(cols[5]), float(cols[6]), float(cols[7]),
                math.radians(float(cols[8])), math.radians(float(cols[9])), math.radians(float(cols[10])),
            ])
    return np.asarray(rows, dtype=float)


def load_gnss(path, starttime, endtime):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.split()
            t = float(cols[0])
            if t < starttime:
                continue
            if endtime > 0 and t > endtime:
                break
            rows.append([
                t,
                math.radians(float(cols[1])), math.radians(float(cols[2])), float(cols[3]),
                float(cols[4]), float(cols[5]), float(cols[6]),
            ])
    return np.asarray(rows, dtype=float)


def compensate_imu(samples, gyro_bias, acc_bias, acc_scale=None, gyro_scale=None):
    out = []
    for t, imu in samples:
        out.append((t, compensate_imu_sample(imu, gyro_bias, acc_bias, gyro_scale, acc_scale)))
    return out


def estimate_bias(cfg, samples):
    """利用起始静止段 IMU 数据估计陀螺/加表零偏。

    原理：静止时陀螺输出 ≈ 地球自转在机体系投影 + 零偏；
          加表输出 ≈ -重力在机体系投影 + 零偏。
    需要已知初始姿态（yaml initatt）将导航系参考量转换到机体系。
    """
    t_end = cfg["starttime"] + STATIC_SEC
    lat, _, h = cfg["initpos"]
    roll, pitch, yaw = cfg["initatt"]
    lat = math.radians(lat)
    c_bn = quat_to_dcm(euler_to_quat(math.radians(roll), math.radians(pitch), math.radians(yaw)))
    omega_ie_n, _ = earth_rates(lat, h, np.zeros(3))
    omega_ie_b = c_bn.T @ omega_ie_n
    g_b = c_bn.T @ gravity_ned(lat, h)

    w_sum = np.zeros(3)
    f_sum = np.zeros(3)
    n = 0
    for t, imu in samples:
        if t > t_end:
            break
        w_sum += imu.dtheta / imu.dt
        f_sum += imu.dvel / imu.dt
        n += 1
    if n == 0:
        raise ValueError("静止段无 IMU 样本，无法估计零偏")

    return w_sum / n - omega_ie_b, f_sum / n + g_b


def initial_state(cfg):
    lat, lon, h = cfg["initpos"]
    roll, pitch, yaw = cfg["initatt"]
    return NavState(
        math.radians(lat), math.radians(lon), h,
        np.asarray(cfg["initvel"], dtype=float),
        euler_to_quat(math.radians(roll), math.radians(pitch), math.radians(yaw)),
    )


def copy_state(s):
    return NavState(s.lat, s.lon, s.h, s.v_n.copy(), s.q_b_n.copy())


def interpolate_truth(truth, time):
    if time <= truth[0, 0]:
        return truth[0]
    if time >= truth[-1, 0]:
        return truth[-1]
    i = np.searchsorted(truth[:, 0], time)
    a, b = truth[i - 1], truth[i]
    r = (time - a[0]) / (b[0] - a[0])
    row = a + r * (b - a)
    row[0] = time
    return row


def nav_error(state, truth_row):
    pos = np.array([state.lat, state.lon, state.h])
    t_pos, t_vel, t_att = truth_row[1:4], truth_row[4:7], truth_row[7:10]
    pos_err = np.linalg.inv(dr_i(t_pos[0], t_pos[2])) @ (pos - t_pos)
    att_err = (quat_to_euler(state.q_b_n) - t_att + np.pi) % (2 * np.pi) - np.pi
    return pos_err, state.v_n - t_vel, att_err


def run_mech(samples, state0, truth, datarate_hz):
    state = copy_state(state0)
    errors = []
    imu_pre = None
    t0 = tf = None
    n = 0

    for t, imu in samples:
        if imu_pre is None:
            t0 = t
            imu_pre = imu
            continue
        state = mechanization_step(state, imu_pre, imu)
        imu_pre = imu
        n += 1
        tf = t
        if n % datarate_hz == 0:
            errors.append(nav_error(state, interpolate_truth(truth, t)))

    return state, errors, t0, tf


def run_kf(samples, state0, truth, gnss, datarate_hz, kf_cfg: KfGinsConfig = None):
    """21-state loosely coupled EKF."""
    cfg = kf_cfg or load_kf_gins_config()
    ins0 = initial_ins_gnss_state(cfg)
    ins0.nav = copy_state(state0)
    errors = []

    def on_step(t, nav):
        errors.append(nav_error(nav, interpolate_truth(truth, t)))

    state, _, t0, tf, nav_records, imu_err_records = run_kf_gins(
        samples, ins0, cfg, gnss, datarate_hz, error_callback=on_step,
    )
    return state.nav, errors, t0, tf, state, nav_records, imu_err_records


def _rms(errors):
    pos = np.vstack([e[0] for e in errors])
    att = np.degrees(np.vstack([e[2] for e in errors]))
    return float(np.linalg.norm(np.sqrt(np.mean(pos ** 2, axis=0)))), \
           float(np.linalg.norm(np.sqrt(np.mean(att ** 2, axis=0))))


def _print_result(label, state, errors, t0, tf):
    """打印单条解算链路与真值对比的误差统计。"""
    pos_rms, att_rms = _rms(errors)
    pe = errors[-1][0]
    print(f"\n{label}（解算时段 {t0:.0f}–{tf:.0f} s）")
    print(f"  位置均方根误差：{pos_rms:.4f} m；姿态均方根误差：{att_rms:.4f} deg")
    print(f"  末端位置误差（北-东-地）：北向 {pe[0]:+.3f} m，"
          f"东向 {pe[1]:+.3f} m，垂向 {pe[2]:+.3f} m")
    return pos_rms, att_rms


def _count_static_samples(samples, t_end):
    return sum(1 for t, _ in samples if t <= t_end)


def run_dataset_experiment(
    config_path=None,
    duration_s=120.0,
    use_align=False,
    use_calib=False,
    estimate_scale=False,
    write_output=True,
):
    cfg_dict = load_config(config_path or DEFAULT_CONFIG)
    kf_cfg = cfg_dict["_kf_cfg"]
    kf_cfg.estimate_scale = estimate_scale
    t0 = cfg_dict["starttime"]
    t1 = cfg_dict["endtime"] if cfg_dict["endtime"] > 0 else t0 + duration_s
    rate = cfg_dict["imudatarate"]

    imu_raw = load_imu(cfg_dict["imupath"], t0, t1, rate)

    if use_align or use_calib:
        kf_cfg, cfg_dict = apply_pipeline_options(
            kf_cfg, cfg_dict, imu_raw, use_align=use_align, use_calib=use_calib,
        )

    gyro_bias, acc_bias = estimate_bias(cfg_dict, imu_raw)
    imu_comp = compensate_imu(imu_raw, gyro_bias, acc_bias)
    truth = load_truth(cfg_dict["truthpath"], t0, t1)
    gnss = load_gnss(cfg_dict["gnsspath"], t0, t1)
    state0 = initial_state(cfg_dict)

    n_static = _count_static_samples(imu_raw, t0 + STATIC_SEC)
    init_err = nav_error(state0, truth[0])[0]
    bg = gyro_bias * 180 / np.pi * 3600
    ba = acc_bias * 1e5

    flags = []
    if use_align:
        flags.append("实验二粗对准替代 yaml 初始姿态")
    if use_calib:
        flags.append("实验一静止段零偏替代 yaml 初值")
    flag_txt = "；".join(flags) if flags else "直接使用 yaml 配置文件中的初值"

    print("\n【实验三】基于 KF-GINS 数据集的惯性导航解算")
    print(f"机械编排采用 Somigliana 正常重力模型（含高程修正）")
    print(f"相对真值的初始位置误差（北-东-地）："
          f"北向 {init_err[0]:+.4f} m，东向 {init_err[1]:+.4f} m，"
          f"垂向 {init_err[2]:+.4f} m")
    print(f"初值来源：{flag_txt}")
    print(f"起始 {STATIC_SEC:.0f} s 静止段（共 {n_static} 个 IMU 样本）"
          f"估计零偏，供纯惯导补偿链路使用：")
    print(f"  陀螺仪零偏（deg/h）："
          f"[{bg[0]:+.3f}, {bg[1]:+.3f}, {bg[2]:+.3f}]")
    print(f"  加速度计零偏（mGal）："
          f"[{ba[0]:+.1f}, {ba[1]:+.1f}, {ba[2]:+.1f}]")
    print(f"解算链路一：{kf_cfg.state_dim} 维扩展卡尔曼滤波，"
          f"GNSS 天线杆臂 {kf_cfg.antlever.tolist()} m")

    s1, e1, a, b, ins_kf, nav_records, imu_err_records = run_kf(
        imu_raw, state0, truth, gnss, rate, kf_cfg,
    )
    dim_label = "21 状态（含比例因子）" if estimate_scale else "15 状态"
    r1 = _print_result(f"链路一：{dim_label}卡尔曼滤波 + 在线 IMU 误差补偿", s1, e1, a, b)
    kf_bg = ins_kf.gyro_bias * 180 / np.pi * 3600
    kf_ba = ins_kf.acc_bias * 1e5
    kf_sg = ins_kf.gyro_scale * 1e6
    kf_sa = ins_kf.acc_scale * 1e6
    print(f"  滤波结束时估计的 IMU 误差：")
    print(f"    陀螺仪零偏（deg/h）："
          f"[{kf_bg[0]:+.3f}, {kf_bg[1]:+.3f}, {kf_bg[2]:+.3f}]")
    print(f"    加速度计零偏（mGal）："
          f"[{kf_ba[0]:+.1f}, {kf_ba[1]:+.1f}, {kf_ba[2]:+.1f}]")
    print(f"    陀螺仪比例因子（ppm）："
          f"[{kf_sg[0]:+.1f}, {kf_sg[1]:+.1f}, {kf_sg[2]:+.1f}]")
    print(f"    加速度计比例因子（ppm）："
          f"[{kf_sa[0]:+.1f}, {kf_sa[1]:+.1f}, {kf_sa[2]:+.1f}]")

    out_dir = Path(kf_cfg.outputpath)
    nav_path = out_dir / "NavResult_py.nav"
    err_path = out_dir / "ImuError_py.txt"
    if write_output:
        write_nav_file(nav_path, nav_records)
        write_imu_error_file(err_path, imu_err_records)
        print(f"\n结果已写入：")
        print(f"  导航解：{nav_path}")
        print(f"  IMU 误差：{err_path}")

    s2, e2, a, b = run_mech(imu_comp, state0, truth, rate)
    r2 = _print_result("链路二：纯惯导机械编排 + 静止段零偏补偿", s2, e2, a, b)

    s3, e3, a, b = run_mech(imu_raw, state0, truth, rate)
    r3 = _print_result("链路三：纯惯导机械编排 + 原始 IMU 数据", s3, e3, a, b)

    print("\n三条链路误差汇总（位置均方根误差 / 姿态均方根误差）")
    print(f"  链路一（卡尔曼滤波）：{r1[0]:8.4f} m / {r1[1]:6.4f} deg")
    print(f"  链路二（惯导+补偿）：  {r2[0]:8.4f} m / {r2[1]:6.4f} deg")
    print(f"  链路三（惯导+原始）：  {r3[0]:8.4f} m / {r3[1]:6.4f} deg")

    return {
        "nav_path": nav_path if write_output else None,
        "imu_error_path": err_path if write_output else None,
        "states": (s1, s2, s3),
    }
