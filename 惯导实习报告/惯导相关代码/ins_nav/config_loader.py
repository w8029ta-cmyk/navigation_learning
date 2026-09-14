"""解析 KF-GINS 风格 yaml 配置，并转换为 SI 单位。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_yaml_value(text: str, key: str):
    match = re.search(rf"^{key}\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        raise KeyError(f"Missing yaml key: {key}")
    value = match.group(1).split("#", 1)[0].strip()
    if value.startswith("["):
        return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", value)]
    if value.startswith('"') or value.startswith("'"):
        return value.strip("\"'")
    if "." in value or "e" in value.lower():
        return float(value)
    return int(value)


def _parse_imu_noise(text: str) -> dict:
    noise = {}
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("imunoise:"):
            in_block = True
            continue
        if not in_block:
            continue
        if stripped and not line.startswith((" ", "\t")):
            break
        m = re.match(r"(\w+)\s*:\s*\[(.+)\]", stripped)
        if m:
            noise[m.group(1)] = [
                float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", m.group(2))
            ]
            continue
        m = re.match(r"(\w+)\s*:\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)", stripped)
        if m:
            noise[m.group(1)] = float(m.group(2))
    return noise


def _deg_per_h_to_rad_per_s(v):
    return np.asarray(v, dtype=float) * (math.pi / 180.0) / 3600.0


def _deg_sqrt_h_to_rad_sqrt_s(v):
    return np.asarray(v, dtype=float) * (math.pi / 180.0) / 60.0


def _mps_sqrt_h_to_mps_sqrt_s(v):
    return np.asarray(v, dtype=float) / 60.0


def _mgal_to_mps2(v):
    return np.asarray(v, dtype=float) * 1e-5


def _ppm_to_ratio(v):
    return np.asarray(v, dtype=float) * 1e-6


@dataclass
class ImuNoise:
    arw: np.ndarray
    vrw: np.ndarray
    gbstd: np.ndarray
    abstd: np.ndarray
    gsstd: np.ndarray
    asstd: np.ndarray
    corrtime_s: float


@dataclass
class KfGinsConfig:
    imupath: Path
    gnsspath: Path
    truthpath: Path
    outputpath: Path
    imudatarate: int
    starttime: float
    endtime: float
    initpos: list
    initvel: list
    initatt: list
    init_gyro_bias: np.ndarray
    init_acc_bias: np.ndarray
    init_gyro_scale: np.ndarray
    init_acc_scale: np.ndarray
    initpos_std: np.ndarray
    initvel_std: np.ndarray
    initatt_std: np.ndarray
    antlever: np.ndarray
    imu_noise: ImuNoise
    estimate_scale: bool = False

    @property
    def state_dim(self) -> int:
        return 21 if self.estimate_scale else 15


def load_kf_gins_config(config_path=None) -> KfGinsConfig:
    default = Path(__file__).resolve().parent.parent / "dataset" / "kf-gins.yaml"
    path = Path(config_path) if config_path else default
    text = path.read_text(encoding="utf-8")
    root = path.parent.parent

    def resolve(p):
        path_obj = Path(p)
        return path_obj.resolve() if path_obj.is_absolute() else (root / p).resolve()

    noise_raw = _parse_imu_noise(text)
    corrtime_h = float(noise_raw.get("corrtime", 4.0))

    return KfGinsConfig(
        imupath=resolve(_parse_yaml_value(text, "imupath")),
        gnsspath=resolve(_parse_yaml_value(text, "gnsspath")),
        truthpath=path.parent / "truth.nav",
        outputpath=resolve(_parse_yaml_value(text, "outputpath")),
        imudatarate=int(_parse_yaml_value(text, "imudatarate")),
        starttime=float(_parse_yaml_value(text, "starttime")),
        endtime=float(_parse_yaml_value(text, "endtime")),
        initpos=_parse_yaml_value(text, "initpos"),
        initvel=_parse_yaml_value(text, "initvel"),
        initatt=_parse_yaml_value(text, "initatt"),
        init_gyro_bias=_deg_per_h_to_rad_per_s(_parse_yaml_value(text, "initgyrbias")),
        init_acc_bias=_mgal_to_mps2(_parse_yaml_value(text, "initaccbias")),
        init_gyro_scale=_ppm_to_ratio(_parse_yaml_value(text, "initgyrscale")),
        init_acc_scale=_ppm_to_ratio(_parse_yaml_value(text, "initaccscale")),
        initpos_std=np.asarray(_parse_yaml_value(text, "initposstd"), dtype=float),
        initvel_std=np.asarray(_parse_yaml_value(text, "initvelstd"), dtype=float),
        initatt_std=np.deg2rad(np.asarray(_parse_yaml_value(text, "initattstd"), dtype=float)),
        antlever=np.asarray(_parse_yaml_value(text, "antlever"), dtype=float),
        imu_noise=ImuNoise(
            arw=_deg_sqrt_h_to_rad_sqrt_s(noise_raw["arw"]),
            vrw=_mps_sqrt_h_to_mps_sqrt_s(noise_raw["vrw"]),
            gbstd=_deg_per_h_to_rad_per_s(noise_raw["gbstd"]),
            abstd=_mgal_to_mps2(noise_raw["abstd"]),
            gsstd=_ppm_to_ratio(noise_raw["gsstd"]),
            asstd=_ppm_to_ratio(noise_raw["asstd"]),
            corrtime_s=corrtime_h * 3600.0,
        ),
    )
