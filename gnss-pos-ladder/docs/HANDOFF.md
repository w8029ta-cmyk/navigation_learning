# 备忘

自己接着做时先看这个。原理和踩坑写在 `docs/技术报告.md`。

## 精度（相对 RINEX approx）

| 阶梯 | 自研 | RTKLIB 2.4.2 |
|------|------|----------------|
| A SPP 3D | 1.81 m | 2.33 m |
| B RTK 平面 | 0.77 m | 0.30 m |
| C PPP 3D | 1.21 m（平面 0.45；U≈+1.12） | 0.82 m |

- 自研汇总：`results/compare/ladder_summary.json`
- 对照：`results/rtklib_ref/rtklib_vs_self.json`（若其中 `self.C_PPP` 还是 1.50，是旧快照；以 `ladder_summary` / `ppp_self_summary` 为准，或重跑 `05`）

## 别误改的默认

- PPP：`use_bds=False`（混 BDS 曾 ~4 m，已回退）
- RTK：`use_ekf=false`（会话固定更稳）
- GEO C01–C05：排除
- 评价：RINEX approx，别为了刷 RMS 把解往 approx 上拉
- RTKLIB 路径只在 `config/experiment.yaml`，不进仓

## 踩过的坑

- rnx2rtkp 2.4.2 无 `-sys` → 用 `-k conf`，`pos1-navsys=33`
- BDS week +1356；MEO/IGSO BDT=GPST−14s
- 精密 CLK 必须加相对论（`peph2pos`：−2 r·v/c²）
- ATX：`data/products/atx/igs20_2425.atx`（IGS 下 gz 解压）
- 武大 FTP 用 Python `ftplib`，系统 FTP 易 502

## 接下来想做

1. PPP 冲 U≈1.1 m（ZTD–高程 / 与 RTKLIB 差分）
2. RTK 调稳 `use_ekf` → 平面往 ~0.3 m
3. 可选：修 GEO；BDS PPP 单独分支（别默认混进）
4. 可选：重跑 `05`，刷新对照 JSON 里的自研 PPP 段
