# gnss-pos-ladder

王明 · 安徽理工大学 · 导航工程

课设里一般只做到伪距单点。我想把同一天、同一批公开数据，自己写成一条能讲清楚的链路：

伪距 SPP（米级）→ 短基线相对定位（分米平面）→ 精密单点定位 PPP（亚米），再用本机 RTKLIB 2.4.2 同数据对照。

说明文档在 [`docs/技术报告.md`](docs/技术报告.md)，回来找文件可以看 [`docs/目录导读.md`](docs/目录导读.md)。

## 做了什么

- 观测方程、选星、估计器、产品解析都是 Python 自己写的（`ladder/`）
- 本机 `rnx2rtkp.exe` 只用来对照，不把对照结果当成我的结果
- 评价相对 RINEX 头文件近似坐标（approx），不是 IGS 周解真值

## 当前数字（相对 RINEX approx）

以 `results/compare/ladder_summary.json` 和 `results/rtklib_ref/rtklib_vs_self.json` 里 RTKLIB 段为准；自研 PPP 看最新 `ladder_summary` / `ppp_self_summary`（对照 JSON 里自研 PPP 若还是 1.50，是旧快照，重跑 `05` 就行）。

| 阶梯 | 自研 | RTKLIB 2.4.2 |
|------|------|----------------|
| A SPP 3D | 1.81 m | 2.33 m |
| B RTK 平面 / 3D | 0.77 m / 1.70 m | 0.30 m / 2.89 m |
| C PPP 3D（平面） | 1.21 m（0.45 m） | 0.82 m（0.55 m） |

## 怎么跑

在 `gnss-pos-ladder/` 下：

```bash
pip install -r requirements.txt
python scripts/00_fetch_data.py
python scripts/01_run_spp.py
python scripts/02_run_rtk.py
python scripts/03_run_ppp.py
python scripts/04_compare_all.py
python scripts/05_run_rtklib_compare.py
```

- 数据：武大 FTP `igs.gnsswhu.cn`（要用 Python `ftplib`；PowerShell/系统 FTP 容易 502）
- RTKLIB 路径写在 `config/experiment.yaml` → `paths.rtklib_bin`，别拷进仓库

## 目前限制

1. 北斗 GEO（C01–C05）算星会发散，先排除了
2. PPP 默认只用 GPS（试加 BDS 变差到约 4 m，已回退）
3. RTK 主路径是分会话取整固定；`use_ekf` 骨架在，默认关（易误固定）
4. 评价参考是 RINEX approx，不是 IGS 周解真值
5. PPP 高程还有约 +1.1 m 系统偏（NGL 与 approx 高程差只有 cm 级，不全是参考坐标问题）

## 目录

```
config/          实验开关 + RTKLIB conf
ladder/          算法
scripts/00–06    拉数 / 三阶 / 汇总 / 对照 / PPP 诊断
results/         结果 + rtklib_ref/
docs/            技术报告、备忘
```
