# RTKLIB 对照速查

详细原理、算法、调试和结果写在 [`技术报告.md`](技术报告.md)。

同数据、同 RINEX approx 参考；自研 vs RTKLIB 2.4.2：

| A SPP 3D | B 平面 | C PPP 3D |
|----------|--------|----------|
| 1.81 / 2.33 m | 0.77 / 0.30 m | 1.21 / 0.82 m |

复现：`python scripts/05_run_rtklib_compare.py`

对照时我翻 RTKLIB 源码的入口：SPP → `pntpos.c`，RTK → `rtkpos.c`，PPP → `ppp.c` / `preceph.c`。
