"""整周模糊度：降相关 + 整数最小二乘搜索（精简 LAMBDA）。

超短基线浮点一般已贴近整数；这里把协方差相关性拆开再搜。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _ldlt(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = Q.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    A = Q.copy()
    for k in range(n):
        D[k] = A[k, k]
        if D[k] < 1e-18:
            D[k] = 1e-18
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / D[k]
            for j in range(k + 1, i + 1):
                A[i, j] -= L[i, k] * L[j, k] * D[k]
                A[j, i] = A[i, j]
    return L, D


def decorrelate(a: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 z=Z^{-1}a, Qz, Z。"""
    n = len(a)
    Z = np.eye(n)
    Qw = 0.5 * (Q + Q.T)
    for _pass in range(3):
        L, D = _ldlt(Qw)
        for i in range(n - 1):
            for j in range(i + 1, n):
                mu = round(L[j, i])
                if mu == 0:
                    continue
                # integer Gauss: subtract mu * col i from col j on L factors via Z
                Z[:, i] = Z[:, i] + mu * Z[:, j]
                Qw[:, i] = Qw[:, i] + mu * Qw[:, j]
                Qw[i, :] = Qw[i, :] + mu * Qw[j, :]
    # Ensure SPD
    Qw = 0.5 * (Qw + Qw.T) + np.eye(n) * 1e-12
    try:
        Zinv = np.linalg.inv(Z)
    except np.linalg.LinAlgError:
        return a.copy(), Q.copy(), np.eye(n)
    z = Zinv @ a
    Qz = Zinv @ Q @ Zinv.T
    Qz = 0.5 * (Qz + Qz.T) + np.eye(n) * 1e-12
    return z, Qz, Z


def _ils_search(z: np.ndarray, Qz: np.ndarray, maxcan: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    n = len(z)
    L, D = _ldlt(Qz)
    # Collect candidates via recursive conditional rounding of nearest few ints
    cands: list[np.ndarray] = []
    costs: list[float] = []

    def rec(k: int, cost: float, az: np.ndarray):
        if len(cands) > 200:
            return
        if k < 0:
            cands.append(az.copy())
            costs.append(cost)
            return
        # conditional mean
        zk = z[k]
        for j in range(k + 1, n):
            zk -= L[j, k] * (az[j] - z[j])
        nearest = int(np.floor(zk + 0.5))
        for d in (0, 1, -1, 2, -2, 3, -3):
            zi = nearest + d
            r = zk - zi
            c2 = cost + (r * r) / D[k]
            if c2 > 1e5:
                continue
            az[k] = zi
            rec(k - 1, c2, az)

    rec(n - 1, 0.0, np.zeros(n))
    if not cands:
        r = np.round(z)
        return r.reshape(1, -1), np.array([float((z - r) @ np.linalg.solve(Qz, z - r))])
    order = np.argsort(costs)
    pick = order[:maxcan]
    return np.array([cands[i] for i in pick]), np.array([costs[i] for i in pick])


def lambda_fix(
    afloat: np.ndarray,
    Qa: np.ndarray,
    ratio_th: float = 2.0,
) -> Tuple[np.ndarray, float, bool]:
    afloat = np.asarray(afloat, float).ravel()
    Qa = np.asarray(Qa, float)
    n = afloat.size
    if n == 0:
        return afloat, 0.0, False
    Qa = 0.5 * (Qa + Qa.T) + np.eye(n) * 1e-10
    z, Qz, Z = decorrelate(afloat, Qa)
    C, chi = _ils_search(z, Qz, maxcan=2)
    zhat = C[0]
    afix = Z @ zhat
    ratio = float(chi[1] / chi[0]) if len(chi) > 1 and chi[0] > 1e-18 else 0.0
    return afix, ratio, ratio >= ratio_th
