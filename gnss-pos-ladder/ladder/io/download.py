"""从武大 IGS 数据中心拉本实验要用的文件。

不用 rtkget（URL 列表很多过时）。本机 PowerShell FTP 对武大会 502，改用 ftplib。
"""

from __future__ import annotations

import ftplib
import gzip
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable

from ladder.util.config import gps_week_dow, load_config, yyyy_ddd


class WhuFtp:
    def __init__(self, host: str = "igs.gnsswhu.cn", timeout: int = 120):
        self.host = host
        self.timeout = timeout
        self.ftp: ftplib.FTP | None = None

    def __enter__(self) -> "WhuFtp":
        self.ftp = ftplib.FTP(self.host, timeout=self.timeout)
        self.ftp.login()
        self.ftp.encoding = "latin-1"
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.ftp is not None:
            try:
                self.ftp.quit()
            except Exception:
                try:
                    self.ftp.close()
                except Exception:
                    pass
        self.ftp = None

    def nlst(self, path: str) -> list[str]:
        assert self.ftp is not None
        try:
            return self.ftp.nlst(path)
        except ftplib.error_perm:
            self.ftp.cwd(path)
            return self.ftp.nlst()

    def download(self, remote: str, local: Path) -> Path:
        assert self.ftp is not None
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size > 0:
            print(f"[skip] {local.name}")
            return local
        tmp = local.with_suffix(local.suffix + ".part")
        with open(tmp, "wb") as f:
            self.ftp.retrbinary(f"RETR {remote}", f.write)
        tmp.replace(local)
        print(f"[get ] {remote} -> {local}")
        return local


def _gunzip(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dst


def _decompress_z(src: Path, dst: Path) -> Path:
    """处理 .Z（Unix compress）。优先系统/RTKLIB gzip，再尝试内置。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    # RTKLIB 自带 gzip.exe 往往能解 .Z
    cfg = load_config()
    gzip_exe = Path(cfg["paths"]["rtklib_bin"]) / "gzip.exe"
    if gzip_exe.exists():
        # gzip -dkc file.Z
        with open(dst, "wb") as out:
            subprocess.run(
                [str(gzip_exe), "-dc", str(src)],
                check=True,
                stdout=out,
            )
        return dst
    # fallback: rename tricks with Python gzip sometimes fails on .Z
    raise RuntimeError(f"无法解压 {src}，请确认 RTKLIB bin 下有 gzip.exe")


def crx_to_rnx(crx_path: Path, out_dir: Path, rtklib_bin: Path) -> Path:
    """Hatanaka 解压。

    RTKLIB 2.4.2 自带 crx2rnx 不认 RINEX3 长名 `.crx`，官方提示用管道：
    `crx2rnx - < file.crx > file.rnx`
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = rtklib_bin / "crx2rnx.exe"
    if not exe.exists():
        raise FileNotFoundError(exe)

    if crx_path.suffix.lower() == ".crx":
        dest = out_dir / (crx_path.stem + ".rnx")
    elif crx_path.name.endswith("d"):
        dest = out_dir / (crx_path.name[:-1] + "o")
    else:
        dest = out_dir / (crx_path.name + ".rnx")

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    with open(crx_path, "rb") as fin, open(dest, "wb") as fout:
        proc = subprocess.run(
            [str(exe), "-"],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode != 0 or dest.stat().st_size == 0:
        if dest.exists():
            dest.unlink(missing_ok=True)
        err = proc.stderr.decode("latin-1", errors="ignore")
        raise RuntimeError(f"crx2rnx 失败: {crx_path}\n{err}")
    return dest

def _pick(names: Iterable[str], keywords: list[str]) -> str | None:
    up = [(n, n.upper()) for n in names]
    for kw in keywords:
        for n, u in up:
            if kw.upper() in u:
                return n.split("/")[-1]
    return None


def fetch_experiment_data(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    year = int(cfg["experiment"]["year"])
    doy = int(cfg["experiment"]["doy"])
    yyyy, ddd, yy = yyyy_ddd(year, doy)
    week, dow = gps_week_dow(year, doy)
    data_root = Path(cfg["paths"]["data_root"])
    raw = data_root / "raw" / f"{yyyy}{ddd}"
    rnx_dir = data_root / "rinex" / f"{yyyy}{ddd}"
    prod_dir = data_root / "products" / f"{week}"
    raw.mkdir(parents=True, exist_ok=True)
    rnx_dir.mkdir(parents=True, exist_ok=True)
    prod_dir.mkdir(parents=True, exist_ok=True)

    rover = cfg["stations"]["rover"].upper()
    base = cfg["stations"]["base"].upper()
    host = cfg["data"]["primary_host"]
    out: dict[str, Path] = {}

    with WhuFtp(host) as whu:
        # --- observations (prefer long RINEX3 Hatanaka) ---
        obs_dir = f"/pub/gps/data/daily/{yyyy}/{ddd}/{yy}d"
        names = [n.split("/")[-1] for n in whu.nlst(obs_dir)]
        for sta, key in [(rover, "rover_obs"), (base, "base_obs")]:
            pick = _pick(names, [f"{sta}00CHN", f"{sta}00", sta])
            if pick is None:
                raise FileNotFoundError(f"测站 {sta} 在 {obs_dir} 未找到")
            local = whu.download(f"{obs_dir}/{pick}", raw / pick)
            out[key + "_compressed"] = local

        # --- broadcast mixed nav ---
        nav_dir = f"/pub/gps/data/daily/{yyyy}/{ddd}/{yy}p"
        nav_names = [n.split("/")[-1] for n in whu.nlst(nav_dir)]
        pref = cfg["data"]["broadcast"]["preferred"]
        fb = cfg["data"]["broadcast"]["fallback"]
        nav_pick = _pick(nav_names, [pref, fb, "BRDM", "BRDC"])
        if nav_pick is None:
            raise FileNotFoundError("未找到混合广播星历")
        out["nav_gz"] = whu.download(f"{nav_dir}/{nav_pick}", raw / nav_pick)

        # --- precise products ---
        prod_remote = f"/pub/gps/products/{week}"
        prod_names = [n.split("/")[-1] for n in whu.nlst(prod_remote)]
        tag = f"{cfg['data']['products']['sp3_prefix']}_{yyyy}{ddd}0000_"
        sp3_cands = [n for n in prod_names if n.startswith(tag) and "SP3" in n.upper()]
        clk_cands = [
            n
            for n in prod_names
            if n.startswith(f"{cfg['data']['products']['clk_prefix']}_{yyyy}{ddd}0000_") and "CLK" in n.upper()
        ]
        erp_cands = [
            n
            for n in prod_names
            if n.startswith(f"{cfg['data']['products']['erp_prefix']}_{yyyy}{ddd}0000_") and "ERP" in n.upper()
        ]
        bia_cands = [
            n
            for n in prod_names
            if n.startswith(f"{cfg['data']['products']['bia_prefix']}_{yyyy}{ddd}0000_") and "BIA" in n.upper()
        ]
        # 优先 30s 钟差
        clk_cands = sorted(clk_cands, key=lambda n: ("30S" not in n, n))
        picks = {
            "sp3": sp3_cands[0] if sp3_cands else None,
            "clk": clk_cands[0] if clk_cands else None,
            "erp": erp_cands[0] if erp_cands else None,
            "bia": bia_cands[0] if bia_cands else None,
        }
        for label, fname in picks.items():
            if not fname:
                print(f"[warn] missing product {label}")
                continue
            out[label + "_gz"] = whu.download(f"{prod_remote}/{fname}", prod_dir / fname)
    # decompress
    rtklib_bin = Path(cfg["paths"]["rtklib_bin"])
    for key, sta_key in [("rover", "rover_obs"), ("base", "base_obs")]:
        comp = out[sta_key + "_compressed"]
        if comp.name.endswith(".gz"):
            # could be .crx.gz or .26d.gz
            inner_name = comp.name[:-3]
            inner = raw / inner_name
            _gunzip(comp, inner)
        elif comp.name.endswith(".Z"):
            inner = raw / (comp.stem)  # strip .Z → keep .26d
            # stem of file.26d.Z in pathlib: file.26d
            inner = raw / comp.name[:-2] if comp.name.endswith(".Z") else raw / comp.stem
            _decompress_z(comp, inner)
        else:
            inner = comp

        if inner.suffix.lower() == ".crx" or inner.name.endswith("d"):
            rnx = crx_to_rnx(inner, rnx_dir, rtklib_bin)
        else:
            rnx = rnx_dir / inner.name
            if inner.resolve() != rnx.resolve():
                shutil.copy2(inner, rnx)
        out[sta_key] = rnx

    nav_gz = out["nav_gz"]
    nav_path = rnx_dir / nav_gz.name.replace(".gz", "")
    _gunzip(nav_gz, nav_path)
    out["nav"] = nav_path

    for k in list(out.keys()):
        if k.endswith("_gz") and k.split("_")[0] in {"sp3", "clk", "erp", "bia"}:
            gz = out[k]
            dst = prod_dir / gz.name.replace(".gz", "")
            _gunzip(gz, dst)
            out[k.replace("_gz", "")] = dst

    # 记一下下了哪些文件
    man = data_root / f"manifest_{yyyy}{ddd}.txt"
    with open(man, "w", encoding="utf-8") as f:
        f.write(f"year={year} doy={doy} week={week} dow={dow}\n")
        for k, v in sorted(out.items()):
            f.write(f"{k}={v}\n")
    out["manifest"] = man
    return out
