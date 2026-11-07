"""KonIQ-10k -> data/koniq_raw224.npy + data/koniq_index.csv。

下载（断点续传）-> sha256 校验 -> 解压 -> 与 data/transforms.py 同尺寸的 uint8
224x224 memmap。归一化不在缓存里做：raw_to_tensor 读取时再减 MEAN 除 STD，
与 scripts/cache_raw.py / data/utkface.py 的既有链路一致。

数据源是 HuggingFace 数据集仓库 xyliu6/koniq10k-1024x768（官方 DaRUS 文件的
未改动镜像，CC BY 4.0）。本机直连 huggingface.co 不通时用 --endpoint
https://hf-mirror.com，sha256 校验保证镜像内容与官方一致。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from data.koniq import scan
from data.transforms import SIZE

REPO = "xyliu6/koniq10k-1024x768"
ZIP_NAME = "koniq10k_1024x768.zip"
ZIP_BYTES = 5608799013
ZIP_SHA256 = "ea73b96de191c4fd5168699abb9d3ae5ec0c459881e413b28fc0c32537ed7592"
CHUNK = 1 << 22
MAX_RETRIES = 3


def zip_url(endpoint: str, repo: str, name: str) -> str:
    return "%s/datasets/%s/resolve/main/%s" % (endpoint.rstrip("/"), repo, name)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, expect_bytes: int) -> Path:
    """断点续传到 dst.part；dst 已齐则跳过。返回 dst。"""
    part = dst.with_name(dst.name + ".part")
    if dst.is_file():
        got = dst.stat().st_size
        if got == expect_bytes:
            print("[zip] 已存在且字节数一致，跳过下载：%s (%d B)" % (dst, got), flush=True)
            return dst
        raise SystemExit("[zip] %s 只有 %d B（期望 %d B）；要重下请加 --force" % (dst, got, expect_bytes))
    dst.parent.mkdir(parents=True, exist_ok=True)
    fails = 0
    t0 = time.time()
    while True:
        done = part.stat().st_size if part.is_file() else 0
        if done == expect_bytes:
            break
        if done > expect_bytes:
            print("[zip] .part 超长，重下", flush=True)
            part.unlink()
            done = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "koniq-builder/1.0",
                                                       "Range": "bytes=%d-" % done})
            r = urllib.request.urlopen(req, timeout=180)
            if done and getattr(r, "status", 200) != 206:
                r.close()
                print("[zip] 服务器不认 Range，从头下", flush=True)
                part.unlink()
                continue
            if done:
                print("[zip] 从 %d B 处续传（总 %d B）" % (done, expect_bytes), flush=True)
            mark = time.time()
            with r, open(part, "ab") as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    fails = 0
                    if time.time() - mark > 15:
                        sp = (done) / 2**20 / (time.time() - t0)
                        print("[zip] %5.1f%% %6.2f GiB / 5.22 GiB  %.2f MB/s  剩余约 %.0f s"
                              % (100.0 * done / expect_bytes, done / 2**30,
                                 done / 2**20 / (time.time() - t0),
                                 (expect_bytes - done) / 2**20 / max(sp, 1e-6)), flush=True)
                        mark = time.time()
        except Exception as e:
            fails += 1
            print("[zip] 第 %d/%d 次失败：%s: %s" % (fails, MAX_RETRIES, type(e).__name__, e), flush=True)
            if fails >= MAX_RETRIES:
                raise SystemExit("[zip] 连续 %d 次失败，已保存进度到 %s" % (fails, part))
            time.sleep(3 * fails)
    part.replace(dst)
    print("[zip] 下载完成 %s (%d B, %.0fs)" % (dst, dst.stat().st_size, time.time() - t0), flush=True)
    return dst


def verify(path: Path, expect_bytes: int, expect_sha: str) -> None:
    got = path.stat().st_size
    print("[zip] 字节数 %d / 期望 %d  %s" % (got, expect_bytes, "一致" if got == expect_bytes else "不一致"), flush=True)
    t0 = time.time()
    sha = sha256_of(path)
    print("[zip] sha256 实测 %s" % sha, flush=True)
    print("[zip] sha256 期望 %s" % expect_sha, flush=True)
    print("[zip] 校验 %s (%.1fs)" % ("通过" if sha == expect_sha else "失败", time.time() - t0), flush=True)
    if got != expect_bytes or sha != expect_sha:
        raise SystemExit("[zip] 校验失败，不往下走")


def extract(zip_path: Path, out_dir: Path, force: bool) -> Path:
    """解压到 out_dir，返回唯一的 jpg 目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if i.filename.lower().endswith(".jpg")]
        ok = sum(1 for i in infos
                 if (out_dir / i.filename).is_file()
                 and (out_dir / i.filename).stat().st_size == i.file_size)
        if ok == len(infos) and not force:
            print("[unzip] %s 已有全部 %d 张，跳过解压" % (out_dir, ok), flush=True)
        else:
            t0 = time.time()
            zf.extractall(out_dir)
            print("[unzip] %d 张 -> %s (%.0fs)" % (len(infos), out_dir, time.time() - t0), flush=True)
        parents = {str(Path(i.filename).parent) for i in infos}
    if len(parents) != 1:
        raise SystemExit("[unzip] jpg 不在同一目录：%s" % sorted(parents))
    root = out_dir / parents.pop()
    print("[unzip] jpg 目录 %s（%d 张）" % (root, len(infos)), flush=True)
    return root


def build(records: list[dict], raw: Path, index: Path, force: bool) -> None:
    for p in (raw, index):
        if p.exists() and not force:
            raise SystemExit("%s 已存在；要覆盖请加 --force" % p)
    raw.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(str(raw), mode="w+", dtype=np.uint8,
                                   shape=(len(records), SIZE[1], SIZE[0], 3))
    t0 = time.time()
    for i, r in enumerate(records):
        img = Image.open(r["path"]).convert("RGB").resize(SIZE, Image.BILINEAR)
        mm[i] = np.asarray(img, dtype=np.uint8)
        if (i + 1) % 2000 == 0:
            print("[npy] %d/%d (%.0fs)" % (i + 1, len(records), time.time() - t0), flush=True)
    mm.flush()
    del mm
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print("[npy] %s shape=(%d, %d, %d, 3) dtype=uint8 %.3f GiB"
          % (raw, len(records), SIZE[1], SIZE[0], raw.stat().st_size / 2**30), flush=True)
    print("[csv] %s %d 行 列=%s" % (index, len(records), list(records[0])), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="https://huggingface.co",
                   help="HuggingFace 端点；直连不通时用 https://hf-mirror.com")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--zip", default="D:/data/koniq/koniq10k_1024x768.zip")
    p.add_argument("--extract-dir", default="D:/data/koniq")
    p.add_argument("--root", default=None, help="已解压的 jpg 目录；默认自动探测")
    p.add_argument("--scores", default=None, help="打分表 .tab/.csv；默认自动探测")
    p.add_argument("--raw", default="data/koniq_raw224.npy")
    p.add_argument("--index", default="data/koniq_index.csv")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（小样本）")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    zip_path = Path(a.zip)
    download(zip_url(a.endpoint, a.repo, ZIP_NAME), zip_path, ZIP_BYTES)
    verify(zip_path, ZIP_BYTES, ZIP_SHA256)
    root = Path(a.root) if a.root else extract(zip_path, Path(a.extract_dir), a.force)

    recs = scan(root, a.scores)
    print("[scan] %d 张有 MOS（jpg 共 %d 张）"
          % (len(recs), len(list(root.glob("*.jpg")))), flush=True)
    if a.limit:
        recs = recs[:a.limit]
        print("[scan] --limit %d -> 本批 %d 张" % (a.limit, len(recs)), flush=True)
    if not recs:
        raise SystemExit("[scan] 没有可用记录")
    build(recs, Path(a.raw), Path(a.index), a.force)


if __name__ == "__main__":
    main()
