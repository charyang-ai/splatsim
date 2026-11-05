#!/usr/bin/env python
"""Download the parts of the upstream Google Drive data that the robot_iphone replay uses.

The README links three zips: test_data.zip (38.9 GB), output.zip (20.8 GB) and
bc_data.zip (76 MB). The apple-on-plate replay reads one COLMAP scene, three
splats and the trajectories, so this script reads each zip's central directory
with HTTP range requests and extracts only those members into data/.

    python radeon/fetch_data.py [--data data] [--views 0 3 120 240 360 480]
"""
import argparse
import fnmatch
import io
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
TEST_DATA = "14D3fFtaPX4GBe9dSJLKAIvUYlgK7fUxS"
OUTPUT = "1rAUkf7l2ZZqG1Bm3ih6cAO5HCd9dSTO-"
BC_DATA = "1NhSBNYMi51hETAspk6vN7F-Ih1134_lt"


class RangeReader(io.RawIOBase):
    """Seekable view of a remote file. Sequential reads share one HTTP stream."""

    def __init__(self, url: str):
        self.url, self.pos, self.resp, self.resp_pos = url, 0, None, -1
        with self._open(0, 0) as r:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _open(self, start, end=None):
        rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
        req = urllib.request.Request(self.url, headers={"Range": rng})
        for attempt in range(6):
            try:
                r = urllib.request.urlopen(req, timeout=120)
                if r.status != 206:
                    raise IOError(f"expected HTTP 206, got {r.status}")
                return r
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"retry {attempt + 1} at byte {start}: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if self.resp is None or self.resp_pos != self.pos:
            if self.resp is not None:
                self.resp.close()
            self.resp, self.resp_pos = self._open(self.pos), self.pos
        got = 0
        view = memoryview(b)
        while got < n:
            try:
                k = self.resp.readinto(view[got:n])
            except Exception as exc:  # noqa: BLE001
                print(f"stream error at byte {self.resp_pos}: {exc}; reopening", file=sys.stderr)
                self.resp.close()
                self.resp = self._open(self.resp_pos)
                continue
            if not k:
                break
            got += k
            self.resp_pos += k
        self.pos += got
        return got


def extract(file_id: str, dest: Path, patterns) -> None:
    zf = zipfile.ZipFile(io.BufferedReader(RangeReader(URL.format(file_id)), buffer_size=8 << 20))
    picked = [i for i in zf.infolist() if not i.is_dir() and any(fnmatch.fnmatch(i.filename, p) for p in patterns)]
    total = sum(i.compress_size for i in picked)
    print(f"{file_id}: {len(picked)} files, {total / 2**20:.1f} MiB")
    done, t0 = 0, time.time()
    for info in sorted(picked, key=lambda i: i.header_offset):
        target = dest / info.filename
        if not (target.exists() and target.stat().st_size == info.file_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            tmp.replace(target)
        done += info.compress_size
        print(f"  {done / 2**20:8.1f}/{total / 2**20:.1f} MiB {done / 2**20 / max(time.time() - t0, 1e-3):5.1f} MiB/s"
              f"  {info.filename}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--robot", default="robot_iphone")
    ap.add_argument("--views", type=int, nargs="*", default=[0, 3, 120, 240, 360, 480],
                    help="authors' renders to fetch for radeon/check_render.py")
    args = ap.parse_args()
    dest, r = Path(args.data), args.robot

    splats = [f"output/{o}/point_cloud/iteration_7000/*" for o in
              ("plastic_apple", "plate", "plastic_banana", "plastic_orange", "plastic_strawberry")]
    renders = [f"output/{r}/train/ours_30000/{kind}/{v:05d}.png" for v in args.views for kind in ("renders", "gt")]
    extract(OUTPUT, dest, [f"output/{r}/cfg_args", f"output/{r}/cameras.json",
                           f"output/{r}/point_cloud/iteration_30000/*", *splats, *renders])
    extract(TEST_DATA, dest, [f"test_data/{r}/sparse/*", f"test_data/{r}/images/*"])
    extract(BC_DATA, dest, ["bc_data/gello/*"])


if __name__ == "__main__":
    main()
