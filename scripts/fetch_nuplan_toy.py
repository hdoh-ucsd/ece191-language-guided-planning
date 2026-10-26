#!/usr/bin/env python3
"""Fetch a tiny nuPlan v1.1 subset (one log + one city map) without downloading the archives.

The public Motional bucket serves the 8.5 GB mini-split zip and the 1 GB maps zip with
HTTP range support, so single members can be read straight out of the remote zips.
Run on the host (no nuPlan tooling needed); only the standard library is used.

Example:
  python3 scripts/fetch_nuplan_toy.py --dest nuplan_data \
      --log 2021.08.24.13.12.55_veh-45_00386_00472 --city us-pa-pittsburgh-hazelwood

Data is CC BY-NC-SA 4.0 plus Motional's additional terms (non-commercial); the LICENSE
file is written next to the data.
"""
from __future__ import annotations
import argparse
import io
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BUCKET = 'https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1'
MINI_ZIP = f'{BUCKET}/nuplan-v1.1_mini.zip'
MAPS_ZIP = f'{BUCKET}/nuplan-maps-v1.0.zip'
MAP_INDEX = 'maps/nuplan-maps-v1.0.json'
RETRIES = 6


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests."""

    def __init__(self, url: str):
        self.url, self.pos, self.fetched = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
            if 'bytes' not in (r.headers.get('Accept-Ranges') or ''):
                raise RuntimeError(f'{url} does not support range requests')

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
                print(f'  retry {attempt + 1} after {e}', file=sys.stderr)
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, RangeFile]:
    rf = RangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(rf, buffer_size=8 << 20)), rf


def extract(zf: zipfile.ZipFile, member: str, target: Path) -> None:
    """Stream one member to `target`; zipfile verifies the CRC as it reads."""
    info = zf.getinfo(member)
    if target.exists() and target.stat().st_size == info.file_size:
        print(f'  exists, skipping: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.part')
    print(f'  {member}  ({info.file_size / 1e6:.1f} MB) -> {target}')
    with zf.open(info) as src, open(tmp, 'wb') as dst:
        while chunk := src.read(8 << 20):
            dst.write(chunk)
    if tmp.stat().st_size != info.file_size:
        raise RuntimeError(f'size mismatch for {member}')
    os.replace(tmp, target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dest', type=Path, required=True, help='nuPlan data root (becomes NUPLAN_DATA_ROOT)')
    p.add_argument('--log', action='append', required=True, help='log name without .db; repeatable')
    p.add_argument('--city', action='append', required=True,
                   help='map location of the log(s), e.g. us-pa-pittsburgh-hazelwood; repeatable')
    p.add_argument('--list', action='store_true', help='only list the logs in the mini split and exit')
    args = p.parse_args()

    mini, mini_rf = open_remote_zip(MINI_ZIP)
    if args.list:
        for zi in sorted(mini.infolist(), key=lambda z: z.file_size):
            if zi.filename.endswith('.db'):
                print(f'{zi.file_size / 1e6:8.1f} MB  {Path(zi.filename).stem}')
        return

    print('mini split:')
    extract(mini, 'LICENSE', args.dest / 'LICENSE')
    for log in args.log:
        extract(mini, f'data/cache/mini/{log}.db', args.dest / 'nuplan-v1.1' / 'splits' / 'mini' / f'{log}.db')

    maps, maps_rf = open_remote_zip(MAPS_ZIP)
    print('maps:')
    extract(maps, MAP_INDEX, args.dest / MAP_INDEX)
    for city in args.city:
        members = [zi.filename for zi in maps.infolist()
                   if zi.filename.startswith(f'maps/{city}/') and zi.filename.endswith('map.gpkg')]
        if not members:
            raise SystemExit(f'no map.gpkg for city {city!r}; known: '
                             + ', '.join(sorted({zi.filename.split("/")[1] for zi in maps.infolist()
                                                 if zi.filename.count("/") >= 2})))
        for m in members:
            extract(maps, m, args.dest / m)

    print('verifying logs:')
    for log in args.log:
        db = args.dest / 'nuplan-v1.1' / 'splits' / 'mini' / f'{log}.db'
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        ok = con.execute('PRAGMA integrity_check').fetchone()[0]
        loc = con.execute('SELECT location FROM log').fetchone()[0]
        n = con.execute('SELECT COUNT(*) FROM lidar_pc').fetchone()[0]
        print(f'  {db.name}: integrity={ok} location={loc} frames={n}')
        if ok != 'ok':
            raise SystemExit(f'{db} failed integrity check')
        if loc not in args.city:
            raise SystemExit(f'{db} is in {loc}, but only maps for {args.city} were fetched')
    print(f'done; transferred {(mini_rf.fetched + maps_rf.fetched) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
