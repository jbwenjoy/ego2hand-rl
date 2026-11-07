"""Fetch only the official single-axis clip group using HTTP byte ranges.

Avoids downloading checkpoint payloads in the 247 MB upstream release archive.
Zip CRCs and the catalog's SHA256 hashes are both checked.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.10.22/wuji-mjlab-v2026.10.22-assets.zip"
ROOT = Path(__file__).resolve().parents[1]


class RemoteFile(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.cache = []
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": "bytes=0-0"}), timeout=30) as response:
            assert response.status == 206, "Server must support ranges"
            self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.downloaded = 1

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        return self.pos

    def preload(self, start, stop):
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={start}-{stop - 1}"}), timeout=90) as response:
            assert response.status == 206
            data = response.read()
        assert len(data) == stop - start
        self.downloaded += len(data)
        self.cache.append((start, data))

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if not n:
            return b""
        end = min(self.pos + n, self.size)
        for start, data in reversed(self.cache):
            if start <= self.pos and end <= start + len(data):
                out = data[self.pos - start:end - start]
                self.pos = end
                return out
        self.preload(self.pos, min(max(end, self.pos + 65536), self.size))
        return self.read(n)


def main():
    destination = ROOT / "src/dex_supermimic/tasks/pen_spin/data/rawdata/clips"
    remote = RemoteFile()
    with zipfile.ZipFile(remote) as bundle:
        marker = "/pen_spin/data/rawdata/clips/"
        selected = [entry for entry in bundle.infolist() if marker + "01_single_axis/" in entry.filename and not entry.is_dir()]
        assert selected
        start = min(entry.header_offset for entry in selected)
        last = max(entry.header_offset for entry in selected)
        after = [entry.header_offset for entry in bundle.infolist() if entry.header_offset > last]
        stop = min(after) if after else bundle.start_dir
        remote.preload(start, stop)
        installed = []
        for entry in selected:
            relative = Path(entry.filename.split(marker, 1)[1])
            assert not relative.is_absolute() and ".." not in relative.parts
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = bundle.read(entry)
            target.write_bytes(data)
            installed.append(dict(path=relative.as_posix(), sha256=hashlib.sha256(data).hexdigest()))
    group = destination / "01_single_axis"
    index = json.loads((group / "index.json").read_text())
    for clip in index["clips"]:
        assert hashlib.sha256((group / clip["file"]).read_bytes()).hexdigest() == clip["sha256"]
    result = dict(url=URL, archive_bytes=remote.size, downloaded_bytes=remote.downloaded,
                  clip_count=len(index["clips"]), files=installed)
    output = ROOT / "outputs/windows_bringup_20261101/pen_minimal_download.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
