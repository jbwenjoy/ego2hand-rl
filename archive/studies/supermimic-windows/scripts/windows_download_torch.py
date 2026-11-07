import concurrent.futures
import hashlib
import pathlib
import time
import urllib.request

url = "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl"
size = 1990608063
out = pathlib.Path(".cache/downloads/torch-2.14.1+cu130-cp312-cp312-win_amd64.whl")
out.parent.mkdir(parents=True, exist_ok=True)
expected = "0a09031e93632d14ef49553acd6211ece3ae9cdc0969a369b8c24e3cc4d15be9"
if out.exists():
    with out.open("rb") as existing:
        if hashlib.file_digest(existing, "sha256").hexdigest() == expected:
            print("Official PyTorch wheel already verified")
            raise SystemExit(0)
with out.open("wb") as f:
    f.truncate(size)
chunk = 32 * 1024**2
start = time.monotonic()


def fetch(offset):
    end = min(offset + chunk, size) - 1
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                assert r.status == 206
                assert r.headers["Content-Range"].startswith(f"bytes {offset}-{end}/")
                data = r.read()
            assert len(data) == end - offset + 1
            with out.open("r+b") as f:
                f.seek(offset)
                f.write(data)
            return len(data)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


done = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for i, n in enumerate(pool.map(fetch, range(0, size, chunk)), 1):
        done += n
        print(
            f"{done / size:.0%} {done / 1024**2:.0f} MiB in {time.monotonic() - start:.1f}s",
            flush=True,
        )
h = hashlib.file_digest(out.open("rb"), "sha256").hexdigest()
assert h == expected, h
print("SHA256 VERIFIED", flush=True)
