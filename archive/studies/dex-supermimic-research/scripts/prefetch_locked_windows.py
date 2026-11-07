"""Fetch slow Windows wheels from this lock in parallel byte ranges, with SHA256."""

import concurrent.futures
import hashlib
import pathlib
import time
import tomllib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = {"scipy", "imageio-ffmpeg", "mujoco", "tensorboard", "matplotlib", "onnx", "wandb", "triton-windows", "sympy", "numpy", "pillow"}
CHUNK = 4 * 1024**2


def main():
    data = tomllib.loads((ROOT / "uv.lock").read_text())
    tasks = []
    selected = []
    for package in data["package"]:
        if package["name"] not in NAMES:
            continue
        wheels = [wheel for wheel in package.get("wheels", []) if (
            ("cp312-cp312-win_amd64.whl" in wheel["url"]) or
            ("cp312-abi3-win_amd64.whl" in wheel["url"]) or
            wheel["url"].endswith("py3-none-any.whl") or
            wheel["url"].endswith("py2.py3-none-any.whl") or
            wheel["url"].endswith("py3-none-win_amd64.whl"))]
        assert len(wheels) == 1, (package["name"], wheels)
        wheel = wheels[0]
        path = ROOT / ".cache/downloads" / urllib.parse.unquote(wheel["url"].rsplit("/", 1)[1])
        expected = wheel["hash"].split(":")[-1]
        size = wheel.get("size")
        if size is None:
            with urllib.request.urlopen(urllib.request.Request(wheel["url"], headers={"Range": "bytes=0-0"}), timeout=30) as response:
                size = int(response.headers["Content-Range"].split("/")[-1])
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with path.open("wb") as output:
            output.truncate(size)
        selected.append((package["name"], path, expected))
        for start in range(0, size, CHUNK):
            tasks.append((wheel["url"], path, start, min(start + CHUNK, size) - 1))

    def fetch(task):
        url, path, start, end = task
        for attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=60) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
                    content = response.read()
                assert len(content) == end - start + 1
                with path.open("r+b") as output:
                    output.seek(start)
                    output.write(content)
                return len(content)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, _ in enumerate(pool.map(fetch, tasks), 1):
            if n % 10 == 0:
                print(f"{n}/{len(tasks)} ranges in {time.perf_counter() - started:.1f}s", flush=True)
    for name, path, expected in selected:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (name, actual)
        print(f"SHA256 VERIFIED {name} {path.name}", flush=True)


if __name__ == "__main__":
    main()
