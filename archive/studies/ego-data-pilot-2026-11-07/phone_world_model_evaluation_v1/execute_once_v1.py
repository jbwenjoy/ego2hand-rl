"""One fixed evaluation child, no retries; records actual OS terminal/log pins."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

H = Path(__file__).resolve().parent
PROTOCOL_SHA = "7b76f9cacfbcf74d4e47d2219b853200f29b9683638acfa8f4f37ef0a1be7ce2"
SOURCE_SHA = "93687fa66636ac060eb614b6730693b4933639797c518dc35319c830cf948439"

def pin(p):
    b = p.read_bytes(); return dict(path=str(p.resolve()), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

def main():
    p = H/"protocol_v1.json"; source = H/"evaluate_phone_wm.py"
    assert pin(p)["sha256"] == PROTOCOL_SHA and pin(source)["sha256"] == SOURCE_SHA
    process = H/"process_v1"; process.mkdir()
    cmd = [sys.executable, "-X", "utf8", str(source), "--protocol", str(p), "--run"]
    start = time.perf_counter(); now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (process/"stdout.log").open("xb") as stdout, (process/"stderr.log").open("xb") as stderr:
        child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with (process/"started.json").open("x") as f:
            json.dump(dict(pid=child.pid, command=cmd, source=pin(source), protocol=pin(p), started_UTC=now), f, indent=2)
        code = child.wait()
    terminal = dict(schema="phone_fixed_final_evaluation_process_terminal:1", pid=child.pid,
        returncode=code, command=cmd, elapsed_s=time.perf_counter()-start,
        source=pin(source), launcher=pin(Path(__file__)), protocol=pin(p),
        stdout=pin(process/"stdout.log"), stderr=pin(process/"stderr.log"), automatic_retry=False)
    with (process/"terminal.json").open("x") as f:
        json.dump(terminal, f, indent=2)
    print(json.dumps(dict(returncode=code, terminal=pin(process/"terminal.json"))))
    raise SystemExit(code)

if __name__ == "__main__":
    main()
