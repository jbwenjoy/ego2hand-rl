"""One saved label export child, no retries; records actual OS terminal/log pins."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

H = Path(__file__).resolve().parent
PROTOCOL_SHA = "29b3238d820599dde1f1c7ab2b910c3d8933a808d9ca3d534c177a46e1089b0f"
SOURCE_SHA = "cb56da625ed282fd30792c14b3ad05dcdcd5cddcf14d1f405e8eb580cde6e9aa"

def pin(p):
    b = p.read_bytes(); return dict(path=str(p.resolve()), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

def main():
    p = H/"protocol_v2.json"; source = H/"build_pad_support_labels.py"
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
    terminal = dict(schema="phone_pad_support_label_process_terminal:2", pid=child.pid,
        returncode=code, command=cmd, elapsed_s=time.perf_counter()-start,
        source=pin(source), launcher=pin(Path(__file__)), protocol=pin(p),
        stdout=pin(process/"stdout.log"), stderr=pin(process/"stderr.log"), automatic_retry=False)
    with (process/"terminal.json").open("x") as f:
        json.dump(terminal, f, indent=2)
    print(json.dumps(dict(returncode=code, terminal=pin(process/"terminal.json"))))
    raise SystemExit(code)

if __name__ == "__main__":
    main()
