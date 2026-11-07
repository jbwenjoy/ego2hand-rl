"""One saved label export child, no retries; records actual OS terminal/log pins."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

H = Path(__file__).resolve().parent
PROTOCOL_SHA = "1d71f3f0fec41f10d35f783f8a96866a5447bd14145975f64ac0056514595b03"
SOURCE_SHA = "c3c7ec1fe61ab2f6bce730718f00809844cf7a2f31bbb7ab8f8ee60c2e950b89"

def pin(p):
    b = p.read_bytes(); return dict(path=str(p.resolve()), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

def main():
    p = H/"protocol_v3.json"; source = H/"build_pad_support_labels_v3.py"
    assert pin(p)["sha256"] == PROTOCOL_SHA and pin(source)["sha256"] == SOURCE_SHA
    process = H/"process_v2"; process.mkdir()
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
