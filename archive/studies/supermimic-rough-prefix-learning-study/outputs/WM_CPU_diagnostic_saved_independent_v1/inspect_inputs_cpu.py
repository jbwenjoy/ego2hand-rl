"""Read saved output schema only; no Torch/MuJoCo/model imports."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/WM_onpolicy_CPU_diagnostic_v1"
report = json.loads((DATA / "report.json").read_text())
for k, v in report.items():
    if k in ("groups", "models"):
        print(k, len(v))
    elif isinstance(v, (list, dict)):
        print(k, repr(v)[:1200])
    else:
        print(k, v)
for name in ("metrics.json", "reward_rows.json", "FD_rows.json", "codec_receipts.json"):
    v = json.loads((DATA / name).read_text())
    print(name, "count", len(v), "first", repr(v[0])[:4000])
for name in ("runplan.json", "protocol.json"):
    v = json.loads((DATA / name).read_text())
    print(name, "keys", list(v))
with np.load(DATA / "predictions_truth_baselines_gradients.npz", allow_pickle=False) as z:
    print("ARRAYS", len(z.files), {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files})
