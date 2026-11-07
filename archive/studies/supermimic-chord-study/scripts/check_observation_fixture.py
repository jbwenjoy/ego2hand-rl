"""CPU data-leak contract checks; keep negative inputs for reproducibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_observation_fixture import FIELDS, export


def check(source: Path, fixture: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source, allow_pickle=False) as z:
        raw = {key: z[key].copy() for key in z.files}
    for key in raw:
        if key not in set(FIELDS.values()):
            fill = -999 if np.issubdtype(raw[key].dtype, np.integer) else np.nan
            raw[key] = np.full_like(raw[key], fill)
    poison = output / "poison_robot_labels.npz"
    np.savez_compressed(poison, **raw)
    poison.with_suffix(".json").write_bytes(source.with_suffix(".json").read_bytes())
    export(poison, output / "poison_export")
    with np.load(fixture, allow_pickle=False) as a, np.load(output / "poison_export/observations.npz", allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    raw["time_s"][1] = raw["time_s"][0]
    bad = output / "nonmonotone.npz"
    np.savez_compressed(bad, **raw)
    bad.with_suffix(".json").write_bytes(source.with_suffix(".json").read_bytes())
    try:
        export(bad, output / "should_not_exist")
    except ValueError as exc:
        reason = str(exc)
    else:
        raise AssertionError("Nonmonotonic input was accepted")
    assert not (output / "should_not_exist").exists()
    report = {"prohibited_label_poison_invariance": "PASS", "allowlist_roundtrip": "PASS",
              "nonmonotonic_time_rejected_before_output": "PASS", "negative_case_reason": reason,
              "gpu_used": False, "claim": "Data-view boundary only; no physical adapter qualification"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.fixture, args.output)))
