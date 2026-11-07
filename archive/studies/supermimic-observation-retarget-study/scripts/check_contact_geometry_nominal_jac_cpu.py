"""Independent central-difference check of penetrating-pair IK derivatives."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from contact_penetration_geometry_nominal import ContactGeometry, ALLOWANCE_M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    g = ContactGeometry(args.model, args.context)
    with np.load(args.candidate, allow_pickle=False) as z:
        qs, poses = z["joint_pos_rad"], z["mapped_object_pose_w"]
    rows, arrays = [], {}
    for frame in (0, 75, 150):
        g.set_object_pose(poses[frame])
        q = qs[frame].copy()
        distances, analytic = g.distance_and_jacobian(q)
        distances, analytic = distances.copy(), analytic.copy()
        fd = np.empty_like(analytic)
        for joint in range(20):
            plus, minus = q.copy(), q.copy()
            plus[joint] += 2e-5
            minus[joint] -= 2e-5
            a = g.distance_and_jacobian(plus)[0].copy()
            b = g.distance_and_jacobian(minus)[0].copy()
            fd[:, joint] = (a - b) / 4e-5
        active = distances < -ALLOWANCE_M - .0001
        error = float(np.abs(analytic[active] - fd[active]).max())
        rows.append({"frame": frame, "active_pairs": int(active.sum()), "max_abs_derivative_error_m_per_rad": error,
                     "passed_fixed_1e-3_gate": error <= 1e-3})
        arrays.update({f"frame{frame}_distances": distances, f"frame{frame}_analytic": analytic,
                       f"frame{frame}_centralFD": fd, f"frame{frame}_active": active})
    np.savez_compressed(args.output / "cases.npz", **arrays)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"status": "passed" if all(r["passed_fixed_1e-3_gate"] for r in rows) else "failed_local_derivative_gate",
              "rows": rows, "query_options": g.query_options,
              "scope": "Three prespecified geometry poses, only pairs deeper than.6mm; not a global smoothness or dynamics/force certificate. Positive/zero-distance anomalies are not used as contact attraction.",
              "script_sha256": sha(Path(__file__)), "geometry_module_sha256": sha(Path(__file__).with_name("contact_penetration_geometry_nominal.py")),
              "candidate_sha256": sha(args.candidate), "model_sha256": sha(args.model), "context_sha256": sha(args.context),
              "dynamics_steps": 0, "physical_reference_qualified": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
