# Constrained prefix nonlinear producer

`scripts/build_constrained_prefix_geometry_cpu.py --plan JSON --output NEWDIR` defaults to check-only. `--check` is explicit check-only; neither path calls an LP or optimizer. `--run` requires a separately authorized Root plan. Current implementation checks did not run either optimization phase or physics.

`ConstrainedPrefixGeometryProblem` in `src/constrained_prefix/problem.py` exposes:

- `controls_from_x(x340)` → `[18,20]`, with source64 P0 fixed outside the variable vector.
- `q_from_x(x)` → `[16,20]`; `sample_q(x,B61)` → `[61,20]`. Evaluation uses `q0 + B_free @ (P_free-q0)`, preserving a constant repeated-q0 curve exactly.
- `fit(x)` → the unchanged 6,596 parent residuals, sparse CSR `[6596,340]` Jacobian, and 16 saved local diagnostics. The parent `[6596,300]` Jacobian is chained through `kron(B16[1:,1:], I20)`.
- `nonlinear(x)` → 122 operational maximum-generated-overlap depths **scaled by 1 mm**, sparse CSR `[122,340]` Jacobian, and 61 complete local contact diagnostics. The upper bound is **2.999** in scaled units; the later qualification bound remains 3 mm.
- `diagnostic_arrays`, `diagnostic_summary`, `native_records`, `counters`, and `stage` provide source, geometry, tie, missing-contact and lifecycle evidence.

The object trajectory is the frozen parent SO(3)/translation quintic driven by mapped poses and full151 rawFD derivatives. No old full151 robot q or failed-prefix q is read as a seed, output, derivative or regularizer. Parent bone, material, soft-overlap, temporal and nominal residual definitions are unchanged. Material bindings stay hypotheses with unknown confidence.

The fixed nonlinear row order is sample-major, then `hand_object`, `hand_self`. Each row selects the exact deepest generated pair, then the first deepest native contact. Diagnostic tolerance ties across pairs and contacts are retained. When a category has no generated contact, its operational violation and Jacobian are zero while its physical gap remains UNKNOWN; the per-pair raw distance/Jacobian stay NaN with a false mask. These are 61 sampled constraints, not a continuous positive-clearance certificate, and selected native contact Jacobians remain local and branch-sensitive.

The accepted plan schema is `constrained_prefix_geometry_cpu_plan:1`, with full frozen `inputs`, `protocol` (0798), `parent_protocol` (94ea), flattened executed `source_pins`, `output`, and fixed fields:

```json
{
  "frames": 16, "joints": 20, "geometry_samples": 61,
  "n_control": 18, "free_variables": 340,
  "maxiter": 80, "max_evaluations": 250,
  "soft_wall_seconds": 220, "owned_process_seconds": 240,
  "planning_depth_bound_m": 0.002999,
  "control_margin_rad": 0.000002,
  "force_margin_Nm": 0.000002,
  "rate_margin_rad_s": 0.0001,
  "automatic_retry": false,
  "retarget_variant_id": "H123_cubic_bspline_constrained_prefix16_v1",
  "optimization_authorized": false
}
```

The two future phases are one aggregate HiGHS L1 nearest-repeat-q0 LP (680 variables; no tie-break), then only after LP success one SLSQP fit. The LP primal/dual tolerances are 1e−9 and its time limit is the remaining whole-worker soft budget. SLSQP has maxiter80, ftol1e−8, and at most 250 objective-function calls; constraint/Jacobian/callback/unique-trial and geometry calls have separate counters. No alternative seed or changed weight is selected after failure.

Every objective/constraint Jacobian and major-iterate callback atomically commits `latest_iterate.npz`, complete raw contact JSON and a SHA-linked header using the reviewed bounded same-payload IO. The header records PID, parent PID, plan SHA, monotonic elapsed time, counters and feasibility separately. A callback is a **major iterate**, never automatically a feasible accepted state. Unique requested trial vectors are saved by SHA. Geometry-only guards reject step/forward/inverse/geomDistance/contactForce; native kinematics, collision and point Jacobians remain allowed. All CPU thread counts are one and CUDA visibility is `-1`.

Only a normally returned terminal `SLSQP.result.x` creates canonical `candidate.npz` / `manifest.json`. LP failure, objective-budget termination or timeout preserve partial evidence without a canonical candidate. Optimizer status is separate from primal feasibility, so a terminal feasible nonoptimal result is reported without pretending convergence. No history-based best selection occurs. All physical, continuous-clearance and FP32 controller qualification remains separate; the independent controller exporter is owned by the other agent.

Canonical candidate keys include `time_s[16]`, `q_goal_rad[16,20]`, `mapped_object_pose_w[16,7]`, `control_points_rad[18,20]`, `optimizer_x[340]`, basis16/basis61, planned61 time/q/object pose, raw per-pair distances/masks/ties, 122 operational depths, residuals and both CSR Jacobians. Manifest schema is `constrained_prefix_geometry_candidate:1`; it binds protocol/parent/source identity/candidate/report/q0/rawFD/material/source pins and explicitly marks physical qualification false. Plan, NPZ, report and manifest additionally bind `retarget_variant_id=H123_cubic_bspline_constrained_prefix16_v1`. The inherited source/window identity denotes the same raw lineage, not the same trajectory, a new independent motion or inherited rest/init qualification.

Small implementation checks passed 32/32 with exactly 77 counted collision calls (16 fit + 61 planned samples), no LP/minimize, no physics, and no Torch/Warp. They cover exact parent residual/chain algebra, hard P0, raw min selection, global ties, absent NaNs, caching, seven forbidden-API negative cases and IO. They are not a global finite-difference contact-Jacobian qualification. Source/plan negatives are separately recorded; Root retains the one default CLI preflight and any formal execution authority.
