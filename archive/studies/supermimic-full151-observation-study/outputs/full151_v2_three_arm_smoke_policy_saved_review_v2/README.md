# Fixed smoke40 policy saved audit

The corrected saved-only audit passed 574/574 engineering checks. Each arm retained all601 states,600 solver samples,150 control events,150 original post-interval forwards and150 native reward captures. All13 original quality gates were recomputed independently per arm. The checks cover FP32 action/EMA/hard-limit/C2 control, startup rate, source archive and exact saved watchdog identity, fixed initialization/actual8, capacity, phase and unchanged Actor evidence. No Actor/WM checkpoint tensors, native model, Env, FK, collision query, solver, physics or GPU were loaded or executed.

| Method | Position max (151), mm | Directed shaft max (151), deg | Penetration max, mm | Target rate max, rad/s | Original quality failures |
|---|---:|---:|---:|---:|---|
| A1L | 39.429124 | 57.593230 | 7.355551 | 14.888389 | position, shaft, penetration |
| P_ONLY | 41.018574 | 58.513057 | 5.299426 | 15.212497 | position, shaft, penetration, rate |
| A2 | 619.677029 | 140.214483 | 17.732941 | 15.224518 | position, shaft, penetration, non-hand object contact, rate, drop |

All three original task verdicts remain false. Actual joint excess is approximately0.000050545mrad in all three, within the unchanged1mrad limit. All have maximum actuator load ratio1.000000034, within the original1.01 gate; this does not mean there was no force saturation. Charge maxima are0.416169,0.425310 and0.227584 respectively. Capacity and the original FP32 clock contract pass. The601-point position and shaft diagnostics are separately saved; they do not replace the151-knot task metrics.

These are three fixed final40 checkpoints from trainingseed0, each evaluated once with evaluationseed0, on the same known-rough reference. They are engineering smoke results, not evidence of method benefit, independent training-seed replication or a reason to select a different checkpoint. The original failed geometric reference and unqualified contact/gradient boundaries remain unchanged.

Runtime Actor comparisons are source-backed saved before/after digests; raw Actor tensors are not independently reconstructed here. The native selfcollision15 values retain their original reward-snapshot phase. Raw hand-object contacts are not substituted for those sensor labels. Solver force corresponds to the completed preintegration solver and is bound to prestate/poststate indices. Construction/reset/graph API counts remain UNKNOWN; episode600/600 step and150/150 forward counts are closed.

The first audit adapter failed before grading because it used the old `input__...` source-index namespace. Its source and failure report remain in `../full151_v2_three_arm_smoke_policy_saved_review_v1`. Corrected per-run auditorv4 explicitly uses the actual `base__input__...` names; cohortv2 also persists child stdout/stderr. No physical output or executed producer source was changed.
