# Long-control evaluation readiness — 2026-11-16

The long model-free PPO trial remains running and incomplete. Its first
optimized checkpoint has passed ordinary full-state reload. Qualification and
startup evidence are recorded separately from closed Fn6 world-model results:
`provenance/pen_long_control_registration_v1/ledger.json`. Final work accounting
is deliberately null until both arms finish. No new goal pause is requested;
the historical first-checkpoint pause was already served and resumed.

The final evaluation matrix is registered in
`runs/pen_long_control_evaluation_registration_v1/registration.json` under
`configs/pen_long_control_evaluation_v1.toml` and
`docs/pen-long-control-evaluation-protocol-v1.md`. It requires both2000-round
arms before running nine fresh mean-policy cases and a repeat0 video, with
the existing strict full7.36s/COM/full-angle/geometry/no-aid gates unchanged.
Trained-arm prefix errors will use one common observed horizon separately.

`runs/pen_long_control_W1_admission_v1/report.json` proves the preparation
path on the first optimized RESIDUAL checkpoint:404 W1 Model arrays and195
scalars match the historical independent W1 plant witness exactly; actor and
normalizer copy readback passes. Only actor/normalizer are copied. The old
witness actor is explicitly different and supplies no physical state or labels.
Fresh W1 features are LIVE although the qualified training path is CACHED32.
Observed native integrations0, optimizer steps0, qualification wall19.6734s.
Policy performance was not evaluated in this admission check.

The shared trace recorder in `tools/preview_pen_support_pilot.py` now accepts
an explicit admission callback; original callers retain their default behavior.
The new module `learning/pen_w1_evaluation.py` is not imported by the ongoing
training process. No existing captured training source/config/protocol changed.
The original training source receipt still verifies live.

Full415 tests pass (`runs/pen_fn_long_W1_tests415_v1.log`,12.534s). Five new
tests reject incompatible observation/action/joints/objective/arm/normalizer,
nonfinite parameters, W32-to-W1 Model broadcasting, nonidentical Model bytes,
early-checkpoint substitution and incomplete training reports. These tests
and unstepped admission do not yet verify closed-loop final policy performance.

Commands after the two-arm job ends successfully:

```powershell
& 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe' -X utf8 tools/evaluate_pen_long_control.py --config configs/pen_long_control_evaluation_v1.toml --output runs/pen_long_control_evaluation_v1
```

Keep training session89759 as the authoritative job handle until terminal.
Do not restart on observation timeout, edit captured bytes, select an early
checkpoint as final, change gates or count this preparation as ego/task success.
The old64-round evaluated video remains the main result until new execution.
All original M0–M5, real ego assets/dynamics and multitrajectory/seed/cost
requirements remain as in research-plan-v1.18; goal ACTIVE and incomplete.
