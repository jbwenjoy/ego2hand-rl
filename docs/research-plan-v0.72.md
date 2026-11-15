# After resumed2000 WM updates: fix response coverage before policy gradients

Previous goal turn made progress to a healthy stopped100-update checkpoint;
the user explicitly resumed. Current continuation27b7f93 reaches2000 with
44/44 exact-health checks. New recursive evaluation and54 actual physical
response probes produce evidence changing the next action:178 tests pass,
378/378 force receipts and383/383 data/control checks pass, yet physical
model gradients are inaccurate and100–200ms recursive predictions drift.
See`pen-wm-2000-and-physical-response-2026-11-15.md` for pinned results.

Do not extend epochs on the same157 rows or start a gradient-driven Actor
on the strength of checkpoint/AD health. Preserve the full original Goal;
this component stage is not a substitute for no-aid retargeting success.

1. Collect a NEW predeclared source-control cohort. Use actual original-clock
   episodes with the selected initial pose and strict device numerical policy.
   Include source anchors at control0/2/4/6 (0/40/80/120ms), applying independent
   role-declared positive/negative PD-target perturbations at each anchor.
   Respect raw/EMA/clamp mapping; always save and use actual executed targets.
   Stop first native terminal and retain failures. No object force/reset aid.
2. Declare disjoint fresh TRAIN/CHECK direction seeds BEFORE collection. Do
   not repurpose current54 physical diagnostics, old80 CHECK or historical
   diagnostic prefixes as TRAIN. Common deterministic prefixes must not inflate
   independent sample counts or leak identical rows into both splits. Only
   the declared branch interval enters the new paired dataset; retain raw
   full traces for source/terminal/physics audits.
3. Compare actual source55/warm26/physics across branch executions. Multi-step
   prefix variation is measured, not repaired/ignored. Use actual recorded
   source states for supervised dynamics; only identical-source branches can
   support a direct physical derivative comparison without a confounding flag.
   Reuse independent EFC force/torque/impulse masks and indexed source-BODY
   wrench/Fn6 contracts. No expert relabeling or fitting questionable labels.
4. Create an explicitly versioned dataset/checkpoint with frozen TRAIN-only
   scales, matched DIRECT/IMPULSE capacity/initialization/batches/update budgets.
   Keep the original trained2000 baseline frozen. Evaluate new action-stream
   CHECK, predeclared physical response probes and recursively predicted
   state/warm rollouts; report control/representation tradeoffs and costs.
5. Only a physical-response-qualified model can be considered for a bounded
   Actor or MPC probe. Retain native no-aid evaluations and original task/
   geometry/timing gates; finite networks, support LPs, improved initial
   contact or short failure prefixes do not satisfy those gates.

Goal remains ACTIVE; no ongoing process is implied by that application state.
There is no new pause request after resumption. Full scope includes cube/pen
control learning, support-objective A1/A2, real ego input/assets, multi-trajectory
total cost and hardware contracts. Keep performance work in shared`src`;
never fork study directories or change frozen source/data/physics silently.
