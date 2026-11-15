# Native five-control receding-horizon intervention with unchanged task contract

Previous turn is PROGRESS: v0.90 read-only diagnosis completed.24 TRAIN parents
contain23 distinct state55/warm81 sources. All four horizons reproduce prior
FULL arrays exactly. Privileged true warmstart alone leaves148/172mm200ms COM
error, not a deployment fix. Constant-velocity20ms errors1.27mm/3.21deg beat
both models. Native57-action20ms tables lose pose/geometry-qualified endpoints
by k10–12; all45 decisions that have a qualified candidate already select one.
Independent actual57x3 tables have such candidates at35/36 declared sources.
No optimizer/native integration this diagnosis, no Actor/task/ego success.
Full Goal ACTIVE. Keep all frozen sources/data/checkpoints and original gates.

1. Implement a declared H=5 controls (100ms nominal) native receding horizon,
   alongside unchanged H=1 (20ms) baseline. Actual controller still applies ONLY
   the first20ms raw action, then replans from the next actual state. Fixed
   wrist, original368-control/7.36s reference clock, same initialized pose,
   original EMA.5/action scale.5/bias/PD limits, CCD50/1e-6 and numerical profile;
   no physical-object assistance or post-start reset. This is diagnostic native
   planning, not learned-policy A1/A2 or new training data.
2. Each of57 candidate IDs defines an open-loop future action sequence: existing
   bounded center plus frozen candidate_offsets[k+j,ID] at each future phase.
   Propagate copied EMA sequentially, use reference joint commands at k+j+1,
   actual unchanged bias/limits, and hold each resulting PD exactly20ms.
   Snapshot restoration and scheduled graph control must not touch live engine/
   controller memory. Do not hold a single PD target100ms or freeze reference.
3. Compare predicted per-control endpoints to original COM/full-SO3/reference
   hand at k+j+1. Aggregate mean of the existing per-control state/residual/
   support costs with unchanged scales/weights. Evaluate support wrench per
  20ms interval against its original valid nominal target; no invented tail
   derivative. Geometry/finite/capacity/nonhand safety spans the full forecast.
   Near clip end use H_eff=min(H,368-k), without extending the reference clock
   or padding fictitious derivative labels. Original actual task gates remain.
4. Preserve default H=1 code behavior. Unit-test future EMA/bias/clipping/clock,
   orientation including axial roll, horizon scoring and endpoint selection.
   BEFORE closed-loop claims, perform matched-source prefix audits: H5's first
  20ms uses the same raw/PD as H1 and must satisfy original state/force/clock/
   quality tolerances. Independently execute representative full schedules,
   decode original EFC/wrench and record actual warm/control. Self-agreement or
   endpoint score alone is not dynamics validation; retain all failures.
5. Run fresh diagnostic H1/H5 x state/support x5/2.5ms actual native episodes
   under full original termination/duration. Save sources, graph predictions,
   controller state, actual traces, source/model isolation and exact receipts.
   Review actual force/momentum/geometry and scheduled-prefix fidelity; render
   actual trajectories if useful. Never relabel failed episodes as experts or
   promote old CHECK/DIAGNOSTIC cohorts into fitting. A failed prefix/quality
   audit precludes a benefit claim even if survival improves.
6. Count actual and all candidate integrations, horizon, setup/compile/copy/I/O
   and unequal survival. H5 spends5x the per-decision forecast ticks of H1 before
   tail adjustment; explicitly unequal compute. Do not report this as fair-total-
   budget policy A1/A2, real-time performance, learned surrogate speedup, or ego
   generalization. No MLP/Actor training until physical/recursive validity and
   an actual feasible control stage are demonstrated or a new protocol chosen.

Native horizon may still fail; sampled local failures do not prove global
reference infeasibility. Any later reference retiming/projection is a separately
labeled task and cannot claim original7.36s success. Full required cube/pen
learned controls, save/resume/unified no-aid gates, body wrench/physical gradients,
multi-trajectory costs, real ego frames/assets/dynamics and hardware contracts
remain incomplete; do not shrink or mark the Goal complete.
