# Resident GPU step, phase and EMA fidelity before moving-contact planning

Previous goal turn was progress (d093fbc): a bounded actual-GPU first-frame
support proposal passes saved review17/17, but original-clock two-dt execution
still fails at0.22/0.20s. Wrench tolerance is lost by5ms.161 tests pass.
The full Goal remains active; this phase starts no neural training.

Extend the maintained resident forward oracle with a separately named STEP
rollout class. Keep the frozen forward/query source pins intact. Restore
qpos/qvel, original solver warmstart, ctrl, time and physical residual EMA
in planning clones. Require no actuator history/activation/plugin/sleep or
equalities, constant wrist mocap and no external aid. Recompute forward at
the same control boundaries as the actual environment. No rewinding of an
executed physical episode is permitted; planning clones are separate worlds.

Implement original scale0.5, residual clip[-0.6,0.6], EMA alpha0.5, legal
absolute PD-target clipping and50Hz reference phase1 for the first action.
Audit every physics tick at5ms and2.5ms against the saved support-aware
constant controller. Compare single-world and batch128 with identical raw
actions, preserving named object-first layout. Include ctrl/EMA, states,
constraint force, clocks, capacities, no-aid and independent EFC wrench checks.

The intended planner horizon is100ms, so require fidelity over this entire
horizon: max hand q error0.002rad, object translation0.0005m, rotation0.01rad,
hand velocity0.2rad/s, object velocity0.01m/s and angular velocity0.2rad/s.
Original source/mocap/warmstart and control recurrence must be exact within
float32 target tolerance2e-7rad. EFC force/moment identity keeps1e-4N/1e-5Nm.
Report the complete recorded failed-prefix discrepancy separately; do not
present a short-horizon gate as full-trajectory deterministic equivalence.
No threshold relaxation or selecting a favorable seed/backend is allowed.

If this bridge passes, preregister a bounded short-horizon action-sequence
search. Use actual moving target/control phase, current physical state and
EMA, full nominal wrench demand in consistent frames, geometric/capacity
penalties and original source clock. Compare pose-only versus wrench-aware
costs with matched query budgets. Execute only each chosen first action in
the unchanged no-aid/no-reset task, stopping first native termination; preserve
all failures, timings and action logs. This is privileged physical planning,
not learned Actor benefit or a substitute for eventual matched SuperMimic
training/WM gradients. Do not fit failed diagnostic prefixes as experts.

All original controlled learning/save-resume, support targets, ego contracts,
independent RGB/full/multi-trajectory total cost and hardware requirements
remain. Future neural training stops at the first healthy saved checkpoint
for restore verification, ETA and user-requested Goal pause.
