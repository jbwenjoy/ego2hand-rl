# Same-pose GPU projected pen contact and dynamics preflight

Previous turn was progress in9564807: bounded geometry clears to0.050mm,
native CPU rollouts fail0.22/0.18s;158 tests. Full original Goal remains active.
No neural training is started in this phase.

Use exactly that saved geometry and constant residual, not another optimizer.
Original and projected variants at original5ms and half2.5ms GPU integration,
original50Hz/369-frame/7.36s reference and first native task termination.
Reuse the shared controlled evaluator/force reader; additive explicit hooks
for pre-start initialization, constant action and declared dt are permitted.
Preserve previous evaluator source bytes and old report pins with a ledger.

After one task reset, set only20 robot joints to the specified float32 pose,
retain source object/wrist qpos and original velocities, initialize physical
residual EMA to delta and original raw action2*delta. Set first-frame PD target
qref0+delta before the support query for BOTH variants. This differs from old
GPU reset ctrl0 and is explicit, matching the native diagnostic initialization;
no physical integration occurs at that point. Reset observation histories only
after correction; no Actor uses them in this constant-controller diagnostic.

Before starting execution, query actual GPU contacts at this same pose with
zero velocities/zero warmstart as a separate static probe. Save actual contact
geometry, EFC indices/forces, full integration state and source parameters.
Use a DETACHED model/data for native contact Jacobians and the existing indexed
LP core, initialized from actual GPU fields. Host-only adhesion=0 is justified
by all-joint actuation and validated against actual EFC-derived force; no GPU
force/constraint/model is patched. Static full6D LP uses the same gravity target
and0.05rad PD interval; also test unbounded-actuator static and previously saved
nominal first dynamic BODY target on the actual GPU frozen cone. Keep all LP
witnesses/certificates. These remain necessary frozen-contact checks, not
compliance equilibrium, expert supervision or trajectory success.

Restore original source velocities AND warmstart before starting. Assert
projected joint float32 representation/object/wrist invariants and no force aid.
Run original reference clock, original .5 scale/.5 EMA and fixed raw2*delta;
stop first native terminated/truncated flag, no resets after start. Save the
existing per-tick corrected contact/constraint/state/control receipts and all
tracking/penetration gates unchanged. No favorable seed/case selection, no
source time stretch, no object aid, no installed driver/CUDA changes.

Compare GPU contact activation with native forward's two-contact projected
picture; the previous all-pair distance proximity was not an actual-contact
certificate. Preserve backend/dt/termination-timing distinctions and numerical
rounding, not a claim that a frozen native cone proves GPU/global infeasibility.
Subsequent support-aware retargeting must use complete dynamic wrench demand
with explicit nominal inertia/derivative uncertainty and actual contact modes.
Do not loosen acceptance or fit failed diagnostic prefixes as experts.

All original controlled learning/save-resume/no-aid qualification, matched
residualA1/A2/support/WM action gradients, ego asset/dynamics/frame/controller
contracts, independent RGB/full/multiple trajectories/total costs and hardware
subversion/observations remain. On later neural training, stop at first healthy
checkpoint, verify restore/ETA and pause per standing user instruction.

First four-case v1 outcome is retained: original4/projected3 actual GPU contact
points; all full-wrench tests and all four trajectories fail (0.16s each).
Before final review, v2 strengthens source/receipt metadata: store the actual
GPU BODY rotation/COM used by the LP, assert warmstart restoration, and pin
loaded Dex task sources. The old evaluator's generic saved-Actor scope string
was inaccurate for this constant-action branch; it is corrected explicitly.
These changes do not alter the declared control or physics. Preserve v1
source/report snapshots and run the same cohort again, without selecting cases.
