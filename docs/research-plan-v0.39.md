# Research plan v0.39: initialize for wrench and actuator feasibility

2026-11-08. The preceding consolidation-status turn was no progress toward
the remaining physical research, although it verified the maintenance home.
This turn completes the open native-authority diagnostic and extends its
saved review to native pyramid generators and checked LP primals. The Goal
remains active. No new learning stage or first checkpoint has started.

## Evidence that changes the next action

The unchanged initial task has zero passing cases among 65 predetermined
action schedules at both dt. At the fixed initial contact points/normals,
even allowing dry friction anywhere in its actual limit box, the native
condim3 pyramid/PD static LP supports at most0.16781 N for the current
residual range, or0.49299 N for the entire actuator target interval. The
phone weight is1.962 N. Numerical LP witnesses and all130 failures are
retained; neither result proves global dynamic impossibility.

Prior functional-tripod material-point geometry search, gap projection,
compression and native preload trials already exist. The loaded-dynamics
report records16 failed cases. Repeating those geometry-only objectives or
silently increasing mass/friction/gain/penetration allowances would not
resolve the observed force/moment/actuator issue.

## Next distinct experiment: mechanically feasible initialization

1. Define a new, explicitly versioned **initialization candidate** contract,
   separate from the original failed task and from the completed online
   checkpoint lineage. Keep the exact nominal MJB, fixed wrist, actual
   actuator limits, mass, inertia, gravity, friction, no-aid rules and native
   evaluation gates. Record all candidates, not only a selected passing one.
2. Search q/phone rigid pose and legal absolute PD targets jointly. Record
   their deviations from the original retargeted reference. Search trust
   bounds are algorithmic limits, not measured reconstruction uncertainty.
   Changing the initial pose requires a new reference/observation/memory
   bundle before any Actor training; do not reuse the original bundle as if
   it described the changed task. A candidate's rigid offset and resulting
   reference discrepancy must be visible.
3. Include simultaneous object wrench balance, hand generalized force
   balance, actual force saturations and dry-friction limits in the search
   objective/constraints. Use current contact patches/normals/Jacobians,
   not fixed signs or support directions transplanted from the old reset.
   Exact current collision/clearance checks remain required for original
   phone, self and table pairs. Proposal gradients or sampled cones are not
   actual native-force derivatives or full collision certificates.
4. Separate proposal scoring from qualification. Check every proposed final
   state afresh with native collision/forces and actual legal PD. First
   demonstrate independent no-aid two-second holding at both dt using the
   original gates, plus an explicit report of source-reference deviation.
   Keep failed candidates, budget, force margins, contact migration and
   all missing tails. A static LP witness alone never qualifies holding.
5. If no candidate qualifies in the declared bounded search, retain a
   negative result and identify which geometric/actuator constraints bind.
   Do not restart the same failed search without new evidence. A rigid
   phone mismatch may justify another existing ego object/clip with known
   mesh as a separate experiment, while retaining this phone result.
6. Once a useful mechanical route exists, create a fresh TRAIN-only finite
   intervention dataset with full legal command history, PRE labels,
   masks and zero-replay barriers. The permanent heldout action-response
   dataset and present authority diagnostics remain outside fitting.
   Rebuild changed-task bundles and declared normalizers as new artifacts.
7. Specify response-supervised WM and same-budget A1/A2 training only then.
   Continue using the shared update/runtime implementation. A new task or
   model recipe is a warmstart experiment, never a resume of online300.
   Honor the standing first-healthy-checkpoint verification/ETA and pause
   request for any new learning stage.

## Full objective retained

The original controlled pen/cube experiments, rigid collision and physical
calibration/robustness, independent ego reconstruction and unknown gravity,
full61-frame trajectory execution, multi-trajectory shared-model training
cost against matched baselines, and hardware joint/zero/gain/latency and
observation contracts remain in scope. Holding a newly initialized nominal
phone would establish only one prerequisite, not ego generalization or a
CHORD/SuperMimic benefit. No driver/CUDA change or old-source edits.

Evidence: [control-authority results](control-authority-2026-11-08.md),
[completed online300](online300-results-2026-11-08.md), and
`provenance/control_authority/summary.json`. Only mark the Goal complete
after its full original requirements have independently passed.
