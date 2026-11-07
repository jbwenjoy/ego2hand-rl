# Bounded same-state action-response acquisition

`collect_causal_branches_v1.py --protocol PATH` is JSON/hash metadata-only:
no NPZ open, MuJoCo/Torch import, model construction or native call. The provided
`protocol_cpu_false_v1.json` keeps CPU execution disabled. Root can separately
freeze an authorized protocol and fresh output, then use the same command with
`--run`. No training is implemented or authorized here.

The runner imports the pinned `NativeSavedActionProbe` (`b10bd019…`) and its
unchanged Reader (`877aa4c5…`). It does not copy or change the native loop.
Each hypothetical branch creates fresh MjData and restores the original full
`mjSTATE_INTEGRATION` signature16383/size300 **before action**, with the original
episode initial qpos/COM as the hard-stop origin. No reset/restore/observer
`mj_forward` is used. The two native h values have the original saved effective
model digests; only timestep differs between them.

Execution order is mandatory:

1. Run all72 frozen original-action baselines. Save every returned branch first.
   Require raw equality for every captured per-tick trace key, source x/u,
   before/after-action/target integration receipts, target qpos/qvel/state58,
   physical world-left delta52, original59 labels and native clock/ticks.
2. Only after **all72** pass, save `all72_baseline_barrier.json`. If any baseline
   fails, stop; **zero** new side branches have been attempted.
3. Traverse all2880 frozen side records: execute2870 available endpoints,
   preserve10 unavailable records without executing them. No clipping, enlarged
   epsilon, retry, replacement or source selection occurs.

The frozen design has2952 total plan records,2942 executable branches and a
maximum282432 native integration ticks. The283392 design-wide count would also
include the10 unavailable endpoints; this runner does not execute those.
There are2 MJB loads,2942 fresh MjData maximum and0 forward/NN/AD/optimizer/GPU
calls. Early native safety stops can reduce ticks; partial traces remain saved.

Real absolute actuator bounds, all20 coordinate/sign order, exact F64 du and
FP32 represented du, the1e−6 availability rule and unchanged original hard-stop
thresholds are checked before model loading. Side outputs retain the entire
du20 vector, actual control in F64/FP32, contact rows24, PRE contact/force/energy
fields, raw POST qpos/qvel/COM-world state58, physical world-left delta52 and
full source/after-action/terminal integration vectors. The ten encoded endpoint
collapses remain explicitly unavailable, not zero-response measurements.
`represented_delta_control20_FP32` records the rounded **FP32 subtraction**.
For any later model finite-difference denominator, cast each saved FP32
endpoint to F64 before subtracting; the rounded FP32 subtraction is not an
exact representation of that endpoint separation. Physical du uses the actual
F64 controls.

Returned safety failures at a complete-H boundary retain their termination
description and the Reader's original validity rule. Partial output has an
observed-prefix target but **no invented target at20ms**. Normal warning/
nonfinite/safety aborts return full recorded prefixes. An unexpected exception
inside a frozen helper preserves previous branches, the active plan record and
terminal native state; if no trace was returned, the missing tick trace is not
fabricated. All remaining plan records stay in `all_plan_records.json`.

At success, `branches.json` retains all2952 plan rows. Separate files are written
for `CAUSAL_FIT_KNOWN_TRAIN_STATE` and
`HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE`, including unavailable/partial
rows with explicit masks. `training_eligible_bool` is the unchanged Reader's
**complete/finite/no-warning label validity**, not permission to fit the
action-check partition. Consumers must use only the FIT file for training;
the held-out action-response file must not be merged into fitting. Original
TRAIN families, native h pairs and full-state aliases preserve their frozen
roles. The original VAL476 diagnostic branches are not used as source data.
Previously exposed TEST families are not relabeled unseen. The12 metadata-only
fresh final families remain unacquired in this job.

This is the original **two-designated-pad15/28** task and model. A third tip is
not silently reclassified as a pad. Nothing is renamed a successful holding
trial. Native decoded PRE contact/force labels are simulation outputs, not
human-force GT; POST state targets and lastPRE endpoints have distinct phases.
Complete-H contact/path observations can be future supervision/diagnostics,
not deployment-time known inputs or a strict zero-action causal certificate.
Current geometry can later be derived separately from source state using the
new geometry provider. No such query or force oracle is added to this loop.

Baseline replay establishes empirical equivalence for these selected saved
states and controls. It does not prove all MjData/solver hidden scratch is
represented, a Markov state58, a true physical Jacobian, ego transfer or a
SuperMimic/CHORD policy benefit.
