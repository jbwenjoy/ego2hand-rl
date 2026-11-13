# Bounded action interventions and100ms ensemble-envelope validation

v0.66 fresh-buffer diagnosis finds comparable100ms spread with fresh/reused
buffers and no solver-cap ticks. Single-step same-input canonical differences
are tiny (J~1e-8, EFC force~2e-7); source row orders vary. This rules out neither
all hidden workspace behavior nor every solver issue, but does not support
uncleared buffers or iteration caps as the sole explanation. The old24/72
trajectory fidelity result remains FAILED, with unchanged thresholds.

Now test a NEW prediction contract, keeping the same100ms horizon and actual
no-reset task: does an ensemble envelope cover fresh physical execution under
bounded action interventions? This is not deterministic replay equivalence,
a shorter-horizon substitute, a planner or a training/generalization claim.

Predeclare nine profiles per dt: baseline and +/-0.04 raw-action perturbations
in four sine directions d[k,j]=sin(pi*(j+.5)*k/20),k=1..4,j=0..19, each scaled
to maxabs1. Perturb the first five20ms actions, then return to baseline raw
action2*selected_delta. Preserve original EMA/scale/clips and all task gates.
Baseline and first direction+/- are CALIBRATION; other directions are CHECK.
All remain NONEXPERT_NOT_FIT, including failed transitions. No seed/profile
selection or data-dependent envelope inflation.

Before each actual case, create128 identical-input prediction worlds sharing
the readonly live full model. Save all five controls/EMA, full states, actual
constraint wrenches and capacities; verify model arrays unchanged. Actual
execution remains in the shared original evaluator, first native termination,
same selected pose/source velocity/object/wrist, original phase1 and both dt.
An explicit action-provider callback is additive; old Actor/constant paths and
source bytes are preserved. No neural network is built for this cohort.

At every POST tick compare actual object position and world SO3-log rotation
relative to the original reference with ensemble componentwise min/max plus
fixed0.5mm and0.01rad component margins. Report raw and margined coverage and
full envelope widths; require ALL100ms ticks covered for a case to pass this
new diagnostic. This is a sample envelope, not a calibrated probability or
confidence guarantee. Report calibration/CHECK separately and keep every
failure. Wider coverage alone does not qualify the task or justify planning.

Use actual saved EFC receipts to preserve contact/action-response evidence.
Future physical gradients need repeated independent action interventions and
uncertainty checks; current finite-difference failures stay failed. Do not
repurpose these CHECK/diagnostic records as training experts. Full Goal and
future healthy-neural-checkpoint restore/ETA pause remain unchanged.
