# Frozen model native one-step suggestions V1

Use all3 fixed final path-study models,each of2 RAW score gradients,all36 original
TRAIN/development sources,all original exclusions and the original invalid BASE.
30 sources have saved gradients;6 original clipped/excluded sources remain
declared missing,no surrogate/fallback gradient. No NN/Actor/optimizer update,
hyperparameter/validation checkpoint selection or new normalization. These source
phases are reused controlled references,not unseen trajectories/ego/final test.

At each observed source,normalize each exported finite nonzero RAW20 score
gradient to Euclidean norm1. This is steepest local RAW ascent;all controller
diagonals at the original base are .25. Generate base_RAW+4*sign*epsilon*u,
epsilon .002/.005 actual PD radians,sign +/-1,two repeats. BASE2 plus3models x
2goals x2amplitudes x2signs x2repeats=50 declared controls/source.36x50=1800
declared;at most30x50=1500 executed/12000 native integrations. Keep zero gradient
or newly clipped proposals unavailable,with the unprojected proposal retained;
do not clamp,replace direction or reduce amplitude silently. Keep all432 declared
model/goal/source/amplitude groups;maximum360 complete. Original invalid BASE
for SUPPORT_N8861_P066 never becomes admitted by new baseline repeats.

Before native work,freeze the proposal bank,exported direction/norm/RAW and
fresh fixed-model prospective scores. Use original x102 state/warm/dt,actual
float32 controller PD for each candidate,unchanged context and frozen buffers.
No straight-through input offset. Fresh batched forward can have finite-precision
variation from prior batch shapes;save fresh base score in the same model batch
for gain comparisons. All models stay byte-unchanged. No optimizer steps or
normalizer fits during bank construction. Exact source/backend/checkpoint/data/
context/row identities must match the independently reviewed path-study ledger.

Native replay uses current exact backend/support runtime,one-source restores
before EVERY control,full arrays/scalars/observation readback,all8 raw native
ticks/contacts/EFC/geometry/window/held PD recorded. One restore-only warmup per
observed source. Use explicitly pinned SINGLE_RESOLVE_FULL_REHASH scope only,
complete per-call content hashes and implementation source/code boundary checks.
Default loader/engine/physical model/source identities remain unchanged;no epoch
or cached hash. Backend_scope active before construction and through close.
Source flags LIVE,W1,original0/.2 support weight,reference,gravity/forces and
native cached versus qpos frames preserved. No applied object force,position
servo,world-state replacement or trajectory retargeting claim.

Source ranges come from both earlier policy arms;each new control records BOTH
native FULL residual/support scores. A proposal's primary goal selects its own
score. Verify executed source-objective reward agrees with recorded components.
Retain geometry/native termination and all physical-invalid branches. Do not
stop merely because a suggestion decreases reward;on actual simulator/loader
exception,retain full partial trace/counters and do not restart as if no cost.

Independent review decodes every executed packet using original EFC/force/torque/
5e-7Ns impulse/warm/source/mount/no-aid/geometry gates,then reconstructs original
full rewards. New physics validity does not remove old67 dataset masks or
rehabilitate the original invalid center. Compare +direction versus BOTH BASE
repeats and -direction. Store per-source gains/repeat envelopes/central response,
predicted linear and nonlinear gains,and all missing/invalid groups. A positive
gain greater than observed repeat envelope is descriptive,not a confidence
bound;use the original .25 sign-consistent two-amplitude/repeat screen separately.
Primary denominators must include declared,executed,qualified and original
center-valid counts;report per-model/goal/role/budget without best-sign/amplitude/
model selection. A fixed positive direction is the suggestion;negative is a
control,not a selectable native oracle in the main claim.

All prospective NN forward/bank/restore/record/decode costs and actual/attempted
native work count. No reward improvement threshold is guaranteed. A single-step
benefit is not multistep EMA/Actor/task or overall SuperMimic acceleration proof.
Full M0–M5/A1/A2,7.36s no-aid task,multi-reference/independent-seed total costs,
real ego asset/gravity/dynamics/RGB and hardware contracts remain active. Old
epoch anomaly unresolved. No driver/CUDA/installed source edit,push/publication.
