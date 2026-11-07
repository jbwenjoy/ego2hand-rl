# Research plan v0.37: unified online collection and matched updates

2026-11-07. The full Goal remains incomplete. This follows the frozen v0.36
plan, final offline1000 results and repository consolidation. Historical
sources/experiments remain frozen; all new implementation lives in this repo.

## Evidence and current limitation

The fixed final offline model improved endpoint fitting on the new tripod
MON data, but all four original two-dt Actor cases lost holding at about
0.121s. On the permanently held-out finite-action response dataset, final WM
COM and rotation response errors still exceed the zero-response prior. Model
AD/FD agreement only establishes internal numerical consistency. See
`research-results-2026-11-07.md`; no new holding or CHORD benefit is established.

The shared core's short fixed-parent benchmark improved from 6.47s to 0.64s
per paired update at micro8, with declared numerical error checks. This is
not an end-to-end multi-trajectory adaptation speed claim.

## New source-bound online experiment

Use fixed final1000 / WM1100 weights, both Actor/Critic/target states and raw
normalizers, physical context and WM loss scales. Start a distinct experiment
with seed1734 and five fresh Adam states. Do not resume the old source-bound
offline experiment or refit normalization/scales.

The template is `configs/phone_online_gpu.toml`, disabled by default. Actual
run configuration and protocol are saved under ignored `runs`. Batch8/H2,
NN FP32, geometry/physical FP64, AMP/TF32 disabled, clip1, gamma0.99, EMA0.01
and learning rates retain the prior planned recipe. Micro8 is an explicitly
declared change from the archived micro2 online draft, supported by the
bounded comparison; it is not a claim of long-training bit identity.

Execute three cycles. Each uses two arms × four fixed exploration profiles ×
two native dt, maximum20H per episode: 16 episodes, maximum320 rows/30,720
native steps. Actions are .95 Actor plus .05 deterministic profile, with no
clipping, auxiliary forces, failure refill or automatic retry. Task loss is
tested at POST-H boundaries; this local collection is distinct from the
two-second tickwise native evaluator.

Persist raw PRE contacts, POST states and all integration receipts before
canonical replay or labels. Retain partial prefix/NaNs, exact aliases,
independent triple-pad coverage and Reader/task/timeout/truncation reasons.
True failure and reference timeout stop TD bootstrap; collection truncation
does not. A complete terminal endpoint remains valid WM supervision.

Pool both arms across all completed collection cycles. Every source row keeps
its pool pin and original row/episode ID beside the cumulative IDs. WM batches
draw 4 cumulative online + 2 original TRAIN684 + 2 tripod TRAIN153. Actor arms
share exactly the same eight complete prealive starts, with per-mode real
rewards. H2 is model imagination, not a two-native-window requirement. The
heldout action-response, MON and CHECK files never enter fitting.

After100 paired updates, the complete cycle checkpoint supplies the next
collection. Checkpoint10 can only resume its existing cycle1 pool; it cannot
supply cycle2. Own resume restores five Adam states, all RNG/sampler state,
models, frozen context, pool identities and immutable log prefixes, without
recollection. Source/selected-input/config/cursor changes fail explicitly.

## First execution and stop

The first executed learning run stops immediately after cycle1/update10 is
saved and read back. Shared health checks cover five-optimizer moments/steps,
all named pre/post gradients, all network/target/norm states, raw probe
predictions, sampler/RNG, source/input/cursor lineage, raw collection receipts
and log byte prefixes. Additional probes are reported separately from learning
calls. Learning calls are online-stage lineage counts; resumed-worker deltas
are not conflated with them.

Report actual collection and setup time, measured update cost, learning-only
remaining ETA and unmeasured future collection/cache/health overhead. Honor
the user's standing request by pausing the Goal after the first healthy
checkpoint and ETA. No further cycle or policy qualification follows that
pause until the user resumes.

## Remaining full-Goal work

After the fixed online budget is explicitly resumed and completed, evaluate
all predetermined A1/A2 two-dt cases with the original no-aid two-second gates
and retain failures. Repeat source-qualified WM contact/finite-response
diagnostics without fitting or checkpoint selection on heldout data.

Keep calibration and robustness of collision/inertia/friction, uncertain ego
gravity, independent ego/RGB hand reconstruction and robot input contracts,
complete61-frame execution, multi-trajectory shared-model cost and matched
baselines, and hardware joint/order/zero/gain/latency/observation contracts in
scope. The known-lineage HOT3D phone example, nominal asset and engineered
material proxy do not satisfy those remaining requirements by themselves.
