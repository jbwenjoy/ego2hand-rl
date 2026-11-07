# Shared-WM, shared-Actor local prior mechanism proposal

Status: **proposal only; not executed or authorized**. This document records a source-level feasible, bounded CPU experiment. No model, objective, gradient, optimizer, Env or simulator was run for this review.

The smallest useful experiment holds one WM, one Actor including its normalizer, one four-row state/observation/memory batch, and the entire control graph fixed. It changes only the coefficient of the declared support term from zero to one. The result would identify a local functional contribution of that term in this chosen computational graph. It would not identify the cause of the previous training outcomes.

The earlier [checkpoint100 comparison](../outputs/full151_schema2_three_arm_prepolicy100_comparison_CPU_v1/report.json), SHA `c5b0549654a55f3a086c8dea25a5e714170cace1454a1ca0704754988a18f29a`, found different WM/failure parameters, normalizers and checkpoint CUDA RNG before any policy/prior forward. Equal raw initial parameters and equal update budgets do not make those later trajectories identical. This proposal makes a new within-graph intervention rather than treating the old three runs as a strict shared-trajectory causal experiment. It is sequential and exploratory: the full3s outcomes and the controlled20ms physical/WM diagnostic are already known. None of their success, stable-case or sign labels selects the four rows below.

## Fixed authorities and data

| Item | Fixed choice | Reason and limit |
|---|---|---|
| WM | A2 primary2000 `model_final.pt`, SHA `70d0a21d5b2f0238ab745264bf24db08515fef65f072cbe00fb6893a33512a0c` | One model and all of its normalizers for both coefficient conditions; conditional on this learned WM, not an unbiased physical model. |
| Actor | A2 primary `model_100.pt`, SHA `6efee26549d967ba32df89599b7f425d74d397be454b7c8d5e189235becaee89` | Prepolicy Actor parameters remain original. Its own saved normalizer is fixed across both conditions; it is not asserted equal to other arms' normalizers. |
| Rows | Archived actual A2 final2000 native evaluation, phases `[0,49,99,149]`, in that order | These four predeclared positions retain the terminal interval and late failed-trajectory states. No replacement, success filter, noise filter or stable-FD filter. They are dependent rows from one source/seed0. |
| State | `native_reward_transitions[k].x0`, FP32 state58 | Preaction state for this archived trajectory, not a new same-Warp donor state. |
| Actor input | `control_events[k].policy_input155`, FP32 `[1,155]` | Exact saved observation. Do not regenerate it from state58, call an ObservationManager or pretend state58 contains observation history. |
| Memory | `native_reward_transitions[k].memory` | Saved native PEN fields, including `c1/c2`, `a1/a2`, limits, start/time/end and encoder bias. No invented zero EMA/history. |
| Task/reference | Original full151 input bundle SHA `59406c6214a085afba7efcc2b33880d5c82b409ee4f6485ec71c073fec19dc76` | Same baseline_u150, source identity, physical xi46, reference and bounds. No u150. |
| Prior | Unchanged binding_v3 and actual training spec SHA `08d9fa9936766bdf21bb13dcf0123eebe3a54639266f641f47f9e7bb522c3962` | Frozen training64 support directions and original hyperparameters; neither 512 substitution nor a new target/material cone. |

The source [policy_probe.py](../src/full151_native/policy_probe.py:211) saves the observation used by the actual Actor before `env.step`; [lines232–248](../src/full151_native/policy_probe.py:232) save control events and transition `x0`/memory. The evaluation's independent saved audit is [188/188](../outputs/full151_v3_A2_primary2000_policy_saved_review_v1/report.json), SHA `67aa96b737f8060790f81d9e3942c15250659adc9976cb7803fa354876d363c8`. Its task failures remain part of the data and are not qualification failures for this saved component analysis.

The checkpoint100 report records Actor parameter fingerprint `d80a1bf127e47b0d93cfaec6debfbc298a3ecde79d9cde576e2f5492d1a732ce`, normalizer digest `0b76e561b107ded87ba327c01be5e267be04e902673bf6dd9cd84233a5016294`, and `norm.count=40192`. The native collector explicitly updates `actor.norm` even during WM-only collection. Thus “prepolicy” means no Actor optimizer update; it does not mean an untouched or shared-across-arms observation normalizer.

## One common forward graph; three scalar VJPs

For each selected phase k, let x, o, m denote the fixed saved state, observation and memory. Let the Actor's numerical parameter values be theta, with frozen saved normalizer buffers.

1. Strictly load the fixed final WM through the existing 105 WM loader, and load the fixed Actor100 through the existing Actor105 evaluation loader. Never give model100 to the final-only WM loader or relabel a checkpoint. Actual Actor100 load qualification is still pending.
2. Use `torch.func.functional_call(actor, (parameter_clones, buffer_clones), (o,), strict=True)`. Detached CPU parameter clones have `requires_grad=True`; the loaded Actor remains `.eval()` with its original parameters frozen and unchanged. This computes a derivative at fixed numerical weights, not an update. The installed functional-call source supports this API. Every original and cloned buffer is hashed before/after; no `.norm.update` call is permitted.
3. Compute `a=actor(o).clamp(-1,1)` and the original controller:

   `rho = .5 * clamp(.5*a, -.6, .6) + .5*m['c1']`

   `u = C2_020(clamp(baseline_u150[k] + rho - zero_bias, outer_lower, outer_upper))`.

   Use the frozen target_inset Torch function with exact saved bounds and zero encoder bias. Do not move an already-filtered reference target. `c1/c2` remain residual EMA, not absolute commands. Save every clamp/C2 component mask, exact-boundary flag and u. There is no command-rate or physical safety qualification of this newly computed control.
4. Call the same frozen FP32CPU `WM(x,u)` once, with fixed physical xi46 and zero174 exogenous vector as its qualified wrapper defines. It returns x1 state58 and native self15 logits. No second EMA/C2 is applied at the WM input; the u constructed above is the absolute WM control.
5. Call the frozen A2 objective once on the **same x1 and preaction memory m**. Use its returned common `geometry_intent_support_match`, `geometry_intent_penetration_prior` and phase weight. Define per-row reward contributions:

   `b_P = .020 * (-.1 * P)`

   `b_S = .020 * (w * S)`

   `b_A2 = .020 * (w * S - .1 * P)`.

   The P-only formula directly consumes the exact same P, as `PenaltyOnlyPrior.forward` does; it is not reconstructed by subtracting support from a large A2 scalar. A second factory/geometry forward is unnecessary for this paired functional intervention. The existing P-only implementation and CPU factory parity evidence are source authorities, not a new actual validation performed here. Log both raw dimensionless terms, weighted bonus and dt-scaled contribution. Apply dt exactly once.
6. Take three VJPs of `mean(b_P)`, `mean(b_A2)` and `mean(b_S)`, each with respect to u, Actor output a and functional Actor parameter clones; reuse this one graph with `retain_graph` until the last VJP. Do not run `.backward()` into original parameters. Multiplying each row of d(mean bonus)/du or da by four gives its local row derivative because these evaluation graphs do not mix rows. Report the shared-parameter gradient of the four-row mean separately; it may cancel across states and is not four independent estimates.

Report the signed reward-ascent gradient; if reporting Actor-loss descent, explicitly define loss `L=-mean(b)` and negate it. Report all zero gradients, nonfinite failures, saturated controls and boundary conventions. A zero-norm cosine is null, never manufactured as alignment1 or success. Compare the algebraic value and gradient difference `A2-P` to direct support-only output; save the raw numerical residual rather than using a data-tuned agreement threshold. The fixed real-number identity does not make floating-point cancellation exactly zero.

This is **H1 prior-only**, not a call to the native optimizer-bearing `Runner.policy_update`. Native policy_update samples replay starts, advances H8 memory, makes random draws, includes native task reward, physical soft survival, optional failure head and a critic bootstrap, then performs Actor and Critic updates. Those operations cannot be silently no-oped and still called a full native policy update. This first experiment deliberately has no critic, native reward, gamma, soft survival, H8 unroll, replay sampling or optimizer. The native reward is common to the coefficient intervention and absent from both measured prior-only contributions. It is not asserted to be negligible in the real policy objective.

At k149 the single action and resulting x150 response are valid. The objective still uses preaction frame149, then ends. No next Actor call, u150, bootstrap or extra decision is introduced.

## Important mismatch with existing physical secants

The Actor100 is re-evaluated at the archived observations. Its resulting rho/u will generally differ from the archived final-A2 commands and from full05 fresh-donor center targets. Therefore **the existing152 physical branches are not ground truth at these new controls**. This experiment neither loads nor reuses physical FD slopes, sign labels, stable cases or force outcomes. Super's separate exact-prior physical-secant protocol uses the existing8 fresh donors/160 physical nextstates and their executed controls; it answers a different question without Actor forward. The two analyses must keep separate dataset IDs and output directories.

## Precision boundary and a separately gated future GPU study

The proposed experiment above is a **FP32CPU shadow graph**, not the original GPU training gradient. This is a source-verified difference, not just a generic precision caveat. The actual A2 training config has `amp=true`, `compile=false`, `compile_policy=false` and `compile_policy_dynamic=false`. `Runner.__init__` passes cfg.amp into `WorldModel`; `WorldModel.forward` normalizes its input, then applies CUDA BF16 autocast to `self.net` only when `self.amp and x.is_cuda`, casts y back to float, and integrates in FP32. The immutable119fa7 CPU loader constructs `amp=False` and rejects an initialized CUDA context. It must not be moved to CUDA to bypass its guard.

`learning_v2.py:270` explicitly sets `torch.backends.cuda.matmul.allow_tf32=True`. The reviewed entry, learning, native Runner and common-network sources do not explicitly set `torch.use_deterministic_algorithms`, cuDNN deterministic/benchmark, cuDNN TF32, float32-matmul precision, CUDA reduced-precision-reduction flags or `CUBLAS_WORKSPACE_CONFIG`. This is **not** evidence of their historical runtime values; these remain unrecorded/UNKNOWN in this source-only review. Seeds and compileFalse do not imply deterministic kernels. The actual config SHA is `eb7a45da30ff730cf364d5434c26e9f69f424f0b56ff5ea95dbee1c8b36661e8`; effective cfg SHA is `374c30de1b2d355ba2401d706a5c81952d03ac3ffd5d99d64d49435549fe1107`.

A later, independently authorized GPU diagnostic could use the same four saved rows, fixed numerical weights/buffers, one shared GPU Actor/control calculation, and two predeclared WM conditions in fixed order: (1) no WM autocast with TF32 allowed, and (2) native WM-net BF16 autocast with TF32 allowed. The first must be named **GPU non-AMP/TF32-allowed**, not strict IEEE FP32. Actor and objective stay outside any newly invented global autocast; native WM's internal BF16 scope is the only condition switch. Each condition needs one WM batch4 forward, one objective batch4 forward and three VJPs (two WM/geometry forwards and six VJPs total); one shared Actor forward and one objective construction suffice. It remains H1, not full native policy_update.

This requires a **new isolated authorized GPU checkpoint consumer**, reusing strict105 namespace, final-health, actual-context and source checks while constructing native WorldModel with an explicit amp condition. No `.to('cuda')` of the CPU wrapper, global monkeypatch, old-loader relaxation, Env or native simulator is acceptable. The new config must explicitly choose and record determinism/TF32/reduced-precision flags and workspace environment, GPU/driver/Torch versions, actual autocast dtype and before/after RNG/state hashes. No historical flag may be imputed from a new process. Until those flag choices and the new consumer receive Root review, GPU protocol/source and actual GPU qualification are **PENDING**, not implied by this feasibility outline. No GPU source or execution is included here.

## Proposed finite budget and artifact contract

No execution has been authorized by this proposal. A future separately frozen CPU config should authorize exactly:

- Two distinct checkpoint files; one strict final WM-loader call performing its original two `torch.load` calls, plus one Actor100-loader call performing one load: **three actual loads total**. Internally this constructs one WM and two Actor objects (the WM loader's validation Actor and the retained Actor100). Do not hide the validation Actor construction.
- One Actor batch forward of four rows, one WM batch forward of four rows, one A2 prior batch forward of four rows, and three scalar VJPs. One CPU objective factory construction. No P-only factory, native/physics query, Env, optimizer or training resume. Factory may passively read its pinned MJB constants and perform its frozen differentiable geometry algebra; that is not a native collision query or physical contact truth.
- Fresh process with `CUDA_VISIBLE_DEVICES=-1`, CPU FP32, four Torch threads, AMP/compile off. Proposed owned CPU soft120/hard180 seconds, no retry, data alternative or timeout fallback. These are preregistration recommendations for Root approval, not current permission.

Suggested future module API:

```text
metadata_preflight(config_path) -> identity/shape/source receipt, no Torch
extract_saved_batch(config, rows=(0,49,99,149)) -> x4x58,obs4x155,memory,bounds,baseline4x20
run_shared_prior_actor_mechanism(config_path) -> report and tensors (explicit CPU authorization only)
```

Minimal future files are one source-bound saved-array adapter/runner module and one CLI/checker. They should call the existing strict consumers and frozen objective, not construct an unused Env/Runner or patch executed sources. Preflight must join pins by **exact SHA plus `Path.samefile`** where file identity is intended, preserving the original pin spellings; D:/ and D:\ routes to one actual file must not create a false scientific failure.

Pre-model fail gates: actual final A2 health and Actor100 prefix health/source scope; exact bundle/spec; one row per fixed k with source frame200+k; archived observation/state/memory shapes and finite FP32 payload; `start=0/end=151/time=k`; c1/c2 native semantics; bounds/zero bias; original evaluation/source audit; complete unique rows. Missing/misaligned data stop the whole attempt; no replacement or normalization of inputs is permitted.

The Actor100 loader has not been executed for this study. A future qualification must require its returned iteration100 and actual file SHA, plus the actual prefix health and the saved checkpoint100 comparison proving zero Actor optimizer updates. The final-only WM loader and its trusted final health contract stay unchanged. A strict consumer rejection is a preparation blocker, not permission to edit/relabel payloads or weaken existing guards.

Before/after hashes cover both checkpoint files; all WM/Actor parameters and normalizers; both original and functional-clone state; objective buffers and source/spec/target files; all input tensors; Python/NumPy/TorchCPU RNG; and CUDA-uninitialized/noWarp status. Fail closed on any mutation, NaN/Inf, model-load/forward/VJP budget overrun, unexpected device/backend or source drift. Floating-point gradient-identity residuals, zero gradients and opposite directions are measured results, not reasons to rerun or select another checkpoint/state.

Expected fresh artifacts: immutable config/source ZIP; `batch.npz` with saved and newly computed controls clearly separated; loader receipts; objective metadata; raw S/P/w/bonus tables; all gradients by named Actor parameter plus u/a arrays; rowwise norm/dot/cosine summaries (null when undefined); boundary masks; algebraic residuals; mutation/RNG/read-count evidence; report/CSV and optional saved-only plot. No physical accuracy, HO force, forceclosure, whole-hidden-state or training-performance qualification is awarded.

The useful conclusion is narrow: “At this fixed WM/Actor/normalizer and these four archived inputs, enabling the declared support coefficient changes the local prior gradient by this amount/direction.” It cannot explain why one old arm dropped the object, establish that support is beneficial, identify a unique source of prepolicy WM divergence, or predict the outcome of applying an optimizer step or a new physical episode.

Source pins, exact API findings and remaining preparation gaps are in [report.json](../outputs/shared_WM_prior_actor_mechanism_proposal_v1/report.json); the machine-readable preregistration is [preregistration.json](../outputs/shared_WM_prior_actor_mechanism_proposal_v1/preregistration.json). All actual execution remains pending.
