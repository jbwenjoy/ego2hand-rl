# Native full-pose pen learning integration, before task training

Previous goal work is PROGRESS: task/frame audit85/85, offline full-pose161
replay3422/3422 on285 sources, original source/control timing retained. Frozen
old PPO reproduces a multi-minibatch bootstrap-autograd failure; shared update
detaches fixed targets and matches the frozen algorithm with only that fix,
including optimizer bytes. Seven focused tests pass, final suite261 tests.
No new actual task training/rollouts. Full Goal ACTIVE, original failures kept.

1. Before edits verify the contract evidence ledger and capture exact live
   sources. Keep original155 checkpoints/tasks and cube-only PPO config guard.
   Build an explicit native pen adapter/collector using shared ppo.py, never
   a copied trainer or the unqualified WM gradient. Existing cube save/load is
   weights/optimizer/RNG resume, not full physics continuation; do not relabel it.
2. Use the separately tagged161 policy contract with perfect full pose first;
   actor has no object velocity/force privileged channels. Camera/ego estimation
   is a later explicitly noisy contract. Reference/current pose origins must be
   COM; current controlled asset has BODY origin=COM, enforce or offset correctly.
   Specify critic layout before implementation and pin it; do not infer dimensions
   or reinterpret old301/155 weights. Proposed critic188=policy161 + true physical
   EMA20 + wrist-local object linear/angular velocity6 + normalized reference
   phase1; this is privileged simulation input, not a Markov proof or ego input.
3. Sampling contract is important: the existing pen action term does NOT clamp
   raw inputs to[-1,1]; it clips scaled residuals to±0.6. For new161 explicitly
   clamp the Gaussian sample to[-1,1] BEFORE env.step, keep unsquashed sampled
   action/log_prob for PPO, and expose the applied clipped action as previous
   action in policy input. Then .5 scale/.5 EMA and original PD bounds apply.
   Record sample, applied raw, EMA, reference phase and actual transmitted PD
   targets. Existing cube sampling/action convention remains unchanged.
4. Controlled native source: same single upstream clip369 frames/50Hz/fixed
   wrist, original physical asset and dt2.5ms/decimation8, original CCD50/1e-6
   and solver/line-search settings. No force aid, gravcomp, equality, material
   repair, reference retiming or mid-evaluation reset. Freeze exact model
   tensors/units/joint order and permitted reset initialization. Existing
   initializer is single-world and requires selected initial q/EMA: generalize
   it explicitly or provide a reviewed reset adapter, not broadcast guesses.
   Keep initial source v/warmstart semantics, zeros for forbidden applied forces,
   and declare initial EMA/phase plus reset RNG. Seeded phase0 first; random-phase
   or extra trajectory sources require distinct protocol/roles.
5. First implement and inspect reward/reset/frame step ordering. Full-SO3 reward
   must distinguish axial rotation; select explicit weights/scales in config,
   preserve original strict evaluation gates separately. Training auto-reset may
   end on original native50mm/shaft60deg/reference end, but evaluation stops at
   first terminal without resetting and checks original10mm/full10deg + RMS5mm/
   5deg over all7.36s, geometry/force/capacity/finite/no-aid conditions. Success
   cannot use training rewards or native survival alone. Timeout bootstrapping
   must be explicit from the actual terminal transition and applied once; do not
   copy the cube current-value timeout rule while claiming next-terminal-value
   bootstrapping. Track both native terminal reason and strict-gate first breach.
6. Native integration qualification BEFORE optimizer: nworld1 then32,16 control
   steps each with declared bounded scripted actions and partial resets. Nominal
   budget528 transitions/4224 integrations, plus explicitly counted simulator
   setup, resets, diagnostics and replays. Resident static reference/index tensors,
   Torch/Warp stream ownership and independent-world state are required. Check
   fullpose observation against actual qpos/xpose, source/reference clock, no
   world mixing, EMA/PD transmission, finite/no aid/capacity and all reset fields.
   Persist failing states and costs; never silently restart a live process.
7. Before training, serialize/restore full state in a fresh env/process: physical
   qpos/qvel/ctrl/time/warmstart plus required solver/collision and model arrays,
   action/command/observation/reset buffers, module/optimizer/norm/RNG/counters.
   Derive the complete closure from actual runtime, not the reduced WM state.
   Compare an uninterrupted bounded sequence and restored sequence using declared
   state/action/reward/contact tolerances; retain known collision-repeat failures.
   In-process rollback alone is insufficient. If exact continuation is unqualified,
   label optimizer-only restart distinctly and do not mark full save/resume done.
8. Only after native contracts qualify, concrete first training budget is32 envs,
   rollout64,4 PPO epochs,8 minibatches,32 iterations,seed7411,AdamW1e-4,
   gamma.996/GAE.95/clip.2/value.5/entropy.001/gradclip1; no WM update.2048
   transitions/iteration,65536 total,524288 nominal native integrations and1024
   optimizer calls; add setup/evaluation/resume costs separately. Save after2
   iterations, validate checkpoint/finite stats/real step counts and measured ETA
   before any continuation. No claim of task success from this short budget.
   Source freshness, validated reward/action/reset/timeout contracts and budget
   config are launch requirements. Existing authorization covers this work;
   no new permission gate is introduced. Earlier first-checkpoint pause was
   already served and user resumed; no new pause request is active.
9. Keep GPU geometry/performance work in this single repository. Profile actual
   native rollout vs policy update, peak VRAM and synchronization after warmup;
   cache static geometry/reference data and batch device work based on measured
   evidence. No fabricated speedup from CPU unit timing or unequal physics calls.
   Later residual/support A1/A2 includes collection, training, review and evaluation
   total costs, original clips/roles and independent seeds/trajectories.

All full-objective requirements persist: learned cube/pen no-aid controls,
portable save/resume/strict unified evaluation, BODY contact wrench targets,
physical WM gradients, fair policy comparisons/multi-trajectory amortization,
true ego frame/asset/dynamics ingestion, and hardware observation/zero/gain/latency.
Negative diagnostics or a model-free baseline do not finish or block that Goal.
