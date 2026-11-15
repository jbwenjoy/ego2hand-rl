# Portable physical-state closure and terminal-safe collection

Previous goal turn is PROGRESS: native full-pose161/critic188 adapter implemented,
GPU worlds1/32,528 transitions per version. v1 failed662/670; v2/v3 pass674/674.
Independent saved-state/FK reviews256/256 each,15 physical array groups match
original H5 exactly in every world.266 tests pass. Total12672 native integrations,
1056 CPU FK,0 actual task optimizer calls. All failed versions retained. v3
removes duplicate policy encoding but does not establish speedup or native repeat
determinism;32-world terminal reset count41 vs42. No trained policy/ego result.

1. Reverify the native policy evidence ledger and exact aliases before edits.
   Keep full Goal ACTIVE/incomplete. Do not fit historical DIAGNOSTIC_NEVER_FIT
   decision/response records or call the scripted rollout an expert. Initial
   joint projection/EMA are declared configuration, not supervised samples.
   Keep original cube guard/155 weights and strict7.36s evaluation unchanged.
2. Derive a portable snapshot schema from the actual controlled runtime. Reuse
   gpu_model_snapshot.model_leaves for complete model/Data array traversal;
   keep ResidentDataSnapshot pointer-identity/in-process semantics unchanged.
   New portable copies must validate keys/shapes/dtypes/scalars before mutation
   and copy into fresh captured-graph allocations, never replace live pointers.
   Preserve inactive scratch bytes including NaN payloads while separately
   checking ACTIVE finite dynamics; array-copy parity is not rollout parity.
3. Inventory full mutable closure with stable paths: physical qpos/qvel/ctrl/time/
   warmstart/mocap/constraint/contact arrays and model fields; entity position/
   velocity/effort target buffers and actuator/delay histories; action raw/EMA/
   target/mask buffers; command phase/trajectory/timers/reset-prepared/sampling
   histories; observation history/delay caches; episode/manual-reset/pending/
   common/physics counters; adapter initial configuration; all relevant RNG.
   Identify which fields are static protocol/topology vs dynamic state. Save
   source/config/asset/reference layout pins, reject missing/extra/incompatible
   state, and explicitly report any mutable leaves outside the closure. Do not
   claim generic all-task continuation from the restricted controlled adapter.
4. First verify immediate restore of all serialized bytes and buffers, then
   uninterrupted versus FRESH-PROCESS restoration in worlds1 and32. Predeclare
   scripted prefix8 controls, continuation8; no policy optimization. Producer
   runs16, restorer8, so nominal6336 world integrations across both sizes. Count
   every actual setup/retry/diagnostic integration additionally, keep failures.
   Same stream/collision/scheduler and model options as v3; fixed sampled-action
   sequence plus separate RNG-next-value checks. Direct equality for clock,
   phases/controller targets and copied bytes; physical rollout tolerances remain
   those in configs/pen_gpu_step_audit_v1.toml, do not loosen for resume success.
   Comparison must identify complete active contact ownership and source fields,
   not only qpos. Single-process rollback is a separate weaker result.
5. Extend independent inspection of raw contacts/EFC/world IDs, active-prefix
   boundaries, geometry/body/digit grouping, force/wrench decoding and capacity.
   Existing674 checks cover finite/no aid/overflow and control/reset contracts,
   not full substep wrench/dynamics qualification. Validate selected world reset
   cannot contaminate another world's contact loads or internal actuator state.
   Known collision repeat failures remain qualification failures if observed;
   do not hide them with model loading success or a different test phase.
6. Once physics/terminal contracts qualify, implement a collector around the
   adapter and shared PPO update. Store unsquashed sampled actions/log_prob and
   applied clipped actions separately; expose true terminal critic188 BEFORE
   reset. For truncated transitions bootstrap with the terminal-next-state value
   once; for real terminations zero it; GAE never crosses reset episodes. Test
   mixed terminated/truncated/continuing worlds and reference phase alignment.
   New checkpoint combines native closure with modules/optimizers/norm/RNG/
   counters and161/188 metadata. No old cube format4/155 reinterpretation.
7. Launch only after native/save-resume/collector contracts pass, using the
   existing v0.96 short budget32env×64rollout×32iterations,4epochs×8minibatches,
   seed7411,AdamW1e-4/gamma.996/GAE.95/clip.2/value.5/entropy.001/gradclip1.
   65536 transitions,524288 nominal integrations,1024 optimizer calls; separate
   every validation/evaluation/restart cost. Save at iteration2 and verify health,
   source/step counts/resume plus measured ETA before continuation. No new user
   approval/pause requirement; the earlier checkpoint pause was served/resumed.
8. Profile real training rollout/update/geometry/synchronization and GPU/VRAM
   after warmup; diagnostic per-step CPU copies are not throughput evidence.
   Later learned residual/support comparisons still require same-total-cost
   A1/A2, independent seeds/trajectories and original no-aid task gates. A native
   PPO baseline bypasses failed WM gradients; it does not qualify SuperMimic/
   CHORD contacts/gradients or real ego/hardware transfer.

Unfinished full scope persists: cube/pen learned control, portable unified strict
evaluation, BODY COM contact wrench support and true physical gradients,
fair residual/support/multi-trajectory costs, real ego frames/geometry/dynamics
and hardware observation/zero/gains/latency. No narrow component completes Goal.
