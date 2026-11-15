# Resident contact metrics before geometry-aware policy comparisons

Previous goal turn is PROGRESS: actual native PPO32worlds×64rollout×32iterations,
65536 transitions/1024 optimizer steps, plus256 normalization-only transitions.
First2/final health10/10, independent combined-load/CUDA clone checks20/20 each;
collector40/40,285 tests. Evaluation readback99/99. Original mean-policy native
terminal moves0.14→1.14–1.18s, but ALL strict7.36s tasks fail. Latest32 policy has
max object/self penetration3.528/1.611mm;2/.2mm gates unchanged. Native count529544
incl qualifications/evaluation/retries;192 discarded diagnostic CUDA updates.
No CHORD/WM-gradient/ego/speedup claim. W32 physical replay remains unqualified.

1. Reverify the native PPO ledger/source aliases, retain all failed raw records.
   Do not continue epoch/budget sweeps on this geometry-unpenalized reward and
   call native survival task success. Same0.14s prefix improves pose but already
   penetrates3.242mm; longer held contact may exploit compliance/geometry. Final
   saved2048 phase range1–54 is limited coverage, not full-trajectory learning.
2. Implement read-only resident per-world metrics using fixed geom/body/robot
   classification and canonical joint/actuator indices on GPU. Required outputs
   are object penetration max, robot self-penetration max, nonhand normal force,
   finger1..5/other-robot Fn6, object hand/all contact WORLD/BODY-COM wrench,
   joint-limit excursion and active-index/capacity/finite status. Respect actual
   compound cylinder+boxes and condim/pyramidal EFC address semantics; do not
   guess a capsule or use raw geometry ID order as digit order.
3. Verify these metrics first on saved raw EFC/contact records and W32 source
   snapshots against independent NumPy/native readers. Use ACTIVE-prefix/world
   masks, detect invalid ACTIVE IDs/indices; never let unused scratch NaN or
   out-of-range indices become physical contacts. Mask inactive storage for index
   safety explicitly, no clamp/repair of real physics. Preserve contact signs,
   normal/tangent/torsion, normal nonnegativity, moment arms at the EFC geometry
   phase and correct WORLD→BODY rotation. Raw decoder parity is not a Newton/
   gradient/repeatability qualification. Never-fit data may validate arithmetic,
   not become supervised task examples or experts.
4. Hook read-only accumulation at EVERY2.5ms native step and fold over the20ms
   action interval, rather than checking just the final forward's contacts.
   Cache static geometry/reference tensors on GPU; keep stream/array ownership
   compatible with existing CUDA graphs and portable schema. Validate1/32 worlds,
   partial resets, no cross-world accumulation, exact reset of metrics and
   source captures before/after. Any new state/transition/checkpoint layout is
   separately tagged; do not reinterpret current161/188 weights or full native
   checkpoints without an explicit weights-only initialization protocol.
5. Profile actual collection/check/update after warmup, count CPU scalar reads,
   repeated feature/geometry work, allocations and CUDA/VRAM. Current~3s rollout
   vs~.12s PPO update identifies collection as priority;67% device utilization
   and low VRAM do not prove the old WM pipeline was optimized. Measure resident
   metric overhead/parity at matched inputs and update/physics counts; diagnostics
   that copy every contact to CPU are not throughput benchmarks. Do not patch
   old study copies or installed drivers/runtime packages.
6. Once metrics qualify, predeclare geometry-aware reward/termination or
   constraint rules on original2mm/.2mm/10mm/full10deg gates and preserve the
   original task timing/strict evaluation. Compare against the frozen PPO32
   baseline with explicit new data/optimizer/total compute budgets; no silent
   physics/reference retiming/asset shrinking or force aid. Keep true reference
   feasibility and partial observability/history/velocity contracts separate
   from reward tuning. Do not replace the full task with an easier hold task.
7. Reuse existing mean-reference wrench/BODY support helpers, audit origin/time/
   rotation/mean-versus-instantaneous conventions before adding a native support
   reward. A reward target is not an applied object force. Later residual/support
   A1/A2 must share geometry rules/initialization/network/data/total native and
   optimizer budgets, declared seeds/trajectories and strict gates. Model-free
   PPO can test a support reward without pretending failed WM action gradients
   are qualified. Continue WM physical/finite-gradient validation separately.
8. Preserve the open numerical branch: W32 same/fresh-process divergence still
   needs complete substep source snapshots at declared4/5/6 around failure, raw
   active/inactive contacts/solver/world fields, and repeated fixed-source noise
   envelopes. Do not treat source-copy passes or a readback99 as deterministic
   physics. Real ego frame/asset/dynamics and hardware observation/zero/gain/
   latency remain future validations under their original contracts.

Full Goal ACTIVE/incomplete: learned cube/pen no-aid full tasks, portable unified
evaluation, BODY contact support/physical WM gradients, fair multi-trajectory
costs, real ego and hardware. First-checkpoint pause was already served/resumed;
no new pause is requested. Continue authorized reversible research, no extra
approval gate or source/data role promotion.
