# Test an explicit kinematic residual prior, not more epochs on the failed model

Previous turn is PROGRESS: native scheduled H5 implemented, source/control/score
contracts verified, eight actual episodes and videos completed. H5 survives
0.90–1.02s versus H1 0.30–0.62s in this batch, but all original tasks fail.
Matrix128/128, implementation4901/4901 pass; actual EFC53/56 and prefix278/285,
pilot full100ms fidelity47/111 fail qualification.390000 native integrations,
no Actor/WM updates. Keep full Goal ACTIVE and every failed source/result.

1. Verify source aliases before edits. Do not refit on CHECK/DIAGNOSTIC or train
   an Actor with the currently inaccurate action gradients. Native H5 episodes
   remain CONTROLLER_DIAGNOSTIC, not experts or new TRAIN. Keep original task,
   reference/fixed-wrist/no-aid contracts and numerical failures explicit.
2. Existing evidence motivates a model intervention: pure constant velocity
   gives20ms COM1.27mm/full angle3.21deg, better than learned models; true memory
   alone does not stop drift. Introduce DIRECT_CV alongside DIRECT, with a
   deterministic source-state baseline in the SAME WORLD-left output chart:
   hand position delta=H*hand_qd, COM delta=H*WORLD_v, world rotation vector
   delta=H*WORLD_omega, H=.02. Other output components have zero baseline.
   This is an additive rotation-vector coordinate residual, explicitly NOT a
   product of finite rotations. Decode the summed vector using the existing
   WORLD-left Exp map; no geometry repair, clipping or hidden teacher forcing.
3. Both arms use the same102-input/90-output Fn6 semantics, parameter count,
   width512, initialization, valid TRAIN3858/CHECK1773 rows, batch2048, seed7411,
   AdamW lr3e-4/wd1e-5/clip1, auxiliary weight.1 and fixed2000 updates per arm.
   Keep existing TRAIN x/y scales as physical loss scales. For DIRECT_CV only,
   center the output head with TRAIN mean(y-baseline(x)); add baseline(x) back
   in prediction. Fn6 nonnegative convention, warm/wrench labels and physics
   unchanged. No extra capacity or normalization fitted to CHECK. It remains a
   component test, not the original learned-policy support A1/A2 comparison.
4. Use shared maintained model/training/evaluation infrastructure, not another
   copied study/trainer. Give the new family explicit checkpoint/config schema;
   reject accidental old optimizer/arm resume. Fresh paired models, first100
   atomic save/reload/next-update checks and conditional ETA, then fixed budget
   under existing user resume authorization. No automatic Goal pause inferred
   from legacy metadata. Count clone updates and all costs separately.
5. Reuse the exact existing physical finite-action CHECK pairs/masks and actual
   tail H1/2/5/10 windows for development regression, including CV/persistence
   comparators, all excluded uncertain labels, finite failures and per-parent
   summaries. Source-only baseline has zero direct PD derivative; any useful
   action gradient must be learned and validated against actual native finite
   responses. Self AD/FD or lower training loss does not establish that.
6. CHECK/tails have guided development and are not untouched final tests. No
   architecture/hyperparameter sweep or extra epochs after these fixed results.
   If the prior does not improve physical/recursive readiness, preserve that
   evidence and move to reference/contact feasibility and newly predeclared
   state/trajectory coverage rather than silently tuning on the same checks.

Original full requirements remain: no-aid learned cube/pen controls and unified
save/resume, physical contact/body wrench/gradients, fair-total-cost residual/
support Actor A1/A2, multi-trajectory adaptation costs, real ego inputs/frames/
collision assets/dynamics and hardware subversion/observation/zero/gain/latency.
Do not replace them with success on a WM prior, short native survival or a
different retimed/projected reference. Goal is incomplete, not blocked/paused.
