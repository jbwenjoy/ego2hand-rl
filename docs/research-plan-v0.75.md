# Bounded native finite-action control after same-source response validation

Previous turn is PROGRESS: complete independent Data forks pass40/40 live
isolation,1320/1320 saved contracts,840/840 response+14/14 actual force checks,
and ALL24 baseline20ms state/force replay cases.190 tests pass.120 diagnostic
responses reveal strong amplitude dependence with negligible observed repeat
spread. Full fixed-model reconstruction matches all404 original array hashes,
host bytes and numerical receipts; original records remain unchanged.

Next pursue actual no-aid finite-action control, not another neural epoch/
weight sweep or a claim that these finite slopes are exact Jacobians.

1. Implement a reusable ONE-world resident native planning fork in shared
   `src`, reusing complete Data/alias/pointer contracts. Retain one source
   buffer and one independent graph; reset only the fork. Capture resets,
   target application,4/8 native steps and score accumulators as a CUDA graph
   where compatible. Source buffers stay device-resident. Live model/data/
   controller/command are read-only during planning. No actor inference or
   model-gradient use, no object force aid or mid-trajectory live-state reset.
2. Predeclare57 finite RAW action candidates per decision: original2*delta
   center, +/-0.08 along20 joint-coordinate bases,16 local Gaussian offsets
   sigma0.04 clipped+/-0.08 from PCG seed8101. Pre-generate offsets for the
   entire original368-control clock. All models/controllers use the same
   candidate-offset table. Apply original0.5 scale/EMA, residual/target clamps
   and actual encoder bias to compute executed PD targets. Raw legality and
   any clamp are recorded. Search radius is a declared exploration choice,
   not a validated smooth-gradient region or a physical-parameter edit.
3. Fixed horizon20ms; evaluate original next reference COM, directed shaft
   error and hand reference, with bounded residual cost and original geometry/
   joint-limit/no-aid constraints. Use native graph q/v and mean WORLD COM
   constraint wrench; retain the decoder/EFC identity check on selected actual
   executions. Reuse computed geometry/indices rather than repeated mesh/SDF
   reconstruction or per-candidate host validation. Time/score/debug costs
   must be measured, not called real-time from simulated control frequency.
4. Compare STATE_ONLY versus STATE_PLUS_SUPPORT. Predeclare state scales:
   COM0.005m,shaft5deg,hand0.02rad; hand/residual weights0.01. Add only
   support weight0.1, mean force scale0.1N/torque0.005Nm. Derive nominal
   MEAN-INTERVAL COM wrench from reference velocity/angular-momentum change,
   using original clock and actual model m/I/gravity (new explicit target
   schema). It is not measured source force. Preserve derivative-valid masks
   at the reference tail; no resampling, invented labels or early success.
5. Use the original full7.36s reference and stop first native terminal. Same57
   candidates/control and horizon isolate the score change per decision;
   unequal survival implies unequal total computation, so report ALL native
   ticks,decisions,candidates,wall cost and compare the common prefix too.
   Do not call this a final equal-total-budget learned-policy A1/A2 result.
   Retain all candidate constraint violations; if none qualify, explicitly
   flag the least-violation fallback rather than silently relaxing gates.
6. Independently compare EVERY selected20ms prediction/tick with its subsequent
   untouched actual execution at original state/ctrl/force tolerances. The8
   calibrated sources do not pre-qualify newly visited states. Record native
   contact labels, task/geometry/clock gates, failures and chosen raw/EMA/PD
   targets. Only a full qualified native trace can be considered for later
   imitation/distillation. Failed prefixes remain nonexpert; no automatic
   favorable-trajectory promotion or real ego/generalization claim.

This controller probe advances the original contact-support hypothesis while
world-model gradients remain unqualified. Learned SuperMimic/CHORD policy
training/save-resume, fair support-objective comparisons, cube/pen evaluation,
multi-trajectory total cost, true ego input/assets/validation and hardware
contracts remain required. Original full Goal remains ACTIVE. No process is
currently running and no new pause request was made.
