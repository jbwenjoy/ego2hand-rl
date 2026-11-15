# Prepare fresh response-learning data with parent-source splits and explicit label validity

Previous turn is PROGRESS: v0.85 observed-source replay completed2784 native
integrations; source/model/control276/276, mass interpretation36/36, EFC83/84,
graph/instrumented11/12, all57 repeats438/456. All484 required response comparisons
pass state/ctrl/time gates;20 force and13 torque failures retained. One same-PD
repeat has an impulse failure, localized to constraint-force/acceleration balance,
not mass-block decoding. Native stopping-source review does not prove final stop
reason. Finite contrasts400/400 interpreted, object signal71/80 and wrench70/80
exceed observed3-repeat envelopes.218 tests pass; failed JSON setup preserved.
No neural/Actor update, successful task or expert. Full Goal ACTIVE/incomplete.

1. Move toward the required learned world model while preserving uncertainty:
   prepare a NEW explicitly named TRAIN/CHECK collection. Every prior
   DIAGNOSTIC_NEVER_FIT parent/row remains excluded; no fitting the80 contrasts
   or relabeling previous failed trajectories as experts. Native nonexpert
   transitions may supervise a simulator-response model, with validity masks,
   without being called successful imitation demonstrations.
2. Freeze split/config before collection. Use NEW actual X4/FULL_SO3 no-aid
   original-clock parent episodes for STATE_ONLY/STATE_PLUS_SUPPORT at5/2.5ms.
   TRAIN phases k1,2,3,5,6,7; CHECK phases k8,9,10; exclude diagnostic k0,4.
   Keep every action/repeat from a parent source together. Missing phases after
   early termination stay missing; no reset/teacher repair to fill them.
   These are disjoint development parent/phase sets on ONE clip, not independent
   held-out trajectories, real ego or statistical confidence samples.
3. At each available source evaluate the fixed57 X4 actions at original20ms,
   three repeats. Preserve entire source arrays/scalars/layout/model/controller,
   future reference and actual PD. Expected36 parent sources if all phases exist:
   24 TRAIN/12 CHECK,6156 candidate rollouts,36936 candidate ticks across matched
   5/2.5ms parents; report actual cost/missing parents separately. Main parent
   controller physics and independent label audits add costs, never hidden.
4. First specify and verify candidate labels against maintained response schema:
   state55 + engineWarm26 + actualPD20 + nativeDt1 input, next52 tangent/state,
   nextWarm26, source-BODY mean wrench6 and contact-normal heads. Reuse existing
   state/rotation/impulse math, not a parallel implementation. Current graphs
   need GPU recording of next warmstart and named normal-load groups. Verify
   canonical Fn6 mapping against current geometry and account separately for
   every other robot contact; if semantics differ, version the schema/checkpoints
   rather than squeezing new labels into old90-output weights. Keep WORLD/BODY,
   COM, quaternion order and original50Hz action/EMA clock explicit.
5. Independently check force/torque reconstruction, per-step momentum, finite/
   no-aid/capacity and repeated-response spread at original gates. Predeclare
   a shared usable-label mask for both future model arms: do not repair or average
   away failed impulse labels, or treat high-spread force targets as exact.
   Save all rejected rows with reasons. Keep task/geometry-success flags separate:
   physically interpretable failure transitions are not policy experts. All
   original complete-task gates still apply to subsequent policy/control tests.
6. Prepare normalization from valid TRAIN only; preserve parent keys, all three
   repeats and response-pair masks. CHECK may evaluate but never fit scalers,
   tune losses or choose a favorable checkpoint in this planned experiment.
   Current102/90 response networks/checkpoint-resume infrastructure may be reused
   only after schema/physics compatibility is verified; old failed training
   remains historical. Dataset batches should stay GPU-resident in future fitting,
   with no repeated geometry or per-sample host synchronization in the loss.
7. This phase delivers reviewed data/contracts first, not a blind long Actor run.
   Then predeclare matched DIRECT/IMPULSE state+contact response fitting and saved
   resume checks. Finite action prediction agreement is distinct from a native
   Jacobian: contact-switch contrasts and a model's own autograd checks cannot
   by themselves qualify a physical contact gradient. Require actual subsequent
   closed-loop full-task evidence before policy/expert/acceleration claims.

Original learned SuperMimic+CHORD equal-total-budget residual/support policy
A1/A2, world-model contact/gradient validation, cube/pen/save-resume, multi-
trajectory total adaptation cost, real ego frames/assets/dynamics and hardware
contracts remain required. No active training/checkpoint wait or new pause
request; full Goal ACTIVE/incomplete.
