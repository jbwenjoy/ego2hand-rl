# Explicit cached training and trained64 W32 timing — protocol v1

Registered before collecting new timings. Preserve prior fixed-frame and
feature-JVP tolerances,all physics/geometry/no-aid/contact-window gates,and
UPSTREAM control. Default mode stays LIVE. New objective config explicitly
selects CACHED at cold construction;old configs/checkpoints remain immutable.

1. Current-source actualCUDA32 qualification:369 phases,clamps,dynamic/NaN,
   saved-mode-before-write and allocation guards. Reuse the existing exact
   offline score receipt only if all four original source/input pins verify;
   no historical simulator source receipt is rebound.
2. Fresh CACHED qualification uses shared calibration8 and both arms4 controls:
   4096 native integrations,zero optimization;exact saved-state/readback,
   independent target/score/reward and existing consistency checks required.
3. Fresh CACHED training uses two arms,2x64x32 transitions each,32 Adam steps
   per rollout. Total65536 native integrations/128 updates. Retain first and
   second checkpoints,health,failures;main64 policies/video are not replaced.
4. Load original SUPPORT trained64 actor/critic and embedded normalizers only
   into fresh W32 physics using the original objective/adapter configuration.
   Compare ALL404 native Model arrays by keys/shape/dtype/bytes,ALL195 scalars,
   host model,reference,full shared objective/observation/action/joints/NN/PPO
   metadata and historical snapshots. Do not broadcast a W1 context. No old
   native,Adam,RNG,counters,rollouts or observations are imported. Native4,
   Python and RNG closure must remain byte-identical across initialization.
   Embedded normalizer data counts are explicitly pinned to the source under
   a distinct training5 weights-seeded schema;new experiment work counters
   stay zero. Existing training2/3/4 count guards retain zero seed offset.
5. Save this initialized current-source state with zero counters/empty Adam.
   Warm two64-control/32-update rollouts,then reload the saved current-source
   state and prove warmup NN/native/RNG/counters discarded. Comparison fixture
   must equal that initial state. Charged warm work remains in the report.
6. Same explicit LIVE/CACHED order as prior8-case comparison:3 clean repeats
   per mode and2 separate CPU profiles. Only three declared mode fields may
   differ in saved source forks;ordinary loader guards stay enabled. Each
   case64x32 controls and32 diagnostic updates. Total163840 native/320 updates
   including discarded warmup;no recalibration. Report raw timings,medians,
   whole phase ratios and reset counts;do not pool earlier timing cohorts.
7. All dynamic feature JVP tolerances remain2e-6,forward2e-7/reference1e-7,
   with exactness separately reported. They do not qualify physical contact
   or world-model gradients. Three repeats on one trained seed/reference do
   not establish universal throughput or full-training/multitrajectory benefit.

Retain failed attempts,all source versions and exact full-path relocations.
After this integration return to full20D action-response coverage and isolated
TRAIN-only loss scaling. Original324 CHECK remains NEVER_FIT;new validation
must be reserved before fitting. Full7.36s task,ego assets and hardware goals
remain incomplete. No installed driver/CUDA/package changes or external push.
