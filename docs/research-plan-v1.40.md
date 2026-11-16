# Research plan v1.40 — interval-supervised model comparison

Preserve every original M0–M5 and v1.39 requirement. Goal ACTIVE/incomplete.

1. Fixed three-arm DIRECT study completed session17860 exit0. Each2000 Adam,
   original3202 TRAIN points/779 qualified pairs,matched architecture/initial
   NN/point+pair streams. RESPONSE_ONLY0/0,PATH_POINT1/0,PATH_POINT_RESPONSE1/1
   extra path point/response weights. Predict interval geometry LOSS only;
   hard failure/error labels still lack positive coverage and are unmodeled.
2.100/2000 owned checks46/46 each,including restored next update.6000 live+
   12 clone QA=6012 Adam,0native/Actor.536 tests pass. Original90-unit buffers
   reused;new TRAIN path-unit computations3 total(production,isolation,review).
   Training+QA23.218s,runner26.962s;devicewide medianGPU23%,not overall speedup.
3. Full final evaluation session30286 exit0,29/29,9.331s;all original groups,
   masks,invalid centers and both observable/full native targets remain. Model
   own AD/FD agrees locally but does not prove native gradients. Independent
   saved arithmetic rechecks scores/chains/screens/FD/aggregations and CPU
   readonly model/Adam metadata;GPU RNG is not migrated to CPU.
4. Development FULL reward AD-relativeL2(residual all386/stable280;support
   all386/stable292):baseline .936/.608/.952/.678,path point .937/.423/.951/
   .464,path point+response .937/.642/.956/.631. Stable subset improves for
   point-only,all-direction error remains near zero-response1. Do not claim
   full gradient qualification or select weights/checkpoints from this inspected
   development evidence. No model admitted for Actor.
5. New baseline is a new matched retraining,not a byte-replay of the prior
   six-arm final models. Same-weight DIRECT objective/gradient parity is checked
   separately. All three model outcomes/forward tradeoffs and both denominators
   remain reported.32 validation pairs with original invalid BASE remain in
   full denominator and excluded from local candidate admission.
6. Next freeze the existing three models/weights and register no-aid native
   one-step candidate-direction/amplitude probes. Measure actual native reward
   change,all failures/invalid centers and every native query cost before any
   Actor update. Keep original physics/source/impulse gates;explicit full-rehash
   scope may be used in the new entry. Do not regenerate data merely to hide
   unfavorable all-direction results or refit validation normalization.
7. Contact-switch/failure coverage,multistep EMA/rolling error and native policy
   improvement remain. Preserve true A1/A2,full7.36s no-aid task,multi-reference/
   independent-seed total costs,ego asset/gravity/noninertial/dynamics/RGB and
   hardware model/control/latency/observation contracts. Old epoch anomaly stays
   unresolved;no epoch/cached hash,driver/CUDA/installed edit,push/publication.
   Goal ACTIVE/incomplete.

See pen-path-study-results-2026-11-16.md and pen_path_study_results_v1 ledger.
