# Fresh paired world-model first100 checkpoint on reviewed parent-split candidate data

Previous turn is PROGRESS: v0.86 complete.36 fresh parents,6156 independently
executed candidate responses; valid TRAIN3858/CHECK1773,525 rows retained and
excluded by shared action-group masks. Independent readback49253/49253,
decoder refactor parity8/8 and222 tests pass. Total95664 integrations includes
parent primary33516, actual588, extra repeats24624 and audit36936, no double
count of labeled primary candidates. All parent tasks fail; no expert/Actor/
real-ego result. First100 entry dry-run passes, no training launched. Goal ACTIVE.

1. Use frozen configs/pen_candidate_fit_first100_v1.toml and exact reviewed data.
   Fresh DIRECT/IMPULSE models, maintained102-input/90-output Fn6 semantics,
   width512, identical NN initialization and CUDA sample streams. Same batch2048,
   AdamW lr3e-4/weight decay1e-5, gradient clip1, auxiliary weight.1,seed7411,
   planned2000 updates per arm. No old model/optimizer resume or CHECK-based
   selection. This is a world-model component comparison, not policy A1/A2.
2. Use only shared valid TRAIN mask and already frozen TRAIN-only scales/physics.
   Dataset batches remain GPU-resident; supervised loss contains no native
   collision/geometry solver or per-sample CPU readback. Keep masked failures
   available for explicit diagnostic evaluation, not relabeling or hidden removal.
   New batch differs from old256: no cross-batch/history acceleration claim.
3. Run existing first100 boundary, atomically save both model/Adam/sampler/RNG/
   scales/physics/cursor/gradient snapshots. Verify exact reload predictions,
   matching next sampled batch and two-restored-clone next-update parity. Account
   separately for four diagnostic clone optimizer steps. Failed health must
   remain on disk and cannot trigger blind continuation.
4. Report loss and valid CHECK blocks, nonnegative Fn6, finite gradients and
   conditional remaining1900 compute ETA from CUDA timings. Also report mask
   coverage and label uncertainty; excluded CHECK rows are uncertain native
   labels, not accurate ground truth. Device telemetry on a short first100 run
   is not a sustained utilization or whole-pipeline benchmark.
5. Keep distinctions explicit: defect-label algebraic closure is not native
   momentum correctness; a model's own autograd JVP/finite differences are not
   physical action gradients; pointwise fit is not recursive stability or full
   task success. Plan physical finite-contrast and held-reference recursive
   evaluation after checkpoint health, using no CHECK fitting or future resets.
6. First100 is a process/checkpoint-validation boundary. The previous standing
   first-checkpoint goal pause was already serviced and the user resumed; no
   new pause request is inferred from a legacy health-file flag. Full Goal stays
   active under current resume authorization. Do not update Actor or claim
   stable control/ego/generalization/acceleration from healthy checkpoint alone.

Original learned policy residual/support fair-total-budget A1/A2, contact and
physical-gradient validity, cube/pen/save-resume, multi-trajectory adaptation
cost, real ego inputs/frames/assets/dynamics and hardware contracts remain
required. No current training/checkpoint wait; Goal ACTIVE/incomplete.
