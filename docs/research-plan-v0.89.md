# Actual-control recursive diagnosis after the candidate fit

v0.88 completed fixed2000 updates per arm with44/44 final health checks;
226 regression tests pass. Valid CHECK object12 MSE is1.30063(DIRECT) and
1.60358(IMPULSE), while TRAIN is0.001599/0.008968. Of240 attempted CHECK axis
pairs,196 pass existing shared quality masks;44 remain excluded. Pooled native
object response AD relative errors1.12663/1.07626 exceed zero-response1.
This is evidence against readiness for model-gradient Actor optimization,
not proof that no useful controller or contact-support formulation exists.

1. Use the same fixed2000 checkpoints without any further fitting or selection.
   Read actual parent episode traces and every consecutive source snapshot.
   Candidate branch order is never interpreted as a trajectory.
2. Evaluate all actual controls from k8 through K-2 in all four parent episodes.
   These are development tails following the training phases, not independent
   held-out episodes or untouched final tests. Last interval K-1 is excluded
   because its actual next engine warmstart was not recorded separately.
   Do not substitute post_qacc or candidate-predicted Warm for that observation.
3. Reconstruct102/90 records with the shared chart/packet target function.
   Obtain source/next warmstart from actual source snapshots k/k+1. Require
   exact source/endpoint qpos,qvel and clock continuity, actual held PD targets,
   original force/torque/linear-impulse, no-aid/capacity/nonhand quality masks.
   Geometry and full-task success remain separate and unchanged.
4. Reuse the maintained recorded-control recursive evaluator at H1/2/5/10.
   Only initial state55/warm26 are observed; later state/warm are predicted.
   Supply actual controls/dt without teacher forcing, resets, clipping or
   physical assistance. Retain all attempted/invalid windows and report
   finite failures. Include state persistence as a comparator, full SO3 angle,
   COM and velocity errors; do not count overlapping windows as IID trials.
5. Archive first100/continuation/physical-response/recursive results and source
   receipts together. Report mask exclusions, near-zero contrast instability,
   component costs, no-Actor status and failed task context. Decide subsequent
   data/model intervention from this evidence; do not blindly extend epochs.

Full Goal ACTIVE/incomplete: original policy A1/A2 total costs, physical
gradients, complete cube/pen no-aid gates, multi-trajectory, real ego asset/
dynamics/frames and hardware observation/controller contracts remain required.
