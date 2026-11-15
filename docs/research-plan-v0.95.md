# Explicit full-pose policy observation contract before native policy learning

v0.94 audit is PROGRESS:85/85 checks pass on all original H1/H5 trajectories,
raw clip and fixed mount. Rebased COM difference<7.5e-8m, rotation<4.7e-7rad;
reference joints stay within PD limits. H5 first full10deg breaches occur at
0.595–0.6175s, mainly shaft rather than solely axial twist. Six H5 decisions
miss an available first-pose-qualified candidate due to mean future costs.
Current155 policy/reward directly encode only the two shaft points: controlled
10/90/180deg axial rotations produce the same feature. This is instantaneous
aliasing, not proof of global unobservability or sole task-failure cause.
PPO remains explicitly cube-only; no guard has been removed.250 tests pass.

1. Keep existing upstream155 observations/rewards/checkpoints unchanged. Add a
   separately tagged161-dimensional full-pose policy contract: reference hand20,
   reference pose9, reference links63, current biased-relative joint20, scaled
   joint velocity20, current pose9, previous applied raw action20. pose9 uses
   wrist-local bounded COM3 and the first two rotation MATRIX COLUMNS6. Normalize
   valid quaternion scale/sign at the boundary; zero/invalid poses remain invalid,
   never silently replaced by identity. No physical geometry repair or clipping.
2. Replay all285 recorded H1/H5 decision sources. Require old reference/current
   shaft features to agree with raw/cached source poses under fixed wrist and
   original phase/units, diagnose qpos-versus-derived-pose timing explicitly.
   Replace only the two pose encodings, retaining all other channels exactly.
   Store source/target timestamps, current/ref poses and new161 features.
3. Test common world-frame transform invariance and quaternion sign, full axial
   distinction, column ordering and dimension/action-contract rejection. Recheck
   clip hash, source q/v/control/clock and every case; do not interpret offline
   reconstructed features as a deployed Actor or an ego observation system.
4. Do not feed161 observations to existing155 weights, reinterpret a cube PPO
   checkpoint, or enable pen PPO by deleting its guard. The new contract stage
   has no optimizer/physics integration. Native vectorized env/collector/reset/
   reward and exact save/resume still need a separately declared integration
   protocol. Any later actual-simulator PPO baseline bypasses the failed WM
   gradient only; it does not qualify SuperMimic/CHORD model gradients.
5. Plan that later baseline with full-SO3 reference/reward observability, bounded
   raw actions/EMA/PD limits, explicit actual-source initialization and no force
   aid. Reuse shared PPO update primitives, verify legacy cube behavior before
   extraction, and validate nworld1/multiworld body/frame/control/load ownership
   before training. Count actual transitions and all optimizer/native costs;
   retain original7.36s no-aid evaluation without post-start resets and later
   same-total-cost residual/support A1/A2. A feature/checkpoint test alone is
   not a training launch criterion; proceed within the already authorized
   research scope with a concrete validation/budget protocol.

Full Goal ACTIVE/incomplete: learned cube/pen controls, unified save/resume/
no-aid gates, body wrench/physical gradients, multi-trajectory costs, true ego
frames/collision assets/dynamics and hardware observation/zero/gain/latency.
Original strict failures remain. No representation change claims task success.
