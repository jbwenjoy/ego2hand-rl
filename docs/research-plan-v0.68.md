# Failure-triggered numerical-quality intervention at fixed source states

v0.67 saves18 real action-intervention trajectories and prospective envelopes.
Calibration6/6 and CHECK12/12 envelope diagnostics pass, but EFC124/126 leaves
two original impulse gates failed: D4_PLUS5ms POST0.12s and2.5ms POST0.055s.
Saved balance decomposition attributes these residuals mainly to final
free-body equation residual, with integration roundoff~1e-11N*s. This is not
a unique solver-cause proof. No labels are promoted to experts/training.

At EACH failed source PRE state, restore exact qpos/qvel, preceding actual
qacc warmstart, already-applied ctrl and time. Require native host topology
and all15 physical fields match the source of the full404-array snapshot.
Restore404 device fields/195 scalars exactly into independent models before
any variant edit. Keep all mass/inertia/geometry/friction/gravity/timestep/
integrator/CCD parameters unchanged. Test128 worlds,three reset probes of ONE
physical step per variant, with exact fixed source input. No selection/rerun.

Three preregistered variants: DEFAULT10iterations/default tolerances; ITER50
only raises iterations to50; TIGHT uses50iterations,50line-search iterations,
tolerance1e-10 and line-search tolerance1e-6. These are explicit NEW numerical
settings, never a retroactive change to original runs or acceptance. Save
iteration counts, source inputs, actual raw EFC forces, state and full6D
free-body equation residual. Independently compute the original impulse gate
5e-7N*s and force/torque EFC identity1e-4N/1e-5Nm. Count all setup/probe ticks.

This post-failure local diagnostic cannot qualify a full task,100ms fidelity,
other contacts/actions or a learned model. Even a successful TIGHT probe needs
fresh original-clock full-case execution and repeated envelope/label checks
before use. Failed DEFAULT and original two impulse checks remain failed.
No installed engine/kernel/CUDA/driver edits or neural optimization. Full Goal
and future first healthy checkpoint restore/ETA pause remain unchanged.
