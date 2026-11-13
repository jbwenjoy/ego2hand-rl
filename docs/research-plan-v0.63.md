# Bounded actual-GPU dynamic-wrench contact proposal

v0.62 actual-GPU audit passes saved static replay, EFC wrench identities and
81-world versus individual queries at the declared force/moment tolerances.
Both finite-difference directional checks fail and active contacts change even
at h/2. Preserve these failures; do not use that Jacobian for optimization.
The batch-query timing is not training or full retargeting speed evidence.

Use derivative-free cross-entropy proposals, seed6201, population512,20 rounds,
32 elites, mean/std smoothing0.7, in the existing model at original5ms forward
dt. There is no integration, neural optimizer or Actor fit. Bounded variables
are20 joint offsets within0.1rad of source and20 PD preload offsets within
0.05rad of the candidate joint pose. Also enforce legal original hard/ctrl
bounds and physical residual within0.1rad of source qref0. Source object/wrist,
original velocities, nominal inertia, gravity and contact materials are fixed.

Score actual BODY-COM contact force and moment against the saved first nominal
dynamic target, normalized by0.03N and0.0015Nm. Include mean robot acceleration
mismatch with the original-clock four-point joint stencil (50rad/s2 scale,
weight0.05), joint regularization0.01 and unchanged2mm object/0.2mm self
penetration penalties. Penalize object-nonhand contact. This is a declared
proposal score, not an expert label or new acceptance criterion for the task.
Save every candidate's bounded inputs, actual wrench/acceleration, contact
metrics and score. Keep reference/previous best in each population; no restart
or unreported seeds. Fixed total10240 candidate slots plus explicit preflight
and final readbacks; count them separately.

At the selected minimum-score candidate, independently decode actual EFC
contacts, check state/control bounds and forces, full6D frozen static/dynamic
cones, and every63 object-hand/189 self convex pair with native distances.
Proposal force/moment gates use vector norms0.03N/0.0015Nm. Geometric gates
remain2mm object and0.2mm self. Rank or sparse engine contact lists cannot
substitute for these gates. Source derivative/inertia uncertainty remains.

Only if the moving-target force/moment, actual full dynamic cone, bounds,
capacity, no-aid and complete geometry gates pass may this candidate advance
to actual no-reset two-dt constant-residual execution under the existing
50Hz/original-velocity reference and task gates. Static-cone feasibility is
reported separately, not silently substituted for dynamic support. A failed
budget is a local search failure, never a global feasibility proof. All failed
states remain diagnostic-only. Full Goal and future neural-checkpoint pause
remain unchanged.
