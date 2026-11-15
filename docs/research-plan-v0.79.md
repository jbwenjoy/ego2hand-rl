# Inspect pair-level contact generation at the recorded pose split

Previous turn is PROGRESS: seven saved anchors x two paths x three repeats,
6960 integrations, independent294/294 contracts and147/147 EFC checks, with
all prediction failures retained. All57 repeated candidates pass770/798; the
180ms selected candidate changes2/2/3.206 tests pass. A committed collision-
only probe adds20 queries andZERO integrations: fixed pose is stable, but the
two recorded near-identical poses split between contact present/absent.
Full Goal remains ACTIVE/incomplete, with no policy update or task success.

1. Keep exact fixed model, geometry, numerical policy, original dt and the
   four recorded inputs (two distinct qpos images). Begin with pair1/29:
   pen and right_finger2_link4 collision mesh. Source model404 arrays/195
   scalars must match; Data static reconstruction remains explicitly limited.
   Do not turn CPU negative distance into an assumed ground-truth contact label.
2. Inspect frozen installed primary collision/CCD source and expose pair-level
   broadphase candidate identity/order plus narrowphase termination/result/
   distance/support-simplex information in an isolated diagnostic wrapper.
   Do not edit installed libraries, drivers/CUDA, original datasets or model
   tables. Preserve source bytes/version and fresh public/diagnostic workspace
   metadata. Existing public Data snapshots do not cover all private contexts.
3. Query all recorded inputs with repeated exact reset, geometry poses and
   capacities saved, with NO force solving/integration. Distinguish pair culling,
   narrowphase termination, contact emission/filtering and row-order artifacts.
   Existing20 queries have13 broadphase candidates<192 capacity, but identical
   counts do not prove the pair identity is present. Keep all outcomes.
4. Compare explicitly matched geometry tables/poses to an independent geometric
   comparator, including mesh/hull representation, margins, precision, filters
   and transforms. Any small SO(3)/ULP perturbation grid or numerical-option
   ablation must be predeclared and called sensitivity diagnostics, not human
   measurement, hardware noise, confidence intervals or a new dynamics model.
5. Choose a numerical-policy or versioned collision-geometry intervention only
   after matched evidence locates the discrepancy. Physical geometry changes
   need explicit deviation/coverage checks and a separate version/cohort; no
   silent proxy replacement just to make a task pass. A local detector fix
   requires repeated/contact-domain verification and full original no-aid
   task evaluation before any expert or gradient qualification.
6. Once relevant20ms fidelity is understood, consider correct multi-control
   reference/EMA/PD horizon advancement with fair budget accounting. Full
   orientation/axial roll, learned SuperMimic/CHORD/save-resume, cube/pen,
   multi-trajectory total costs, true ego assets/input/dynamics and hardware
   contracts remain required. Do not substitute these probes for the full goal.

Original failed episodes/import enumeration stay unchanged. Reconstruction-
only capacities are not original observations; new fresh episodes record
observed scalar/layout metadata. No training/checkpoint is waiting, no new
pause request exists, and Goal stays ACTIVE.
