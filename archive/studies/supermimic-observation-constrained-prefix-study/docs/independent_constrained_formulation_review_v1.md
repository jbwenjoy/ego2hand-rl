# Independent constrained-prefix formulation review, before any solve

The existing failed candidate is retained unchanged: prefix16 candidate SHA `1e67d99b25b6d991fa8e5209f521ea5c269b3212a45b054fe7353ab5b7277ae4`; curve/control receipt SHA `f5e45ac4b42cac400b6a723473087aac0da465d34457da48c03b86edbb639489`. The current main STUDY_STATE was read. The old candidate has legal sixteen q knots but a15.626615mrad quintic lower excursion,28.484651rad/s planned post-C2 target rate, track/hold reference-state raw-PD cap violations, and5.141163mm generated hand-object overlap. Its ftol stop is not gtol/global convergence. None of those failures proves the observation motion impossible.

This is a new restricted planner, not a retry with changed weights or a physical experiment. Object motion, source window, fixed-wrist redesign and uncertain contact hypotheses must remain explicit. Success on0.3s does not qualify full151-frame/3s tracking, force prediction, shared learning or real ego generalization.

## Basis and fixed boundary

A clamped cubic B-spline with eighteen control points has340 free scalar variables when P0 is hard-fixed to the canonical q0. With uniform interior knots at.02,...,.28s, it spans the same fifteen20ms intervals. Use physical-time derivatives: differentiating a normalized interval coordinate requires1/h and1/h² factors. Control-point hard bounds imply all-curve q bounds through nonnegative partition of unity. This is sufficient, not necessary: some admissible curves have exterior control points. The fixed source64 q0 lies2.22e-16 above the lower bound at joint3; do not clamp or move it. Bound P1 implies an inward/nonnegative initial derivative at that joint.

Node residual Jacobians chain through the basis: frame-local J_q times each free B_p(t), with control-point-major20-joint flattening and no P0 columns. Temporal residuals stay the declared raw q differences, not undeclared derivative rescaling. Independently test partition of unity, derivative sums, endpoint q0, a local basis/CSR directional FD and exact variable ordering before fitting.

## Linear reference/control certificate

The known actual gain/bias model gives `tau = kp*u + b0 + b1*q + b2*qd` for the explicitly checked unit-gear hinge actuators. Transporting the same fixed static prior and compensating reference damping makes each proposed u_k affine in spline control points. Constrain u before C2 to the actual hard∩ctrl interior, then separately re-evaluate the FP32 proposal, C2 result, actual planned14 updates and raw-PD predictions. A float64 linear certificate cannot silently justify a different rounded/clamped command.

Within each20ms interval, u_k is held fixed while q is cubic and qd quadratic. After degree elevation, raw-PD prediction is cubic. Its Bernstein coefficients are affine in the control points. Bounding every coefficient by both actual actuator and joint force caps is a whole-interval sufficient certificate for this **reference-state algebra**, not a necessary condition for a feasible motion. Preserve physical-seconds derivative factors and the ZOH command index; use the same newly designed u0 for hold and track.

For hold, repeat u0 through all fifteen intervals, but evaluate its certificate against the same moving desired q/qd. This is a deliberately conservative, counterfactual planner restriction. The actual hold trajectory will generally differ from that reference; violating this prediction is not proof that a real hold saturates, and passing it is not a closed-loop dynamics or safety certificate.

Root's declared numeric design margins are2e-6rad for proposed controller interior and2e-6Nm for the force coefficient bounds. The1e-12rad curve-certificate tolerance concerns arithmetic/coefficient error only. Report all actual excesses. These margins are not established robust uncertainty bounds for collision accuracy, FMA, model error, contact dynamics or hardware. Do not change them after seeing a failed solve.

## Sampled generated-contact constraints

At the same61 planned5ms instants, generated hand-object and hand-self depth is constrained to≤2.999mm. The1µm difference from the unchanged3mm qualification gate is only a numeric planning buffer, not a collision-accuracy guarantee. This does not prove continuous-time clearance between samples or constrain future simulated actual states.

Ungenerated contacts have unknown gap. A fixed positive-slack/zero-Jac constraint for an absent category is bookkeeping, not a positive gap oracle, observed no-contact label, valid support normal or force certificate. Keep constraint dimensions fixed for SLSQP. If using the deepest generated contact per category, expose both ties within a pair and ties across pairs; a deterministic selected branch is not global differentiability. The old collision Jacobian's feature/tie/numerical qualification limits remain. A small basis/CSR FD pass must not be renamed as full contact-gradient qualification.

Frozen hand-material/object-local anchors across all times encode a prospective sticking-like correspondence, not an observed contact-mode sequence. Rolling, sliding or changing material contacts may be physically valid but outside this hypothesis. The new constrained parameterization does not repair an incorrect correspondence by itself; source observation/scale/handedness/confidence remain unqualified.

## Two-phase and result semantics

One L1 LP may find the closest-to-repeated-q0 point satisfying the declared linear restricted-domain certificates. Its objective, matrix/order, source pins, solver status, exact returned point, residuals and cost count must be retained. LP infeasibility only rejects this sufficient-restriction class, not all splines, all controllers, dynamics or the observed motion.

One subsequent SLSQP call may optimize the original sixteen-frame fit with the fixed sampled geometry constraints, using the LP point only as initialization. Starting linear-feasible does not imply contact-feasible. Keep80-iteration/250-evaluation/220s soft/240s owned caps and count actual objective, constraint, Jacobian and geometry requests. Preserve failed/interrupted/infeasible iterates and source-PID identities. No projection, new initial guess, parameter sweep, smoothing substitute or retry may be used to reinterpret failure as success.

Only independent final residual/linear/basis/FP32 schedule/61-geometry checks can qualify the planned CPU candidate within this scope. Solver success alone is not full KKT/global feasibility, measured force, physical initialization or training readiness. GPU remains separately gated and unauthorized during this review. No LP, optimizer, collision or physics execution was performed to write this formulation review; source-specific checks are pending actual implementation.
