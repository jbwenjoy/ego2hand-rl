# Saved matched-cone numeric audit

All 38 checks passed by reading saved arrays with NumPy only. No LP, native model, geometry query, WM, environment or GPU was run. The audit independently rebuilt the six normal-force-normalized rays per declared contact, the lever and direct torsional moments about COM, and the fixed analytic inertial demand. The H123 matrix is the exact first 18 columns of H12345 in every frame.

All 302 saved witnesses are finite and nonnegative. Maximum scaled primal residual is 1.7764e-15 N-equivalent for H123 and 5.5512e-16 for H12345, below the frozen 1e-7 N gate. Original torque residuals are at most 2.8207e-17 and 1.3880e-17 Nm. The all151 human contact masks remain unknown; no target or mask was changed.

The solver-reported minimum total normal force ranges from 0.2961 to 2.0814 N for H123 (median 0.5459 N), and from 0.1601 to 1.7095 N for H12345 (median 0.3824 N). Five minus three is at most 2.22e-16 N: 142 frames are strictly lower at the predeclared 1e-7 N comparison allowance and nine are equal within it. Frame0 totals are 1.02937 and 1.02625 N. Full per-frame values are preserved in `normal_force_by_frame.csv`.

These are idealized cone-model force lower bounds, not measured hand forces or upper capacities. Primal feasibility and nesting were independently checked; solver optimality was not independently dual-certified. The extra declared contacts cannot mathematically reduce the feasible set, but the hypothetical contacts may be unreachable, absent, or inconsistent with actuator balance, material-frame orientation and dynamic contact laws.

Both matched H123 and H12345 are feasible at all151 frames. Earlier H123 achieved only23/151 with a different mu0.1, nine unit-force rays per contact and no direct torsion. That comparison changes material and cone representation, so the feasibility change cannot be attributed to adding fingers. The matched difference here concerns the conditional normal-force objective only. None of these results qualifies a real grasp or the failed geometric trajectory physically.

The separate lifecycle/source audit is `outputs/full151_matched_material_cones_lifecycle_saved_independent_v1/report.json` (SHA edd3032d7c494968a61455ea78d39484f07a7b6365fb7c0eb2b1c3accbba7aef); its source/OS checks are separate from this numerical audit.
