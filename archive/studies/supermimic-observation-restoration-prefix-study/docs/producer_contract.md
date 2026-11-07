# Restoration → feasible-fit producer contract

This is a separately frozen method using protocol `3c4113a6…b387c` and variant `H123_cubic_minimax_restoration_filtered_prefix16_v1`. It reuses the old geometry problem `e0edf889…e8808`, basis `145117f0…773dfd`, exact 5,960×340 saved linear contract, unchanged 6,596 fit residuals and terminal seed `64aa2cfe…ad9b43`. The previous failed experiment and its outputs remain unchanged. No old upstream robot-q tail is read or appended.

The entry is `scripts/build_restoration_prefix_cpu.py --plan JSON --output NEWDIR --check`; check is also the default. `--run` requires a separately authorized Root plan. Plan schema is `restoration_prefix_cpu_plan:1`, with:

- `protocol`: the new 3c411 protocol; `target_protocol`: 0798; `parent_protocol`: 94ea.
- Full unchanged `inputs`, `seed_candidate`, `linear_contract`, `linear_contract_metadata` and flattened executed `source_pins`.
- `retarget_variant_id`, `output`, `optimization_authorized`, `automatic_retry=false`.
- `R_maxiter=60`, `R_max_objective_calls=200`, `R_ftol=1e-9`; `F_max_major=8`; `QP_maxiter=25`, `QP_max_objective_calls=100`, `QP_ftol=1e-9`; whole-worker soft340/owned360 seconds.

Root's new protocol uses `parent_protocol` for 0798, `input_parent_protocol` for 94ea, and `linear_metadata` for the metadata artifact. The plan maps these to `target_protocol`, `parent_protocol`, and `linear_contract_metadata`, respectively. The implementation rejects a different protocol hash, source, seed, matrix, variant or budget. All five CPU thread counts are one and CUDA visibility is `-1`. No new LP is called.

## Mathematical implementation

`src/restoration_prefix/math.py` contains pure algebra and decision gates. `problem.py` wraps the frozen geometry implementation without editing it. The R variable is `[x340, slack_mm]`, with objective **slack only**. Its seed is exactly the old normally returned terminal coefficients; slack is `max(0,max(depth_mm)-2.999)+1e-6`. The restoration box is the original coefficient box intersected with seed±.05 rad. R has one SLSQP attempt; fit cost is recorded but never added to its objective.

Only a normally returned R terminal vector is tested for the next phase. All 61 constraint sample caches are cleared and 61 fresh native collision calls are required. The original 122 generated-overlap inequalities, original linears, coefficient box and R trust box decide R→F. R solver success and artificial slack are separately recorded and cannot authorize F.

Arithmetic tolerances are those fixed in the new protocol: depth 1e−12 m / 1e−9 scaled mm, each linear row 1e−12 in its original unit, coefficient/trust bounds 1e−12 rad. Exact raw booleans and numerical residuals remain visible beside these tests. They do not overwrite the previous 0798 result or represent collision accuracy. The original qualification depth remains 3 mm, with 2.999 mm planning depth.

At most eight F majors each create one fixed linearized QP. The objective is `.5||r+J*p||² + .5*1e-4||p||²`; its gradient uses `g=J.T*r`, `H=J.T*J+1e-4I`. It retains original coefficient bounds and linears, adds the linearized 122 geometry inequalities and a .02 rad infinity step bound, and starts from zero. The R trust box is **not** carried into F. One QP SLSQP has maxiter25, ftol1e−9 and at most100 objective calls.

Only the normally returned terminal QP direction is considered. It must independently satisfy its fixed linearization and bounds, be finite, have max|p|>1e−12 rad, change x and satisfy `g.T*p<0`. Status alone does not qualify it. Six alphas are tried in the fixed order `[1,.5,.25,.125,.0625,.03125]`. The first finite changed actual geometry state passing original arithmetic feasibility, **strict actual original-cost decrease**, and Armijo with c1=1e−4 is accepted. There is no history-based choice, extra alpha or alternate QP after failure. Every inner QP log is explicitly prediction-only; actual geometry validation is logged separately.

## Evidence and selection

Each actual evaluated state has an immutable archive containing complete NPZ diagnostics, native contact JSON and a SHA-linked record. It stores original cost, max/summed depth violation, exact/numerical primal tests, selected pairs/contacts, ties, branch and CSR Jacobian signatures, plus actual geometry creation deltas. Missing native gaps remain NaN/masked; a missing category has no generated violation and does not acquire positive-clearance meaning. Native Jacobians remain local, approximate and branch-sensitive.

`trace_events` and `trace.jsonl` distinguish R trial/major, F base/actual alpha/acceptance, and final fresh validation. QP inner calls are in separate prediction logs. Every R objective or nonlinear-constraint Jacobian and major callback commits a partial checkpoint. Each F accepted state commits `last_accepted.npz/json`, including sequence, source, PID, actual plan SHA, metrics and independent feasibility. NPZ copies and headers use the reviewed same-payload bounded IO. One shared `g` total is reported without adding its parent counter again.

Normal algorithm termination exports the **last algorithmically accepted** state, using the raw feasible R terminal at F0 if no fit step was accepted. It is labelled `F_NO_ACCEPTED_STEP` in that case. Before output, all 61 native constraint samples are refreshed again. A changed numerical contact branch can fail final qualification; the same coefficient vector remains the canonical algorithm result and no older state replaces it. Timeouts, R objective-budget exhaustion or engineering interruption preserve partial evidence only; they do not commit a pipeline canonical. A controlled QP100 budget is a normal F stop retaining the current accepted state.

The canonical requires normal worker completion and the committed `manifest.json`. Interrupted files without that completion are partial. A physical or full-motion qualification is never inherited from a normal optimizer return, an arithmetic-tolerance pass, a zero-rest result or an old source identity.

## Final ABI

Manifest schema is `restoration_prefix_geometry_candidate:1`, with the new full variant, original raw lineage `source_identity`, `protocol`, `target_protocol`, `parent_protocol`, **`source_plan`** (actual absolute Root plan path and SHA), candidate/report/source pins and independent selection provenance.

`canonical_candidate` records `source=normally_terminated_pipeline_final_x`, `normal_termination`, `selected_phase`, `last_accepted_state`, `accepted_seq`, `F_accepted_steps`, `F_stop`, `pipeline_return` pin and `final_x_sha256`. R and QP optimizer outputs are separate artifacts; neither is a pipeline final result.

`pipeline_return.npz` stores `final_x[340]`, `last_accepted_x[340]`, `R_terminal_x[340]`, `F_accepted_steps`, `accepted_seq`, `F_stop`, `normal_termination`, `source_plan_sha256`, `protocol_sha256` and `retarget_variant_id`. Final and last accepted x must be bitwise identical.

`candidate.npz` keeps the existing geometry ABI: `time_s[16]`, `q_goal_rad[16,20]`, `mapped_object_pose_w[16,7]`, `control_points_rad[18,20]`, `optimizer_x[340]`, basis16/basis61, planned61 q/object pose and complete residual/contact/Jacobian diagnostics. It adds new variant/plan/source/PID metadata and the fixed restoration seed/trust bounds. The retained trust bounds describe R and do not become F constraints.

Independent reference/init/control export remains pending after a real normal result. Future analytic qdot, native forward FD, requested initial state, actual measured state and executed control stay separate; no GPU or training is authorized by this implementation. A 0.3 s prefix does not complete the full motion, SuperMimic+CHORD, shared learning or true ego-data goal.

## Checks completed before any solve

The saved-data/plan/AST/IO self-checker passed **34/34**, without importing MuJoCo, Torch or Warp, and with zero geometry or solver calls. The independent reviewer passed **18/18** synthetic checks of the actual algebra helpers, including strict descent, unchanged x, floating Armijo equality, original-primal versus slack, and R/F trust separation. Compilation passed. These are implementation checks, not a native derivative or feasible-reference certificate. Root retains the single default native check and any later explicit formal-run authority.
