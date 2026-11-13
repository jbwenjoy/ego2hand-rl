# Research plan v0.57: fixed120ms native horizon diagnostic

2026-11-14. Previous turn was progress: 6711cd0 records actual eight-case native
feedback, saved review8/8, public observation contract and verified video. No
case qualified at2s;140 tests pass. The Goal stays active, full scope unchanged.

## Hypothesis and fixed comparison

The60ms constant-action planner delayed REFERENCE abort from.519063/.527188s
to.868750/.901563s, but did not solve it. This result does not prove physical
infeasibility or that native planning cannot succeed. Test a120ms lookahead
under the same initialization, numerical plant, goal clock, action box/mask,
EMA, residual formula/scales, ridge, line search and one-GN-iteration controller.

`configs/mug_native_horizon6.toml` inherits the pinned old configuration. It
overrides horizon3->6 and restricts this diagnostic to REFERENCE/NATIVE_MPC at
the two original dt. HOLD and ZERO are not rerun or relabeled. Original60ms
case traces/metrics and their successful saved-data review are the comparison.
No reset change, material/gravity change, new fitted data or NN is involved.

The planner sums state residuals over its horizon, so adding future states
changes the duration of the objective relative to once-per-action regularizers.
The exact residual formula and coefficients remain; do not call this an isolated
decoder change or a matched-time comparison.30 native branches per decision
and up to twice integration per branch versus H3, fully charged. Budget4M
oracle integration steps total, execution/restore/parity costs separately shown.
No GPU speedup, learned-policy or shared-model amortization claim.

## Execution and qualification

The existing immutable NativeTrackingPlanner and NativeBranchOracle perform all
control math. A reusable case driver stores actual per-tick native execution,
complete source/endpoint states, controls/EMA, every proposal/residual/merit,
decision timing, restore checks k0/5 and forecast-versus-executed-state checks.
Each executed complete first-H state must equal the selected forecast exactly.
Partial terminal rows remain explicit. The original summarize function and
all original physical/2s tracking/static-hold gates are used without modification.
No source pose is copied into the simulator during execution.

Goal-conditioned inputs include current complete native state and actual EMA,
previous action and known future reference. This is a privileged CPU oracle
baseline; its output is not an Actor using the separate public144 contract.
Initial45deg/50mm aborts remain; original reference range is within that envelope.
Only1/2/remaining horizon truncation at the finite2s endpoint is allowed, with
actual source-clock goals and no extrapolation or time stretch.

Boundary tests cover six-step residual dimensions, legal control bounds and
end-of-task truncation; full141 tests pass before experiments. The initial
native oracle18-case byte parity remains a prerequisite, with checks repeated
on each selected complete first control interval. Preserve failures and all
metrics; do not tune the protocol against the resulting CHECK cases.

## Next actions retained

Inspect actual qualification, contact/actuator/velocity failures, control-box
usage and local predicted merit reduction. If longer horizon alone is inadequate,
separate control authority from constant-action/local-search limits before
claiming infeasibility. Future action-sequence or real-native policy experiments
must declare their own budget, independent data roles and public observation
contract. Failed diagnostic trajectories are not successful expert supervision.

Original FULL objectives remain: controlled fixed-wrist cube/pen loading/
playback, short learning/save-resume/no-aid evaluation; object-frame support,
matched A1/A2 and contact/action gradients; ego collision/inertia/material/
gravity/frame/input/controller contracts; independent RGB/full trajectories;
multi-trajectory TOTAL shared-vs-per-trajectory cost; hardware subversion/joint/
zero/gain/latency/observations before deployment. Same nominal mug is not the
full task. No driver/CUDA changes or new workspace study roots. No NN training
is launched here; keep Goal active rather than inventing a checkpoint pause.

Evidence: original `runs/mug_native_tracking_v1/report.json` and
`runs/mug_native_tracking_review_v2/report.json`; new results belong to
`runs/mug_native_horizon6_v1/report.json`, if actually generated.
