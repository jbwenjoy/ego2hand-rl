STATE-VELOCITY RESERVE CONTROLLER: DESIGN ONLY

No filter function, simulator, GPU, policy evaluation or training has been run.
The scalar Python file is an unvalidated reference for review, not a production
controller. Candidate parameters are null; no launch configuration is enabled.

Condition C1_STATE_VELOCITY_RESERVE is separate from frozen A1L/A2/P_ONLY.
The first comparison, if approved, is native A1L versus native A1L+C1. Keep all
original rewards, WM loss/error_floor, self-collision supervision, reference,
physics, actuator limits and1mrad actual-q/target acceptance unchanged. Do not
combine a contact prior, state barrier, new loss weights or anti-windup into C1.

At each20ms control tick:
 v+ = relu(qd); v- = relu(-qd)
 d+ = dt*v+ + v+^2/(2*a_emp); d- analogously
 command domain = original hard-joint / enabled actuator-control intersection
 inner domain = command domain intersect [hard_lower+m0,hard_upper-m0]
 upper risk if q+d+ >= hard_upper-m0; lower risk is symmetric
 if upper risk, shrink target upper to min(current_upper,q-m0-dt*v+)
 if lower risk, raise target lower to max(current_lower,q+m0+dt*v-)
 project proposed target into the resulting nonempty interval

If that interval is empty, set an explicit infeasible flag and use a bounded
inward PD target for the larger-risk side, with the original force limit. Log
which reserved constraints remain violated. Never clamp/write/reset q, qd,
object state or an imagined state. Never turn infeasibility into success.

m0 is a new controller engineering reserve, NOT a change to the1mrad scoring
margin. a_emp is a calibrated empirical braking scale, NOT a guaranteed lower
bound under contact or joint coupling. The rule is not a CBF, safety guarantee
or proof of next-q feasibility. Native force saturation is retained; no new
torque/rate feasibility guarantee is claimed.

PARITY AND SAMPLE/HOLD
Real uses current true q/qd at process_actions, once per20ms. Cache u_exec.
apply_actions runs at four5ms substeps and ONLY writes that cached command;
it must not recompute the filter from newly moved q/qd. Reset/init needs one
explicit filter/caching step without changing the frozen physical initial state.
Imagined control_target must receive x_current explicitly and use its q/qd.
One common FP32 tensor kernel must implement both paths. Same-input parity does
not imply real and WM-predicted states/trajectories are equal or accurate.

The first candidate is stateless: no rate limiter, previous-u dependency,
velocity estimator state or anti-windup. c1 remains EMA residual. Existing
executed/frozen target caches contain u_exec. Real actuator target, replay u,
WM input and PD torque estimate must all use u_exec. Existing executed_target
reads an action cache, so independently verify it equals actual simulator ctrl.
Any future rate filter with prev_u is another condition requiring a complete
memory/replay/reset/imagination/observation contract.

TIMING LOGS
u_tick[k] controls [k,k+1]. Existing rollout target[k+1] records that interval's
command. Log q/qd before tick, proposed/bounded/filtered/backend-readback u,
control_tick_id held across four substeps, actual q/qd/torque at every5ms,
infeasible flags, first/peak overshoot and reset/assist counts. Do not align a
previous interval target with the next proposed action by mistake.

EVIDENCE AND FUTURE TESTS
The already viewed nine joint-limit traces and present n3 runs are seen data.
Parameter choices informed by them are post-hoc engineering. They do not count
as unseen qualification. Freeze independent calibration/heldout case IDs and
parameters before confirmation; use fresh training seeds4/5/6 for a paired
A1L-vs-C1 confirmation. These are new stochastic seeds on the same known clip,
not new-motion/video generalization. Keep every physical result, not best runs.

See proposal.json and test_plan.json for bounded stages. No stage is authorized
by these design files; root review and a separate action are required.
