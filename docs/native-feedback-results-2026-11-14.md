# Actual native reference feedback: fixed protocol results

The eight-case protocol in research-plan-v0.56 has completed. All cases fail
the predeclared full2s qualification; no trained policy or ego-retargeting
success follows. Frozen ZERO traces remain byte-identical to the original
native baseline. This is actual no-object-aid execution, not a WM video.

|Task|Control|Coarse dt duration|Fine dt duration|Qualified|
|---|---|---:|---:|---|
|HOLD|ZERO|.5190625s|.5271875s|0/2|
|HOLD|NATIVE_MPC|1.1875000s|1.1956250s|0/2|
|REFERENCE|ZERO|.5190625s|.5271875s|0/2|
|REFERENCE|NATIVE_MPC|.8687500s|.9015625s|0/2|

All terminations are orientation drift beyond the original45deg abort. The
longer survival is measurable progress, not complete holding/tracking. Fine-dt
HOLD MPC also reaches.302374mm native self penetration, above.2mm gate. Tail
force coverage is unavailable because no case reaches the tail window; it is
not counted as passing. Reference cases use the actual anchored object goals,
not a static substitute or time-stretched reference.

## Reproducibility and independent saved-data review

The observed plant's complete next-H state equals its selected forecast in
206/206 complete intervals. Partial terminal intervals remain explicit. All8
planner decision restores at k0/5 reproduce actions/candidate costs exactly.
Independent saved-data review8/8 passes: legal control/EMA reconstruction,
no applied object aid in stored states, raw forces/couples/moving-COM wrench,
pose/velocity, task metrics, historical ZERO parity, proposal objective/merit
and total native budget.164342 raw contact rows were reviewed. Normal/other/
table force and penetration reconstruction errors are0; mean wrench maximum
3.33e-15, COM velocity1.20e-14, metrics3.89e-15. Warning counters are retained
online evidence, not independently reconstructed from compact integration state.

Reviewer v1 had two not-applicable per-case fields initialized toTrue; v2 uses
null for ZERO forecast checks and MPC historical-ZERO checks, with appropriate
cohort filters for aggregate checks. No simulation result or metric changed.
Original v1 source/report are snapshotted; v2 is the authoritative review.

## Cost and optimization limits

The full eight-case run took245.41967s, including1,889,280 oracle steps and30,072
executed steps. Original parity checks separately charged5184+5184 steps.
Median decision costs were.72939/1.38552s for HOLD and.71130/1.37832s for
REFERENCE, coarse/fine. They are roughly36–69 times the20ms control period;
the baseline is offline privileged native planning, not real-time deployment.
No GPU/NN training was launched in this phase. Aggregate wall time excludes
earlier neural fitting, parity, independent review/rendering and code work.

About18–25% of optimized action entries are at the residual box boundary across
these MPC runs; late-window fractions are22–33%. Median one-step local proposal
cost improvements over zero/previous incumbents are4.5–8.7%. These diagnostics
do not prove global infeasibility, insufficient hardware authority, or optimality
of this one-iteration60ms/constant-action search. It is premature to distill its
failed rollouts as successful demonstrations or compare compute savings at
unequal task performance.

## Public Actor observation contract and label ambiguity

The new144D packet is current state55, physical-radian EMA20, previous raw
action20, normalized phase1, three relative future object goals15 at offsets
1/3/5 control intervals, and validity3. Future goals are expressed in the
current object BODY axes: relative COM3, rotation columns6, relative COM
velocity3/angular velocity3. Near the finite horizon, endpoint padding is
marked invalid, never extrapolated. Object truth is available in this simulator;
hardware observation/velocity estimates are not established.

It excludes solver warmstart26, native integration dt, future actual states
and future contact forces. Public Markov sufficiency is not assumed.316 actual
pre-action packets are saved as DIAGNOSTIC_ONLY_NOT_EXPERT_NOT_FIT. No Actor is
trained on them. Public-equal initial packets from the two dt can have distinct
native planner actions: max raw-action difference.248081 for HOLD and.636799
for REFERENCE within the SAME controller. This can reflect numerical state,
local optimizer sensitivity/nonuniqueness or integration dt; it is NOT proof
that public observations cannot support physical control. ZERO and MPC labels
are intentionally different policies and are kept separate in alias diagnosis.

Public report v1 mixed controller contexts and measured differences from the
first action only; v2 reports same-context pairwise differences and separately
marks mixed-controller groups. Its packet NPZ bytes are identical to v1. Both
old source/report snapshots are preserved with explicit correction provenance.
All140 repository tests pass after the contract and context-grouping tests.

## Artifacts and next research action

- `runs/mug_native_tracking_v1/report.json`: full executed protocol and traces.
- `runs/mug_native_tracking_review_v2/report.json`: authoritative saved review.
- `runs/mug_public_control_v2/report.json`: public packet and alias contract.
- `runs/mug_native_tracking_video_v1/reference_zero_vs_native_mpc.mp4`: actual
  coarse-dt REFERENCE comparison,4x slow, terminal freezes explicitly labeled.

Next separate limited control/search horizon from deficient state/geometry/
training coverage. Predeclare a native horizon/action-sequence comparison with
all computational costs, retain independent qualification and dt checks, and
collect fresh explicitly separated TRAIN/CHECK only when learning is justified.
Do not simply fit the failed diagnostic packets or reuse old validation probes.
The public Actor interface can be used for that later reference-conditioned
study, while privileged numerical state remains an oracle/WM distinction.

The full original controlled cube/pen, no-aid save/resume/evaluation, object-frame
support, matched A1/A2/contact-action gradients, ego geometry/dynamics/frame/
observation contracts, independent RGB/full and multiple trajectories/total
costs, and hardware version/joints/zero/gains/latency/observations remain. No
driver/CUDA changes or new workspace study roots. Goal remains active; no
neural checkpoint was started, so the standing first-healthy-checkpoint pause
boundary was not triggered in this phase.
