# Preregistered geometry/support feasibility pilot v1

Freeze this file/config/source receipt BEFORE any pilot optimization. Full task
remains369 reference frames,50Hz,2.5ms physics/8 native samples per control,
fixed wrist, original compound30g pen, zero body aids and strict evaluation.

Both arms share the SAME calibrated native4 source, all initial actor/critic/
normalizer tensors and optimizer state. Calibration uses initialized raw actions
for8 controls, once, followed by the original full pre-start reset. The actor's
mean-head weights start zero and bias equals initialized raw; Gaussian std=.05,
clamped[.02,.2]. This is explicit common NETWORK initialization, not a different
latent/action transform: sampled/logged actions remain total raw controls under
the existing clamp/scale/EMA/PD contract. No old trained weights are imported.

Both arms get2 updates x64 controls x32 worlds=4096 transitions/32768 native
integrations and64 Adam steps. Shared calibration costs2048 integrations.
Count qualification, failures, evaluation and reviews separately. A single seed
7411 and one trajectory are a feasibility pilot; online trajectories/noise can
diverge after reset/policy changes. Do not claim identical online datasets or
statistically established support benefit.

Every control records8 native maxima. Geometry failure is object depth>2mm,
robot self depth>.2mm, joint excursion>.001rad, nonhand load>.1N or invalid
active/capacity status. It terminates at the20ms CONTROL BOUNDARY, not mid-
integration. Keep the old native50mm/shaft60deg termination too. Original strict
COM5mm RMS/10mm max and fullSO3 5deg RMS/10deg max remain EVALUATION gates.

Geometry loss is the sum of log1p(squared maximum/limit) for the four geometric/
nonhand quantities. Reward is original tracking reward minus .25*geometry loss
minus2*geometry-failure indicator. This reward transformation never clamps or
repairs physical penetration. SUPPORT alone adds .2*exp(-.5*support loss), gated
off on geometry/native failure and invalid reference intervals. RESIDUAL weight0
still computes all support fields, sharing that compute path.

Support loss is mean squared force error/.3N plus mean squared torque error/
.003Nm. Nominal mean WORLD COM wrench is built by the existing reference
momentum helper on the ORIGINAL clock, using actual native mass/inertia/gravity.
USED phase k maps interval[k-1,k], so index=k-1. Last3 intervals are invalid and
receive no support reward, not extrapolated/retimed targets. Reference geometry
is composed in FP64 at the canonical origin to avoid world-origin derivative
roundoff. Compare measured hand wrench about MOVING actual COM in SOURCE BODY
axes(window slot5) to the nominal moving-COM vector rotated by actual source R.
Do not substitute fixed-source-COM slot7 or average changing BODY axes silently.
Targets are a simulator prior, not measured ego forces or applied object force.

The training3 container reuses maintained training2 collector/update/validators
through an explicit internal structure adapter, after enforcing its new schema,
objective/arm and immutable target contract. Old schemas must reject public
cross-loading. The shared initial fork changes ONLY the reward-weight scalar
and top-level arm metadata, with the parent pin and patch recorded. No physics,
reference, network/normalizer/data arrays are edited in this fork.

Evaluate each saved pilot actor mean from a FRESH W1 source with explicit
weights/normalizers-only loading; no W32 physical state transfer, no post-start
reset, up to full7.36s or native/geometry termination. Save all failures. A short
legal prefix or better training reward is not full task/CHORD/ego success.
