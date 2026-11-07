# Fixed5000 known-state response evaluation

`evaluate_fixed5000.py --protocol PATH` defaults to JSON/hash metadata only:
no Torch import, NPZ decoding, NN construction or inference. The supplied
`protocol_cpu_false_v1.json` passed this preflight and binds the completed fixed
5000 checkpoint `344bac45…dc9a7`, its healthy15/15 receipt and terminal exit0.
Root alone can freeze a CPUTrue/fresh-output protocol and add `--run`.

Actual budget is **one own CPU checkpoint load, one frozen NN constructor,
two forwards** (CHECK1472 + VAL114 =1586 rows), two structured decodes,0 AD/
optimizer/training/native/Env/GPU. Current geometry is computed with the frozen
providers and cached by identical input-state bytes. No geometry or normalization
is fitted here. The constructor is inside a scoped RNG fork; weights,
normalizers and raw Python/NumPy/Torch RNG must remain unchanged.

The main CHECK archive retains all1476 original plan records:36 baseline
references,1436 available side predictions and4 unavailable unknown rows.
All1440 side records and720 plus/minus pair records remain present with explicit
defined masks. No unavailable target/prediction is replaced or extrapolated.
Six source families, both native h values, all20 axes and both signs remain in
the output. Original source states were already TRAIN; only the new action
responses were held out. This is not an unseen-state, ego or blind trajectory
confirmation. TEST and TRAIN/FIT NPZs remain hash-only; any separately collected
fresh-family evaluation is a different artifact and budget.

`load_evaluation_data` mechanically joins branch/context/family/slot IDs and the
saved CHECK-only pair-label archive. `execute` imports the frozen224a trainer
factory for the exact architecture and366 feature preprocessing; it **never**
calls `prepare_data`, fits a normalizer or opens the FIT targets. All scales and
physical context come from the checkpoint. NN parameters are frozen and all
inference runs under `inference_mode`.

Endpoint outputs include decoded COM/world state58, physical head178, per-pair
J22x6, meanFn22 and coverage probabilities, plus raw222 and actual normalized366
F32 inputs. Full SO3 geodesic, raw R9, joint/COM/velocity errors keep their units.
Per-pair force impulses [Ns] and moment impulses [Nms] remain separate. Groups
include all endpoints, all new sides, baseline references, each family, both h
values and both source branches. No model/case/axis winner is selected.

CV and continuous-ballistic baselines explicitly use constant joint velocity
and world angular velocity; ballistic adds the frozen simulation gravity to
COM position/velocity. These are kinematic priors, not native contact solves or
complete torque-free anisotropic rigid-body integration. Their J/Fn/coverage
comparison uses a declared zero-contact prior. CV's prescribed state is not
momentum-balanced under gravity; encoding its pose/joint178 components with
zero impulses does not make it a structured physical solution.

Every available side is compared with its same-state baseline in the common
source chart. The saved head response is `head(side)-head(baseline)`, not a
difference of point errors `D(pred,target)`. Physical delta52 responses also
include the decoded v/omega changes. The primary native/model denominator is
the **same actual F64 selected-control step**, with no epsilon or rescaling.
Separately saved representation diagnostics are:

* F64 difference of the two raw physical-control endpoints after casting each
  to F32; the rounded F32 subtraction is stored separately.
* The actual normalized NN input's F32 coordinate separation, cast to F64 and
  multiplied by checkpoint input std to obtain its represented physical step.
  This is the secondary model secant denominator: training normalized F64
  inputs **before** casting to F32, so a raw-u F32 difference is not identical.

Undefined denominators/unknown endpoints stay masked. Actual normalized inputs
must match in all other365 coordinates. Requested.01 steps and bound-reduced
available steps are described separately, with a fixed floating representation
allowance rather than treating rounding as clipping.

For all20 axes and all22 pairs, native-response strata are exact numerical F64
zero, nonzero F64 with identical F32 endpoints, and F32-resolved endpoints. Raw
native/model response magnitudes and componentwise endpoint-ULP ratios are
preserved, including zero-count strata. Object delta52 gets the corresponding
four unit-separated position/rotation/velocity/omega descriptions. These are
empirical signal/precision descriptions, **not** native noise bounds, new
accuracy/stability gates, complete-H contact-path certificates or true Jacobians.
Zero net impulse alone does not imply zero displacement/orientation response.

VAL114 is a fixed state58 monitor, with its original row IDs and the6 excluded
partial IDs retained. Original VAL also has the two designated padcoverage
labels. It does **not** provide meanFn22 or impulse22 labels in this interface:
those metrics are explicitly null; min/lastPRE force is not substituted for
meanFn. The checkpoint remains fixed5000 regardless of VAL/CHECK results.

The result is a component-level check of the empirical action responses and
the historical spurious-response failure, not an Actor/CHORD/holding benefit
claim. No new fitting, source physics, AD tests, old-WM ablation or task rollout
is silently appended. Saved arrays/statistics are sufficient for subsequent
interpretation without another inference.
