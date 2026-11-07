# Fixed WM response evaluation source

The source compares **warm geometry/momentum WM5000** and **fixed offline
paired1000/sharedWM1100**. It never selects a checkpoint or axis from observed
errors. The real native dataset is328 planned/executed records:8 predeclared
MONITOR contexts, each original action and20 axes×2 signs. Its global8-baseline
raw-bit barrier, terminal and10/10 saved integrity report are pinned. That
review is integrity evidence, not an independent physical-label certificate.
The evaluator has not loaded any checkpoint/NPZ or run NN, AD or native code.

Default command (JSON/hash only):

```powershell
& 'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe' -X utf8 `
  'D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/evaluate_WM_action_response_v1.py' `
  --protocol 'D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/protocol_CPUFalse_v1.json'
```

Final1000 protocol/result/health/terminal remain pending. After those actual
files exist, `prepare_protocol_v1.py --bind --final-protocol ... --final-result
... --final-health ... --final-terminal ... --probe-saved-review ...
--protocol-output <fresh-absolute-CPUFalse-path> --output-name completed_v1`
binds the exact fixed experiment. It retains CPUFalse. Root must separately
authorize a fresh actual protocol before appending `--run`.

The fixed CPU4 budget is2 own checkpoint loads/2 WM constructors, at most20
model forward invocations and16 Jacobian API invocations. Per model: one
available-record endpoint batch,8 B1 baseline action Jacobians, one batched
model-only central FD at±1e-4rad on actual-bound-available axes. Every Jacobian
uses reverse-mode `torch.autograd.functional.jacobian(vectorize=True)`: one
forward and a batched VJP carrying274 cotangent directions. Across16 calls that
is4384 scalar reverse directions equivalent, not16 scalar AD operations or a
GPU/kernel profiler count. Outputs274 are decoded world-tangent52, physical
head178, meanFn22 and marginal coverage22. Parameters are frozen and `.grad`
must stay None; there is no optimizer or native/GPU call.

H1 baseline x is fixed. Thus action u enters only the raw control slots,
normalization, NN and physical decoder; current geometry is action-independent.
This does not test geometry-q derivatives, H2 state→geometry gradients or
native/contact infinitesimal Jacobians. The local left-world SO3 tangent at
each model baseline is compared with the model's geodesic central FD.
Rotation9 is not an angle. Self-AD/FD statistics split the six physical-head
unit blocks (including interleaved Jforce/Jmoment indices), along with state,
meanFn and coverage blocks; raw274 Jacobians remain saved.

Native finite actions have nominal±.01rad, with authoritative actual F64
endpoints and one-sided bound room. They are different interventions from the
model's±.0001rad self-check. Native endpoint response is compared to model
response at the **same actual endpoints**. All320 sides and160 central pairs
are retained, with actual denominators, unknown/unavailable/partial masks and
raw outputs. No true noisy-force floor or model-accuracy gate is invented.
The data's Actor+EMA reachability is a descriptor:290 native sides are within
that current mapping and30 are not; neither group is silently discarded or
called a measured policy gradient.

Input arithmetic is physicalF64→normalizedF64→NN FP32. The saved response
arrays distinguish actual F64 du, cast-raw-control FP32 endpoint separation
and the **actual normalized NN FP32 endpoint separation multiplied by its
frozen input std**. Encoding collapse/undefined masks remain separate; a
zero model response after collapse does not imply zero physical response.

State58, physical178, source-BODY/moving-COM impulse22, meanFn22 and marginal
coverage22 are compared by fixed cohorts/all20 axes, with per-pair zero/F32
collapse/resolved native signal arrays. Joint simultaneous three-pad coverage
is native truth only. No model joint head exists and no product of marginal
heads is fabricated. Baseline state/hidden receipt empirical reproducibility
does not make state58 a fully observed Markov state or prove arbitrary native
restoration. These are local same-lineage MONITOR contexts and heldout new
action responses, not unseen ego states/independent trajectories or a policy
benefit claim. This diagnostic dataset is permanently excluded from fitting.

Outputs retain all328 predictions,320 side responses,160 central records per
model, all identities, model-only AD/FD arrays/statistics, fixed endpoint and
finite-response summaries. Missing targets remain NaN in arrays; JSON masks
and null statistics describe unknowns. Failure is saved without automatic
retry or changing tolerances. The frozen online/Actor/native evaluation
sources and old data/weights are untouched.
