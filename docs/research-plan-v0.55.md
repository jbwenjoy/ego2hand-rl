# Research plan v0.55: finish the original impulse representation budget

2026-11-13. Previous turn made progress: e4ce5ec committed measured-wrench
impulse decoding, explicit nonzero numerical-defect targets and the matched
DIRECT/IMPULSE recipe. First checkpoint100 saved with38/38 checks and requested
pause. The user resumed the Goal. Continue the original remaining1900 updates;
no new hyperparameter/model/data selection follows from intermediate CHECKs.

## First100 evidence

Small-action CHECK AD NMSE versus zero-response prior1, DIRECT/IMPULSE:
delta52.833326/.833183, object12.992879/.994373, contact9 1.018593/1.020664.
There was no established object/contact response benefit at this boundary.
At200ms, COM error4.016/3.934mm; angular error.061520/.062765rad. Linear momentum
defect RMS5.61e-4/1.40e-7Ns versus native4.63e-8Ns; angular1.27e-5/2.32e-10Nms
versus native8.77e-11Nms. The coupled representation reduces an algebraic defect,
but this cannot certify causal dynamics, contact geometry or physical control.

The regular99-update loop took.69822s; retained first step/restore-clone checks
took2.3850s. Three device-wide regular-loop samples reported utilization18%
and used VRAM1993MiB/32607MiB; this short workload is not a utilization/speedup
benchmark for full SuperMimic. Original preparation and source/data pins remain.

## Exact fixed-budget continuation

`configs/mug_impulse_resume2000.toml` selects the healthy100 checkpoint and the
original2000 endpoint. Shared `impulse_stage` functions restore each distinct
arm, original head schemas, physics/scales, Adam100, RNG and CUDA sampler.
No weight/head reset, optimizer reset, sampler reseed, normalizer refit, changed
loss or new data. Same3258 TRAIN rows, same batch256 and matching sampler
sequence. All1920 new CHECK rows, old probes and56 recursive cases are read-only.

Retained step101 is compared with one expendable restored clone update per arm:
actual batch, losses, clipped gradients, full model/optimizer and sampling cursor
must be byte-exact. The two extra clone updates are counted as restoration
verification, not inserted in the retained fit. Abort on mismatch without
restarting or silently replacing evidence. Both nonzero-Adam restore tests and
all135 tests pass before research fitting starts; real CUDA checks occur in-run.

Updates102..2000 call the same shared update and pure-array decoder. All data,
scales, physics and labels stay resident; no geometry, native simulation or host
health scalar transfer inside the loop. The FIRST and ONLY new checkpoint of
this resumed stage is2000, Adam2000 per arm, completing the predeclared budget.
Atomic full-state save/readback verifies schema/cursor, optimizer state, model
and buffers, RNG, next batch, unchanged physics/head scales, matching sampler
cursors and finite prediction/gradient/loss. No later fitting just to improve
validation. Report1899 regular updates separately from real step101/clone
verification, startup/save/diagnostics. At2000 zero planned updates remain;
no research-completion ETA is implied. After the healthy new checkpoint/report,
PAUSE Goal as the user requested.

## Additional predeclared diagnostic detail

Keep original absolute error, native finite-action/JVP and20..200ms recorded-
control recursion reports. Add read-only strata for BOTH amplitudes and ALL20
control joints, splitting hand40, COM position3, rotation3, COM velocity3,
omega3, force3, torque3 and Fn3. Preserve all original rows, fixed output
normalizers and source roles. Zero native effects retain their zero native RMS,
predicted RMS and undefined ratio/cosine=None, not an epsilon denominator or
filtered subset. Empty axes in generic inputs are explicitly marked, not called
success. Component/joint reports do not choose models or change fitting weights.

These comparisons can separate pose-head errors from force-to-velocity errors
and reveal false action effects. A coupled mean wrench/velocity still lacks
full contact geometry, contact timing and verified pose integration. Actual
no-aid control is required before saying the representation enables ego retargeting.

## Work after the next explicit resume

Use the fixed-budget outcomes, source coverage and genuine native numerical
floor to choose the next hypothesis. Establish physical reference-conditioned
feedback/Actor observations with controller EMA and future goals; validate
short-horizon native returns and matched task-level baseline/contact support
objectives. Treat learned momentum consistency and scalar fitting error as
separate from physical action/contact gradients and successful control.

Keep the FULL original objective: fixed-wrist Hand2 cube/pen physical loading/
playback, short learning/save-resume/unified no-object-aid evaluation; object-
frame support and matched A1/A2; trustworthy contact/action gradients; collision,
inertia, material, gravity, frame/input/controller contracts; independent RGB
ego and full trajectories; multiple trajectories with shared-vs-per-trajectory
TOTAL acquisition/fit/compile/validation/retry cost; hardware subversion/joint/
zero/gain/latency/observations before deployment. A nominal single-mug task is
not a replacement. No driver/CUDA changes, duplicate workspace study root or
claim of full Goal completion.

Actual completion evidence belongs to
`runs/mug_impulse_resume2000_v1/health_2000.json` if generated, plus the separately
pinned `response_strata.json`. This predeclared plan is not execution evidence.
