# Actual GPU pen initialization comparison

The v0.61 protocol is complete. All four GPU runs fail at 0.16 s; no neural
training occurred. Geometry correction activates three fingertip contacts,
but full force and moment support remains infeasible. The full Goal remains
active, including controlled learning, matched support objectives, world-model
gradients, multi-trajectory cost and real ego validation.

## Fixed protocol and results

`runs/controlled_projected_pen_GPU_v2/report.json` contains original zero
residual and the saved geometry correction at 5 and 2.5 ms. Original 50 Hz,
369-frame, 7.36 s reference, velocities, wrist/object pose and physical
parameters are preserved. Only 20 robot joints change, by at most 0.03409 rad.
Corrected raw action is twice the physical residual under the original
scale/EMA contract. Both variants explicitly initialize first PD targets to
reference plus residual rather than old reset ctrl0. This is a declared
diagnostic comparison, not a pure qpos intervention or newly trained Actor.

| Variant | GPU dt | Duration | COM RMS | Max penetration | Qualified |
|---|---:|---:|---:|---:|---|
| Original | 5 ms | 0.16 s | 29.844 mm | 3.801 mm | false |
| Original | 2.5 ms | 0.16 s | 28.875 mm | 3.801 mm | false |
| Geometry correction | 5 ms | 0.16 s | 31.022 mm | 1.829 mm | false |
| Geometry correction | 2.5 ms | 0.16 s | 29.620 mm | 3.335 mm | false |

All stop at the first task position-error termination. No post-start resets,
object force aid, gravity compensation, equalities or time stretching occur.
The corrected coarse case meets penetration only; tracking and duration fail.
Native CPU's corrected 0.22/0.18 s did not transfer to GPU. Backend and
termination timing differ and remain explicitly separate.

## Actual frozen GPU support and review

The separate pre-start static query zeroes velocity/warmstart and restores
both exactly before execution, without integrating physics. Actual GPU
physical tables and EFC contact records are saved. A detached native model
supplies Jacobians. The demonstrated copied-host adhesion omission is cleared
only in host query records and checked against direct GPU EFC decoding;
GPU constraints, force and model are not patched.

Original geometry has four eligible link4/tip contacts; corrected geometry has
three fingertip contacts, versus two in native CPU forward. Every bounded-PD
static LP, unbounded-actuator static full6D cone and first nominal dynamic
full6D cone is infeasible. Matrices have rank six; neither rank nor contact
count establishes positive support. Static/dynamic separating certificates
are independently checked. These are local frozen-cone statements conditional
on modes, nominal inertia and derivative estimates, not global impossibility.

`runs/controlled_projected_pen_GPU_review_v2/report.json` passes 76/76 checks
of initialization, clock, carries, original EMA/PD control, wrench basis,
LPs, certificates and metrics. Its EFC subreview passes 28/28. PD target,
wrench-basis and metric reconstruction errors are zero. Physical tables match
across cases. These verify recorded failures, not manipulation success.
`runs/pen_gpu_projection_final_tests159.log`: 159 repository tests pass.

## Reproduction and preservation

Use existing external Hand2 Python from the repository root:

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/run_gpu_projected_pen.py --config configs/controlled_pen_gpu_projection_v1.toml --output runs/FRESH_GPU_RUN
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/review_gpu_projected_pen.py --report runs/FRESH_GPU_RUN/report.json --output runs/FRESH_REVIEW
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/render_gpu_projected_pen.py --report runs/FRESH_GPU_RUN/report.json --output runs/FRESH_VIDEO
```

Both dt videos in `runs/controlled_projected_pen_GPU_video_v1` replay actual
saved states at 20x slow motion with explicit terminal stills and no
interpolation. Each decodes to 145 frames, 1280x800, 30 fps; decoded frames were
inspected. Rendering forward calls refresh graphics without integration.
They show a controlled upstream reference, not ego video or learned success.

v1 source/report bytes remain in
`provenance/pen_gpu_projection_v1_metadata_revision`. v2 adds BODY rotation/COM,
warmstart assertions and loaded Dex-source pins, and corrects constant-branch
scope text. Prior v4 evaluator and first reviewer bytes are also preserved.
The provenance ledger maps old source hashes to exact snapshots; raw reports
are not overwritten. Archives, external environments and drivers are unchanged.

## Next phase

Search a bounded arrangement with actual dynamic wrench support and
controller-compatible forces. Keep geometry, nominal derivative uncertainty
and actual execution as separate checks. A resident batched GPU forward-query
path can reduce repeated geometry work and fine host synchronizations; no
new support-aware solver or speedup is established by this result. Failed
prefixes remain diagnostics, not expert supervision. The standing first
healthy neural checkpoint/restore/ETA pause applies to future training.
