# Two-pad static-reference labels (additive v2)

Use **`completed_v2/report.json`** and
`completed_v2/all960_pad_only_other_and_static_reference.npz`. The first attempt
under `completed_v1` failed on a metadata field before any raw contact NPZ read;
it is preserved. The corrected exporter is `build_pad_support_labels_v3.py` with
`protocol_v3.json`: only the executed-step-count source changes to the frozen
original acquisition identities. Physics, two analytic cases, grouping, formulas,
targets and masks are unchanged. See `execution_correction_receipt_v3.json`.

All960 source intervals remain:912 complete H20ms and48 actual partial prefixes,
with prior TRAIN/VALIDATION/TEST, family and episode identities. A raw contact's
other-geom ID determines **pad15 /pad28 /all other phone-boundary**. There is no
normal-force threshold. The other group includes table if present; table is not
silently counted as designated support. Reader-captured table normal force is0
in this cache. Group sums reproduce the frozen all-contact wrench to8.882e−16.

Each group has `*_world_mean6`, `*_BODY_mean6`, `*_world_impulse6` and
`*_BODY_integral6`. Six-vector order is force3/torque3 about current COM. Means
retain N/Nm and integrals Ns/Nms; no mixed-unit6D norm is formed. BODY axes follow
the actual PRE orientation, so BODY mean is not the source-fixed-axis mean.

The controlled-static reference is `Wreq_world=[−m*g,0 torque]`, here upward
1.962N for the assumed .2kg native model. It is rotated by **each actual PRE
Rbody.T**, then averaged for `static_reference_BODY_mean6`. The pad-only errors
are `pad_only_minus_static_reference_world6` and `..._BODY6`; the reference is
for the **sum of the two pads**, not duplicated as a full-gravity target per pad.

Across912 complete intervals, pad-only mean force vector RMS is.918N and torque
RMS.02952Nm; other-contact contribution is.307N/.02263Nm. Relative to the static
reference, pad-only world force error RMS is1.629N and torque error RMS.02952Nm.
Mean upward pad force is.519N. These summarize moving simulated transitions,
many losing support: they are **not** a new holding necessary gate, force GT,
sample filter, controller, or proof of global infeasibility. Motion need not
instantaneously satisfy a static target.

These labels support future two-designated-pad wrench reward wiring without
mistaking all-contact/other-link forces for the intended support. A future third
designated finger requires a new set/schema; this artifact remains fixed15/28.
No native/NN/physics/training was run, and frozen v1 labels were not modified.
