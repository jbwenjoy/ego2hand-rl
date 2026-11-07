# Arithmetic forensic interpretation — author-side, not independent audit

The original dense consumer failed and remains failed. The separate forensic run completed serialization and diagnostics; it did not qualify or execute dense proxy scoring. These notes use only saved arrays and retain the complete 452 endpoints, 160 anchors, 1808 solver poses and 2260 conditioning rows in the accompanying figures and CSV.

Only five endpoint components exceed the original bound, all in reader/wrist-frame angular velocity. q, qd, position and linear velocity differences are zero; rotation-entry maximum difference is 4.76837158203125e−7 and remains inside the original bound. All 160 leading anchors and all 1808 solver-pose position/rotation comparisons pass that unchanged rule.

The following is the complete indexed exception list, alongside—not replacing—the all-row artifacts. Values and absolute errors are rad/s.

|Fresh path / phase|State component|CPU − GPU reference|Old bound|Operand/output descriptor|Closer to secondary F64 shadow|
|---|---:|---:|---:|---:|---|
|A0 / 21|55|−1.2516975403e−6|9.5367431641e−7|7.380194|GPU reference|
|A2 / 10|57|−5.2452087402e−6|3.8006241994e−6|2.180031|GPU reference|
|A2 / 14|57|−4.2915344238e−6|9.5367431641e−7|3.387635|GPU reference|
|A2 / 40|57|−1.3113021851e−6|1.2363520909e−6|4.367209|CPU|
|A2 / 45|57|−1.6689300537e−6|1.4293513004e−6|6.329182|CPU|

These descriptors are finite and modest rather than an automatic demonstration of extreme cancellation. Neither backend is uniformly closer to the secondary F64 calculation at these five points. The shadow's normalization and reduction order are themselves a chosen arithmetic reference, not physical truth. The saved observations do not isolate a unique kernel, normalization step, TF32 effect or harmless-roundoff explanation.

The native state definition rotates raw freejoint BODY angular velocity into world coordinates and then into the fixed reader/palm frame. The original bit-exact q/qd checks and all position/pose checks support the tested mapping/phase alignment, but they do not turn the five angular failures into passes or prove every arithmetic operation qualified.

The frozen geometric penalty reads q, object position and rotation, not qd or either velocity. A separately declared **geometry-input** qualification could therefore be evaluated on its actual dependencies while retaining the full-state angular failure as a diagnostic. That would require a new explicit protocol/consumer scope; it is not a retroactive change to the original full-state gate, and these figures grant no execution permission or scoring eligibility.

No model, converter, geometry, objective, optimizer or physics was run for these notes. Contact's independent saved-array audit remains separate.
