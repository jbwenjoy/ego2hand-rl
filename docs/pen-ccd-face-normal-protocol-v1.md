# Same EPA face normal V1

Same484 fixed geometry/pair/native-witness inputs and original1e-6 tolerance,
50 iterations,96/factor20 scratch and bounded axis retries.968 GPU queries
comparing pinned original retry source to a uniquely cached face-normal
variant.0 new oracle/FK/integration/Adam. Preserve all old failures/costs.

Original EPA returns depth=-sqrt(face_norm2), but contact frame normal is the
difference of two WORLD-coordinate weighted witnesses. The isolated variant
retains the selected EPA face index for cylinder/mesh (static multicontact
branch still excludes cylinders) and forms its single-contact normal from
that face's origin projection. Contact positions/materials/physical shapes,
thresholds, force and all failure flags remain. Both original and retried EPA
results carry the chosen index. Retain exact source edits and actual kernel
identities; no installed-module changes or state source rebinding.

Use the same5um depth/5deg signed normal/0.1mm midpoint criteria and1um boundary
classification. Compare old backend contact bytes and record all changes.
Even passing does not establish integrated contact dynamics, state/gradient
admission, task success or performance. Additional independent scenes and
fresh-backend integrated validation remain before training.
