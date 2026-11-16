# Derived scene pair isolation and capacity V1

Input is the explicitly derived original252/row63 scene whose full installed
collision query returned world31 raw EPA_HORIZON256. No exact original quat
or original SUPPORT206 cause is asserted. No integration/training/backend
migration. Fresh fixture construction must preserve the pinned complete
Model/Data input bytes, original materials, mesh graph, frames and numerics.

Capture original broadphase context with installed NXN/SAP and replay complete
narrowphase once; require raw masks to match the saved scene. Isolate ALL
world31 candidates from that context, at most275, using their original ordered
geom IDs, predefined pair/sensor IDs and world31. Use unchanged installed
_narrowphase one candidate at a time; retain every result. At most4 failing
pairs may proceed; otherwise stop with evidence. No nearest-pair selection.

For EVERY isolated failing pair, query original scene plus eight one-FP32-ULP
object-quaternion neighbors (+/- each component), and three nonoverflow
controls: exact parent, original adjacent rows61/62. Run capacities24/48/96.
24 uses installed source. 48/96 use isolated exact source copies with ONLY
the unique epa_horizon allocation literal replaced, pinned and imported under
separate module names. Keep Apache header. Do not patch installed modules or
their globals, algorithm conditions, tolerance, mesh support, assets or state
metadata. Dispatch convex and primitive paths as production does.

Fixed maxima:1 full broadphase/narrowphase replay;275 candidate pair queries;
144 paired capacity queries (4 pairs x12 scenes x3 capacities);32 additional
32-world FK builds (4x8). Report actual attempted/completed work, compilation/
wall, raw flags and finite contacts, stable sorted contact-byte controls and
all quaternion/rotation perturbation sizes. Larger buffers can expose further
algorithm failures; do not assume larger is correct. Even capacity-only success
on this DERIVED fixture is not production-state admission or a final task
comparison. Preserve all negatives/exceptions and require fresh explicit
qualification before restarting matched control training.
