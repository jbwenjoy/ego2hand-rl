# Derived retained collision scenes V1

No training or integration. Pin valid recoveryV2 RESIDUAL201, the original
failed252 memory, the prior model evidence and all collision/kinematics source.
Use a new isolated Warp Model/Data, explicitly reconstruct its captured array
layout and numerical options, then require complete Model/Data field bytes
equal to the valid parent before constructing any derived scene. This is an
isolated collision query, not a native training restore or source migration.

First audit original failed world31 rows61,62,63: previous terminal policy
equals current policy, previous time/phase equal current source clock/phase,
source_R/C recomposed into wrist-local policy features differ by <1e-5, raw
rotation is near orthonormal/right-handed. Bias/default must be exactly0.
Derive hand qpos from direct policy block92:112; derive object center from
source_C and quaternion by explicit FP64 matrix projection/inversion and FP32
cast. Quaternion bytes are unavailable; resulting scene is always DERIVED.
Fixed wrist/model/mesh/materials stay pinned. No claimed exact original geom
pair or raw incident mask follows solely from this reconstruction.

Fixed budget5 full32-world kinematics/collision queries: exact parent geometry,
derived-parent geometry, and original failed rows61/62/63 with only world31
changed. Compare raw-parent FK with saved geometry, then measure every geom's
derived-parent error. Require reconstruction max position/matrix error <=1e-6
before querying derived original scenes; this diagnostic consistency tolerance
does not replace task/overflow/physical geometry gates. Keep all failed gates.
Report actual FK builds, world geometry/collision queries, broadphase/contact
counts, raw per-world flags and valid contact prefixes, source clocks/fields,
all numerical scalar reconstruction diffs, wall and source receipts.

Use the unchanged installed full collision pipeline, not the official helper
whose mesh support/cutoff differs from production. Do not run optimizer or
integrator, relax overflow gates, change production scratch capacity, or claim
successful repair from a non-reproducing query. Any reproduction still needs
pair isolation, numerical sensitivity and controlled capacity testing.
