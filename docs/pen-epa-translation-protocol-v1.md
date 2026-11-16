# Pair translation diagnostic V1

12 pinned derived scenes from capacityV2, one isolated physical pair[1,55].
Installed horizon24 and all algorithms remain unchanged. This is a diagnostic
scene query, not modified production collision or a training state restore.
For each scene query WORLD, translate all world31 geom centers by geom1 center,
and by the pair midpoint. Compute shifts/subtractions in FP64, cast to FP32;
report relative-position changes caused by rounding rather than assuming
translation invariance in floating point. Keep all rotations/mesh/sizes fixed.
Each query uses the installed single-pair narrowphase and independent FP64
mj_geomDistance at50 iterations on the same translated FP32 geometry inputs.
Exactly36 GPU pair queries and36 CPU distance queries maximum;0 integration,
FK or Adam. Record raw flags, contacts/distances, finite outputs, source and
translation errors. Do not infer a production repair from this fixture-only
translation; an algorithmic pair-local frame variant would require a separate
source artifact and broader contact/geometry/state qualification.
