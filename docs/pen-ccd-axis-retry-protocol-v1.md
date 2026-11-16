# Bounded GPU CCD axis retry V1

All44 frozen challenge scenes, two DISTINCT kernel identities: scratch96/factor20
control and same scratch with bounded recovery. Maximum88 pair queries and132
extra CCD attempts; count actual attempts from diagnostic printf.0 native/FK/
reference queries/Adam. No CPU fallback or new physical geometry/force.

Only cylinder/mesh needs_epa and dist>=gap after EPA can retry, excluding
sensors. Original GJK-separated and valid-contact paths remain. Up to3 fixed
orthogonal initial GJK search directions; reset mesh search indices for each
attempt and use original support functions, thresholds, scratch, witness and
EPA solve. Accept the FIRST finite negative-depth result with contact count>0;
otherwise retain original result. Preserve every raw overflow flag from all
attempts; no capacity/gate masking. Imported ccd helper does not perform
contact injection or bypass its geometry. Initial points are search guesses,
not new geom positions; n0/near-boundary behavior remains part of qualification.

Kernel factories/calls must be uniquely namespaced and object identities
checked before query. Keep all contact witnesses, frames and false positives/
negatives against pinned native reference, with <=1um boundary reported
separately. Passing44 scenes only promotes further diagnosis, not production
state/control/gradient admission or training/performance/ego claims.
