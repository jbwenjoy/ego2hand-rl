# Current-state phone/hand geometry provider

`geometry_provider.py` provides actual compiled fixed-wrist hand FK using Torch
tensor arithmetic, with **no `nn.Module`, learned parameters, simulator calls or
future contact/force inputs**. It is a new phone input contract, not the old pen
chart and not a geometry-conditioned WM that has already been trained.

```python
from geometry_provider import PhoneGeometryProvider, validate_features

provider = PhoneGeometryProvider(".../completed_v1/compiled_constants.json")
# x: Torch [...,58], u: same dtype/device [...,20]; h: [...], n: integer [...]
features = provider.features(x, u, h, n)
validate_features(features)
poses = provider.poses(x)
# Optional purely geometric convenience calls:
phone_vertices_W = provider.mesh_vertices_world(poses, 71)
capsule_a_W, capsule_b_W, radius_m = provider.capsule_centerline_world(poses, 8)
```

`x[..., :20]` is the actual robot q20; there is no second, possibly inconsistent
q override. `u` is the absolute current command, not a future target label. FK
depends on q and the phone pose, not on u. Leading batch dimensions and F32/F64
are supported. Native h and n remain explicit; only h=.0003125/n=64 or
h=.00015625/n=128 with nominal H=.02 are accepted by the feature constructor.

`poses` returns every body/geom in the compiled native ID order, plus the 22
phone-boundary relative geom poses. Exported constants preserve the actual
parent tree, hinge axis/anchor, qpos0 reference, local body and geom transforms,
joint/control bounds and explicit collision-pair identities. The fixed wrist
is the original compiled pose. Body rotation is **BODY→world**, not the inertial
principal-axis rotation. From phone state58,

```
p_root_world = p_COM_world - R_body_world @ body_ipos_body
p_other_geom_BODY_COM = R_body_world.T @ (p_other_geom_world - p_COM_world)
R_other_geom_BODY = R_body_world.T @ R_other_geom_world
```

The relative origin is COM, while the axes are phone BODY/material axes. The
phone geom offset and rotation are separately bound in the metadata. Each
hinge uses `q - compiled qpos0` and the rotation-about-anchor translation. The
formula matches native `mj_kinematics1`, including off-center joints.

Schema `phone_current_pair_geometry_features:2` replaces neither the frozen
v1 causal contract nor its files. It separates `pair_pose_known` and
`pair_geometry_binding_known` (all true) from `pair_gap_known`,
`pair_lower_bound_known`, `normal_known` and `normal_unique_known` (all false).
Gap/lower-bound/normal arrays are **NaN**, not a fabricated zero distance.
Do not feed these NaNs to a network without the explicit mask/encoding choice.
No signed-distance, uniqueness, collision avoidance or full-H zero-causal-path
certificate is issued.

The 23 bound geoms are phone71, designated original pads15/28, the remaining
phone-boundary hand geoms and table0: 12 meshes, 10 capsules, one box. The phone
mesh has 352 compiled vertices/700 source triangles; pad15 has 1982/3956 and
pad28 has 1848/3690. These whole sensor collider meshes do not certify a
functional sensing face, soft material or measured friction. A future third
designated pad needs an explicit task/schema change; it remains nonpad here.

Compiled `mesh_vert` is already in the compiled geom frame. Apply the geom
world transform **once**; do not reapply exported diagnostic `mesh_pos/quat`.
Raw source triangles are retained for identity, not represented as an exact
convex-hull triangulation or SDF. Capsules use radius `geom_size[0]` and a
local-Z centerline of half-length `geom_size[1]`. This agrees with the pinned
[MuJoCo 3.11 collision support code](https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_collision_convex.c),
which binds `mesh_vert` and the geom pose directly (lines659–702), and its plane
support path transforms directions into the geom frame (lines955–969).

## Bounded numerical check

The pre-frozen protocol selected only original TRAIN current states:

| family | native h (s) | interval | input phase |
|---:|---:|---:|---|
| 0 | .00015625 | 0 | episode initial, before action |
| 8 | .0003125 | 2 | prior raw POST, before action |
| 24 | .00015625 | 5 | prior raw POST, before action |

One actual process completed exit0, 7/7 checks. It used one pinned MJB load,
three fresh MjData and three `mj_kinematics`, six saved-state NPZ reads, six
Torch FK calls and one pure-tensor q Jacobian. There were **0 forward/comPos/
integration/NN/optimizer/GPU calls**. No VAL or TEST arrays were opened.

* F64 maximum position error over all29 bodies/all72 geoms: 1.67e−16 m;
  maximum geom rotation-element error: 4.27e−15.
* F32 geom position/rotation-element differences: 6.15e−8 m / 2.28e−7,
  descriptive precision measurements, not new physics accuracy gates.
* Geometric q→thumb-relative-center Jacobian vs the same native states'
  `xaxis × (p_geom−xanchor)`: maximum difference 1.53e−16.
* Nonzero directional finite difference vs Torch derivative: 6.72e−11.
* Raw Torch RNG bytes unchanged. No generated random parameters.

These establish the tested geometry/coordinate implementation. The q Jacobian
is **not** an action-to-state, native dynamics, contact-force or policy gradient.
The module can be evaluated at a predicted valid state, but it does not make
state58 Markov or resolve solver history. Raw improper/nonfinite SO3 is rejected
without projection. The independent geometry query module can be connected
later with its own distance/normal scope, masks and differentiability contract.

`completed_v1/report.json` binds sources, data pins, compiled metadata/arrays,
the selected input identities and full numerical evidence.
`process_v1/terminal.json` binds the exact invocation, exit status and complete
stdout/stderr (stderr0 bytes). Frozen sources and original data/model were not
changed.

## Upstream integration boundary

This supplies a missing **current state→geometry** function. A future feature
consumer must explicitly adopt schema2 and a mask/unknown encoding; it cannot
silently reinterpret the old `pair_geometry_known` as a known gap. Current
geometry may be a deployable input. Full-H contact graphs/forces and intervention
outcomes remain supervision/diagnostic labels, never inputs available at the
start of H. No geometry-aware WM head, causal zero-mask, Actor/reward migration,
contact dynamics update, Hand2 task benefit or ego generalization is implemented
or claimed by this artifact.
