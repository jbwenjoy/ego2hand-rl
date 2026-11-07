# Phone physical-world delta interface v2

This is a usable tensor/array interface, not a trained world model. It reads no
checkpoint, constructs no neural network, and changes no simulator or old study
source. The original transition dataset and scalar codec remain immutable.

## Exact coordinates and API

`phone_adapter.py` uses `phone_COM_world_state58:1`:
`[q20, qd20, pCOM_world3, Rbody_to_world9(row-major), vCOM_world3, omega_world3]`.
The input action is the **actual absolute position-actuator command u20 in rad**
held for 20 ms. It is not the Actor's normalized action and is not processed a
second time by EMA, C2, clipping, or a bias subtraction.

The new `phone_physical_world_delta52:2` is
`[dq20, dqd20, dpCOM_world3, Log(Rnext @ Rcur.T)_world3, dvCOM_world3, domega_world3]`.
Decode is `Rnext = Exp(drotation_world) @ Rcur`; other blocks add directly.
All quantities remain in their listed physical units. There is no normalization,
rotation projection, angle wrapping, inherited pen scale, or checkpoint loader.

```python
import phone_adapter as A

# x: Torch (...,58); u: (...,20); same float32/float64 and device.
features78 = A.state_action_torch(
    x, u, state_schema=A.STATE_SCHEMA, action_schema=A.ACTION_SCHEMA)
next58 = A.decode_torch(
    x, physical_delta52, state_schema=A.STATE_SCHEMA, delta_schema=A.DELTA_SCHEMA)

# Optional one-call callback bridge; the caller supplies a new phone predictor.
next58, physical_delta52 = A.predict_next_torch(
    x, u, predict_physical_delta,
    state_schema=A.STATE_SCHEMA, action_schema=A.ACTION_SCHEMA,
    input_schema=A.INPUT_SCHEMA, delta_schema=A.DELTA_SCHEMA)
```

`state_action_numpy`, `physical_delta_numpy`, and `decode_numpy` provide batched
NumPy equivalents. Encoding delegates the pinned original F64 world-Log codec;
its half-turn ambiguity and strict SO(3) domain reject the whole call. Torch
decode preserves dtype/device/autograd and permits only the declared raw rotation
roundoff tolerance (F32 1e-5, F64 1e-10), with no repair. Exp has a finite analytic
derivative at zero; this is a decoder derivative, not a physics Jacobian.

## Saved loader and split boundary

`saved_dataset.load_saved_dataset(report_path, expected_report_sha256=...,
overlap_audit_path=..., expected_overlap_sha256=...)` binds the original report,
schema/output pins, and alias receipt, then reads only seven required NPZ arrays.
It returns read-only `x58/u20/target58/physical_delta52`, identities, original
branch/boundary/H2 indices, and the alias component. `branch_rows()` gives every
original H2 pair without shuffle or deduplication.

All 322 rows remain: 161 planning branches, two H20ms transitions each, from seven
consecutive source boundaries of one failed-hold prefix. Six exact cross-boundary
receipt-and-control aliases connect **all seven boundaries**. The loader checks
those original aliases and `require_single_partition(labels)` rejects any
train/test/validation partition, including a user-assigned cohort split. A single
`fitting_only` partition is allowed but explicitly has no independent holdout
qualification. There are no independent trajectories or new ego measurements.

State targets are POST20ms; the available endpoint pad-force scalars are last
solver PRE (19.84375ms at h=0.15625ms), not POST contact geometry or complete force
trajectories. The pure time contract supports n=64/128 at H=.02 using integer
ticks; this particular saved dataset is n=128 only. A new h or mass/geometry does
not acquire qualification from the matching 58/52 dimensions.

## Necessary integration changes, not implemented here

* Native `state.py:20-38` uses `Rcur @ Exp(dr)` and `Log(Rcur.T @ Rtarget)`.
  Its chart52 is incompatible with this world-left physical delta52. Schema
  arguments therefore reject a native-chart tag explicitly.
* `sim.py:95-110` reads object root position/velocity in palm coordinates. The
  phone contract uses world COM position/velocity and world omega, including the
  nonzero body COM offset. A new Reader must use that contract on actual POST
  qpos/qvel; the old root fields must not be relabeled COM.
* `networks.py:11-61` normalizes dynamics+u and delta52, then calls the old
  integrator and emits a separate contact head. New phone scales/context and
  left decoding must be explicit. This raw78 bridge does not invent xi46,
  dynamics278, self15 labels, impulse targets, or a replacement contact head.
* `runner.py:413-417,529-543` fits the old difference/normalizers/contact loss;
  imagination calls old task rewards (`runner.py:734-739`). Those call sites,
  task memory/reward, replay grouping, failure semantics, and same-budget
  SuperMimic/CHORD binding still require a separate phone integration. The
  fixed-wrist state codec does not establish a Markov state: warmstart/HISTORY
  and other solver context are absent.
* The full151 factory asserts four 5ms substeps (`full151_native/native.py:65-66`,
  `learning_v2.py:298-299`). The phone's 64/128 substeps and late PRE labels need
  explicit new timing/buffer contracts, not a silent dt edit.

No training, normalization fit, policy update, native forward, physics step,
model/checkpoint load, or GPU execution is included. Existing pen weights and
322 correlated planning transitions provide no phone accuracy, holding, Actor
benefit, real-ego dynamics, or generalization claim.
