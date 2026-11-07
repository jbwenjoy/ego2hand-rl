# TRAIN-only causal intervention design

This directory contains an **executed metadata generator**, not a new physics or
learning run. `build_design.py --protocol protocol_v1.json --build-metadata`
produced `generated_v1/report.json`. `validate_generated.py` checked the saved
ordering/full-state layout and fail-closed input schema. No model/FK/native step,
NN/AD/training was run; no VAL or TEST NPZ was opened.

## Ready acquisition plan and budgets

`contexts72.json` selects original TRAIN replicate0 for response fitting and
replicate5 for action-response checks, for each of3 profiles and2 source mappings,
both dt values, and fixed intervals0/2/5. Original family roles never change.

| New role | Original TRAIN families |
|---|---|
| Response FIT | 0,8,16,24,32,40 |
| New-action CHECK at known states | 5,13,21,29,37,45 |

Every source includes the raw before-action and after-action300-value native
integration vectors in `source_before_action_restore_arrays.npz`, their original
state-spec/hash receipts, actual x58/u20, and original episode COM/quaternion for
hard-stop references. State velocities and hidden integration fields are retained;
they are not reset to zero or rebuilt from state58. A future executor must restore
fresh Data from the **before-action** vector, set the branch command, preserve the
matching after-action receipt, and step without a new observer forward. Existing
four-state restore proof is pinned, not extended here to all72 sources.

`all_baselines_then_sides_plan.json` has all72 baselines first, then2880 records
for20 axes×negative/positive. The nominal step is.01rad, reduced to half the true
remaining control margin without clipping or stepping up. Native sides require
actual |du|≥1e−6rad: **2870 available and10 unavailable** records remain. All10 are
also FP32-collapsed model endpoints and cannot be called new represented model
inputs. Other19 controls are byte-identical. F64 and represented-F32 deltas and
original-TRAIN input aliases are explicitly retained.

The executable plan is2942 branches/282432 native steps; the conservative all-record
upper bound is2952/283392. No steps are authorized or executed by this artifact.
Every unavailable, failed restore, warning, nonfinite or partial branch must remain
with its reason/mask. Do not replace a source/family or select cases using outcomes.
Preserve original-episode hard-stop references and all baseline/side records.

## Causal masks and geometry inputs

The frozen model's explicit22 phone pairs override individual geom collision bits.
Every one of20 actuators has a descendant with a possible direct phone-contact
path. Hand-hand contacts/coupling may add paths. Thus **all strict72×20 object
zero masks are false**. Twelve unperturbed baseline H intervals have no generated
phone contact; that is a target-side empirical observation, not an intervention
or swept-H separation certificate. An empty current contact list is even weaker.

`causal_contract.py` provides:

* `validate_current_geometry_inputs(batch)`: exact current/predicted-state geometry
  ABI; proper body/relative rotations, model/state identity, h/n contract, finite
  commands, known masks, unit unique normals and lower-bound semantics. Unknown
  geometry may remain NaN outside known masks. Reflections and future-label fields
  are rejected.
* `strict_object_zero_mask(N)`: v1 issues no strong certificate, so always false.
* `loss_response_mask(...)`: only actual available/full/finite executed pairs can
  supervise finite responses; it is not a zero-target mask or analytic Jacobian.

The current PRE input column and target-only column are separate in
`geometry_contact_input_contract.json`. Full-H path certificates, contact traces,
native endpoint responses and decoded impulses are **supervision only**, never
known deployment inputs. Geometry at imagined steps must be recomputed from the
predicted q20 and full phone COM/bodyR using differentiable FK and validated compiled
collision geometry. This provider is not implemented/qualified here. Unknown
features require an explicit masked encoding, not NaN silently relabeled zero gap.
Whole tip-sensor meshes do not establish functional-pad faces.

This task stays on original designated pads15/28. The third fingertip remains in
other/nonpad; a future TRIPOD task needs new data/head/design semantics.

## Two levels of final checking

The new action CHECK families use states previously fitted by the original WM;
they can test new counterfactual responses only, not unseen states or ego scenes.
Current476 VAL diagnostic branches never enter TRAIN. Original6 TEST families
were already evaluated and are not relabeled unseen.

`new_final_family_plan.json` separately freezes **12 new family IDs1000–1011**,
unused SHA seed namespaces, distinct literal q8 perturbations and20-axis control
curves, bothdt grouped, and no clipping. These were checked against all original48
initial/control hashes; no new outcomes are acquired or used for model selection.
Their future maximum budget is240 intervals/23040 steps. They offer fresh controlled
initializations/actions under the same assumed model/source geometry, not independent
real ego trajectories. Keep these labels sealed until the future method is fixed.

## SuperMimic/CHORD integration boundary

`upstream_interface_map_v1.json` pins exact call sites and required migration.
Current phone state58/time and left-world delta52 are usable; the old right-chart
integrator, palm/root Reader, dynamics278/xi46, self15 head, pen reward/reference,
replay partitioning and prior hook are not plug-compatible with this contract.
No Actor/critic/reward update is implemented here. A matched phone study still
needs common warmup/replay/initial policies, the same scene/controls/physical budget,
and distinct learning/physics counts.

The pinned `phone_momentum_contract_v1` can link source-BODY-axis/moving-COM impulse
targets to world momentum residuals. It neither determines p/R nor contact causality.
See `momentum_source_review_v1.json` for the source-only frame review and scalar/
(...,1) mass-broadcast restriction. No owner numerical proof was rerun.
