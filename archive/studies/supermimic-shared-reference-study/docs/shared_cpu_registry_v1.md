# Shared reference registry: CPU stage v1

This stage materializes and actually loads derived reference catalogs. It does not run shared training, physics or checkpoint transfer. All ten observation-derived references remain physically unqualified. The frozen G1 reader, geometry batch, native BODY ABI exporter, previous single-clip entry and root README remain unchanged.

`configs/derived_reference_registry_v1.json` binds every source SHA, observation-export SHA, source window `[0,151)`, geometry/native artifact SHA and original lineage. Its five recipes were resolved and loaded by the real `SingleHandTrackingReader(device="cpu")`: training eight (1,208 frames), validation009 (151), test010 (151), seen-cost ten (1,510), audit ten (1,510). Train8's physical catalog contains only 001–008; external009/010 cannot enter that track. Seen-cost explicitly permits training on the same ten references and cannot be reported as holdout. All ten share the conservative parent-unknown lineage; external trajectory evaluation is controlled within-lineage, not independent ego generalization.

The existing geometry batch independently checked relative object/body transforms and native BODY21 ABI. This builder additionally checks each track's actual resolved ordering, prefix boundaries, exact native q and per-clip finite-difference qd, and computes the same byte-hash inputs used by native PenMDP (q, palm-local object pose, native BODY links and trajectory ends). These are CPU reference signatures, not evidence that a trainer has accepted the registry. Macro palm/link1/link3/link4/tip arrays are never renamed to native BODY21.

## Actual interfaces

```python
from shared_reference.reference_registry import (
    open_reference_registry, resolve_init_requested,
    validate_initialization_identity, require_training_ready,
)
reg = open_reference_registry(path, expected_sha256=explicit_pin)
t = reg.track("within_lineage_train8")
candidate = resolve_init_requested(reg, "01_single_axis/clip_007",
                                  velocity_design="zero_rest")
```

`resolve_init_requested` requires an explicit `zero_rest` or `derived_native_FD` design. It separates q goal, native FD goal qd, requested FP32 initial q/object pose, requested qd, and designed object world linear/angular velocities. Neither design is video-measured velocity or a real initialized state. `init_measured`, actual executed u and PD preload are null. A future physical worker must restore exact physical context and default mocap, convert world angular velocity to the native freejoint convention, and save measured FP32 state/controller readback. The registry performs none of those runtime operations. Clip007 `[0,151)` rejects evidence from historical `[200,351)` even when the source-file SHA is identical.

`TargetRegistry` in `target_registry.py` binds source ID, exact native artifact SHA and window before mapping native `start/end/current time` to local target frames. All ten real bindings are `MISSING_INDEPENDENT_BUNDLE`; `gather` raises `MissingTargetBundleError`, rather than creating zero support or treating unknown as known absence. The available-bundle branch was exercised only with temporary synthetic lookup fixtures, deleted after testing. No contact target, force observation or physical teacher was produced. Unknown/no-contact semantics in a future shaping objective still require their own tests; this registry is not that objective.

`make_transition_metadata` in `replay_metadata.py` records the actual supplied executed absolute u independently of q goal, with source identity, track prefix, current local phase and episode. IDs are metadata, not network inputs. `replay_window_mask` rejects unmarked episode/trajectory switches and continuing phase/prefix drift, retains a terminal transition, and masks the subsequent reset suffix. These functions have not been installed into native replay or collection.

`shared_budget` and `assert_matched_resource_counters` implement only DRAFT aggregate transition/update/teacher counter arithmetic. `assert_exact_transfer_identity` checks reference/catalog exposure, physical context, model, controller and layout identities; it refuses old checkpoint formats 2/3/103. It neither loads a model nor resumes optimizer, RNG, replay, environment or iteration state. `require_training_ready` always rejects this stage, including when callers claim targets are ready.

## Concrete remaining integration work

- Native `commands.py:697` already concatenates references and computes qd separately per clip. `pen.py:80` hashes the concatenated reference and `pen.py:117` selects current phase using start/end. No additional clip-ID feature is necessary for the current reference-conditioned actor. Input q/body reference labels must remain the observation→IK→native-FK products; actual sim x58 and native selfcollision15 remain simulator supervision, not ego observations or measured hand-object forces.
- The controlled entry `supermimic-chord-study/scripts/train_controlled_pen.py:250` still requires exactly one frozen clip, and line547 assumes one contact timeline. A new explicit entry must resolve one permitted registry track, independently bind its contact bundle, use identical phase in native and imagined reward, and preserve executed-u/EMA/target-filter parity. This CPU stage does not relax the old entry.
- Native replay `buffer.py:11` partitions environments, not trajectory lineage. `runner.py:392` updates actor normalization from eligible observations, and `runner.py:676` samples imagination starts across eligible replay environments. Holding out some environments cannot establish external trajectory holdout: 009/010 must stay out of collect, normalizers, WM fitting and actor start sampling. Replay must call these metadata/mask contracts or implement an independently tested equivalent.
- Registry requested initial states have no reset/physics qualification. Real reset must bind the exact clip/window/model/context/velocity design, preserve q goal vs measured q vs executed u, and verify terminal/first-next reference cadence. Existing warmseed physics qualification cannot be copied across windows.
- Native exact-reference transfer checks (`runner.py:635`, `runner.py:1116`) and controller-aware format103 from-scratch rules remain intact. Any future shared pretraining/adaptation transfer needs an explicitly different, tested contract; this metadata guard cannot bypass those loaders.

## Costs and status

`configs/shared_cpu_stage_contract_v1.json` keeps pretraining, each new-trajectory adaptation, evaluation, teacher generation, failed attempts and setup/compilation as separate charged stages. The DRAFT primary comparison A1L/P_ONLY/A2 matches actual transitions and WM/failure/actor/critic optimizer counts; secondary uses the latest complete checkpoint before total wall deadline, never the best result. Shared speedup is unmeasured. Pretraining all ten before an N1 experiment is forbidden free exposure. Nested stage/process/simulator times must not be summed as disjoint costs.

CPU checks: 29 passed, including old-window initialization rejection, external training exclusion, pending targets, phase/reset boundaries, unequal teacher/update rejection and old-format transfer rejection. Synthetic fixtures only test interfaces. No engineering training smoke, formal seeds, physical initializations, GPU parity, full transfer/resume, multi-trajectory trained controller or heldout performance has been demonstrated by this stage.
