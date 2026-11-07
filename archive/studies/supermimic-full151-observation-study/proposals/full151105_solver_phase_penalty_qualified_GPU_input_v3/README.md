# Qualified GPU state inputs for the same-phase dense P diagnostic

Status: **source-only protocol draft; no scoring authorization**. The current draft is `protocol_draft_v2.json`, SHA `c1463665e0cfb996d5c8f0734a25ed1c674c82db7c083e6858a19be1f776422e`. Its first draft and delta reports remain beside it. No qualified packet NPZ, future dense output, model or objective was opened/executed to write this proposal. Only replay result/config/packet-pointer and lifecycle metadata were read after Root permitted that inspection.

The proposed change replaces the failed CPU state conversion with the exact, independently confirmed output of the new matched native CUDA batch-one replay. It does not replace or relax the original full58 comparisons. The GPU producer's qualification is an input prerequisite; its `dense_scoring_eligibility=False` and `proxy_scoring_authorized=False` remain accurate. A separate Root CPU config and current-source CPU proof are still required.

| Item | Old dense source00e1 / protocol1ec18 | New draft |
|---|---|---|
| Scoring inputs | One CPU native-state conversion, 2260 rows, then use first1808 | Read only the qualified GPU packet's `native_GPU_prestate58[1808,58]` bytes |
| State qualification | All452 endpoint and160 anchor full58, plus1808 solver-pose checks | The same gates, independently confirmed and bound to exact replay report/config/packet/OS12; no new converter |
| Explicit passive model read | One frozen MJB constant/index/catalog load | Retained once for the exact original native IDs/masks/type/range and optional mapping agreement, never state reconstruction |
| Factory-owned passive model read | One source-accounted read inside the exact factory | Retained and reported separately |
| Penalty | One original P_ONLY factory, same08d9 spec and actual geometry | Unchanged |
| Work | 1808 rows, batches14×128+16, 15 objective forwards | Unchanged |
| Native primary | Valid WORLD0 generated hand-object depth strictly greater than3mm; all exact deepest ties | Unchanged; no force-positive filtering or firstargmax |
| Coverage secondary | All strict>3mm pairs, separately named overlapping categories | Unchanged |
| Proxy/math | Original1344 fixed samples /21 sampled geoms, rawP and BP=−.1P, original SDF threshold | Unchanged; full native catalog is never inferred from sampled geometry count |
| Original failure | CPU all58 converter FAIL | Preserved FAIL, never relabeled |

The seven substantive parent sections (`question`, `prior_knowledge`, `all_rows_schedule`, `phase_contract`, `native_depth_label`, `actual_geometry_binding`, `fixed_analysis`) are JSON-value identical in the delta report. The separate exact-tie rule makes the already-frozen source behavior explicit, without changing its definition.

Proposed input ABI additions are `qualified_GPU_prestate1808`, `GPU_replay_report`, `GPU_replay_config`, `GPU_replay_protocol`, `GPU_replay_identity` and the still-pending `GPU_replay_saved_independent`. The packet keys come directly from frozen replay source9ff9: `native_GPU_prestate58`, `completed_mask`, `source_pre_row`, `palm_pos`, `palm_rot`, `hand_qpos`, `hand_dof`, `object_qpos_start`, `object_dof_start`. Require exact FP32 shapes, finite all rows, all-true completion, and `source_pre_row=arange(1808)` in original physical_samples order. Integer mapping and body/geom/mocap namespaces remain distinct.

The consumer must fail closed before its first factory if the independent replay review, any source/input pin, full58 flags, solver-pose evidence, complete row identity or original physical capacity/lifecycle provenance is missing or inconsistent. Numeric thresholds, sample masks and classifications cannot be changed to clear this prerequisite.

This covers one source/seed and dependent fresh-action-source donor/branch states. Native contact absence is not clearance or exhaustive collision truth; native generated HO depth is not a positive force, selfcollision, global task failure or human-contact label. Proxy agreement/disagreement cannot grant dynamics, force, physical-gradient or method-benefit qualification. The proposal does not authorize any new GPU, physics, training, WM/Actor call or dense proxy execution.
