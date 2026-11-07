# Native104 saved-data / world-model interface review

Saved-only review, 2026-11-04. The three actual primary evaluations contain enough data for a separately validated CPU reconstruction of the native 58-dimensional physical state and 15 executed controls per trajectory. They do **not** contain directly saved native self-collision labels or native state features. This review ran no reconstruction, WM inference, environment, collision query, forward, step, or GPU work. The machine-readable inventory and exact source/checkpoint pins are in [report.json](D:/workspace/supermimic-rough-prefix-learning-study/outputs/native104_WM_raw_interface_review_v1/report.json).

## Exact state and control mapping

The preferred codec calls the actual [`SimReader.state`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/sim.py:95) on a lightweight CPU object supplying only `env.sim.data.qpos/qvel`, compiled joint/free-joint address tensors, and fixed `palm_pos/palm_rot`. It must not construct a `PenSimReader`, `Env`, `MjData`, or `Runner`. Resolve addresses from the pinned model and declared joint order; do not assume `qpos` and `qvel` use the same addresses.

The native [`pack`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:16) order is:

| State slice | Saved input / native transformation | Meaning |
|---|---|---|
| `0:20` | `full_qpos[:, robot_joint_q_adr]` | Actual robot joint position |
| `20:40` | `full_qvel[:, robot_joint_v_adr]` | Actual robot joint velocity |
| `40:43` | `R_palm.T @ (object_position_w - palm_position_w)` | Object position in fixed reader frame |
| `43:52` | `(R_palm.T @ R_object).reshape(9)` | Row-major flattened orientation matrix |
| `52:55` | `R_palm.T @ object_free_qvel[:3]` | World linear velocity expressed in reader frame |
| `55:58` | `R_palm.T @ (R_object @ object_free_qvel[3:6])` | Body angular velocity rotated through object frame to reader frame |

Use the native [`quat_to_mat`, `mm`, `mv`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/rotations.py:25), including quaternion normalization and elementwise product/sum operations. Quaternions are `wxyz`. The angular conversion uses the **post-integration state's** object orientation, not the preceding solver orientation. The pinned object has `body_ipos == 0`; no body-origin/COM translation correction is needed here. The native reader does not subtract palm-frame velocity; the present palm is fixed. This mapping should not be generalized to a moving wrist without a new contract.

The codec must bind the palm frame explicitly. [`refresh_frame`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/sim.py:87) obtains the entity root link pose; [`EntityIndexing.root_body_id`](D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/entity/entity.py:67) and [`EntityData.root_link_pose_w`](D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/entity/data.py:264) determine its native frame. The palm-rooted asset is built by [`_palm_rooted_xml`](D:/workspace/dex-supermimic-research/src/dex_supermimic/assets/wuji_hand2.py:301).

For this exact MJB, body 3 is `robot/mocap_base` (parent world, mocap ID 0) and body 4 is `robot/right_palm_link` (parent 3, no joint, zero local translation, identity local quaternion, mocap ID **-1**). Thus these two link frames coincide. The saved bridge wrist is the palm link pose; in all three evaluations its FP32 cast equals raw mocap 0, and that raw mocap remains constant across all 61 states. A codec may rely on this asset-specific equality only after asserting the exact topology/transforms and saved equality. It must not claim that the palm body itself has mocap ID 0, or treat arbitrary mocap data as a palm pose. The review's static metadata records these fields.

Use raw indices `[0, 4, ..., 60]`: 16 physical state knots with 15 transitions at 20 ms. For transition `k`, use `control_events[k].actualu[0]` (20 absolute executed joint targets), independently crosschecked against raw `ctrl_rad[4*k+1 : 4*k+5]`. Raw state index 0 contains the startup control and is not the first learned interval target. Do not reapply EMA, hard clipping, C2, or reference addition to `actualu`. Policy residual/action, executed `u`, actual joint `q`, and reference `q_goal` remain distinct.

These are reconstructed native features. CPU execution of the original method is stronger than a hand-written approximation, but no saved native `state58` exists to establish bitwise replay. Backend/device and quaternion arithmetic differences must be reported rather than hidden.

## Reward/collision phase and label boundary

The native [`ManagerBasedRlEnv.step`](D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py:435) executes four 5 ms physics substeps, then termination and reward, then one final forward and observation update. [`PenSimReader.attach_capture`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:351) saves `self.state()`, `self.contact()` and actuator force when reward is computed. [`Runner.collect`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:378) trains the WM against `cap['physics']`, not an arbitrary post-forward sensor snapshot.

At transition `k`, cap physical state reads post-integration `qpos/qvel` at `20*(k+1)` ms. These channels are unchanged by the final forward for the saved no-reset/no-state-write evaluation. In contrast, the contact/torque sensor evidence belongs to the just-completed last solver: raw step `4*k+4`, prestate index `4*k+3`, solver time `20*k+15` ms. It must not be replaced with a new contact query at the 20 ms poststate. [`mujoco_warp.step`](D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp/_src/forward.py:1364) performs forward then integration; the saved solver samples explicitly retain this distinction.

The 15 logits are **hand self-collision**, not hand-object contact/support. [`PenSimReader.contact`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:347) concatenates 10 `link2/link3` found flags, followed by five `link4 OR fingertip` flags. The compiled 25 underlying sensors match each finger geom against the palm body subtree. Link1 flags are not directly included in the 15 targets. Sensor found is a matched-contact count comparison, not positive normal force.

The raw files save geometry IDs, distance, position, frame and solved force, but not `sensordata/found`, `contact.type`, or `efc_address`. The Warp [`_contact_match`](D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp/_src/sensor.py:2375) checks the contact's `CONSTRAINT` bit. Therefore direct native 15-label truth is **UNKNOWN** in this saved-only contract. There are zero-normal-force contacts (18 A1L, 15 P_ONLY, 18 A2); force-thresholding would change semantics. Neither hand-object contacts nor geometric penetration depth may be substituted for native self15 labels. A future conditional label reconstruction would need an explicit contact-type/sensor equivalence certificate or direct logging at the capture phase; this review grants neither.

Consequently existing saved data support state prediction errors and descriptive self15 logit inspection, but not native self15 BCE, calibration, precision/recall, or classification success claims.

## Strict104 CPU WM consumer

All three final checkpoints are actual format 104, iteration 2000. Each WM has input 298 = physical state 58 + fixed simulator context 46 + disturbance 174 + actual executed target 20; output 67 = tangent delta 52 + self-collision logits 15. Hidden widths are `(512, 512, 512)` and all WM tensors are finite. The task/FailureHead context width 368 is a different interface and must not be passed as WM dynamics.

[`PenObservations.dynamics`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:432) concatenates state, simulator context and disturbance. [`disturbance`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:413) selects the 174 external-wrench components from auxiliary observations, excluding the 21-element palm/prefix block. The controlled evaluations save zero applied external force; the separately validated consumer may use a zero 174 vector after checking this invariant and the pinned fixed model/context. Contact forces are endogenous dynamics, not an input disturbance to insert here.

A minimal new consumer can instantiate the actual [`WorldModel`](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:11) on CPU with `dynamics_dim=278`, `contact_dim=15`, saved hidden widths and `amp=False`, under a preserved CPU RNG scope. Strict-load only `saved['modules']['wm']`, retain all norm buffers, set evaluation/frozen parameters, and call `model(x, actualu, cat(x, xi, zeros174))`. It must validate format 104, controller/reference/bundle/source/model/context identity and all shapes/finiteness before inference. The constructor's random initialization is not new training; preserving RNG avoids changing other CPU work.

Do not mutate checkpoints to format 3. The old [`FrozenPenWorldModel`](D:/workspace/supermimic-chord-study/src/chord_study/world_model.py:22) deliberately rejects 104 and the old rollout helper expects different world-velocity fields. The old format-3 gradient-probe phase indices extend past this 16-frame reference. Neither is a drop-in consumer. CPU FP32 predictions also are not numerical parity with GPU AMP BF16 training/inference.

## Bounded next CPU diagnostic

No new data or physical sampling is needed for a first descriptive check:

1. Validate all three existing saved-policy audits; reconstruct each trajectory's 16 native state knots and 15 actual targets with the native CPU codec and record its assertions/reconstruction status.
2. Evaluate each of the three WMs on each of the three saved trajectories: a complete 3-by-3 matrix, 15 one-step transitions per cell. Keep each trajectory/source checkpoint and temporal index in results. Include a persistence baseline. These are three policy trajectories from one training seed/reference; frames are not independent experimental N.
3. Report native tangent `difference(predicted, actual_next)` grouped into joint q/qd, object position, SO(3) tangent angle, linear velocity and angular velocity; retain both native training normalization and physical units. Shaft angle is a separate task diagnostic from full SO(3). Report all 15 self15 logits only, with truth `UNKNOWN`; do not call them CHORD contact/support predictions.
4. If desired later, fixed overlapping H4/H8 open-loop windows are a separate descriptive test (12/8 windows per trajectory). They do not establish action gradients, contact-force accuracy, physical stability, or generalization. No actor/RNG/environment is needed for these saved-input tests.

Physical causal direction tests remain a separate future protocol. The saved raw qpos/qvel do not contain full `mjSTATE_INTEGRATION`/warmstart, full controller/Manager state or the exact counterfactual restore contract. An existing format-3 local-gradient probe cannot certify this 104/domain merely by changing its loader. No causal action-gradient comparison or new physics is authorized by this interface review.

The prior local contact-proxy and compiled-precision failures remain part of the research evidence. Current three single-seed 0.3 s task passes are useful saved outcomes, not full151 motion success, true ego observation qualification, contact-force prediction qualification, or completion of the overall goal.
