The saved physical engineering audit passes **194/194** checks in [report_v2.json](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_saved_independent_v1/report_v2.json). This is an independent NumPy/stdlib read of terminal artifacts, source bytes and ZIP contents. It does not execute an environment, MuJoCo, Warp, Torch, a world model, a physics query or an optimizer.

The audit binds the actual OS-launched configuration SHA `02753b7defccc30a5b8bf9a1ee32c5e2bbc17f1dba8280765af6b96e1fd81c15`, all 415 current source files and their 415 ZIP entries, and all 485 recursive configuration pin occurrences. It checks the saved worker PID/parent/creation chain, normal exit, no timeout, and the post-terminal CIM evidence for the four owned PIDs. The old manifest's copied-config-path rejection is retained separately; it is not treated as a physical failure or used as qualification here.

The complete raw scope is 16 snapshots, 560 restore transactions with 1,120 begin/end event receipts, 48 baseline branches, 512 perturbed branches, 2,480 scheduled substeps, and 620 reward captures. Each path has 903 raw capacity records: 60 donor substeps, all branch begin/substep/end reads, the initial pre-reset read, donor-terminal read and one final read. All four external host OR ledgers independently reconstruct to zero, with no unknown reads. Public interval forward count is 620, with eight reset forwards separately recorded. The four constructor scopes additionally report eight nested Warp step API calls, eight nested Warp forward API calls and four CPU forward calls; those API scopes must not be added to public graph replays as a kernel count.

All 48 original baseline gates pass. The largest errors against each fresh donor transition are:

| Channel | Largest error | Frozen tolerance |
|---|---:|---:|
| q | 5.96046448e-8 rad | 2e-5 rad |
| qd | 9.53674316e-7 rad/s | 2e-4 rad/s |
| object position | 0 m | 2e-5 m |
| rotation matrix entries | 0 | 2e-5 |
| object linear velocity | 1.86264515e-8 m/s | 2e-4 m/s |
| object angular velocity | 7.15255737e-7 rad/s | 2e-4 rad/s |
| primary native FP32 reward | 0 | 2e-5 |

Three-repeat baseline contact geometry, pairing, constraint type, raw found values and self15 labels agree exactly. Some solver forces and continuous states are not bitwise reproducible: the maximum contact force6 difference is `3.27825546e-7` and the maximum efc-force difference is `6.92903996e-7`. They are preserved per snapshot in the report. Force bitwise equality was never a preregistered baseline gate; no new tolerance or threshold was introduced to obtain this result.

All 2,480 actual controls match the declared held FP32 target, with four substeps per phase and zero logged exogenous force. The beginning qpos/qvel/time of all 560 restored branches matches its fresh donor state exactly. Actual eight-field model context is checked against the authoritative context row, after both copy and `set_const_0`, for all four paths. Full initial qpos/qvel/ctrl/mocap matches the archived setup. The 16 snapshot NPZs include integration, PEN memory, native Warp/Torch buffers, explicit arrays and RNG buffers. Their independently rebuilt value hashes match every live restore-inventory receipt. There are 3,392 saved empty Warp-array occurrences, so the corrected empty-buffer path was actually exercised.

The distinction between evidence levels remains explicit: saved arrays can independently reconstruct inventory digests, initial/raw state and raw-control equality. Python object/container identities, closure cells, pointer identity, and live RNG restore equality were asserted in process by the pinned helper. Complete post-restore Python object graphs are not serialized, and this audit does not replay them.

Solver COM and position equal the preintegration raw state exactly. Native body xquat differs from the unnormalized raw qpos quaternion in 475 samples by at most `1.19209290e-7` per component; after normalization the maximum component difference is `2.40274604e-8`. This is a kinematic FP32 arithmetic relation, not a different force phase. Raw state is postintegration; contact, actuator force and self15 labels retain the original reward-capture/last-solver phase before Manager forward. Raw sensor cache contents are not relabeled as newly computed post-forward contacts.

All 620 native self15 vectors are zero, independently reconstructed from the 25 raw found values using ten middle-link components and five distal-or-tip components. These are native self-contact labels, not hand-object labels or force estimates. This data alone cannot establish positive self-contact classification performance.

Fresh donor/archive differences are recorded in six channels for all 61 states, the four selected snapshot states and four selected next states. A0 is exact; A1L, P_ONLY and A2 have nonzero archived drift, including A1L's body-angular-velocity maximum difference of `2.95809919 rad/s` over its full trace. The protocol compares branches to their fresh same-world donors; these differences do not select a donor, replace a trajectory, or regrade any archived task outcome.

Costs are nested scopes: `70.283916 s` native worker section, `84.007373 s` total worker, and `85.188000 s` watchdog. They must not be summed. Complete physical sampling and repeatability gates permit the separately authorized CPU comparison to consume these observations; they do not establish physical Jacobian accuracy, gradient alignment, learning benefit, full action-domain coverage, or independent-trajectory generalization.

The original [report.json](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_saved_independent_v1/report.json) and [audit_saved_cpu.py](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_saved_independent_v1/audit_saved_cpu.py) retain the first reviewer run, 162/165. Its failures were an obsolete donor-drift field name, an inappropriate raw-qpos/xquat bitwise predicate, and an invented solver-force bitwise predicate. The new checker corrects only those reviewer assumptions and adds actual six-channel drift reconstruction. It does not modify raw data, production code, original numerical gates, the physical execution, or the old failure artifacts.
