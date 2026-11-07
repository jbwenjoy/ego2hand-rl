# Independent saved-only WM diagnostic audit

The final saved-only audit passed 68/68 checks: [report_v2.json](D:/workspace/supermimic-rough-prefix-learning-study/outputs/WM_CPU_diagnostic_saved_independent_v1/report_v2.json), SHA `3c4e58104981ed4b4f6f8c1879d5bfa377e15df207ed45dfe6e140789f4d5ef0`. It used NumPy and the standard library only. No checkpoint tensors/model were loaded, no WM forward or autograd/FD rerun occurred, and no environment, physics query or GPU was used. All original diagnostic files and sources remain unchanged.

The input is the existing completed [WM_onpolicy_CPU_diagnostic_v1](D:/workspace/supermimic-rough-prefix-learning-study/outputs/WM_onpolicy_CPU_diagnostic_v1/report.json), bound to runplan `f0076db1c26954d324878933cccb36e942f28ac4cdf800d5b16eaa5208867ef6`, protocol `18ef4ad5b39c8ea4ef178665c78d44e9cb3f34f59a56ca2917b9403defd415f7`, entry `4812b8dafdec7e9a2a2c0cf01946650e7df748714d2b2c0b9657e44bf67acdb3`, codec `8321538887c660456fc50ce8fc37d3b130760212fa63c905310bbd6a079d5156` and WM loader `ab15f23b569d259cbca587e2b6f883c509ae15584567fcc8c27443787f294a20`.

## What was independently checked

- Current hashes matched every registered source/input pin (374 unique files at that checkpoint in the audit), including actual final2000 model files and each saved-data binding. Source copies match executed registered bytes; archived JSON copies match the original pinned documents semantically.
- All 92 NPZ arrays are finite. Encoded `x/x_next/u` are byte-equal to the earlier actual 32-check reconstruction outputs. All four paths (A0, A1L, P_ONLY, A2), including quality-failing A0, remain present. The 15 executed targets match all four corresponding raw substeps. No startup control replaces the first learned target.
- The full 3-model × 4-path matrix contains 180 predictions. All eight identity/constant-velocity baseline groups, 300 phase rows and 600 k/k+1 reward rows are complete and unique. Summary groups exactly match the saved metric table.
- Every saved state-error scalar/per-joint metric and 1 mrad joint-limit mask was recomputed from saved predictions and truth. Identity is exact. Constant velocity uses `Exp(dt*omega_palm) @ R` on the **left**, since omega is in the fixed palm frame. Its maximum independent FP64 versus native FP32 array difference was `8.5797865e-8`.
- Native quaternion-based FP32 `so3_log` results were compared with an independent FP64 `atan2` angle from relative matrices. Maximum per-phase angle difference was `3.8995511e-8 rad`; it is retained as arithmetic roundoff, not corrected in original results.
- Object-only position/endpoint reward terms were recomputed for current-reference phase k and secondary tracking phase k+1. The saved four FP32 terms reproduce the saved sum difference and the externally applied `dt=.02` exactly. This is the declared object-only reward diagnostic, not a regrade of the physical task or the complete native reward.
- All 288 predeclared FD cases, PRNG vectors/order, FP32 ±epsilon targets, exact domain masks, skipped nulls, gradient-dot-direction AD values, effective FP32 directions, discrepancy formulas and paired-epsilon disagreement were independently checked without executing a model. No target was clamped or resampled by the reviewer.

## FD coverage and evidence limits

| WM | Path A0 valid / 24 | Path A1L valid / 24 | Path P_ONLY valid / 24 | Path A2 valid / 24 |
|---|---:|---:|---:|---:|
| A1L | 20 | 8 | 8 | 0 |
| P_ONLY | 21 | 8 | 8 | 0 |
| A2 | 24 | 8 | 8 | 0 |

There are 113 domain-valid pairs and 175 skipped pairs. Every skipped pair keeps `FD`, discrepancies and effective-direction AD as null. All 24 pairs on the A2 trajectory are skipped for each WM; their finite autograd gradients do not supply an FD comparison. The largest saved valid absolute AD–FD discrepancy is `5.4925039e-4`; the largest relative discrepancy is `0.0813929351`. These have no predeclared pass/fail accuracy threshold and do not imply a physical action-gradient qualification.

The FD rows do not save the two ±epsilon reward values. Thus this reviewer independently verified AD from the saved gradient and all FD discrepancy/domain formulas, but could not independently reconstruct the FD slope itself without a forbidden model rerun. The executed pinned source supports how those slopes were computed.

All three strict loader receipts refer to the intended final104 checkpoints. Saved wrapper-state before/after digests are equal. Those wrapper digests include `physical_xi` and intentionally differ from the loader's WM-only digest namespace. Model/RNG in-memory before/after raw states are not separately serialized: their unchanged flags are source-backed runtime evidence, while the checkpoint/source file bytes were independently rehashed. This audit does not overstate those flags as a new raw RNG/model-state recomputation.

Native self15 truth remains **UNKNOWN**. Saved BCE/F1/accuracy remain null; probability arrays are merely sigmoid(logits). No hand-object contacts, force thresholds or declared geometric intent targets were substituted as self15 truth. State prediction accuracy, internal AD–FD consistency and this audit do not establish contact-force prediction, same-backend physical Jacobians, full151 motion, new ego data, or method benefit.

## Preserved first audit and checker correction

The first audit's [report.json](D:/workspace/supermimic-rough-prefix-learning-study/outputs/WM_CPU_diagnostic_saved_independent_v1/report.json) and [audit_saved_cpu.py](D:/workspace/supermimic-rough-prefix-learning-study/outputs/WM_CPU_diagnostic_saved_independent_v1/audit_saved_cpu.py) remain unchanged. It recorded 65/67 checks; both failures were reviewer comparisons:

1. It compared the raw hash of a reserialized archived JSON document against the original document's raw pin. The original raw file hash and semantic archived content are both valid; v2 checks them separately.
2. It applied an absolute/relative comparison budget to a near-zero difference of two roughly order-10 reward sums, ignoring cancellation. V2 instead verifies the original FP32 sum/subtraction/dt exactly from the four saved terms, while keeping the independent FP64 geometry difference descriptive. The maximum FP32-versus-FP64 cancellation difference, `1.7790442e-6`, is retained in `report_v2.json`. All independent reward component comparisons satisfy the original declared arithmetic budget; no tolerance or original result was changed to obtain a pass.

The audit's fixed F32 arithmetic budgets, all numerical discrepancies and limits are recorded in `report_v2.json`. The original diagnostic's 4.05936 s internal wall and this audit's wall are distinct nested/scoped costs; this report makes no acceleration or equal-compute claim.
