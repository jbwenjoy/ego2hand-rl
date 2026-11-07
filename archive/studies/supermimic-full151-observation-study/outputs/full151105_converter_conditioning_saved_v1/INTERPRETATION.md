# Saved converter conditioning analysis

The original eight-eps32 full-state gate remains **FAIL**. This saved-only analysis evaluates the equations frozen in `FREEZE.json` (`2e1b9af7…f689e`) before the new diagnostic raw errors were read. It performs no state conversion, model/objective construction, MJB load or physics. Its float64 formula evaluation is a conditional diagnostic, not an outward-rounded interval certificate or a replacement acceptance rule.

All 2260 rows are retained: 1808 solver-pre rows and 452 post20 endpoints. The component CSV has 40,680 rows (18 components per state), with native CPU output, float64 shadow, conditional one-side envelope, operand scale and zero-output status. All 452 CPU/GPU endpoint differences, old bounds and flags are retained. The 160 leading anchors are compared separately.

| Donor endpoint | Feature | CPU−GPU (rad/s) | Old bound | Frozen conditional pair envelope | Error / envelope |
|---|---:|---:|---:|---:|---:|
| A0, k21 | omega0 | −1.251698e−6 | 9.536743e−7 | 1.480883e−5 | 0.08452 |
| A2, k10 | omega2 | −5.245209e−6 | 3.800624e−6 | 6.353153e−5 | 0.08256 |
| A2, k14 | omega2 | −4.291534e−6 | 9.536743e−7 | 3.838521e−5 | 0.11180 |
| A2, k40 | omega2 | −1.311302e−6 | 1.236352e−6 | 2.064109e−5 | 0.06353 |
| A2, k45 | omega2 | −1.668930e−6 | 1.429351e−6 | 3.551776e−5 | 0.04699 |

These are five elements in five rows, all unperturbed donor endpoints (`branch_id=null`). They are not five different semantic error categories. The remaining endpoint q/qd/p/v differences are zero; R differs by at most 4.768372e−7 and passes its original gate. All original 160-anchor and 1808-solver-pose gates pass, as independently established by the raw saved auditor.

Every tested component lies inside the evaluated conditional envelope: CPU versus float64 for all 2260 rows, GPU versus float64 for 452 endpoints, CPU versus GPU for 452 endpoints, and CPU versus GPU for 160 anchors. Across all endpoints, the largest angular CPU/GPU residual/envelope ratio is 0.12753, and the largest R ratio is 0.17061. The five originally failing angular components have positive conditional pair margins of 1.36e−5 to 5.83e−5 rad/s.

The envelope does include upstream quaternion normalization and polynomial error, rather than only the final dot-product conditioning. The five failing components have nested absolute-product sums of 2.33–9.49 rad/s, output-conditioning ratios of 2.18–7.38 and maximum BODY angular inputs of 5.01–23.44 rad/s. Their ratios are moderate; describing all failures as extreme cancellation would be unsupported. Across all 2260 rows, the largest BODY angular component is 55.23848 rad/s, the angular conditioning ratio reaches 1317, and the angular CPU–float64 difference reaches 1.131237e−5 rad/s. The maximum endpoint GPU–float64 angular difference is 3.446272e−6 rad/s. Thus the CPU/GPU discrepancy does not establish which backend is closer to an exact transform at every row.

The saved CPU normalized quaternion differs from the independent float64 normalization by at most 8.428448e−8; its rotation matrix differs by at most 4.076092e−7. Both are inside the provisional normalization/polynomial model for every component. Quaternion norms range from 0.9999999374 to 1.0000001458. The producer float64 shadow and independent native-expression float64 shadow differ by at most 3.55e−15. Recorded palm position and rotation each have one distinct bit pattern across all rows.

This pattern is consistent with conditioning-sensitive normalization/polynomial/reduction arithmetic. It does not establish a unique cause: GPU normalized-quaternion and intermediate rotation values were not captured, and the specific CPU/CUDA reduction/sqrt/division dispatch and rounding assumptions remain unproved. Native `mm/mv` use elementwise products and sums, so TF32 GEMM cannot simply be named as the cause. Sparse small differences in R/omega, exact q/qd/p/v and passing anchor/pose checks provide no positive indication of a broad frame/address/phase mismatch; agreement alone cannot prove its complete absence.

## Root-selected next condition, not an authorization

Root selected a new **matched native GPU batch-one state replay**, preserving the original full58 qualification. The planned input is all2260 saved FP32 qpos/qvel/palm rows from this forensic run. The original unbound `SimReader.state` will be called individually with B=1, matching the original one-world full05 execution shape. The plan has one fixed mode, no fallback or mode search, no Env, physics, model/weights, MJB, factory or AD. It copies each output back as CPU bytes.

The new condition must require all452 endpoint and all160 anchor comparisons to pass the unchanged all58 rule: q/qd exact FP32 bits and original eight-eps32 bounds for the remaining18 components. The same1808 solver-pose crosschecks remain required. Any failed gate leaves eligibility false. No value is repaired, zeroed or silently replaced, and the old CPU run remains a preserved FAIL.

This is a new backend/shape-matched experiment, not a numerical relaxation justified by the conditional envelopes. It requires a new preregistration, source closure, config and explicit Root GPU authorization. The present saved-only analysis performs none of that replay and grants no dense-scoring, physical or gradient qualification.
