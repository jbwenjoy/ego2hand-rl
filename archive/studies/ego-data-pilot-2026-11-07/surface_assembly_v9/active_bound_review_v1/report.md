# v9 active bounds and selected-point geometry — saved description

All four final cases and all32 optimized-joint rows are retained in `report.json`; all four common-A direction calculations are in `antipodal_geometry_v1.json`. This analysis read saved JSON only. It ran no LP, FK, model, collision query, physics or learning. Near-bound labels use predeclared numerical descriptors (1e-6 rad for q/rotvec and1e-7 m for translation), not new physical gates.

| Case | Native4ray scaled loss | Minimum torque slack Nm | LP finger1 / finger2 Fn N | Algorithm bounds reached |
|---|---:|---:|---:|---|
| seed0/rank0 | .218302700 | .0975550 | 1.11392 /2.23983 | index q1 true-hard lower; q2 lower trust; q3 upper trust; q4 lower trust |
| seed0/rank1 | .219577147 | .0998798 | 1.10946 /2.24809 | index q1 true-hard lower; q2 and q4 lower trust |
| seed1/rank0 | .218302700 | .0975550 | 1.11392 /2.23983 | index q1 true-hard lower; q2 lower trust; q3 upper trust; q4 lower trust |
| seed1/rank1 | .219821890 | .1002826 | 1.10909 /2.24975 | index q1 true-hard lower; q2 and q4 lower trust |

In all four, index q1 is−1.046999999 rad, the actual−1.047 hard lower limit plus the original1e-9 numerical planner margin; it cannot be relaxed beyond the physical limit. Index q2=.11698231918 and q4=.64707769289 are source−.45 algorithm limits, not true lower limits (−.698 and−1.047 rad). The two rank0 q3 values1.83569266831 hit source+.45, below the actual upper limit2.094 rad. Rank1 q3 values retain .02043/.02388 rad of algorithm margin. No thumb-joint algorithm bound is near active.

None of the six phone t/rotvec bounds is near active: minimum translation margins are3.680–4.025 mm and minimum rotvec-coordinate margins .1540–.1682 rad. The remaining12 joints are bit-identical to the source. The current LP witnesses have positive torque slack and Fn below3 N; these results do not support blaming the present witness residual on a tight PD/normal cap. They also do not prove those constraints irrelevant elsewhere.

The narrow next range is to remove the artificial source±.45 q8 box while retaining every actual hard joint range plus the existing1e-9 planner margin, the same10 assembly equations, designated-pad window, other20 phone/189 self inequalities, phone t/rotvec boxes, wrist and physical parameters. Exact proposed q8 ranges are saved under `next_range_proposal.q8_range_rad`. Existing four feasible finals can initialize that new bounded local search; it would still not be a global feasibility test.

There is a separate fixed-material geometric issue. For the line from common A_thumb to A_index, the thumb inward-force normal makes58.314387° (tan=1.620048); the index normal makes5.868208° (tan=.102779). These values are essentially identical across all four. The optimistic outer circular μ=.7 cone has half-angle34.992020°: the selected thumb's collinear antipodal direction is outside even this outer cone. Passing the outer-circle test on the index side would not certify the native four-ray cone. Whole-grasp rotation along the same fixed-material assembly manifold does not materially change these relative angles, although it can change the direction of gravity relative to the grasp.

The phone triangle thumb anchor is `[.0344433,−.0033321,.0497725] m`, with outward normal `[.483614,−.875282,~0]`, whereas the original ego scan target is `[.0363767,−.0005921,.0532241] m`: a4.812 mm displacement. The index displacement is4.738 mm. This provenance difference can change the selected face/normal; no original human contact normal or force has been measured here. It should be investigated alongside, rather than hidden by, the active trust limits.

This two-point direction descriptor does **not** evaluate the seven actual native contacts, the entire hand's force closure, or gravity-specific support. Antipodality is not imposed as a necessary2 s holding condition; two torsionless3D point contacts do not themselves certify full6D force closure. All four actual strict witness solves failed and no holding case was enrolled. The frozen local optimizer results are retained, with no unique failure cause or global impossibility claim.
