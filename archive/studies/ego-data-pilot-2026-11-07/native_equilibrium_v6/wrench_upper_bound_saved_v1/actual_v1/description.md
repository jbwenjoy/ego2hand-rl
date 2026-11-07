# Fixed three-contact optimistic wrench bound

The two preregistered LPs used exactly seed0/rank0's three generated native contact points and frames: one thumb contact and two index contacts. All were retained regardless of saved normal-force magnitude. For each point, four force rays are `sign*(normal ± .7*tangent1/2)`, with nonnegative coefficients. Moments use the saved world COM. There is no free contact torque. The first LP caps the sum of all normal coefficients belonging to each finger at 3 N; the second removes only those caps. Both minimize the maximum absolute residual after scaling force by 1.962 N and moment by .151074 Nm.

| Case | HiGHS status | Minimum scaled L∞ residual | Thumb normal N | Index normal N |
|---|---|---:|---:|---:|
| Each finger ≤3 N | Optimal | .282214751819 | 1.540341974 | 2.648421688 |
| No force cap | Optimal | .282214751823 | 1.540341974 | 2.648421688 |

The residual is approximately `[-.553705,+.553705,-.553705] N` and `[-.0426353,-.0426353,-.0101117] Nm`. Removing the caps does not remove it, and the capped optimum is already below both force caps. Numerical inequality violations are about 1.4e-11 and 6.7e-12 in the corresponding LP constraints; these describe solver arithmetic, not a new physical acceptance gate.

Within this fixed saved point/normal geometry and native pyramidal friction convention, the numerical optimistic force allocation therefore does not balance gravity exactly. This is a local geometric obstruction under the stated model, even before imposing joint torque/PD realizability or the soft-contact force law. It does not establish global infeasibility: changing actual contacts, contact locations/normals, or robot configuration is outside this two-solve experiment. No native model, forward, integration, NN or GPU was called; no actual holding reference is qualified.
