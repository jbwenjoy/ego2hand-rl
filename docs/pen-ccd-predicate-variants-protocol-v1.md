# Predicate precision V1

Reuse exactly the44 pinned geometry inputs and their independently queried
native distances from the preceding recovery probe; no new reference query,
scene selection or data fit.220 pair queries,0 FK/integration/Adam.

Controls: installed24/factor5 and original math96/factor20. Three isolated GJK
source variants promote only tetrahedron _same_side cross/dot, _S3D cofactors/
sum/sign comparisons, or both to FP64. Shapes/state/simplex arrays and returned
barycentric coordinates stay FP32; original thresholds and all branch outcomes
are determined by the new predicate arithmetic. No closest-face retry variant
is included. All three use96/factor20 scratch. Preserve exact source diffs,
input identities, all flags/contacts/witnesses/distance errors and near-boundary
cases within1um. Separation false contacts must stay visible. No package/
driver/CUDA or production state change; fixture success is not full backend
admission and is not a policy/ego/generalization/speedup result.
