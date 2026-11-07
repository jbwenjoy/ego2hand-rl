# Current-pose directional gap features

`DirectionalGapProvider(geometry_catalog_v1.json).features(state58)` supplies all
22 phone boundary pairs in the frozen provider's order. It has no parameters,
action/force inputs, native calls, or future-state inputs. The source geometry is
the compiled collision geometry, not visual STL geometry or functional-pad truth.

All supports are expressed in phone BODY axes with the phone COM as origin.
Phone vertices are `geom_R71 @ compiled_vertex + geom_pos71 - body_ipos`.
Other mesh vertices use the provider's BODY-COM relative geom pose once.
Capsules use their actual centerline endpoints and radius; the table box uses
its actual eight corners. No diagnostic `mesh_pos/quat` is applied again.

For each nonzero direction, the scalar is
`(min_H dot(n,H) - max_P dot(n,P)) / norm(n)`. The descriptor is the maximum over
the finite catalog of phone triangle normals, their negatives, and BODY axes.
In ideal real arithmetic for full-dimensional convex supports, writing
`C=P-H` gives `gap(n)=-support_C(n)/norm(n)`. Maximizing over every unit direction
gives distance to C when disjoint and negative minimum translation depth when
overlapping. A finite subset therefore cannot exceed that ideal signed value;
it may be loose or substantially more negative. This is not a scan SDF, closest
point/normal, contact patch, force label, or a claim about contact over 20ms.

F64 is the numerical reference. FP32 is explicitly approximate. The exported
scale-based guard is a numerical heuristic, **not** directed rounding or a
proven total FK error bound. `directional_gap_known`, pose/binding knowledge,
and exact-gap knowledge are separate. Exact gap, closest/unique normal,
formal floating certificate, and full-H causal-zero masks remain false even
when the descriptor is positive. Argmax directions are maximizing planes,
not measured or unique contact normals. Min/max are piecewise differentiable;
their derivative at ties is not a unique physical derivative.

The bounded check uses four analytic support cases and two named saved
compression states in F64/FP32, totaling six pure Tensor FK calls, one analytic
support AD and one descriptor AD, plus two finite-difference descriptor calls
already included in the six. All44 pair results and differences from saved
exact query rows are retained; AABB-only rows are not treated as exact truth.
The 1nm F64 and 5um FP32 comparison allowances are arithmetic diagnostics,
not changed geometric eligibility thresholds. No native model/FK/forward,
physics, learned network, optimizer, or GPU is used.
