# Fixed material and native pyramid contract

The single passive MJB/context check passed 17/17. It created no MjData, environment, Torch/Warp runtime, contact query, LP or physical step. The contract binds the exact nominal MJB and the actual shared FP32 friction context; it does not infer material from observed or policy contacts.

There are 26 listed hand collision geoms, of which 21 are enabled, and three object geoms. All 63 enabled hand/object combinations have hand priority 1 versus object priority 0. With no explicit pair override or global override, the installed mixing code therefore chooses hand condim 4 and hand friction. Equal-priority componentwise maximum and solmix averaging are inapplicable to these pairs. Model cone is PYRAMIDAL (0). Exact nominal and actual values, masks, solmix, solref and solimp are preserved per geom in `all_candidate_materials.json`.

The actual friction vector is `[0.699999988079071, 0.699999988079071, 0.004999999888241291, 0.00009999999747378752, 0.00009999999747378752]`. Only the first three entries are active at condim 4: the first two are dimensionless sliding coefficients, while the third is torsional length in m = Nm/N. Rolling is inactive.

`material_cone_contract.json:/A_local6x6` stores contact components by ray columns, in row order `(Fn, Ft1, Ft2, Tn, Tt1, Tt2)`. Every column has normal force 1. The three pairs are sliding tangent 1, sliding tangent 2, and torsion about the normal. In particular, each torsion ray also has normal force 1. With nonnegative coefficients lambda in normal-force N:

`Fn = sum(lambda)` and `abs(Ft1)/mu1 + abs(Ft2)/mu2 + abs(Tn)/mu_spin <= Fn`.

This is not a Cartesian product of an eight-edge sliding cone and an independent torsion interval. Do not normalize the force parts to unit length. Transform direct torque as well as force, then add the COM moment `cross(point-COM, force)`. Torque normalization by the enclosing radius, if used, is applied once to both cone and demand.

The independently executed explicit native C ABI dim-4 proof passed 9/9 using arbitrary unequal fixture friction, not actual material values. Its seven decoder calls agree with this six-column formula. The earlier Python wrapper fixture failure remains preserved: official matching 3.11.0 wrapper source passes `mu.size()` as C dimension despite allocating for one higher dimension. This is a binding-size issue; neither SDK nor simulation code was changed. The links in `READY.json` preserve both results and the archived official source review.

Source projection normals and Frisvad tangents are a declared hypothetical frame, not an observed native contact frame. The finite pyramid depends on tangent orientation even when the sliding coefficients are equal. These data do not prove that any contact exists, is reachable, has available normal-force capacity, satisfies hand/actuator balance, or will occur dynamically. Disabled geoms, absent contacts and unknown confidence remain distinct from positive clearance and release labels. This contract authorizes no LP, training or physical run by itself.
