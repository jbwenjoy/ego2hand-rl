# Conditional COM-axis LP: no feasible proposal

The one frozen LP completed normally with HiGHS status2 (infeasible). `result.delta_bodyZ_m` and `result.proposed_ctrl20` are both null. No source, tolerance or constraint was changed and no second solve was attempted.

The LP held the original three world contact points, native four-ray frames/friction, hand Jacobians and robot q/PD context fixed. It enforced `sum(F)=-mg` exactly, all three moment equalities about `COM0+axis*delta`, each finger's summed normal load≤3 N, legal unsaturated inverse-control torque intersections, and `0≤delta≤.075 m`; it minimized delta. The moment column is `-cross(axis,Ftarget)`, valid because the net-force equality is hard.

The previously calculated approximately59.55 mm displacement removes only one two-point-line axial moment and is not sufficient for these full constraints. This result does not isolate whether another inequality or moment component prevents a solution, and does not rule out actual translated phone poses: their native points/normals/manifold and associated Jacobians can change. There is no proposed control or pose to consume as qualified. No native model, forward, physics step, NN or GPU was called.
