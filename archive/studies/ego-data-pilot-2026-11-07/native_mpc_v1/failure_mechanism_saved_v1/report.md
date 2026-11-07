# One MPC failure: saved author description

All seven control intervals, all 42 line-search records, every actual tick, and all 20 controls are retained in `report.json`. This is an author description, separate from the peer saved gate review. The script read two saved trajectory NPZ files and no model, framework, forward or physics step.

Both runs use the same original seed0/rank0 pose/control/COM and fine dt .15625 ms. MPC aborts after 822 steps (.1284375 s), constant-u after 784 (.1225 s); both exceed the 50 mm early-loss threshold and fail the original 2 s gate. At the same .1225 s tick, MPC has 44.838 mm COM drift versus constant-u 50.078 mm, but 38.405° rotation versus 32.744°. The additional 5.9375 ms before abort is not a successful hold or a method benefit qualification.

| Interval | Exact H2 hold-previous-u cost | Selected cost | Actual end s | COM drift mm | Rotation ° | Both pads >.1 N fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1035.978 | 1003.834 | .0200 | 1.185 | 1.043 | .187500 |
| 1 | 2497.872 | 2432.513 | .0400 | 4.604 | 4.271 | .054688 |
| 2 | 4685.248 | 4631.802 | .0600 | 9.855 | 10.243 | 0 |
| 3 | 7813.849 | 7609.902 | .0800 | 17.197 | 18.790 | 0 |
| 4 | 11641.222 | 11641.221 | .1000 | 28.005 | 28.009 | 0 |
| 5 | 16881.959 | 16881.958 | .1200 | 42.723 | 37.249 | 0 |
| 6, partial | 23358.203 | 23358.203 | .1284375 | 50.107 | 41.150 | 0 |

Every interval's selected exact compound cost decreases relative to holding the last selected control from that same current state. This does not mean every component improves: for intervals 1/2/3, selected predicted terminal angle becomes 9.587/17.643/28.009°, versus 9.467/16.189/25.251° under the respective hold-previous-u branches. The final three cost reductions are only .000945/.000876/.000541. The compound objective can decrease while orientation and the overall actual trajectory worsen.

The first uninterrupted dual-pad positive-normal-force run ends at PRE tick14, 2.1875 ms, before the first 20 ms control boundary. There are later isolated reacquisitions: the final dual-pad sample is PRE tick189, 29.53125 ms. All dual-positive samples also exceed .1 N in these saved runs. MPC has only 31 such samples total; constant-u has 23 and its final isolated dual sample is 40.625 ms. These sample sets are not sustained contact durations. Thumb positive force ends at29.53125 ms; index positive force persists intermittently until75.15625 ms. No phone nonpad/table force or self-penetration is recorded. Missing generated contacts/force do not prove geometric clearance or identify a unique cause.

Controls remain inside the previous saved compiled control ranges; all interval trust intersections pass and the other twelve controls remain bit-identical to u0. Maximum per-interval change is .050000000000000044 rad, a floating representation of the .05-rad boundary. Maximum cumulative departure from original u0 is .2 rad; the trust rule permits accumulation across intervals. Bounds here are checked against the saved compiled interface, with unchanged-model provenance and runtime bound assertions delegated to the producer; this analysis does not reload the MJB.

Initial interval0's previous-control branch is the matched constant-u H2 rollout. At later intervals, “hold previous” uses a different current state and a previously selected control; it is not the original constant-u counterfactual at that state. The saved common-time full-trajectory comparison is kept separately. The source's six full-interval raw integration equality assertions are recorded; actual full 300-value integration vectors were not exported, so their complete scratch/history equality cannot be independently reconstructed from the qpos/qvel trace alone.

The narrow next evidence is the preregistered optimistic wrench upper bound at the unchanged rank0 contact geometry: all three actual generated points, native four-ray friction pyramids, with and without the fixed per-finger 3 N cap. This distinguishes a local point-force geometry limitation from failure of this bounded control search to realize support. It does not certify native soft-contact equilibrium, justify changing the original physical gates, or establish global infeasibility.
