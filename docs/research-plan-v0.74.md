# Common-source physical identification after the finite-response loss ablation

Previous turn is PROGRESS: four matched POINT/RESPONSE x DIRECT/IMPULSE models
reach2000,97/97 healthy at100/2000, POINT controls reproduce frozen V2 exactly.
Point predictions improve but physical AD remains near zero-response quality
and IMPULSE pooled AD worsens.185 tests pass. Resident full dynamic-array
restore also passes14/14 on actual native GPU Data(169 arrays/8 scalars), with
14/14 subsequent actual force receipts. Copy fidelity alone does not prove
physics determinism or20ms/later-source replay.

The next action is physical identification, not another loss-weight/epoch
sweep and not a learned-model Actor. Preserve the full Goal and all frozen
data/model roles. No process is currently running.

1. Capture one ACTUAL original no-aid source at each control anchor0/2/4/6
   (0/40/80/120ms), both native dt, from the selected pose/fixed clip with the
   existing strict numerical policy and original baseline raw2*delta. Save
   full dynamic arrays/scalars, command phase/controller EMA, actual held PD
   targets and fixed actual model arrays/scalars. Check required state fields
   and no unsupported activation/history/plugin/sleep assumptions.
2. Build an INDEPENDENT compatible Data fork plus its own step graph. Preserve
   array layout, dtype, capacity and alias semantics (or reject unsupported
   aliasing). Existing ResidentDataSnapshot.restore intentionally requires
   original pointers, so it cannot be misused to restore into a new graph's
   allocations. Copy source bytes explicitly to the fork. Never restore into
   the live evaluation state after episode start; captures are read-only.
   Prove all live source/model bytes unchanged around diagnostic work.
3. Predeclare diagnostic-only responses: one unit-L2 direction from new local
   PCG seed7801, PD amplitudes0.002/0.005rad, both signs plus baseline,3
   execution repeats at every anchor/dt (120 total20ms responses). No learned
   model chooses cases or perturbations. Hold actual absolute PD targets over
   all4/8 physics ticks, preserve time/gravity/no applied force/no nonhand aid.
   Record actual clamps; clipped responses never become action derivatives.
4. Compare EVERY baseline repeat/tick with untouched actual native execution
   under the SAME executed targets. Reuse original tolerances in
   `configs/pen_gpu_step_audit_v1.toml` (joint0.002rad,object0.5mm/0.01rad,
   joint velocity0.2rad/s,object velocity0.01m/s/omega0.2rad/s,ctrl2e-7rad,
   EFC force1e-4N/torque1e-5Nm), with existing numerical impulse5e-7Ns.
   Compare proper indexed/world/body quantities, not raw contact row order.
   Preserve failed ticks/gates; passing byte copies or one initial step cannot
   stand in for20ms fidelity at all later sources.
5. Only after this check can repeated +/- contrasts be interpreted with an
   observed execution-spread/amplitude-stability assessment. All120 are
   DIAGNOSTIC_NEVER_FIT and not independent final task/ego samples. Any future
   TRAIN cohort needs fresh roles declared before collection, not promotion
   of favorable diagnostic records. No silent physics or threshold edits.

Cube/pen native no-aid baseline control/learning and full support-objective
A1/A2 remain required alongside WM validation, ego input/assets, multi-trajectory
total costs and hardware contracts. Stable task control cannot be inferred
from any local derivative, support LP or short failed prefix. Goal ACTIVE.
