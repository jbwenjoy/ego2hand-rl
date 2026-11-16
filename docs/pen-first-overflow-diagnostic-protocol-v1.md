# Bounded first native-overflow diagnostic V1

This is numerical diagnosis, not another long training or a final arm comparison.
Preserve the original25 collection checks, full native checkpoint guards,
objective, cached feature mode, physics, environment and both historical failures.

Load the valid RESIDUAL201 checkpoint from recoveryV2 once through ordinary
full native/NN/Adam/RNG restore. Keep the named main.step interface. A separate
external observer owns all scratch; nothing is attached to runtime inventory.
On the owned Torch/Warp stream, copy pre-operation fields, execute the original
STEP or FORWARD, then latch each world's FIRST nonzero raw native int32 mask,
event counters, pre/post phase/time/qpos/qvel/warmstart/ctrl and geom poses.
The latch survives reset. No per-operation CPU scalar reads are added. Field
bytes are exact; this is not a complete solver-state or collision-pair trace.
An overflow already set before an operation is explicitly visible in pre data.

Before physical replay, qualify the observer using synthetic CPU and GPU
events, including reset and a later conflicting mask, and unchanged live
buffers. Stream order is explicit. Retain all failed startup/QA work. Unchanged
W32 future trajectories are not guaranteed; altered scheduling can expose
different collisions even with no physics writes. Do not claim reproduction.

Fixed maximum99 collections202..300, 64 controls,32 worlds,8 native steps per
control: at most1,622,016 native integrations. At most98 valid diagnostic PPO
updates /3136 Adam calls (no update at the last collection). Stop at the FIRST
collection boundary with any observed event OR any failed original check,
before updating that collection. No batch with an intermediate overflow may
be optimized even if an end-boundary flag was cleared. Preserve the final raw
memory and explicit NOT_RESUMABLE policy/native diagnostics, observer packet,
per-collection health, actual attempted/completed native/Adam counts and walls.
No fabricated success report, engine/source metadata rebinding or overflow
gate relaxation. Exception path also retains counters and available evidence.

Pin original host MJB and all current native Model arrays/scalars for later
collision fixture construction. Post geom transforms are from the operation's
latest collision forward, whereas post-step qpos may have integrated. No exact
faulting geom pair is inferred solely from a per-world bit; reduce candidates
in a separately admitted collision diagnostic. Installed engine/driver/CUDA
remain unchanged. Isolated capacity variants require positive controls and a
new state/physics contract before any new matched control trial.
