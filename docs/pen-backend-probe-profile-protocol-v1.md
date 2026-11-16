# Current-backend strict restore/branch attribution V1

Freeze this diagnostic before executing new native work. Use all36 actual new-
backend TRAIN/development-validation sources with original source/engine/actor/
Model/hidden-backend metadata. Old source-epoch anomalies remain unresolved;
do not enable read-locked/source-epoch mode or import old backend states here.
No code/config/driver/CUDA/physics change;no optimizer or trained policy update.

For each actor construct fresh LIVE W1 under the exact explicit backend,restore
every one of its18 source states once as warmup then3 measured/cProfile strict
restores.108 measured+36 warmups=144 full restores,0 native integration. Keep
array/scalar and observation readback guards. File load outside these restore
timings,as in the original per-source branch loop. Collect separate ATen operation
counts on1 restore per actor atTRAIN N8841 P024,not timed as ordinary restore.
No binding/source-check cache or relaxed guard;all checks retain defaultSTRICT.

At R/S N8841 P024 use the original10 profiles:2BASE repeats plusdirection0 at
both original PD eps.002/.005,2 signs,2 repeats. Execute each once uninstrumented
then once withcProfile:40 controls/320 native integrations. Every branch begins
with full strict restore and exact observation check. Copy/record all8 ticks
using original OwnedProbeRecorder and retain all packets/windows. Total restores
144+2 dispatch-count probes+40 branches=186,0Adam/Actor. Constructors are counted
if any native step unexpectedly occurs;do not subtract hidden/failed work.

Record top-level nonoverlapping host stage intervals:strict restore+observation,
recorder begin,step including8 recorder callbacks,finish/window copy,compression
and final synchronization. Boundary synchronizations/cProfile affect execution;
these are instrumented attribution,not active GPU time or an optimized speedup.
Whole branch includes bookkeeping/hash overhead beyond stage sum. Nested cProfile
cumulative times overlap and cannot be summed;exclusive self time includes waits.
ATen dispatch is separately perturbed and excludes Warp/driver work. Existing old
collection/restore profiles are context only,not new-backend measurements.

All original online geometry/no-aid/PD/capacity flags retained. Independently
decode every new profile packet with maintained EFC/force/impulse/state/warm
math and original1e-4N/1e-5Nm/5e-7Ns/nohand/geometry masks. Preserve failures;
profile is not newly admitted TRAIN/validation data. Compare same-source input
bytes and observed endpoints across instrumented/uninstrumented repeats without
claiming deterministic W1 future. No row is promoted by a passing restore.

No simulation/source optimization is admitted by timing. Future performance
changes need separate matched-budget/numerical tests,including prior unresolved
source-epoch evidence. Full task/M0–M5/A1/A2,multi-reference/seed costs,ego asset/
gravity/dynamics/RGB and hardware requirements remain active.
