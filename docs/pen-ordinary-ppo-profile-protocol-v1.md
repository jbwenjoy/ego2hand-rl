# Ordinary32-world collector/PPO profile,registered before execution

Use the actual maintained SupportPenRuntime/SupportPenPPO,centered support.2,
32 worlds,64 controls and32 Adam steps per update. Construct a NEW current-code
fixture;do not load historical native state or silently transfer weights. Use
existing7411 seed,initial mean/std and8-control calibration. Warm two ordinary
iterations to establish kernels/Adam;each must pass original consistency gates.
Save a full current-source checkpoint at that boundary,including native/task/
cache/modules/Adam/normalization/memory/counters/RNG. Not the final64 actor.

From this SAME fixture restore before CLEAN_A,CLEAN_B,CPROFILE,TORCH_CPU_PROFILE,
CLEAN_C. Cold restore/source/byte/observation validation stays outside timing.
Each case executes collect64,original consistency,update32;no skipped checks,
changed resets,aid,geometry,phase,stream,solver or physics settings. Preserve all
outcomes and actual work. Record phase wall and CUDA event timing with explicit
boundary fences,reset-world/termination counts,full rollout and final modules.
Numerical differences across restores may occur;do not assume future bitwise
physics or replace a case. Three clean repeats are timing observations,not
trained seeds or task-success evidence.

CPROFILE records Python/native call counts and cumulative/self time separately
from clean timing. TORCH_CPU_PROFILE deliberately captures CPU operator/API
durations/counts and a trace,without claiming a GPU-kernel timeline or occupancy.
CUDA event time and optional device-wide utilization telemetry are separate.
Profiler overhead is not a speedup. Analyze scalar extraction/nonzero/transfer,
runtime pending/reset checks,reference validation,geometry/contact sampling,
native solver/graph launches and PPO updates before choosing a change. Do not
attribute a GPU bottleneck from an unmeasured static suspicion.

If all runs complete:calibration2048 native +warm32768 +five cases81920 =116736
native world integrations. Optimizers64 warm+160 measured=224;the fixture cursor
returns to64 before each measured case and96 after,not224 cumulative live state.
Count executed work rather than saved cursor. No Actor promotion,new task/ego
benefit or new WM result follows. Keep previous failures and all frozen sources.
No installed-package,driver or CUDA changes. A measured optimization needs its
own matched clean timing and source/state/geometry/reward/reset verification.
