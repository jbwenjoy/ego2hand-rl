# Longer real-control baseline — protocol v1

Register before current-code qualification/training. Main64-round policies
fail the full7.36s controlled pen task. WM loss/gradient diagnostics have not
qualified an imagined Actor. Independently test a larger MODEL-FREE budget;
do not use any unqualified WM in action selection,updates,rewards or resets.

Same full-pose161 actor/188 critic,20 raw actions,clamp/scale.5/EMA.5/PD,
20ms control/eight2.5ms native ticks,reference/joints/mass/inertia/friction,
geometry2mm/.2mm/.001rad and nonhand.1N gates,force/torque scales.3N/.003Nm,
point geometry penalty.25/failure2,centered support and weightsR0/S.2. Default
UPSTREAM driver;explicit qualified CACHED32 fixed-frame features. No installed
package,driver or CUDA changes. Actual pen movement comes only from contact.

Fresh shared initialization/seed7411,zero mean-head weights/initial raw bias,
std.05(bounds.02–.2),calibration8. Passing current-source target/score/frame/
fork/collector/native-state qualification required (4096 native,0 updates).
Old score evidence verifies critical live source and all frozen input pins.
No old64 NN/Adam/native/RNG is resumed. RESIDUAL and SUPPORT fork only declared
weight fields from one common untrained calibrated state.

Each arm2000x64 controls x32 worlds=4096000 transitions,32768000 native steps,
64000 Adam updates. Two arms total65536000 native/128000 Adam,plus separately
charged qualification4096. Equal work,one seed/reference,not independent seeds
or shared-world-model A1/A2. Train sequentially using the same maintained PPO.
Keep all collector health/geometry failures,normalizer counts,times and sources.
Source/config/protocol bytes stay fixed while this job runs.

Save initial,iteration1,every100 and2000 checkpoints and corresponding memories.
At first optimized checkpoint verify finite NN/Adam and full ordinary reload
readback of native arrays/scalars,NN/Adam/observations/memory/normalizers before
continuing. This is byte-readback,not W32 future-trajectory determinism. Both
arms receive the same declared first-reload intervention. Retain failures and
stop that run if any check fails. Never silently restart or overwrite artifacts.
Subsequent checkpoint cursor/finite state checks remain. No success claim from
increasing training reward or reaching later phases.

Report first healthy checkpoint and ETA;update ETA using median up to50 recent
loop times (startup/reload/next-arm setup/final evaluation excluded). The user's
historical first-checkpoint pause was served and explicitly resumed;continue
this registered budget under the active goal without introducing another pause.
No dependency on all WM directions passing is imposed on model-free control.

After fixed budget,fresh W1 mean-policy execution,original strict no-aid/geometry/
position/full-orientation/full-duration gates and repeated executions are
required. Admit weights only under explicit observation/action/joint/reference/
full plant/config compatibility;do not convert historical native state. CACHED
W1 still needs separate qualification,so evaluate with declared LIVE W1 features.
Render actual saved native poses with visible terminal freezes and repeat/task
failures. New video must reflect real new policy execution,not WM predictions.
Reserve original trajectories/independent trained seeds for later conclusions.
