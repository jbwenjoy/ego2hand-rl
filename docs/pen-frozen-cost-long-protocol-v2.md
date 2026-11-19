# Frozen Actor original/guarded-cost long W1 evaluation v2

This is fresh continuous evaluation of the existing INITIAL/RESIDUAL/SUPPORT
Actor parameters and frozen normalizers,not adaptation or new Actor training.
Keep the qualified face-normal collision backend,original OR stopping/50 caps,
physical parameters/reference/controller/reward and all physical/task gates.
W1 only;do not infer W32 or ego/hardware admission from this experiment.

First admit all three actors on both numeric backends with zero native steps:
explicit source/target identities,full historical W1 Model witness,policy161/
RAW20/joint order/reference/physical compatibility,exact parameter/normalizer
readback,fresh phase1 state and CPU-reconstructed initial observation. Old
native Data,optimizer,critic,RNG,counters and episodic state are never loaded.
Fresh runtimes may have backend-dependent derived solver caches;retain their
complete initial snapshots and compare physical starts rather than assuming
whole-Data equality. No inherited world-model/proposal controller is used.

Population is three numerical repetitions x two algorithms x three fixed
actors =18 complete declared executions. Reverse algorithm order in repetition1;
use original then guarded in repetitions0/2. All start fresh with the existing
seed/reference reset. These are numerical repeats of one training reference,
not independent data,random-seed replication or held-out generalization.
Each case has at most368 controls/2,944 native ticks/7.36 seconds. Whole cap
is52,992 native integrations,zero optimizer/normalizer work. Include admission,
setup and any failed/partial work explicitly;never resume/restart partial output.

Advance continuously without query branching or restoring states between
controls. Save full typed owned states at initial/end only. At every20ms control
record source qpos/qvel/warm/time,controller EMA/previous applied RAW,trajectory
and phase,cached object axes/fixed wrist mount,actual policy observation/mean
RAW,raw native/contact/EFC eight-tick packets and contact/reward windows.
The reduced mathematical view has a distinct evaluation-only schema and cannot
be passed to any checkpoint/restore loader. The full initial snapshot supplies
immutable model/reference/controller constants;compare them at the end.

Every control keeps the same raw decoder,physics/geometry limits and reward
reconstruction as the previous local comparison. In particular the linear
impulse gate remains5e-7 Ns;geometry limits remain2mm object,.2mm self and1mrad
joint excursion. Save all flags and the first invalid executed window. Stop
at a physical audit failure or task termination/reference completion;record
the unexecuted future explicitly as censored. A physical audit stop is not
automatically an object-drop/task failure. Never reset after a stop.

Maintain canonical strict tracking gates from controlled_evaluation_v4.toml
in addition to every physical-window decision. Full success requires all368
physically valid windows,reference completion without task failure and all
strict trajectory gates. End-to-end executed duration and physically valid
prefix are separate fields. Preserve every failure and missing tail.

Pre-register worker and separate saved reader after zero-step admission and
CPU checks. The independent reader must verify sealed files before/after,
recompute raw/EFC/geometry/reward/stop decisions,physical source chains and
EMA/RAW feedback,regenerate policy161 from source/reference,check frozen CPU
Actor means with predeclared scaled-L2 bounds1e-5 for observations and1e-4 for
RAW,verify no updates/full state/model invariants and recompute strict summaries.
Cross-device neural bounds do not relax any physical gate. Reader uses no
native integration or forward replay. Export only actual saved poses,repeat0,
all three policies per backend,with physical/task stop labels and no fictitious
future motion. Rendering forward calls are display-only and separately counted.

Wall time includes host copies/audits/storage and is not an isolated solver or
training speed measurement. Previous full54/long evaluation results stay
immutable. The full M0–M5/true A1/A2,contact/world-model gradients,multi-reference
shared protocols,ego asset/dynamics/coordinates and hardware contract objectives
remain active whatever the measured policy outcome.

V2 execution correction:retain V1 registration/output/failure,which ended
before any actual/attempted native integration because a one-element MuJoCo
view required .item(). Correct only the scalar read,no structural/physical gate
change. Reuse the unchanged six zero-step admissions:identical actor/config/
plant/backend semantics;V2 differs in output/source/protocol paths. New native
cap remains52,992;V1 spent0. All old files remain immutable.
