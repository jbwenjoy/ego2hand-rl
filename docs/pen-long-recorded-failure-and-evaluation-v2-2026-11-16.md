# Recorded failure diagnosis and final evaluator v2

## New observational evidence

Offline analysis of the retained RESIDUAL1000 episode independently rebuilds
native hand/nonhand normal loads, per-finger groups, penetration, resident
window means, target frame rotation and saved evaluation metrics:11/11 checks.
No new native integration or optimizer work. The first report remains intact;
v2 also records the five largest reference rotation intervals.

Reference interval1.64–1.66s rotates15.0973deg in20ms (position step2.3849mm).
The preceding1.62–1.64s rotates13.6252deg. Actual full-angle error first exceeds
5deg at1.6600s and10deg at1.6825s;COM error exceeds5/10/20/50mm at
1.6725/1.7125/1.7800/1.8600s. Hand normal first falls to at most0.1N at the
native pre-integration contact sample nominal time1.7200s.

The descriptive1.5–1.6s interval has hand normal mean7.3389N,versus0.3619N
at1.7–1.8s. These are total opposing normal loads,not net support forces.
Mean measured WORLD vertical support is0.3075N and0.1913N respectively;
nominal reference targets are0.3138N and0.2894N. Preserve force/frame/origin
distinctions. Contact fields come from native pre-integration forward/solve;
pose is post-integration;20ms means use the original intervals. GPU float32
clock drift is retained and plot axes use the declared nominal native clock.

Raw mean actions exceed the clamp in1/1860 components (control index82,
component11,max abs1.0017257). This is not evidence that executed absolute
PD targets or drives were unsaturated:those fields were absent in the1000
record. These associations identify a rapid-rotation/contact-transition
interval for inspection,not a causal explanation or proof of infeasibility.
In particular,reference angular speed alone does not prove physical failure,
and contact loss occurs after the first pose-error threshold crossings.

Reports:`runs/pen_long_midpoint_failure_diagnosis_v1` and`_v2`.
Figure:`runs/pen_long_midpoint_failure_diagnosis_v2/failure_timeline.png`.
All intervals are descriptive after viewing this single trajectory,not unseen
validation. Do not select a policy or change this trial using these metrics.

## Final evaluator measurement amendment

V2 is registered before either final2000 checkpoint exists. It supersedes the
V1 final execution command while preserving all V1 registrations/admission
records and historical source bytes. Only recording/admission instrumentation
changes:no training code,physics,reward,reference,gates,case order,repeats,
budget,initialization,checkpoint selection or video-repeat selection changes.

The shared recorder now saves pre-qpos/qvel,actual GPU ctrl,native actuator
force and generalized actuator force per native tick;also reference joints,
initial EMA,encoder bias,absolute target bounds and explicit joint/control IDs.
Every final case audits raw clamp/scale0.5/EMA0.5/reference/bias/absolute-limit
targets against transmitted GPU ctrl,eight-substep hold,and state continuity.

Actual qualification uses the fixed first optimized checkpoint,11 controls/
88 new native integrations,0 optimizer updates. All6 control audits pass;
GPU ctrl reconstruction max error0rad,all states contiguous and drives finite.
The observed early policy outcome is retained as instrumentation QA,not final
performance;it is not substituted for a final policy. Wall10.7510s. Offline
analysis of this shorter QA episode independently passes17/17 including the
control audit,without new native work. Full419 tests pass in9.930s;four new
tests distinguish raw clipping from absolute target limiting and reject altered
control/state/reference/bias/binding/phase/nonfinite-drive data.

Closed added work:88 native integrations,0 optimizer. The earlier midpoint
744 integrations and startup qualification4096 remain separate prior cohorts;
long training is still ongoing and its final actual work remains unreported.
No WM Actor admission or ego/generalization claim is made.

Final command after successful completion of BOTH2000-round arms:

```powershell
& 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe' -X utf8 tools/evaluate_pen_long_control.py --config configs/pen_long_control_evaluation_v2.toml --output runs/pen_long_control_evaluation_v2
```

Use the unchanged strict full7.36s/COM/full SO(3)/geometry/no-aid gates and
three fresh numerical repeats per policy. These remain one trained seed and
one controlled reference,not independent seed/ego validation. Finish and
report the full registered experiment before deciding any follow-up to the
rapid-rotation/contact-transition,reference momentum-target or drive limits.
