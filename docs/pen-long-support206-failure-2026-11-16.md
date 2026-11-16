# Long trial stopped at SUPPORT206;bounded replay did not reproduce it

Original session89759 is TERMINAL exit1. RESIDUAL2000 completed;SUPPORT205
updates completed,then collection206 failed an original consistency gate
before optimizer206. Original failure.json records36143104 native integrations
and70560 optimizer steps. No report.json exists for a completed two-arm trial.
Do not run final evaluatorV2 or claim equal2000-round results from these files.

The original exception handler saved counts/phase but omitted the failed
health flags,memory206 and policy/native state. Checkpoint200 is the last
saved SUPPORT state;successful health201–205 are scalar evidence,not recoverable
full checkpoints. The exact failed invariant and failed data are unavailable.
EPA horizon24 warnings appear in the console,but do not prove which check failed
or that the warning was its cause. No gate is relaxed and no bad batch is fitted.

Separately registered diagnostic replay loads SUPPORT200 ordinarily under
unchanged captured source/config/physics. Immediate full reload passes. It
executes exactly6 further64-control rounds201–206,and all pass25/25,so all6
are optimized. Added work98304 native/192 optimizer,not original trial work.
Each raw memory,health and post-update checkpoint is retained. Replay session
95314 is TERMINAL exit0. It is not the original run continuing or a replacement
for its missing failed data.

The first replayed aggregate already differs:iteration201 mean raw reward
1.342902899 vs original1.368985772;iteration202 terminated worlds23 vs24.
This supports the existing limitation that immediate full-byte reload does
not establish future W32 trajectory identity. It does not identify the source
of divergence or reconstruct the original failed gate. Both original and
replay max logprob errors are0 in retained201–205 health records;the original
206 check itself is unknown.

Next,keep original training/source/checkpoint bytes immutable and build any
failure-aware orchestration as a new tool. It must save health flags and raw
invalid batches BEFORE rejection,retain nonfinite values in tensor artifacts,
write resumable checkpoints only at valid boundaries,and count all actual
work. Do not silently resume the original from200,hide lost work,or call replay
206 a valid original checkpoint.

A possible new matched recovery study resumes BOTH arms from their recorded
200 checkpoints with identical extra restore interventions,then executes the
remaining1800 rounds each under unchanged physics/objective/source contracts.
This requires a new explicit protocol/config and final-evaluation registration,
equal cumulative policy budgets and separate accounting for reused prefixes,
discarded original work,diagnostic work and actual recovery compute. It has NOT
been launched. An unmatched SUPPORT-only recovery is not the registered trial.

The full423 tests and ego gravity audit remain valid in their own scopes.
All original M0–M5,strict control/WM gradients,real-ego assets/calibration,
multi-trajectory/seed total-cost and hardware requirements remain active.
Goal incomplete. There is currently no live training or training ETA.
