# Empty-signal upstream query removal:matched ordinary PPO comparison

Register before execution. Isolated profile has~1.8s collect64 versus~.12s
update32. Torch nonzero costs~.54s including waits for earlier GPU work;do not
assume that is removable overhead. Upstream env.step queries termination IDs
even though our declared upstream termination manager has zero active terms.
Real geometry/tracking termination and explicit resets live in the outer adapter.

Add opt-in EMPTY_SIGNALS control driver in the single maintained adapter/collector.
Its specialized empty-signal chronology calls the SAME action/physics/scene/
empty termination/reward/metrics/forward/command/sense/observation/recorder code
in the same order;only omit materializing known-empty upstream termination IDs.
Keep ordinary UPSTREAM default and both manual/outer pending guards. Reject
auto-reset,active upstream terms/rewards/metrics/curriculum/recorders,unexpected
termination manager or step/interval events before action/physics. Unit tests
compare installed upstream chronology/counters/outputs and count exactly one
omitted nonzero call on deterministic fixtures;test pending/unsupported cases.
No installed-package,geometry,physics,solver,phase/reset or reward edits.

Construct a NEW same-source current-code32-world fixture,8 calibration controls,
two ordinary64-control/32-update warmups. Do not rebind historical source receipts
or resume old native checkpoints after a source edit. The two selectable drivers
coexist in the same source version;full current fixture restores before each
case,with byte/scalar/observation/source guards outside clean timing.

Clean order BASE_A,FAST_A,FAST_B,BASE_B,BASE_C,FAST_C balances position partly;
then separate TORCH_BASE/TORCH_FAST CPU traces. Each64 controls/32 updates.
No monitoring threads/processes. Three clean cases per driver,not independent
seeds. Count165888 actual native world integrations,320 executed optimizer steps
including calibration/warmups;no Actor promotion. Preserve every case and failure.

Compare clean collection/consistency/update timing,operator counts and actual
reset/termination counts. Independent review must check original collector
health,8 native samples,no-aid/no-overflow,geometry/reward/phase/EMA/PD/reset
consistency and common fixture full readback. Save full rollouts/modules.
Measure any numeric/reset divergence against the three observed BASE repeat
envelopes;these are not probabilistic bounds. Do not relax a geometry gate or
silently drop a case to claim equivalence or speedup. If variation exceeds
the observed baseline,retain that result and keep adoption explicitly incomplete.

No timing claim until all work finishes. Report profile overhead separately,
absolute seconds,all clean repeats,work budgets and reset variability. One
removed query is not a solution to all GPU utilization/geometry/launch issues.
Keep UPSTREAM default until evidence supports changing a declared caller.
