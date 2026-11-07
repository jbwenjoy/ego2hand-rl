# Phone matched online learning source

Status: **SOURCE_READY_PENDING_RUNTIME**. The first docstring retains the
historical draft label to keep the reviewed/pinned `ab7a9fde...` source bytes
unchanged. No new-stage model, optimizer, NN forward, native collection or
training has been executed. The existing offline GPU run is separate.

Default metadata command:

```powershell
& 'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe' -X utf8 `
  'D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_online_learning_v1/train_online_matched_v1.py' `
  --protocol 'D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_online_learning_v1/protocol_CPUFalse_v1.json'
```

The delivered template intentionally has four pending root artifacts. After
offline paired1000/WM1100 actually completes and its final health/terminal are
saved, this pure JSON binder creates another **CPUFalse** protocol; it does
not authorize collection or learning:

```powershell
& 'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe' -X utf8 `
  'D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_online_learning_v1/prepare_online_protocol_v1.py' --bind `
  --root-protocol '<actual final offline execution protocol>' `
  --root-result '<actual fixed1000 result.json>' `
  --root-health '<actual paired1000 health.json>' `
  --root-terminal '<actual final training process terminal.json>' `
  --output-name gpu_run1 --protocol-output '<fresh absolute bound-CPUFalse protocol>'
```

Root must separately review the bound protocol and explicitly authorize CPU,
selected GPU and native collection, retaining the fixed recipe and a fresh
output. Only then is the actual entry the default command plus `--run`.
Resume adds `--resume-checkpoint` and `--resume-health` to the binder, a new
`--output-name` and fresh protocol. The binder keeps authority flags False.

The executable loop is3 real collection/update cycles. Each collection is an
isolated CPU child process using frozen collector v3, both Actors, four fixed
exploration profiles, two native dt and at most20 complete20ms intervals per
episode:16 episodes/320H/30720 steps per cycle, at most92160 native steps total.
The child gets its own environment copy with CUDA masked; the GPU parent's
environment is unchanged. All early failures/partial prefixes remain in raw
files, without refill. Actual per-arm samples can differ despite identical
maximum budgets. Collection and raw→canonical labels finish before learning
reads the pool. Future-H force/impulse labels never enter Actor or WM inputs.

Each cycle then performs100 matched paired updates. One shared WM and both
arms' Actor/Critic/target/norm/physical state are warmstarted from the same
fixed offline1000 artifact. Five AdamW optimizers start fresh at seed1734.
WM batch8 is4 cumulative online +2 old originalTRAIN684 +2 old tripodTRAIN153.
Normalizers, WM loss scales and physical context remain frozen from offline
final1000; no online/monitor/CHECK/fresh/TEST fit. The new stage uses absolute
WM supervision, not the historical counterfactual loss.

Actor/Critic use the same online batch8 with replacement from the common
cumulative pool (both acquisition arms pooled). Each arm takes its own saved
real reward, shared WM and same data indices. Real TD absorbs actual boundary
task/Reader failure and reference timeout; collection truncation may bootstrap.
The complete transition ending at a terminal remains a valid WM/TD example;
partial unknown targets cannot become valid through replay NaN conversion.
Imagined Actor H2 uses four microbatches of2, not a claim of native H2 windows.
The A2 same-material/normal support term remains an engineered reward proxy,
not measured contact, force, closest normal or physical support qualification.
Both arms' experience and shared WM are deliberately coupled; these are not
independent learning histories.

Pool identity is `(dataset pin SHA, original row/episode ID)`. Every cumulative
merge writes explicit stable global row/episode IDs and preserves original
IDs/pins, masks, NaNs and both reward columns. No episode can silently cross a
collection boundary because local IDs repeat.

The first formal learning checkpoint is cycle1/update10, with10 new WM updates
and10 Actor/Critic updates per arm. Raw five-Adam state, named PRE-update
gradients, AFTER-update weights, target critics, frozen norms/context/scales,
CPU/Python/NumPy/selected-CUDA RNG, sample generator, collection/dataset/raw
protocol pins and log-prefix bytes are saved. Health performs one own CPU
readback,5 cloned networks/10 diagnostic forwards, exact saved/live state,
Adam/gradient/RNG and fixed-prefix checks. Health+ETA is written and the
process exits so Root can hand off and pause the Goal. This is not completion
of the3×100 target.

Resume restores all state and the exact previously collected pool order; it
does not recollect cycle1 at step10. Previous process log segments remain
bound by path/firstNbytes/SHA/global range. At update100 the next healthy
checkpoint is saved before its updated Actors supply the next collection.
Only full-cycle100 Actor checkpoints may supply a new collection; step10 is
for resume only. `global_paired_updates` and new Adam steps are online-only
1..300; explicit lineage fields are1000+global paired and1100+global WM.

ETA uses measured update times (up to the most recent25;10 at the first stop)
for remaining learning updates only. Setup, current geometry cache and health
IO are separate. Future policy-dependent collection/setup/health costs remain
unmeasured, with prior child elapsed times disclosed rather than inventing an
exact end-to-end ETA.

Preparation performed one JSON/hash-only metadata preflight and one bounded
pure NumPy merge fixture (8/8). That fixture uses two pool views with repeated
local row/episode IDs, partial NaNs and different mode rewards; it makes zero
NN/optimizer/native/NPZ calls. Source review is not runtime qualification.
No ego generalization, holding improvement, physical action-gradient accuracy
or complete SuperMimic/CHORD goal is established. Original2s double-dt native
policy evaluation remains a separate requirement; these0.4s acquisitions do
not replace it.
