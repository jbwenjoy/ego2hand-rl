# Matched offline phone A1/A2 trainer

`train_shared_WM_A1_A2.py` is executable training code. Its default invocation
checks JSON and file hashes only. No Torch/native import or array payload is
needed for default metadata. `--run` requires explicit CPU authorization and,
for `cuda:0`, GPU authorization in a fresh protocol/output directory. The
delivered `protocol_CPUFalse_v1.json` authorizes neither.

The fixed recipe uses one common WM, warmstarted from the physical checkpoint
5000 weights with its input/output normalizers frozen. Adaptation loss scales
are fitted to the equal mixture of old TRAIN684 and new TRIPOD_TRAIN153 only.
New 22-pair impulse/force/coverage labels supervise absolute transitions. The
historical counterfactual loss is not continued in this recipe. MONITOR52 is
used only for fixed checkpoint WM-loss reporting, never normalization,
checkpoint selection or reward tuning. CHECK, old TEST and fresh-final data
are not opened.

Common WM adaptation100 is followed by1000 matched paired rounds. Each round
has one common WM update, one Critic update and one Actor update per arm.
A1 uses tracking/joint/rate reward; A2 adds the frozen engineered material
support proxy. Both arms have identical fresh Actor/Critic initialization,
norm exposure, real replay sample indices, WM and optimizer settings. Actor
imagination is B8/H2, four microbatches of2. All153 complete live TRAIN source
states may start imagined H2; this is not a claim of153 real two-step windows
(the dataset contains129 native H2-eligible starts).

The trainer saves only zero-update modules before common adaptation. After
common100 it writes counters/loss JSON, no updated model/Adam `.pt`. The first
formal learning checkpoint is paired100: WM200 updates and each arm Actor100,
Critic100. It saves raw state dictionaries, target critics, five optimizers,
named PRE-update gradients, normalizers/context/scales, sampler and raw
CPU/Python/NumPy/selected-CUDA RNG. Readback checks restored predictions,
weights/norms/Adam/gradients/RNG and fixed JSONL prefix. Health plus last25-round
training ETA is written and the process exits at paired100 so Root can hand
off and pause the Goal. Resume from that own healthy checkpoint restores the
entire state and continues rounds101..1000 in a new directory. ETA excludes
setup/common100/checkpoint-health IO; resume setup is reported separately.

```powershell
& 'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe' -X utf8 `
  'D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/train_shared_WM_A1_A2.py' `
  --protocol '<absolute reviewed protocol path>'
# Only Root-authorized actual protocol: append --run.
```

This is an offline matched SuperMimic-style component pilot, not a completed
online SuperMimic/CHORD implementation. It has actual Actor/Critic/replay,
supervised WM adaptation, real-TD critic targets and differentiable imagined
Actor optimization. Policy-native collection, feedback adaptation and the
original double-dt2s evaluation are separate remaining steps; this recipe
makes zero native calls. Its checkpoint `arms[mode]` exposes `stateactor`,
`statecritic`, `statecritic_target`, `norm` for the separately qualified native
policy evaluation callback. Whole-module pickle and old PenMDP/state charts
are not used.

The new tripod task15/28/41 is OOD relative to the warmstart's old2pad task.
All32 acquisition episodes lost boundary holding after6–7 intervals. TRAIN
nonpad/table peaks were0; MONITOR had two nonpad events up to2.3888N, not used
to change this recipe. The A2 same-material/same-normal1mm proxy is not true
contact, force, closest normal or native4ray support. Learned physical action
response remains imperfect. No legal holding, ego generalization, physical
gradient qualification, CHORD benefit or policy advantage is established by
source checks, replay fitting or checkpoint health. Boundary POST-H task loss
absorbs TD; collection truncation may bootstrap; any-native-tick loss is a
separate descriptor. Partial/unknown H targets remain masked in the pinned
source rather than being promoted by replay's numeric NaN handling.

Preparation performed one generator-internal JSON/hash preflight and one
direct default-CLI check (exit0), zero NN/native calls. The
CPUFalse/source/recipe pins are in `metadata_CPUFalse_v1.json`.
