# A1 compile/throughput probe — 100 iterations

Status: passed, exit code 0. This run used unqualified clip001 initial conditions;
it establishes a bounded compiled training pipeline and cost measurements only.
It is not a task-success, CHORD-benefit, ego-retargeting, or speedup result.

- 128 parallel environments; a pure crop of source clip001 frames `[0,151)` at 50Hz (3 seconds, both endpoints included).
- World model: 512×3 hidden layers, batch256, 2 fits/iteration, 8-step supervised window.
- Policy: batch128, imagination horizon8; 20 WM-only iterations then 80 policy updates.
- Actual optimizer steps: WM200, actor80, critic320; auxiliary failure-head200, retained for matching upstream cost but unused by policy.
- Native 15-dimensional finger-self-collision supervision restored to weight0.1. Contact output weights changed by L2=1.75934. This head is not CHORD hand-object contact.
- AMP and `torch.compile` enabled for the upstream model/actor/task-function compilation paths. 24 captured graphs, 7,733 calls captured, graph-break counter empty. No eager fallback occurred.
- WM, actor, critic parameters remained finite and changed. Checkpoints `compiled/train/model_100.pt` and `model_final.pt` are readable at iteration100.

| Timing (seconds) | Measured |
|---|---:|
| Total process wall time, including source/reference preparation | 328.20 |
| Environment initialization | 3.01 |
| Runner initialization | 2.49 |
| Learn loop | 316.98 |
| Simulator collection, including warmup | 27.51 |
| WM training calls | 95.00 |
| WM holdout calls | 34.57 |
| Policy/critic calls | 159.83 |
| Checkpoint writes | 0.044 |

Steady iterations30–90 took 0.5174 seconds/iteration. The first WM stage took
70.66 seconds, and the first policy stage about153.76 seconds, including first
compiled calls/autotuning. These are phase timings, not isolated compiler timings.
The measurements are not an eager-versus-compiled comparison.

Cache condition: the initial script inherited the host's
`C:/Users/ASUS/.unsloth/studio/cache/torchinductor` variable; Triton defaulted to
the project cache. This was cache reuse only, with the isolated Dex virtual
environment unchanged. Inductor recorded35 FX-cache misses and1 hit. Therefore
this probe is not labelled wholly cold-cache. The current script now explicitly
sets project Inductor/Triton cache roots and saves before/after cache inventories;
the formal pilot records those conditions independently. The exact probe script
is frozen under `source/train_controlled_pen.py`.

Training used fixed play-scene physics, clean observations, no action masks or
external disturbance events, no first-frame hold and no RSI. A local reset
adapter pins only command phase and controller target to frame0 after the
upstream dt=0 reset; it does not add object-state writes. Training termination
thresholds are5cm and60° and are distinct from independent task acceptance.
The loop reset3,216 worlds including initialization; this number is not a
success rate. The upstream `task/rot_error` log is not a pen reference metric.

`replay_control_trace.npz` records actual actions, executed targets, state,
flattened controller/reference memory, context, observed collision labels,
torques and termination flags for a bounded sample. Its sampled external-wrench
channels and post-step `push` corrections were exactly zero. Full physics fields
were saved separately in `fixed_physics_context.npz`. The0.309GiB peak is only
the PyTorch allocator, excluding Warp/driver allocations.

For an initial formal pilot, keep the tested128env/WM512×3/batch256×2/pi128/
horizon8 architecture and use2,000 bounded iterations, WM-only200 and checkpoints
every100, as selected by the root protocol. At the measured128env steady rate,
2,000 iterations project to about17.2 minutes plus startup/compilation and changes
in contact workload; this is a planning estimate, not a measured completion time.
The candidate256env configuration has not been timed and should not inherit that
estimate as a result. A1/A2 must retain equal budgets and separately charge all
compilation, verification and retry costs.

Primary evidence: `summary.json`, `compiled/result.json`,
`compiled/metrics.jsonl`, `compiled/effective_training_config.json`,
`reference/provenance.json`, and `source/inventory.json`.
