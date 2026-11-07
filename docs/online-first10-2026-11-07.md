# First unified online checkpoint

2026-11-07, source commit `9b38f97`. The new online stage ran on cuda:0 and
stopped at cycle1/update10. Training and collection children exited0; the
first serialized checkpoint passed24/24 health checks. This is pipeline and
serialization evidence, not a new policy holding result.

The checkpoint is `runs/phone_online_v1/checkpoint_0010.pt`, 39,222,409 bytes,
SHA256 `ed5efc600facec7738d0b36c320b8cfd08c559cac8bc1b26b2622e3871ff91f1`.
The health report is `runs/phone_online_v1/health_0010.json`, SHA256
`6ed96893d6b3942c60a370be099a9b270f7a7357ab33538781a7c8ae7d95ccec`.
Small copied reports are under `provenance/online_first10`; original pins are
retained, with large tensors/raw traces outside Git.

Separate saved-data reviews passed18/18 for the serialized checkpoint and
10/10 for the collection. The former loaded only this checkpoint and the
fixed root on CPU, with0 network construction/forward/backward/optimization;
it checked all five Adam states at10, finite named gradients, actual weight
changes and raw frozen norms/scales/context. The latter used NumPy/JSON only
to verify112 raw/decision/canonical chains,300-value receipts, control bounds,
POST state and canonical22 impulse/coverage labels. These checks do not
replace a policy evaluation or physical-gradient qualification.

## Actual collection and learning

Cycle1 loaded the fixed offline1000 Actors, not the newly updated checkpoint.
Sixteen arm/profile/dt cases produced112 complete20ms rows,56 per arm,
without partial rows. Each episode ended after7H at POST-H task loss. These
short failed collection cases are not successful two-second holding and are
not the tickwise holding evaluation of the new checkpoint.

Native counts:2 MJB reads,16 MjData,16 initial forwards,10,752 integration
steps,17,295 contact-force reads,352 receipts,0 observer forwards. Both arms
used5,376 native steps; no failed episode was refilled. The CPU worker made
112 Actor forwards,0 WM/Critic forwards and0 optimization updates, preserving
Actor/norm/RNG state and the original nominal physical model.

Learning used the pooled112 rows plus explicit original TRAIN684 and tripod
TRAIN153 caches: WM4online+2old+2tripod, common Actor batch8/H2/micro8 and
per-mode real rewards. Five fresh Adam states each reached step10. Online
lineage calls:10 WM supervised forwards/backwards/steps,40 Actor forwards,
40 WM imagination forwards,40 real Critic forwards,20 Critic bootstrap
forwards,20 Actor microbatch backwards and20 Actor/20 Critic optimizer steps.
Readback probes/constructors are separate health diagnostics. Raw root
normalizers, loss scales and physical context remained frozen.

## Timing and GPU scope

Actual supervisor runtime was39.68s: setup10.50s, CPU collection child19.13s,
cycle cache1.09s, ten updates averaging0.7675s, plus health/I/O and process
overhead. There were no further learning updates after the healthy checkpoint.

The measured remaining290 learning updates estimate222.58s (about3.7min),
including the first-update warm-up in the ten-update mean. Two later
policy-dependent CPU collections, cache/health/I/O and resume setup are
additional unmeasured overhead. A paused run has no scheduled finish time;
this is work remaining after an explicit resume.

Thirteen GPU samples inside actual update wall windows give50% median
utilization and8,891MiB maximum reported whole-device memory. Torch peak
allocated memory was4.878GiB. Whole-device memory includes desktop/driver and
other allocations; it is not the same as Torch allocated memory. CPU
collection/setup/health samples are excluded from the utilization estimate.
The sample count is small and this is not an end-to-end speed comparison
against the earlier offline experiment.

## Resume boundary

The disabled template `runs/protocols/phone_online_resume1.template.toml`
passed metadata/source/input/recipe/collection checks. It names the saved
own checkpoint/health through the registry. The next update is11, using the
same cycle1 pool; cycle2 collection requires the complete100 checkpoint.
Actual resumed optimization remains subject to the same source-bound
checkpoint, five-Adam, pool-map, log-prefix and RNG validation.

The full research Goal remains incomplete. The standing user stop request
applies at this first healthy checkpoint; no policy benefit, independent ego
generalization, physical calibration or multi-trajectory cost claim follows.
