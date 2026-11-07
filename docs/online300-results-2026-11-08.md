# Unified online300: completed learning, failed holding

The resumed experiment completed all300 paired online updates and three
collection cycles. All four predetermined native Actor cases still lost
holding at about0.121s. There is no measured two-second holding or CHORD-proxy
benefit in these cases. The full research Goal remains active and incomplete.

## Execution and exact resume

Source commit `9b38f97` remained unchanged. The first10 checkpoint had already
passed the user's pause boundary; the explicit Goal resume continued this
same experiment from11 through300, without another first-checkpoint pause.
The290 resumed updates and two remaining CPU collection children exited0.
Supervisor wall time was216.45s. The resumed update mean was0.58937s, GPU
utilization median56% across311 samples inside update windows, and maximum
whole-device reported memory8,693MiB. This is a different data/model workload
from the short performance benchmark, so it is not a matched end-to-end
speedup comparison. The final checkpoint's Torch allocation peak remained
about4.88GiB.

Final checkpoint: `runs/phone_online_resume1/checkpoint_0300.pt`,39,226,569B,
SHA256 `439c77a0cc78aee7cf396b8477bcba61b23351349550dbded650942930efe621`.
Final health: `health_0300.json`, SHA256
`3cffd2eec7a8ef24fcca61bc00d2332e25a9718b33ff6e1b355da059131c798d`,
24/24. Online-only counters are WM300 and paired300; both arms have300 Actor
and300 Critic steps. Lineage totals are paired1300 and shared-WM1400, keeping
the original offline1000/WM1100 distinct from the new online stage.

The separate saved review passed18/18 with2 CPU checkpoint reads and0 neural
construction/forward, optimization, native or GPU calls. It replayed every
CPU sampler draw for all290 updates from the saved step10 generator state;
all4+2+2 WM row selections and common Actor8 IDs matched, as did the final
generator bytes. Global Python/NumPy/Torch/CUDA RNG bytes were restored from
step10 and preserved. Five Adam states reached300 with finite moments and
matching parameter/gradient shapes. Normalizers, loss scales and physical
context remained byte-identical; weights/targets actually changed.

## Collection scope

Each of the three cycles retained16 episodes and112 complete20ms rows,
56 per arm:336 rows total,32,256 native steps. Each episode terminated at
POST-H task loss after7H (0.14s); no partial or refill was introduced. Source
Actors were respectively fixed offline1000, complete online100 and complete
online200. Cycle1 was reused on resume, not recollected.

These are nominal-phone, known-initial-state, policy-generated TRAIN data,
with both arms pooled. Shared initial-state aliases are retained. They are
not independent ego trajectories or successful holding demonstrations. The
POST-H collection stop is distinct from the tickwise original holding gate.

## Original no-aid physical evaluation

The outer tool uses the unchanged shared native loop and gates, CPU Actor
inference, the exact qualified initial q/u, and both original native dt.
No WM or Critic forward, backward, optimization, observer forward or GPU
operation occurs. Integrity checks passed9/9; failures remain results.

| Policy | dt ms | Native steps | Exit time s | End COM drift mm | End angle deg | Holding gate |
|---|---:|---:|---:|---:|---:|---|
| A1 tracking | 0.31250 | 386 | 0.12062500 | 50.1423 | 38.8725 | FAIL |
| A1 tracking | 0.15625 | 773 | 0.12078125 | 50.0034 | 38.4511 | FAIL |
| A2 engineered proxy | 0.31250 | 386 | 0.12062500 | 50.0564 | 37.9122 | FAIL |
| A2 engineered proxy | 0.15625 | 773 | 0.12078125 | 50.0777 | 38.7729 | FAIL |

All cases ended `ABORT_LOSS_OF_HOLD`:2,318 steps,28 Actor forwards. Each has
only6 POST command frames and94 missing tail frames. Tail velocities and
triple contact coverage remain null, not zero. Contact/joint/table/warning
gates passed, while duration, endpoint and tail gates failed. Neither mode
qualifies for two-dt consistency because neither completed two seconds;
small differences between failed prefixes do not change that conclusion.

## Fixed heldout action-response diagnostics

All328 saved branches and320 finite-action sides from the permanently heldout
probe are retained, including all20 axes and both signs. Nothing was fitted
or selected using these data. The frozen offline1000 reference reproduces
25 saved prediction/response/AD/FD array leaves byte-for-byte through the
shared implementation, supporting the diagnostic port's numerical parity.

| Response error, component RMS | offline1000 | online300 | Zero-response prior |
|---|---:|---:|---:|
| Joint q, rad | 0.000762474 | 0.000717345 | 0.000711276 |
| COM, micrometres | 21.6017 | 21.6567 | 21.4117 |
| Endpoint world rotation, mrad | 0.834592 | 0.816819 | 0.659967 |
| vCOM, m/s | 0.00222383 | 0.00222345 | 0.00230053 |
| World omega, rad/s | 0.0750932 | 0.0744517 | 0.0768843 |
| J22 force, N s | 0.000147301 | 0.000148187 | 0.000152190 |
| J22 moment, N m s | 0.00000916359 | 0.00000917552 | 0.00000952733 |

Improvements and regressions are mixed. In particular COM/rotation response
errors remain above the zero-response prior. On86 exact-zero native COM
responses, phantom model response vector RMS changes17.0368→16.6330µm.
The endpoint pose and response metrics are distinct; no combined unitless
winner or inference of useful control gradients follows from absolute loss.

Two CPU WMs used20 forwards and16 Jacobian APIs, each with274 cotangent
directions (4,384 equivalent scalar reverse directions), with0 updates/native
steps/GPU. All160 bound-available context/axis pairs had no FP32 input
collapse. Online300 model-only AD/FD relative-L2 errors span approximately
0.1303–0.2410%. This is internal H1 numerical consistency, not a native
infinitesimal Jacobian, H2 geometry derivative or contact-truth qualification.
Marginal22 coverage predictions do not constitute simultaneous triple contact.

## Evidence and next boundary

Small reports, immutable pins and retention mapping are in
`provenance/online_completed`. Raw checkpoints, datasets and traces remain
under `runs` or the external artifact registry. New tests preserve original
integer-POST tracking math and missing-tail masks;50 tests passed.

The next research step is a bounded native feasibility/control-authority
audit under the current legal residual/EMA mapping, followed by a separately
declared TRAIN intervention dataset if appropriate. Do not reuse the permanent
heldout probe for fitting or relax the success gates. Further blind Actor
updates are not justified by the current response and physical evidence.
