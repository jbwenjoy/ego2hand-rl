# Same-arm segmented ETA v2

The new source is `scripts/estimate_primary_eta_cpu_v2.py`, frozen before proof at SHA `50bc470d2337bfa536eebd96c03ac7f69369bd28364ec2d0e37f669f6902fd27`. The executed V1 source and its original A1L pause report remain unchanged.

The WM-only steady window, policy-cost function, 20-second overhead allowance and `.5–1.8` planning range retain V1 mathematics. A measured primary policy window is preferred. When not available, the only permitted proxy is `outputs/smoke_{method.lower()}_v2/metrics.jsonl`, with that smoke's actual config checked for exact arm/seed/source/controller identity. There is no cross-arm proxy override.

Config and healthy100 receipt pins must match exactly. `--metrics-snapshot` additionally requires adjacent `report.json` to bind config, health100, committed prefix SHA and last iteration. Snapshot mode is historical replay, so absolute finish predictions are null; it cannot be mistaken for a current ETA.

The real saved A1L199 prefix replays the original point estimate **1531.705229113698 seconds exactly**, with the same 148→198 base window, five steady smoke policy differences, 20-second allowance and range `[765.852614556849, 2757.0694124046563]`. All28 proof checks pass, including P_ONLY/A2 own-smoke routing and wrong arm, wrong seed, wrong source, wrong health/config pin and altered snapshot rejection.

P_ONLY/A2 route checks use in-memory routing descriptors and their completed smoke JSONL only. They do not represent actual P_ONLY/A2 primary ETAs, healthy100 runs or new training. No Torch/Env/GPU/physics was used. An initial proof's Windows text-newline assertion failure is retained in `outputs/primary_eta_v2_cpu_proof_v1`; the ETA source was unchanged, and the new checker uses explicit fixture bytes.

```powershell
& D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe `
 D:/workspace/supermimic-rough-prefix-learning-study/scripts/estimate_primary_eta_cpu_v2.py `
 --config ACTUAL_PRIMARY_CONFIG --health ACTUAL_HEALTH100 --output NEW_ETA_DIR
```

The estimate is a planning range, not a confidence interval, GPU-busy measurement, deadline guarantee or task-quality claim. Historical construction and engineering costs remain separate.
