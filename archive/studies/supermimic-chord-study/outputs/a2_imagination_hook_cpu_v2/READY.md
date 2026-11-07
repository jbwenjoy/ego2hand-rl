# A2 geometric-intent hook — CPU preparation complete

The optional A2 branch is implemented in `scripts/train_controlled_pen.py`.
It uses `GeometricContactObjective.from_bundle` with independently frozen
bundle/model/reference/rule/code SHA256 values, copied into each run's source
snapshot. It never constructs contact targets from current policy contacts.
Construction restores Python, NumPy, Torch CPU and existing CUDA RNG streams.

Only `RecordedRunner.policy_update` temporarily replaces `reward_step` with:

`native_reward + 0.02 * geometric_bonus`

The native cage state and native terms object are returned unchanged. A `finally`
block restores the original callable before collection or world-model fitting.
Environment rewards, actual simulator collection and native WM-fit reward parity
remain native. A1/A1L bypasses prior construction and the A1 reward path performs
no additional tensor or RNG operation.

The objective itself is the contact agent's frozen formula:

`phase_weight * support_match - 0.1 * penetration_prior`

The penetration prior remains active in unknown/known-no-contact phases. The
module does not multiply bydt internally. Training uses64 fixed basis directions
and64 hand-surface points per geom; independent evaluation retains512 directions.
This is **CHORD-inspired geometric intent shaping**, not a qualified contact/force
predictor or a successful dynamics-gradient claim. The failed local proxy
qualification remains explicit in config and module metadata.

Five CPU checks passed: A1 callable identity/bitwise output/RNG preservation;
exactly-once0.02 scaling and gradients; native cage/terms preservation; exception
restoration; real fixed-module construction/input gradients and SHA mismatch
rejection. Evidence: `checks.json` and `constructed_metadata.json`. These are
engineering checks, not physical task qualification.

The real Warp target-limit parity check also passed separately:16 samples ×2
environments,18 out-of-range input components, max actual/imagination/ctrl target
difference0rad (`outputs/control_target_limits_warp_v1/checks.json`). This checks
target processing/application only. It does not imply that full3s trajectories
are bitwise equal; root's independent bounded rollout measured small differences.

Queued engineering smoke commands (not run by this agent):

```powershell
D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/train_controlled_pen.py --config outputs/training_configs/a1l_bounded_10iter_smoke_seed0.json
D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/train_controlled_pen.py --config outputs/training_configs/a2_bounded_10iter_smoke_seed0.json
```

Both use10 iterations,2 WM-only iterations and every2-iteration checkpoints,
with the same bounded controller, data, seed, architecture and native optimizer
budget. The first2-iteration checkpoints allow checking the common WM prefix;
physical GPU nondeterminism must be assessed rather than assumed absent. These
smokes are distinct from subsequent paired2,000-iteration A1L/A2 experiments.
GPU wall time, compilation, prior setup and any failures are recorded separately.
The old A1 trained before target limiting remains a distinct successful runtime-
postprocessed demo, not a bounded-training control group.
