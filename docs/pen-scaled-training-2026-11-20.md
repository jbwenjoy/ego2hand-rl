# Guarded pen scaling on RTX 5090, 2026-11-20

The guarded solver completed six controls at 32 worlds, including a partial reset and a fresh-runtime full-state restore. All 1,536 world integrations were counted and the saved raw-contact oracle and graph checks passed. The first attempt incorrectly required hand normal force above 0.1 N in every world in every 20 ms window. The corrected reader checks the entire contact window against the independent raw oracle, including zeros, and requires observed contact in each world across the six controls. Both attempts are retained.

The collector benchmark used a fresh process per size, cached reference features, the same residual PPO collector and an 8-control calibration. Each process discarded the first 64-control collection as warm-up and timed four more, with CUDA synchronization at the boundaries. The collector sampled with initial weights and made zero optimizer updates. The first benchmark launch used the default JIT execution mode during construction and failed its exact initial-state comparison at each size. The timed commands use the same JIT execution mode at construction and collection as the W32 qualification.

| Worlds | Control transitions/s | World integrations/s | Median GPU utilization | Peak sampled board memory during collection (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1,173 | 9,384 | 54% | 2,241 |
| 128 | 4,203 | 33,624 | 56% | 2,920 |
| 256 | 7,722 | 61,776 | 59% | 3,857 |

The board-memory monitor sampled at approximately 0.25 s intervals. Each measurement covers 64 controls per world, so larger batches gathered proportionately more samples. The comparison measures short collection throughput, including observations, policy sampling, terminal handling and resets. The 128/256 runs passed collector consistency checks. Full raw-contact oracle and reboot/restore coverage in the six-control diagnostic applies to 32 worlds.

The selected run uses 256 worlds, seed 7411, an initially untrained residual PPO policy, and three hours of measured training. Each iteration collects 16,384 transitions, then performs 32 optimizer steps. Model construction, checkpoint storage and separate W1 evaluation add elapsed time. Each rollout retains a 369-bin histogram of used reference phases passing the training geometry, engine, applied-force and non-hand contact checks. These training filters exclude selected invalid samples. Detailed physical impulse verification is performed by the dedicated W1 evaluation. Phases 185–368 form the second half and 277–368 the final quarter. Numerical repeats on this single reference assess full-task success under the existing 7.36 s evaluation.

The trainer checks and reloads its first full checkpoint, saves a rolling full checkpoint every 250 iterations and at evaluation boundaries, and exports an Actor for separate W1 evaluation after the first update, at hourly boundaries and at completion. The final evaluation runs three numerical repeats. The first checkpoint restored 854 arrays, 890 scalars, the optimizer, both modules, observations and the normalization state. Its first evaluation failed the task at 0.24 s (0/1). Later results are recorded in the run directory.

Benchmark commands use `tools/benchmark_pen_scaled.py WORLDS OUTPUT` for WORLDS 32, 128 or 256. The training command is `tools/train_pen_scaled.py 256 runs/pen_scaled_train_v1 --hours 3`. Each command requires a fresh destination. Qualification, benchmark and start-of-training receipts are listed in [the evidence index](../provenance/pen_scale_20261120/start.json).
