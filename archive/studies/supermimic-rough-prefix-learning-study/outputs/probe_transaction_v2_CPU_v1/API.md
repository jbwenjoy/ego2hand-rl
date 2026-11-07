# ProbeTransaction v2

Import `ProbeTransaction` from `rough_prefix.probe_transaction_v2`. The original module/checker paths have been restored to their executed 33-check versions; v2 is independent. No frozen native/runtime/controller/capacity source was modified.

```python
observer = CapacityObserver(env, stage=path,
                            fixed_capacities={"nconmax": 192, "njmax": 4096})
tx = ProbeTransaction(
    env, reader, capacity_observer=observer,
    source_action=actions[k].copy(),  # explicit finite FP32 [1,20]
    label=f"{path}/{k}",
    external_roots={"ctx": ctx, "samplefile": samplefile, "rewardfile": rewardfile,
                    "pure_module": pure, "wp_module": wp, "torch_module": torch,
                    "mujoco_module": mujoco, "mjwarp_module": mjwarp,
                    # also all persistent logs/counters/records/instrumentation holders
                    },
    extra_sidecars={},  # explicitly register any used foreign mutable class
)
with tx.branch(str(branch_id)) as restore_receipt:
    # Worker separately checks actual INTEGRATION get/set/get bytes and actual8.
    # Apply one declared absolute target after original source action processing.
    # Run only the original native interval, then save its actual measurements.
    ...
```

`begin(nonempty_string_id)` is one-use: external capacity read/latch/raise, full native snapshot restore, explicit NumPy/closure/local-generator/custom-sidecar restore, global RNG restore, bitwise inventory and complete PEN-memory verification. It returns `restore_qualified`, the source-action hash, capacity record and inventory digest. `finish(id, failure=None)` must always run; its capacity record is external. Passing a body exception marks the transaction failed permanently. The context manager preserves a primary body exception if the finish capacity read also fails, retaining the secondary capacity error and OR ledger. It never retries a failed/duplicate branch.

`verify_current()` can be called immediately after the worker's separately required integration roundtrip. `describe()` returns profile/entry/array/tensor/closure/local-generator counts, opaque handle identities and external-reference paths. No raw state setter, forward, step, reset, contact query or sensor computation is performed by this helper. `reader.memory()` is read to check native memory values and shapes; this includes the original c1/c2/action/phase/limits/camera/frozen-target semantics.

The ledger must be a direct `CapacityObserver(env,...)` instance kept outside env/reader, not the context manager that attaches an observer to env. It must not be in any native captured container or restorable extra sidecar. Ledger flags are read before every restore and after every branch, even if a branch body fails. The worker still owns donor-terminal/end reads and no-physics-after-final rules.

The default factory is deferred native `WarpTaskSnapshot(SimpleNamespace(env, reader))`; custom factories are permitted only with `strict_native=False` for explicitly unqualified CPU fixtures. Native entry identities/scalars, every captured native Warp/Torch value, explicit arrays, closure cells, local RNG state, source action and complete `PEN_MEMORY_DIMS` are checked. Byte comparison preserves NaN payload and signed zero. Unknown foreign mutable classes fail capture until declared; native external MuJoCo/CUDA handles are identity-only inventory entries, not falsely claimed to have Python-copyable internals.

Native field inventory is based on the installed sources. At k0, `reset_buf/reset_terminated/reset_time_outs` may not exist until the first step; absence is preserved rather than manufactured. `_manual_reset_pending`, `_command_dt`, observation caches/history/delay/class instances, action raw/EMA/target/mask state, motion phase/reset state and sensor lazy cache fields are explicitly required. Instrumentation closures must list external log/branch holders and imported module handles; intentionally external state is not rewound.

The 40-check report uses fake stand-ins, never a native environment or Warp. It establishes these boundary contracts only. Actual native profile coverage, full readback byte equality, integration compatibility, baseline self-replay and measured overhead remain PENDING until a separately authorized bounded worker executes. Full-array CPU readback can be expensive; neither this report nor the helper permits bypassing required inventory checks or increasing the agreed budget.
