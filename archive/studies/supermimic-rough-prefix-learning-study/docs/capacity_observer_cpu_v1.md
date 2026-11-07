# Reset-aware capacity observer: CPU preparation

[capacity.py](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/capacity.py) is separate from frozen runtime5736 and the current native A0. It has not been installed into a physical run or training entry.

The single backend field is `env.sim.wp_data.overflow`. Before every call through this Env instance's wrapped `reset`, the observer reads that field once, copies it, records its actual dtype/shape/values and supplied reset IDs/stage, then ORs every integer entry into a cumulative latch. Flag-array shape is never labeled as number of worlds or mapped to particular reset IDs. A partial reset may clear global flags; the accumulated history remains independent of that clear.

Nonzero flags raise `CapacityOverflowError` before native reset can clear them. Missing/unreadable/empty/noninteger flags yield `CapacityUnknownError`; UNKNOWN never becomes zero. The exception and in-memory record are retained. The observer never clears flags, changes capacities, writes a physical state, steps/forwards, touches an actor/WM, or retries a reset/process.

```python
with capacity_observer(env, stage="smoke_warmup_and_updates",
                       fixed_capacities={"nconmax": nconmax, "njmax": njmax}) as cap:
    try:
        # Construct Runner here, so its initial reset is observed too.
        # Run the separately authorized bounded work.
        ...
    finally:
        try:
            cap.final_read()  # after the last physics operation, before env.close
        finally:
            write_capacity_receipt(cap.snapshot())  # Root's bounded metadata I/O
```

`final_read()` is idempotent: an explicit call plus context exit produces only one final backend read. No physics or reset may follow it. Context exit performs the read if omitted and restores the original instance/class reset binding even on exceptions. A body exception is preserved if the final read also fails; the snapshot still records UNKNOWN. Environment close must occur after this context, not before the last read. The caller must persist the snapshot in its own checkpoint/failure/final receipts; this small module does not add an arbitrary callback or own a filesystem path.

`set_stage(name)` changes the next record's stage. `snapshot()` returns a deep JSON-safe copy with cumulative OR, unknown count, per-reset and final read counts, full records and configured fixed capacities. Those capacities are explicitly **caller-supplied configuration metadata**, not an independent allocation measurement. Allocation/configuration identity remains Root's separate native-source contract. `engineering_capacity_qualified` is true only after the final read and an entirely zero, known history; it says nothing about solver convergence or physical accuracy.

[28 CPU checks](D:/workspace/supermimic-rough-prefix-learning-study/outputs/capacity_cpu_contract_v1/report.json) cover normal/partial resets, original argument/return preservation, pre-clear overflow interception, OR remaining nonzero after a later fake clear, final-only overflow, read failure/malformed flags, multidimensional raw masks without world attribution, original exception precedence, method restoration, duplicate installation and final-read idempotence. Fake backend only: no real reset, flag/GPU read, physics or training. The old runtime SHA and A0 sources remain unchanged. A future smoke source version must explicitly install and audit this observer before it can claim capacity across episodic resets.
