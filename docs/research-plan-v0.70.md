# Explicit device tolerance after the library float32 conversion floor

v0.69 factorial has no all-world passing coarse variant; no calibration winner.
Full v1 DEFAULT/fallback16 cases all fail, EFC109/112. Requested/host/device
receipts expose a configuration discrepancy: BOTH_STRICT host1e-12 is still
GPU1e-6. The installed MuJoCo Warp io.py:427–430 intentionally floors native
tolerance to1e-6 for float32, avoiding needless oscillating iterations. The
previous snapshot/local probes explicitly set the device array after conversion;
plain environment configuration did not. Preserve all v1 outcomes and sources.

Add an explicit opt-in numerical_device_override flag. Default false retains
the library floor. When true, apply only declared validated numerical settings
to existing device arrays BEFORE reset/execution. Scalar iteration caps must
already match configured capture-time values; fail otherwise. Array pointers
stay fixed and graph nodes read updated values. Record before/after native/GPU
receipts and flag. No installed-source, mass/material/dt/CCD/driver changes.

Repeat the SAME16-case cohort with a new frozen config: DEFAULT remains the
unchanged library policy; fallback BOTH_STRICT explicitly requests the device
override. There is still no calibrated winner. Compare actual solver counts,
force/moment EFC identities, original impulse gate and complete task gates.
Do not use the configuration fix or passing one case as a global solver repair,
100ms deterministic-fidelity pass, learned-policy gain or expert data.

Verify all source/control/state/physical-table invariants and requested versus
actual float32 numerical values. Preserve mismatched v1 rather than rewriting
it as if the requested tolerance had been active. This is a diagnostic numeric
intervention below the library default floor and needs measured outcomes.
Full Goal and future first healthy neural-checkpoint restore/ETA pause remain.
