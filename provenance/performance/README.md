# Performance evidence

`phase1` records explicit current-state geometry reuse, immutable device
constants, reduced scalar synchronization and the first micro8 comparison.
`phase2` records batched guarded hinge rotations and the final quiet timing.
The phase2 summary maps original local report paths to retained byte-identical
copies. Original JSON pins are historical and have not been rewritten.

Large tensor fixtures, raw GPU samples and profiler traces remain outside Git
under `runs/performance`. A copied report is readable without those files;
re-running the saved-only comparator requires its pinned external fixtures.
All timings exclude setup, fixture transfer and profiling. See
`docs/performance-work.md` for the numerical and sampling limitations.

The first micro2 batch-rotation timing overlapped CPU validation; the retained
quiet repeat addresses that interference. Neither timing is deleted. No
resumable policy checkpoint, native qualification or policy improvement claim
is produced by these private benchmarks.
