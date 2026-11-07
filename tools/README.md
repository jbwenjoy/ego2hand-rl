# Tools

- `consolidate_sources.py`: copy-only historical import; dry-run by default.
- `benchmark_gpu_v2.py`: current fixed-parent private GPU benchmark, including
  exact measured-interval GPU sampling. Weights are discarded; no resumable
  policy checkpoint is saved.
- `compare_gpu_benchmarks.py`: NumPy-only comparison with a fixed tolerance
  preset and an explicit microbatch-change flag.
- `benchmark_gpu.py`: frozen preliminary v1 tool retained to match its report
  hash. Its whole-process GPU median includes CPU profiler export/aggregation
  idle time; use v2 for new measurements.

These tools share the implementation in `src`. They do not maintain separate
research runtime copies or authorize production training.
