# Performance work after consolidation

The migration keeps current numerical behavior. These changes are the next
performance work, not claims of speedups already measured.

1. Measure forward, backward, geometry, validation/synchronization, sampling,
   logging and checkpoint costs on the same recorded batches and update
   budget. Keep setup and checkpoint health separate from steady-state time.
2. Compute current geometry once for a particular state and share poses,
   relative poses, directional features and known masks among observation,
   world-model and engineered-support consumers. Static replay caches and
   differentiable imagined-state caches have different lifetimes. Never reuse
   geometry across different predicted states or detach a required gradient.
3. Consolidate device checks and CPU transfers at meaningful boundaries.
   Preserve failure detection, finite checks and explicit unknown masks;
   removing all validation would change the research contract.
4. Audit tensor residency by category. The current offline trainer already
   places its encoded replay and targets on CUDA. Repeated construction of
   constants, CPU sampling/index transfers and hot `.item()`/`float()` calls
   still need measurement. It is inaccurate to classify all fixed data as
   remaining on CPU.
5. Evaluate vectorization and batching for the batch8/H2/microbatch2 workload,
   then consider compilation. Keep the original Float32 NN / Float64 geometry
   and physics, masks, normalizers, source-frame impulses, real batch IDs and
   update counts as the comparison baseline. AMP/TF32 or different batch
   budgets require a separately declared comparison.

Use the shared runtime once. Do not patch every archived study copy. Report
wall time, GPU utilization, allocated/reserved memory and numerical/gradient
differences together; utilization alone is not the acceptance criterion.
