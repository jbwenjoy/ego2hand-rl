# Registered whole-batch restore timing extension

Before these measurements,use the same6 current-code sources from the matched
benchmark. Default loader and read-locked epoch run in fixed balanced blocks:
STRICT,EPOCH,EPOCH,STRICT. Each block restores every source3 times,18 restores.
One strict warmup per source precedes timing. No physics or optimizer updates.

Wall timing starts BEFORE epoch construction/entry and ends AFTER exit/handle
cleanup. Thus both full SHA boundaries,canonical path resolution,file locks,
identity checks and cleanup are included. Default blocks include every per-load
strict check. Report all4 block durations plus each pair's ratio and pooled
ratio. This extension addresses costs not fully measured by the original
per-restore-only timer; do not claim the latter already included these costs.

Every restore must retain exact arrays/scalars and regenerated observations.
Whole batch byte checks and zero native steps are required. Inputs remain
CHECK/NEVER_FIT. A result only applies to the measured restore batch,not ordinary
PPO collection,geometry kernels,training GPU utilization or end-to-end research.
