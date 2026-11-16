# Research plan v1.53 — phase-aligned ray recording, CPU components implemented

Preserve v1.52 and all original M0–M5/true A1/A2 requirements. Goal ACTIVE.

1. Offline source/endpoint audit is complete15/15 over960 saved ticks and35
   failed peaks. Final alpha and final gradient/search belong to different
   phases. ls_exhausted is a noise-floor test,not a line-search iteration-cap
   indication. Quadratic costs are already shifted. Keep every historical
   physical mask and the completed comparator evidence unchanged.
2. The diagnostic kernel-factory generator is implemented in shared src. Its
   physical AST is exact after removing diagnostic writes. Actual original/
   generated CPU kernels preserve physical outputs over seven branch/overflow
   fixtures. The initial undefined-noise_floor CPU failure and corrected source/
   logs are preserved. Independent FP64 fixed-ray evaluation and analytic/FD
   tests are implemented; full610 CPU tests pass in10.969s. This is not native
   dynamics or action-gradient truth.
3. Next implement the same-iteration post-gradient copier and coupled observer,
   validate owner/alias/binding/graph cleanup,then implement the worker and full
   independent saved reader. Generate an explicit observation identity and
   register exact inputs/schedule before new native work. Follow
   pen-paired-ray-trace-protocol-v1.md:all10 conditions,2 algorithms,3 repeats,
   2 processes,solve-exit-only versus solve-exit-plus-ray scopes. Prospective
   total cap240 queries/1,920 integrations,zero updates; no current live run.
4. Use actual ray inputs,reductions,trial deltas and bracket decisions plus
   paired post-update states to separate cache/epoch mismatch,reduction effects,
   trial-cost behavior and stopping. Preserve unsupported rows,scope effects,
   overflows,partial failures and complete costs. No guessed precision fix,
   relaxed gate,tolerance/iteration change or candidate training admission.
5. Keep main-task requirements in scope:controlled reference feasibility,
   true A1/A2 and multi-step/action-gradient checks,independent references/seeds,
   full-task gains,ownership-defined fitting/validation,GPU performance/total
   adaptation costs,real ego assets/dynamics/gravity/RGB and hardware subversion/
   observation/control/gains/latency. These remain incomplete.

Evidence: pen-linesearch-epoch-audit-results-2026-11-17.md,CPU factory
qualification receipt and the completed prior comparator checkpoint. Source/CPU
progress does not complete the native diagnostic or the full research goal.
