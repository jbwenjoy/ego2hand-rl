# Explicit single-resolution full-rehash verifier candidate V1

Current STRICT native profiling identifies repeated file reads/canonical-path
resolution. Original verify_file_pin resolves once,then file_pin resolves that
same target again. Candidate removes only the second resolution,retaining each
call's full1MiB-chunk read/SHA256/size check,canonical returned path and ordinary
exceptions. No mtime/content/path cache,read-lock epoch or silent native hookup.
No default source/engine/manifest verifier or installed file changes. Neither
original nor candidate supplies an atomic filesystem snapshot under concurrent
mutations;do not claim a stronger security boundary.

Tests compare missing/bad size/bad digest,relative dot/parent paths,multichunk
files and same-size byte changes with restored mtime. Preserve original behavior;
do not call stamp equality a byte validation. Test artifacts are temporary only.

Register two actual source catalogs before timing:TRAIN R N8841 P024 and reserved
development R N8861 P018. Preserve exact snapshot/observed/engine pin order,all
duplicates and bytes. One warmup per method/catalog,5 timed alternating method
rounds/catalog,and one separately instrumented cProfile pass/method/catalog.
Every call rereads and hashes the complete file;compare canonical returned path
lists. Count Path.resolve calls and full read calls in separate profile passes.
Expect resolve count half,read count equal;do not require a positive speedup.
Report raw timings/medians/order and instrumentation independently.

This is a fixed-file verifier microbenchmark,0native/optimizer,not an integrated
restore/collector/learning speedup or numerical-state qualification. Default
native path remains STRICT unchanged. Integration needs a separate registered
scope/exact binding inventory,negative guards and matched native/physical tests;
old unresolved source-epoch anomaly is not bypassed or declared resolved.
Full M0–M5/A1/A2/task,shared-cost/seed/reference,ego/hardware scope remains active.
