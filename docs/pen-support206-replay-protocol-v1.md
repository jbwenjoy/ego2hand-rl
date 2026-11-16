# Diagnostic replay of the stopped SUPPORT206 collection

Original session89759 exited1 at SUPPORT206,after collection and before its
optimizer update. Retain its failure.json and all old checkpoint/health bytes.
Original work36143104 native/70560 optimizer;RESIDUAL2000,SUPPORT205 updates
plus unoptimized collection206. The original fixed-budget trial is incomplete.
Its failure record omits failed consistency flags and memory206;that evidence
cannot be recovered directly from the exited process.

Load the saved SUPPORT200 checkpoint using the ordinary unchanged-source/
native/NN/Adam/RNG loader into the same32-world CACHED runtime. No source,
config,physics,gates,reward or counter rebinding. Run at most6 further64-control
collections (through206),and update only after all original consistency gates
pass. Fixed maximum98304 native/192 optimizer;stop on the first failed gate.
This is separately counted diagnostic work,not the original trial continuing.

Save every raw memory batch before testing consistency,all health flags and
post-update checkpoints. On a failed batch,save CPU modules/optimizer/counters
and the failed raw memory without pretending this is a loadable native resume.
Keep any NaNs/metric errors. Record actual native count independently and do
not optimize an invalid batch. Source receipts verified before/after.

Immediate ordinary reload byte equality does not establish future W32 physics
repeatability. Replayed trajectories may differ and need not reproduce the
original failure. Report differences honestly;do not infer the original failed
flag solely from a replay or EPA warning. No final2000 comparison or success
claim is authorized by this diagnostic.
