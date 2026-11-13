# Fresh-buffer control and canonical contact/constraint-row diagnosis

Previous goal turn progressed in4a70400:404 device arrays/195 scalar leaves
restore exactly, but100ms object fidelity remains failed (24/72), with
identical-world spread up to3.44mm/0.166rad.192/192 saved checks and163 unit
tests pass. Full Goal stays active; no MPC or neural training is started.

Use exactly the v3 saved actual model/state, both5/2.5ms. Compare REUSE_DATA
and FRESH_DATA,128 worlds, three repetitions each, at source times0,20,40ms.
At time0 execute the same original five control intervals/100ms. At20/40ms
query one physical step from the exact saved qpos/qvel/ctrl/time and actual
last-step qacc warmstart, with the next original phase/raw action. Preserve
EMA, fixed mocap, no-aid, original parameters and scalar options. Explicitly
count constructor warmup integration outside measured episode/probe ticks.

Save actual PRE/POST state and dense EFC/contact receipts for the first60ms
of time0 and the single-step anchors. Record solver iteration counts and
overflow. Match contact rows by world/geoms/geomcollisionid/edge, and other
rows by type/id, with explicit duplicate/missing/index guards. Keep original
source row addresses as evidence; canonical reordering is OFFLINE READBACK
ONLY, never changing the actual engine, solver, contacts or physical forces.

Compare identical inputs across worlds and repeats: canonical J,D,aref,
contact positions/normals, smooth generalized force/acceleration, actual
constraint force/acceleration and integration output. First locate whether
input, collision geometry, constraint assembly or solver output diverges.
Track the first state divergence and first contact-mode difference rather
than attributing all drift to atomic row allocation. Preserve failed100ms
gates with unchanged v0.64 tolerances; do not select a favorable data mode.

If solver iteration caps are observed, a separately preregistered numerical
quality intervention may follow. It must compare both actual execution and
prediction with declared numerical settings and physical parameters fixed;
it cannot retroactively qualify the default engine. Fresh buffers alone
passing one step cannot qualify100ms or the full7.36s task. No installed
engine/CUDA/driver edits. No failed diagnostic prefixes become experts.

Keep original controlled training/save-resume, matched residual/support and
WM gradients, ego/multiple trajectory/total cost/hardware scope. Later neural
training stops at first healthy checkpoint for restore/ETA and requested pause.
