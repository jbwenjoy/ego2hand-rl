# Multi-control native-screened suggestion design — not registered or launched

This is a concrete next-stage design. Implementation,live capture/restore
qualification and storage checks remain prerequisites. It is not a completed
controller,registered launch,Actor admission or a replacement for genuine A1/A2.

Use fresh LIVE W1 exact-backend resets and final R/S2000 actor/normalizer weights
through the existing independent full-W1 Model witness. Do not import W32 native
state. Goal follows each teacher arm. Preserve original368 controls/7.36s,actual
2.5ms physics,zero object aids,fixed wrist,original geometry/force/torque/5e-7Ns
impulse gates. All three fixed WM families and both same-network score modes
OBSERVABLE/FULL remain in the design;the inspected decomposition does not select
a winner or delete the explicit path term from all future comparisons.

Per teacher:6 model/mode controllers plus UNMODIFIED_ACTOR,NATIVE_SHAM and
NATIVE_RANDOM=9 cases. Two teachers give18 cases per numerical repetition,three
repetitions54 cases. Repeats are not independent training seeds or unseen
references. All terminal/missing/invalid controls stay declared. This remains a
controlled perfect-state study,not ego/hardware validation.

At a live node,capture complete owned native4 state and keep actor/WM modules
external to runtime inventory. Reconstruct input102 from actual qpos/qvel,
warm26 and actual float32 controller PD,with original reference phase/EMA/bias/
limits and dt. Retain sampled RAW before outer clamp and applied RAW separately;
never replace actor RAW with its clamp silently. Clipped coordinates may have
zero derivative. Any newly clipping proposed coordinate,zero/nonfinite gradient
or nonfinite WM prediction falls back to a declared duplicate BASE proposal;
do not project or tune a new amplitude. The fixed nominal trust radius is the
original smaller radius .002,RAW .008;actual PD movement after limits is recorded
and need not retain nominal norm outside the original unclipped probe domain.

For each screened controller,query BASE and a positive normalized-gradient
proposal from the SAME complete source. NATIVE_SHAM queries BASE twice;
NATIVE_RANDOM uses a fixed stateless seed/index-derived unit direction with the
same raw radius/feasibility rule. Every screened decision has the same two-query
cap/recording and no hidden oracle selection of signs or radii. UNMODIFIED_ACTOR
uses one query. Model input/gradient evaluation occurs before branching;the
original policy/WM weights and normalizers never update.

Audit both native packets before acceptance. Baseline invalidity ends the case
with an explicit physical-audit failure;it is not hidden by a model rescue.
Accept a nonfailure physically valid candidate only if its original full goal
reward exceeds BASE by the fixed1e-4 margin;otherwise restore the FULL captured
end-BASE state. Save complete restoration readback including Python/RNG/warm/
cached/metric state. A terminal BASE may be replaced only by a valid nonterminal
candidate satisfying the SAME fixed rule;completion/truncation is distinguished
from failure. This is offline simulator branching,not an executable hardware
oracle or an imagined-Actor update. Actual and attempted queries are counted by
an external counter that is never rolled back with runtime counters.

Worst per repetition:16 screened cases x368 decisions x2 queries x8 ticks plus
2 unmodified cases x368x8=100096 native integrations;three repeats max300288.
Count failed setup/qualification/partial queries separately,all source capture/
restore/NN/audit/IO wall time,selected-path native time and discarded queries.
Do not claim a native-budget or wall-time win against PPO without total upstream
data/model/setup costs and genuine later matched adaptation experiments.

Before launch qualify the complete source/end-BASE capture,branch and rollback
on GPU,clipping/zero-gradient/terminal/failure/phase transitions and unchanged
modules. The new input adapter's36-source/30-input regression alone does not
prove these lifecycle properties. No step/phase manipulation or old source
schema relabeling to bypass a loader contract.

Storage is a launch precondition. Current frozen source payload is about6.44MB;
full per-node duplicate states can exceed available space across54 cases.
Qualify lossless full-snapshot compression/deduplication or retain complete node
sources with endpoint readback/digests and periodic full endpoints. Compression
must reconstruct identical bytes/types before any restore;never discard an
audit-required state or edit/move legacy assets to meet capacity. Measure actual
footprint and keep a reserve before registering the executable run.

Full native force audits and source receipts remain;no epoch/hash cache,
physics/default/installed file or driver/CUDA change. Original controlled task,
M0–M5/true A1/A2,multi-reference/independent-seed total costs,real ego assets/
gravity/noninertial/dynamics/RGB and hardware contracts remain active.
