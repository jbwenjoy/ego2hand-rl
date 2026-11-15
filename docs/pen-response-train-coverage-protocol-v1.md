# Contact-rich TRAIN coverage, registered before collection

This cohort addresses the failed old WM physical-response diagnostics. It does
not train a policy or establish a WM-gradient benefit. The existing324 CHECK
responses remain NEVER_FIT. This protocol and config freeze before execution.

Use the two fixed final64 actors, each in two newly constructed W1 runtimes.
Transfer actor and normalizer weights only through the explicit maintained
frozen-weights loader. Verify historical frozen snapshots, objective/adapter,
joint order, action/observation and all404 W1 Model arrays plus scalar equality
against the historical W1 context. Do not load old native state, optimizer,
critic or RNG. The CHECK context supplies only a full Model compatibility check;
none of its physical states, controls or response labels is TRAIN data.

Before policy execution, materialize71 raw Gaussian noise20 draws for each
PCG64 seed8801/8802, sigma.01. At each original control use actor mean plus the
scheduled noise through the unchanged clamp/scale/EMA/PD pipeline. Log submitted
actions and clipping counts. Pair schedules between arms; this is NOT two
independent trained seeds. Each execution starts at phase1, has no post-start
reset, and offers sources at phases12/24/36/48/64/72 (times.22/.46/.70/.94/1.26/
1.42s). Complete all four executions BEFORE any counterfactual restore.

Source gates retain nonterminal, eight previous native samples, geometry2mm/
.2mm/.001rad, nonhand<=.1N, mean hand normal>.1N, finite, no metric errors, no
object aids and unclipped actor-mean baseline. Preserve every unavailable or
excluded declared source with no replacements. Baseline is the actor mean,
with the actual noise-history EMA restored, not a new stochastic draw.

Generate four fixed unit Gaussian directions20 with independent PCG64 seed9301.
For each source use two base repeats plus four directions x two PD amplitudes
.002/.005rad x two signs x two repeats =34 branches. Alternate sign order by
repeat. Submit base_raw+4*sign*epsilon*direction; require raw unclipped and
actual held PD20 equal baseline_PD+sign*epsilon*direction within2e-7rad. Execute
exactly eight original2.5ms native ticks, without assistance. Keep every raw EFC
packet, pose, qvel, actual qacc_warmstart, geometry, PD, time and observer audit.

Use default STRICT restoration for every branch. The read-locked epoch prototype
is not enabled: its original impulse failure remains unresolved. Source/state/
observation checks remain unchanged. Never relax a physical threshold to retain
a row. Native budget if all states available is2272 prefix+6528 probes=8800;
zero optimizer steps. Missing/excluded sources reduce probe work, not the
declared denominator. Record actual work, including failures.

Independently decode actual GPU EFC edges using the maintained decoder and
state55+actualengineWarm26+actualPD20+nativeDt1 chart. Output delta52/nextWarm26/
source BODY moving-COM wrench6/Fn6. Require original force1e-4N, torque1e-5Nm,
impulse5e-7Ns, overflow/no-aid/nonhand and rotation-chart checks; independently
retain geometry legality, finite and online-source masks. Do not substitute
host contact force/adhesion for actual EFC force. Preserve rejected rows and
all individual failed gates in preparation; never silently replace them.

No WM fitting is authorized by this data-only protocol. A subsequent registered
fit must use common initialization/data/update budget, scales from valid TRAIN
only, declared response losses and separate CHECK diagnostics. Current CHECK
was previously inspected and is developmental NEVER_FIT, not a final test.
These sources use the SAME original reference as CHECK. Independent noise
histories improve state coverage, but do not demonstrate original-trajectory
generalization. Final claims require whole-original-trajectory splits and
multiple independent trained seeds with full total-cost accounting.
