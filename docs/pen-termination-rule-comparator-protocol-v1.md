# Prospective termination-rule comparator v1

This is a controlled numerical-backend experiment, not policy training. The
full54/actual V2 checkpoint remains immutable. The original5e-7Ns impulse gate,
all other physical/task gates, RAW/PD actions, model parameters, tolerance,
implicitfast integrator, collision backend and max50 solver iterations remain.

## Identity and one changed expression

The candidate kernel in src/supermimic_research/runtime/
pen_termination_rule_kernels.py preserves the complete installed _solve_done
function AST except its name and the done assignment: remove only
`improvement < tolerance` from the original OR expression. It retains the
gradient and Newton-decrement predicates, iteration update, already-done return,
shared/batched tolerance indexing, _rescale implementation and active-world
atomic decrement. pen_termination_rule.py rejects any second AST change and
attests loaded function code against compiled pinned source before binding.
Installed files are never edited. Exclusive process-local binding synchronizes
and restores even on exceptions. It does not itself own a runtime or transfer
historical state.

The composite candidate identity includes the qualified collision-backend ID,
installed solver pin, candidate source pin, adapter source pin and named rule.
It explicitly starts physically_qualified=false and
old_source_transfer_qualified=false. A new graph and physical transition must
carry that identity. The old collision ID alone is insufficient identification.
The original arm must explicitly identify installed-original termination.

Actual CPU tests cover unchanged/deleted stopping branches, a fully
unconverged case, strict equality at tolerance, cap50, already-done worlds,
single and batched tolerance, active counts, exclusive binding and cleanup
including synchronization error. Source and tests are part of the prospective
qualification receipt. These tests do not qualify CUDA graph execution,
state transfer, native force closure or task benefit.

## Fixed population and budget

Reuse all ten declared preparation sources and exact RAW20/held PD:
SUPPORT_P058/P057, UNMODIFIED_P027/P026, SHAM_P064_ALT/BASE,
SHAM_P110/P109 and RANDOM_P067/P066. Do not select only the newly reproduced
failures. Historical observations are inspected, so this is not a blind test.
For each source:3 numerical repeats x2 fresh processes x2 observation scopes
(PACKET_BASELINE and PACKET_PLUS_SOLVE_EXIT) x2 termination algorithms.
Total240 queries,8 ticks each =1,920 actual/attempted integrations maximum;
120 queries/960 per fresh process. Optimizer and normalizer budgets are zero.
Counterbalance algorithm/scope order. Save an exact schedule before execution.
No continuation beyond one historical control per query; no policy loaded.

## Explicit source transfer and graph prerequisites

The changed backend cannot silently restore an old-backend snapshot. Implement
an explicit import receipt which references the original compressed source,
retains every native/Python/RNG value and old provenance, and adds only the new
declared backend metadata. Compare the entire imported numerical tree and
inventory with the old source before mutation and after readback. Reject any
unlisted difference, missing field, changed model scalar or reconstructed
numeric array. An external runtime identity registry can avoid injecting new
mutable leaves into the owned Python tree. This importer and runtime identity
are pending, not supplied by the kernel-binding helper.

Keep old sources/pins intact; store imported sources and transition receipts
separately. The import establishes a deliberately initialized candidate
experiment, not equivalence of physical futures. Bind the selected solver rule
before graph construction; retain graph/context owners and attest graph/binding
identity. Compare whole native4/Python/RNG and clock/counters before/after every
graph capture. Preserve and count any unexpected setup advance, then stop.
The first native query is CUDA qualification within the same budget. Do not
restart a partial run or erase failed work.

## Evidence and interpretation

Save full original packet traces, actual solve-exit raw28/physical26/empty-Mgrad
copies, source/end trees, all original/new masks and task flags. Separately
recompute force/impulse arrays and stored solver predicates in a saved reader.
Report all240 conditions and all matched original/candidate and scope pairs,
not just rescues: valid->invalid and invalid->valid counts, peak full original
impulse and components, iteration/cap/decrement/gradient/improvement values,
state/force changes, nonconvergence and setup/query/audit/total wall costs.
Earlier49/60 byte-equal scope pairs preclude assuming observer neutrality.
Scope effects and ordinary numerical variation must remain in the analysis.

Dropping improvement-only exit may merely run a stalled solve to50 iterations;
that is a possible negative result. Keep the gradient/decrement OR unchanged
to isolate this candidate. Reaching50 is not convergence. Improved local force
closure would still require physical-backend requalification and new data
ownership before WM/Actor training, full-task trials, independent references/
seeds, true A1/A2 or real ego/hardware claims. A new solver backend cannot
retroactively repair the54 historical results or admit action derivatives.
