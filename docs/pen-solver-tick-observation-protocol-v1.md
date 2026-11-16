# Same-solve-exit observation: preparation protocol v1

This prepares a targeted diagnostic, not another training run or an engine fix.
Do not run native replay until the existing full54 producer AND final V3 review
are terminal and successful workflows. Task/physical stops inside those54 are
outcomes, not reasons to skip final review. Keep all old failures and masks.
This protocol changes no installed files, native physics, tolerance, integrator,
feasibility gate, backend identity, optimizer or normalizer.

The five original impulse failures in the fixed676-query diagnostic are all
included. Each has a previously qualified comparison condition. Four comparisons
are previous controls with different sources: they are descriptive controls,
not same-state action contrasts. The SHAM P064 ALT/BASE pair shares a source and
nominal action. No case is selected based on whether a new replay reproduces it.

Use two fresh processes A/B,3 numerical repeats per process, both observer
scopes and all10 conditions:120 controls,960 native ticks maximum. Alternate
scope order by process/repeat and rotate condition order within each teacher.
Restore the entire owned native4/Python/RNG source before EVERY scope, verify
all copied arrays/scalars and original observation, and use the recorded RAW20.
Never migrate old W32 states or fabricate a source schema. All queries, failed
restores, attempted native ticks, graph-construction work and discarded outcomes
must be accounted separately. Setup/graph capture is not credited as physical
execution without observing actual native progress. A failed contract stops
that run and remains an immutable failure report.

The baseline uses the existing step graph and original8-tick packet observer.
The instrumented scope builds a separate step graph in its fresh process with
a temporary process-local wrapper of the pinned solver `_solve`. Call the
original function unchanged, then copy read-only data/context into external
buffers before returning to integration. Restore the binding in a finally
block. No host readback inside CUDA capture. No solver branch/iteration change,
and no observer buffers inserted into the owned runtime. The graph, Python
context owners and copy buffers must outlive all replay. The future worker must
attest the original code object and installed source pins, reject compact/sleep/
island or unexpected solve counts, and distinguish the step graph from the
later env-forward graph. This capture implementation still needs GPU
qualification; CPU preparation does not claim the hook works.

At solve exit copy qacc, qacc_smooth, M, efc.Ma, qfrc_smooth, constraint/bias/
passive/actuator/applied forces, pre-integration qpos/qvel/time, nefc,
solver_niter, and SolverContext grad,grad_scale,grad_dot,newton_decrement,
improvement,done,ls_exhausted,alpha,state/quad change counters. Record full model
M_mulm_rowadr/col/madr and solver/integrator/timestep/tolerance/meaninertia.
Packed M may be present even when model.is_sparse=false; use actual array
layout. Independently compute captured M*qacc in CPU F64; optionally a separate
GPU product may be recorded with its dtype, without writing native state.
Keep original post-tick packet/EFC/actual controls and compare copied qacc/
constraint with the packet, and solve-exit pose/velocity/time with the same
tick's pre-state. Do not substitute end-of-control forward caches.

For object WORLD linear DOFs define G=grad_scale*grad, D=independent M*qacc,
A=stored efc.Ma, S=qfrc_smooth, C=qfrc_constraint. The acceleration/force part
of the original impulse residual has the exact algebraic decomposition:

    dt*(m*qacc-C-m*g)
      = dt*(m*qacc-D) + dt*(D-A) + dt*G
        + dt*(A-S-C-G) + dt*(S-m*g).

These terms separate mass-chart/coupling, accumulated-Ma drift, actual scaled
gradient, gradient/force-recovery mismatch, and smooth/gravity contributions.
Keep signs/vectors; component norms do not add. Combine with the original
velocity-update, contact-label and gravity-representation terms only after
same-tick ownership and all original decoder checks are verified. Arithmetic
closure does not prove physical correctness or solver convergence.

Record all terminal OR predicates (improvement, gradient norm, Newton decrement
below tolerance and iteration cap) rather than assigning a unique stop cause.
F64 descriptive recomputation near float32 branch boundaries is not evidence of
the actual iteration-local branch. In particular, a tiny scaled gradient need
not eliminate a defect caused by accumulated Ma. Do not claim a root cause
until failing-tick evidence separates these terms.

Compare full raw output arrays across scopes and repetitions and report all
differences, even when nominal actions/restored sources match. A changed
trajectory does not automatically invalidate a diagnostic, but prevents a
byte-identical-observer claim. Report original/new impulse masks for all120
conditions and event locations. Preserve non-reproductions and nonfinite rows;
no new fitting or Actor admission follows from successful observation.

CPU preparation verifies the10 original records, exactsource hash/tree/layout,
the old raw/EFC decoder/masks, packed-M product and positive mass/dt/settings.
It records prerequisites without requiring the still-running full54 to have
finished. It performs0 native steps/forward/optimizer/normalizer operations.
It cannot establish failing-tick gradients, reproduce physical failures, prove
the observer has no effect, or fix any label. Full M0–M5/A1/A2/ego/hardware scope
and v1.45 remain in force.
