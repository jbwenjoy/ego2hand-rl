# External solve-exit observer: CPU checks and pending-stage refusal

This extends v1.46's103/103 input/arithmetic preparation. The same full54
producer continues unchanged. No solve-exit graph has been built or executed
on the GPU, no new native integrations/optimizer updates/normalizer fits were
performed, and no failing-tick private context or solver root cause is claimed.

The maintained implementation is
`src/supermimic_research/runtime/pen_solve_exit_observer.py` and its tiny external
counter kernel `pen_solve_exit_kernels.py`. It adds:

1. A CPU-only terminal prerequisite check before any CUDA-related imports or
   observer allocations. Require the exact full54 producer schema/8 workflow
   checks and final V3 review schema/5 checks, complete54 unique cases in
   registered order, matching producer/review/registration pins, exact native
   accounting and zero optimizer steps, with no producer failure file.
   Physical/task stopping remains a valid recorded outcome, not an automatic
   workflow rejection.
2. An exclusive process-local `_solve` binding scope. Reject another model,
   another Data, compact solve, nested scope or unexpected changed binding.
   Execute the original function unchanged before the exit-copy callback;
   preserve its return and restore the binding even when solving/copying fails.
3. A separate observed step-graph implementation retaining its private context
   owners and external copy buffers. Copy Model/Data/context read-only inputs
   at solve exit before integration; never insert buffers in owned runtime
   inventory. Validate actual layouts and reject destination/source aliasing.
   Preload the external counter kernel before graph capture. Keep the original
   step graph as baseline and leave the env-forward graph unchanged.

The graph class accepts only the expected W1/26DOF Newton/pyramidal/implicitfast
CUDA configuration and pinned installed solver binding/source. Its external
counter is intended to distinguish solve-copy graph executions from Python
graph construction; it is NOT the authoritative physical-integration budget
counter. A future worker must still count actual/attempted native execution,
verify capture/setup state effects, validate original full native4 restores,
and compare every same-tick packet/epoch as the protocol requires.

Focused12/12 tests pass in1.335s. They cover original-before-copy ordering,
unchanged return values, both failure cleanup paths, wrong-model/Data/compact
rejection, nested/changed binding restoration, missing/partial/failure/mismatched
final reports, duplicate/order/optimizer rejection and successful closed
workflow eligibility even with physical-stop outcomes. Two tests execute actual
Warp CPU copies and the external counter kernel, retaining the context owner
and checking unchanged source arrays; aliased destinations are refused.
These CPU copies do not test CUDA graph capture/replay or the private solver.

Full580/580 tests pass in11.799s with CUDA explicitly disabled in that test
process. This is separate from the active RTX5090 evaluation. No isolated GPU
performance or end-to-end adaptation-speed conclusion follows.

The actual currently pending configuration is exercised through the real graph
observer constructor by `tools/check_pen_solve_exit_observer_guard.py`.
`runs/pen_solve_exit_guard_v1/report.json` passes5/5. It refuses with
"Wait for both complete full54 producer and final V3 review" before inspecting
a simulator or importing a GPU/simulator module; an import blocker records
zero attempted simulator imports. This expected refusal is a passing guard
check, not a failed physical replay. It is not proof the closed-stage graph
construction will succeed.

Guard report SHA256:
`57c027bdffd8ce04a161df66986a5e7fbe64c020c0a9fa2a1228d308845658f0`.

Next finish/review/aggregate all54 and final fixed-repeat videos. Then complete
the replay worker, freeze its source/registration, qualify the graph on GPU
with exact restore/readback and baseline/observed scopes, and run the bounded
two-process120-query/960-tick diagnostic only if those contracts pass. Preserve
all partial failures/non-reproductions and numerical observer differences.
The original5e-7Ns gate, tolerance/integrator/physics/backend, installed files
and drivers/CUDA remain unchanged. No model/Actor/ego admission is implied.
All v1.46 and original M0–M5/A1/A2/multireference/independent-seed/total-cost/
real-ego/hardware requirements remain open.
