# Saved endpoint epochs and prospective ray recorder

The previous termination comparator is closed at240 queries/1,920 integrations.
This work uses saved data and actual CPU arithmetic fixtures only; no new
simulation integration,training or normalizer update occurred.

`runs/pen_linesearch_endpoint_audit_v1/report.json` passes15/15. It reads all960
observed ticks from both fully reviewed processes and all35 failed-query peaks
(23 original,12 candidate),verifies the exact installed solver source pin and
saves a byte-identical source copy. Two source regression tests check actual
call order and reject changed flag/acceptance semantics. All original masks
and the13/13 archived comparator checkpoint remain unchanged.

Three source facts change what the next diagnostic must measure:

1. _solver_iteration runs line search before constraint and gradient/search
   updates,then applies stopping. The final alpha/improvement/noise flag and
   final gradient/search do not share the same sub-iteration phase. The saved
   search is the updated direction; its negative gradient dot product alone
   cannot prove the previous rejected ray was a descent direction.
2. ls_exhausted is assigned abs(alpha)<noise_floor. It is a step-noise marker,
   not a line-search loop-budget flag. Earlier uses of “exhausted” refer to
   that recorded field,not proof that50 line-search iterations were executed.
3. The pyramidal/equality quadratic helper already evaluates shifted costs.
   Friction branches also contain absolute-cost differences. A blanket
   total-cost cancellation diagnosis is not supported by the existing copies.

All35 failed peaks have state_changed_count1,quad_changed_count0,
search_unchanged=false,grad_scale1. Their last alpha/improvement are zero and
noise flag true. Source confirms that the flag forces a state-change count,
which selects gradient/search rebuilding. This is consistent with the rebuild
path,not evidence that the old ray inputs were identical to final saved values
or that rebuilding the Hessian was required when no quadratic state flipped.
Public qfrc_constraint is recovered from Ma,smooth and scaled gradient after
the solve; it is not an independent reconstruction of the earlier ray reduction.

The factory prototype now records actual ray/GPU reduction/bracket variables
at their original kernel sites. Its canonical generated AST is verified,all
added writes target dedicated diagnostic outputs, and removing diagnostics
recovers the original physical function AST exactly. The generated module
aliases original pinned helper objects. No installed file is edited.

Actual CPU kernel fixtures compare original and generated kernels,not fake
arrays: direct acceptance,bracketing,already-done,equality,quadratic friction,
saturated-friction crossing and diagnostic-cap overflow. Physical outputs are
byte-identical across all seven fixtures. Tests also verify that copied qacc/Ma
are the entry values before the physical output update,valid counts and row/
vector columns are correct,and an overflow flags an error without changing
physics. This does not establish GPU parity,neutrality or all model cases.

The first CPU prototype placed the noise_floor telemetry before its definition.
Warp rejected that generated kernel before any observed-kernel launch or native
work. Preserve revision a5636ab and runs/pen_ray_trace_factory_v1_tests.log.
The corrected placement is in c9f91e5; later tests include the additional branch
fixtures in abd6635. Earlier full607 logs remain distinct historical stages.

The independent FP64 fixed-ray reference is also implemented. Analytic values,
active-set/friction branch changes,finite-difference derivatives,tiny shifted
costs and explicit unsupported-input rejection pass three focused tests. This
reference evaluates the recorded ray inputs; it is not full-dynamics truth or
a native action-gradient oracle. The existing solve-exit payload cannot yet
feed it the missing same-phase row/ray inputs from the failed Hand2 cases.
The full610-test CPU suite passes in10.969s at
runs/pen_ray_trace_tests610_v1.log. The child process deliberately disables
CUDA visibility; its no-device warning is not a driver or GPU failure.

Follow pen-paired-ray-trace-protocol-v1.md for the minimum aligned payload,
two new observation scopes and a prospective240-query/1,920-integration cap.
The coupled live observer,post-gradient copier,graph/alias guards,native worker,
independent complete saved reader and registration remain pending. Factory
qualification alone is not native readiness. Candidate solver admission,
full-task benefit,true A1/A2,reference feasibility,GPU performance,ego dynamics/
assets/gravity/RGB and hardware observation/control/latency remain open.
