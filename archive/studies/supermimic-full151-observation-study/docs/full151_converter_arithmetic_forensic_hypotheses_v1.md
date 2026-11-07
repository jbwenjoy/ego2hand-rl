# Converter forensic hypotheses fixed before new raw diagnostics

Status: **source-only, conditional interpretation proposals; no new raw errors read**. The old source00e1, config160c, failed output and original eight-eps32 output-scaled gate remain unchanged. No MJB, converter, model, objective, WM, FK, collision query, Env or GPU was executed for this review.

The known old outcome is only that the first 452-endpoint gate accepted the first40 q/qd entries exactly and rejected the concatenated other18 p/R/v/omega entries. Its exception text names the whole group; without saved arrays it does not establish which individual channels, rows or magnitudes failed. One passive MJB and one2260-row conversion returned; factory/forward counts remained0. That is an engineering failure preserved as such, not a physical/proxy result.

## Exact source path and semantic alternatives

`SimReader.state` reads raw qpos/qvel from `env.sim.data`, applies the native normalized-wxyz quaternion map, then packs:

```
[q20, qd20, A*(p_world-p_palm), A*R(q), A*v_world, A*(R(q)*omega_body)]
A = recorded_palm_rot.T
```

The frozen CPU converter00e1 constructs FP32 views with the same named MJB indexing and calls the original unbound method. It does not replace it with a float64 formula. The original DataBridge is a zero-copy wrapper of `wp_data`; its qpos/qvel views do not explicitly convert object angular coordinates. Native source applies the BODY-to-world rotation once before the palm rotation. Correct q/qd does not by itself prove object freejoint addressing, frame identity, timing or group joins; those remain explicit forensic checks.

Before discussing arithmetic, the new diagnostic must preserve all2260 raw inputs, all452 endpoint references, the160 leading anchors and all1808 solver object poses. Record mapping addresses and named joint/body IDs, the exact group key/order, raw/pre/post ticks, saved palm matrices/positions, dtype and FP32 byte representability. The semantic checklist is:

1. Same group key `(path,k,branch_id)` and correct final post endpoint, not a nearest or next interval. Per-group pre/post continuity and raw times remain separate from restored-branch concatenation.
2. Correct object freejoint7/6 address, q quaternion orderwxyz, world body-origin linear velocity and BODY angular convention. Handq/qd equality supports the hand mapping only. COM0 does not license a second rotation or a body/world reinterpretation.
3. Same actual saved reader palm frame. Parent world-body ID0 and `body_mocapid==0` are distinct.452 recorded frame values plus fixed-root structure do not measure1808 mocap poses. A copied palm quaternion reconstruction must not replace the saved matrix.
4. No extra state mutation or timing shift between the last raw substep and reward capture. The producer's `raw()` synchronizes and reads postintegration fields; `PenSimReader.attach_capture` captures state before the original reward computation, with no new integrator call in between. This supports the intended alignment but does not substitute for saved per-group evidence.
5. Native `quat_to_mat` normalizes each raw q first. Do not compare an unnormalized polynomial to it or use derived15ms xquat for a post20ms state gate. Keep solver-pose and endpoint conversion tests separate.
6. Exact original `pack` order is q/qd/p/row-major R/v/omega. No rotation geodesic or shaft-only metric may silently replace the old elementwise R gate.

These are plausible semantic failure modes to exclude; none is diagnosed as the cause from the old exception alone.

## What source establishes about precision

The actual physical worker a24c explicitly sets `torch.backends.cuda.matmul.allow_tf32=True` at line362. However native `rotations.mm/mv` are broadcast elementwise multiply followed by sum, deliberately avoiding GEMM. A claim that these small rotations automatically ran TF32 GEMM is unsupported. `quat_to_mat` is also an explicit polynomial after `F.normalize`; no BF16/autocast scope appears around the producer's Reader state path. `with torch.no_grad()` is not mixed precision.

The source proves the intended flag assignment, not a historical runtime flag snapshot. The later WM GPU precision experiment's flags are not evidence about this earlier physical process. CPU/GPU reduction trees, normalization/sqrt/division implementations, FMA/fusion and subnormal handling can differ even when tensors are FP32. `F.normalize` calls `input.norm(...).clamp_min(1e-12)` then divides. Installed `NormTwoOps` expresses square-accumulate/combine/sqrt, but that header alone does not prove every dispatched CPU/CUDA kernel, accumulator type, fast-math flag or error constant used for this length4 call. The conditional bounds below must retain those assumptions.

## Conditional componentwise bounds, not replacement gates

Let u=2^-24 (unit roundoff; eps32=2u), gamma_n=n*u/(1-n*u). Treat saved FP32 inputs as exact real numbers. Bounds require finite normal operations, no overflow, no significant subnormal/flush-to-zero effect and an operation implementation satisfying the stated rounding model. If these assumptions or needed inputs are unknown, label the corresponding interpretation **UNKNOWN**. These formulas are frozen for interpreting the next diagnostic only; they do not change the old gate or authorize scoring.

For a three-term dot, use the conservative one-side envelope

`E_dot(A,z) = gamma_5 * (abs(A) @ abs(z))`.

For two independent CPU/GPU evaluations of the same A,z, a conservative pair envelope is `2*E_dot`. This scales with the actual product magnitudes, not just the final cancelled output. It is not an assertion that errors attain that envelope.

For p, define exact z=p_world-p_palm. A conservative one-side input subtraction envelope is `e_z=u*(abs(p_world)+abs(p_palm))`; subtraction is often much more accurate (Sterbenz cancellation can be exact). Then

`E_p = gamma_5*abs(A)@abs(z) + (1+gamma_5)*abs(A)@e_z`.

For v, exact saved v_world is a direct input:

`E_v = gamma_5*abs(A)@abs(v_world)`.

For quaternion normalization, let L=||q||2, h=q/L; the existing near-unit guard keeps the1e-12 clamp inactive. Under four squared terms, ordinary rounded accumulation, sqrt and division, a conservative provisional norm-relative envelope is

`eta_L = max((1+u)*sqrt(1+gamma_7)-1, 1-(1-u)*sqrt(1-gamma_7))`.

Then a one-side normalized-component envelope is

`e_h[i] = abs(h[i])*((1+u)/(1-eta_L)-1)`.

This expression is a **conditional operation-model proposal**, not a verified PyTorch-kernel theorem. If actual dispatch or approximate sqrt/division violates its assumptions, do not certify it by fitting a larger constant to observed errors.

Propagate h and e_h through the exact native rotation-polynomial tree, without a single arbitrary quaternion tolerance. For one rounded multiplication x*y with input absolute bounds e_x,e_y, a conservative propagated bound is

`e_mul = abs(x)*e_y + abs(y)*e_x + e_x*e_y + u*(abs(x)+e_x)*(abs(y)+e_y)`.

For rounded addition/subtraction use

`e_add = e_x+e_y + u*(abs(x)+abs(y)+e_x+e_y)`.

Apply these rules to the native `1-2*(y*y+z*z)` and `2*(x*y-w*z)` style entries to obtain one-side E_R. Constants1/2 are exact; using generic multiply rounding is conservative. A future numerical implementation should use outward-rounded intervals or explicitly account for evaluator error; ordinary float64 arithmetic is a useful independent diagnostic, not automatically a rigorous enclosure.

For the output palm-frame rotation:

`E_Rout = gamma_5*(abs(A)@abs(R)) + (1+gamma_5)*(abs(A)@E_R)`.

For nested angular transformation, set t=R*omega_body and

`E_t = gamma_5*(abs(R)@abs(omega_body)) + (1+gamma_5)*(E_R@abs(omega_body))`,

`E_omega = gamma_5*(abs(A)@abs(t)) + (1+gamma_5)*(abs(A)@E_t)`.

Two-sided CPU/GPU comparison may use the sum of their separate envelopes (2E only if the same assumptions/budget apply). The term `E_R@abs(omega_body)` can be substantial for large angular inputs, while `A*(R*omega_body)` has a tiny component because of cancellation. The old `8*eps32*max(1,abs(output))` does not model that input conditioning. Whether this phenomenon occurred here is **not known before the saved diagnostic**.

Report separate per-component condition indicators such as

`C_v[i] = (abs(A)@abs(v_world))[i]`,

`C_omega[i] = (abs(A)@abs(R)@abs(omega_body))[i]`,

and C/abs(output), with exact-zero output handled explicitly (infinite/undefined flag, no invented denominator). Preserve units: p in meters, R dimensionless, v in m/s and omega in rad/s. Do not hide discrepancies in one mixed-unit norm.

## Interpretation locked before raw errors

Save original old-gate flags and residuals unchanged for every row/channel. Report the proposed input-aware envelopes and ratios separately, marked conditional. Residuals within a conditional envelope are merely compatible with that arithmetic hypothesis given its assumptions; they do not prove semantic equivalence, restore qualification or a passing new science gate. Residuals outside it mean the proposed arithmetic model does not cover the observation; they do not uniquely prove a semantic bug while kernel assumptions remain unknown. Missing inputs or unsupported rounding assumptions stay UNKNOWN.

The planned converter-only diagnostic must always stop before its first factory/geometry call, even if some old flags happen to pass. No old output is overwritten, no old threshold is loosened and no repaired value replaces a saved reference. Any later revised operative gate requires a new preregistered condition and independent review; these proposals are not a posthoc numerical route to pass.

The machine-readable frozen interpretation and source pins are in [the review directory](../outputs/full151105_converter_arithmetic_source_review_v1/report.json).
