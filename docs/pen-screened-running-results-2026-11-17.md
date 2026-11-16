# Full54 running checkpoint and recorded impulse localization — 2026-11-17

Goal ACTIVE/incomplete. The registered full54 producer remains running;the
checkpoint here is immutable independent review V3 completed_10. It covers all
9 residual-teacher controllers of repetition0 and the first support baseline,
not all54 outcomes. No new Actor/WM training or normalizer fitting.

## Fixed repetition0 residual-teacher preview

All9 declared controllers shown in fixed order,no best-case selection. Each is
a fresh LIVE W1 final2000 actor with the original backend and no object aids.
Original/sham/random controls plus3models x OBSERVABLE/FULL. Models propose a
fixed positive RAW .008 direction;native two-query screening chooses outcomes.
This is an offline simulator oracle,not an imagined-trained or hardware Actor.

| Controller | Executed selected path (s) | Physically qualified prefix (s) | Stop |
|---|---:|---:|---|
| Original actor | 0.54 | 0.52 | Physical audit |
| Same-action sham | 2.20 | 2.18 | Physical audit |
| Random direction | 1.34 | 1.32 | Physical audit |
| RESPONSE_ONLY OBSERVABLE | 3.48 | 3.48 | Task failure |
| RESPONSE_ONLY FULL | 3.48 | 3.48 | Task failure |
| PATH_POINT OBSERVABLE | 1.22 | 1.20 | Physical audit |
| PATH_POINT FULL | 1.30 | 1.28 | Physical audit |
| PATH_POINT_RESPONSE OBSERVABLE | 1.78 | 1.76 | Physical audit |
| PATH_POINT_RESPONSE FULL | 2.62 | 2.60 | Physical audit |

Full7.36s successes0/9 at this numerical repetition.898 selected controls,
1763 native queries/14104 native integrations. Seven physical-audit stops and
two task failures remain separate. Executed time includes the final failing
control;qualified prefix excludes an invalid physical control. Earlier videos
used their registered task stop rules;retain this per-control EFC-audit stopping
rule when comparing durations. These outcomes do not establish a model ranking,
independent-seed improvement or unseen-reference/ego generalization.

The sham accepted one byte-identical RAW proposal because identical restored
inputs can have different numerical futures. It is not evidence of a learned
improvement. All54 repetitions/controls and this control remain in the study.

The nine-panel preview uses selected saved2.5ms native qpos/mocap poses,without
interpolation or new dynamics. Original full-state chains were independently
checked;only selected branches appear. Top row controls,middle OBS,bottom FULL.
20x slow,stop/failure panels explicitly freeze.5/5 encoded/decode checks,manual
start/mid/end image QA complete. Preview is not the final54 result. Rendering
overlapped the running support/random case for61.886s;its before/after producer
observations are saved. Device/case wall times include concurrent CPU review/
graphics activity and cannot be treated as isolated speed benchmarks.

## Completed-case review and input ownership

Shared review implementation recomputes every native/EFC/input/reward packet,
positive proposal,selection/missing-branch rule,restore label and full native/
Python/RNG source chain. It also recomputes model gradients on CPU with fixed
broad cross-device math bounds(values scaled L2<=1e-5,RAW<=1e-4),not new physical
tolerances or proof of native-gradient truth. V3 has16/16 checks per completed
case,including full file capsules captured BEFORE the review and fully hashed
AFTER. Capsules include gradients/proposals/selection/raw packets and transports.
First4 summaries exactly reproduce prior partial reviews.

V1/V2 partial outputs remain unchanged. V2 replaced V1 mutable progress with
immutable numbered indices;V3 rechecks cases once to add previously missing
before/after numeric-input pins. Shared scientific math,model/selection bounds
and original physical masks are unchanged. Canonical continuation tool is
review_pen_native_screened_rollout_v3.py. Final all54 review still pending.

## Why the impulse audit needs a same-tick solver investigation

Fixed diagnostic subset:16 node-qualification queries+281 lifecycle queries+
all379 queries of the first3 full54 controls=676. All original data and5e-7Ns
impulse masks retained,including5 failing queries. Recorded F64 algebra splits
the original impulse residual into velocity-update,acceleration/force-balance,
contact-label/native-force,and native/decoder gravity-representation terms.
All original decoder residual arrays reproduce exactly;all recompositions pass.

At all5 failing peaks,the signed projection of the acceleration/force-balance
term onto the total residual is .999437755–1.000011549. Projections can exceed1
when the other vector terms cancel;term magnitudes do not add as scalars. The
other terms are much smaller. This localizes the recorded gap but does not
prove its optimizer/rounding cause. Free-object damping,armature,frictionloss,
spring,fluid and direct actuation checks pass in these models;no object aid.

The first diagnostic is preserved as FAILED3/6:its Euler-only assumption is
false(all9 model files use implicitfast=3),and the terminal cached acceleration
was wrongly compared with the final native substep. Installed environment code
performs sim.forward() after the8 integrations. In SUPPORT_P058,qpos/qvel still
match the saved last tick while qacc/constraint caches change. Those terminal
Ma/smooth/constraint fields are not a failing-tick SolverContext observation.
Corrected V2 passes6/6,reusing only the prior676 partitions after their algebra/
decoder/mask subchecks pass,and separating endpoint cache context explicitly.
562 CPU unit tests pass;no new native integration/optimizer work in this analysis.

Installed solver code defines a gradient using Ma-smooth-constraint and can
recover constraint force from this gradient. Numerical early stopping is also
described in the [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/stable/computation/).
These sources motivate collecting SAME-TICK internal gradient,Ma versus direct
M*qacc,and smooth/bias/passive/actuator terms after full54 closes. They do not
establish whether optimizer residual,incremental Ma arithmetic or another engine
path caused the5 failures. No native threshold,integrator,physics,installed file,
driver/CUDA or running experiment has changed. Figure shows all676/all5,not a
selection of favorable cases;V1 failed diagnostic remains linked in V2.

## Next work and unchanged scope

Finish the same live full54 job without restarting after observation timeouts.
Review newly sealed cases with V3;after all54 close,produce aggregate outcomes,
complete-case failure/SHAM comparisons,cost accounting and both fixed-repeat
videos. Only then design the small same-tick solver observation experiment if
the balance gap remains material. No permission or deployment stage is implied.
Full M0–M5/true A1/A2,multireference/independent training seeds,total adaptation
cost,real ego assets/gravity/noninertial/dynamics/RGB and hardware contracts
remain active. Earlier67/23 masks,bad centers,failed8-tick serialization and
epoch anomaly remain in the parent chain. Goal is not complete.
