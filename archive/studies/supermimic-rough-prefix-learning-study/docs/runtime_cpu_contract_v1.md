# Rough-prefix runtime CPU contract v1

Implementation: [runtime.py](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime.py), SHA `5736dd1fd678a39f87867461d30d0710e6242af10e42f6fa2d57f6c9301b1ff9`. It is a new physically unqualified research mode; no old native source or checkpoint was changed.

```python
with rough_runtime(ManagerBasedRlEnv, SingleHandTrackingCommand,
                   PenMDP, DexSuperMimicRunner,
                   bundle['arrays'], bundle['descriptor']) as runtime:
    with rough_controller(ManagerBasedRlEnv, PenMDP, DexSuperMimicRunner,
                          bundle['baseline_u15'], bundle['descriptor']):
        # Root constructs Env, restores/verifies actual8, then constructs Runner.
        ...
```

The entry must use the patched base Env directly, without the old ControlledPenEnv reset override that subsequently wrote q0 as the target. Runtime owns command/reset/finite-end/replay/policy eligibility; controller owns baseline+EMA targets, actual-u capture and checkpoint104. Both scopes restore their original methods and buffer bindings.

The original command fixed-wrist method runs first. Runtime replaces `_frames` joint velocity with the frozen analytic qd16 and object world velocity/omega with the frozen quintic 61-table sampled at16 knots. The reader/group FD is retained and copied to a diagnostic field. This occurs during Command construction before the first reset. Adaptive starts and first-frame holds are rejected.

Reset runs the native reset path, then explicitly installs the declared phase0 q/qd/object world state and u0 in selected environments, clears their action/EMA histories, flushes and performs native forward/sense before recomputing corrected observations without a second history advance. q is never projected. It uses the Entity world-angular API once; native body-angular readback is checked separately with a stated 1e-6 rad/s FP32 API arithmetic tolerance. q, qd, pose, cache and control are checked exactly. The independent physical evaluation thresholds are unchanged. Startup pen-dimension events remain before Root's actual8 restoration; unexpected additional reset events are rejected. GPU reset/partial-reset equivalence remains untested.

`reset_readback(env, ids)` returns selected raw fullq/fullqd, object native twist, ctrl/cache/residual/history, mocap, time, phase and episode generation without an extra query/forward. The latest result is `env._rough_runtime_last_reset`.

There are15 action intervals. The real timeout uses preaction `time_steps>=14`; after the final native step, observation/reference15 is retained. It is timeout/done rather than a forced failure label. Physical termination labels are not overwritten. No real phase15 action is valid. The imagined end uses the matching `time+2>=end-start`; inactive rows use controller's finite u14 placeholder and cannot contribute reward, bootstrap, critic normalization or loss. Hard eligibility is **phase<15**, not thresholding the native soft physical termination probability.

The policy adapter pins the actual native runner source and applies11 explicit replacements. Native recurrent loop count, random call sites/order, active formulas, configured actor/critic call counts and candidate minibatch partitioning remain. It checks finite values before masking, detaches inactive state propagation, and masks critic normalization and loss. Critic loss denominators are the active count in each original minibatch (clamped to1 for an empty batch), not the padded H×B row count. All call sites still execute, so empty batches do not silently reduce the update budget. New arms must use this same objective version.

`policy_evidence(runner)` materializes the latest starting phases, H×B active mask, eligible normalization count, each critic minibatch active denominator and optimizer call counts as JSON. Root's stage/checkpoint writer must save this evidence; absence is pending, not zero-pass. Replay add/sample wrappers attach phase, generation, source frame200+phase, native prefix0/16 and reference ID. They preserve the original collector order and terminal transition, validate the original buffer's done/valid suffix mask, and report effective window entries. Metadata is not a network input. Root should record `_rough_runtime_replay_counts` and `_rough_runtime_latest_window` separately.

CPU evidence: [28 method/gradient contracts](D:/workspace/supermimic-rough-prefix-learning-study/outputs/runtime_cpu_contract_v1/report.json), [12 rejection/finally contracts](D:/workspace/supermimic-rough-prefix-learning-study/outputs/runtime_scope_cpu_v1/report.json), and [saved source RNG-call comparison](D:/workspace/supermimic-rough-prefix-learning-study/outputs/runtime_scope_cpu_v1/policy_rng_source_review.json). The first receipt includes actual native command fixed-wrist/action reset/apply/action-manager reset, ReplayBuffer, and native Runner policy methods. Env/Entity I/O is explicitly a CPU stub. The policy clients are tiny differentiable fixture networks; optimize call sites backpropagate but **no optimizer is stepped**. Long nonterminal native policy losses/gradients match within1e-6 and norms exactly. Mixed starts0/7/14/14 withH8 yield8/8/1/1 active rows,19 normalization entries,32 candidate rows and8 critic calls. Changing inactive finite state/reward/value placeholders by1000 leaves actor/critic losses, gradients and normalization unchanged. Inactive NaN is rejected. Terminal replay row is retained and reset suffix is masked.

All40 CPU checks passed, CUDA remained uninitialized, and there were no native Env construction, physics steps, research training or GPU runs. Native CUDA reset, manager cadence, compiled H8/AMP, whole entry composition and saved checkpoint/actual8 qualification still require separate bounded engineering authorization. CPU stubs do not establish those results.
