# H8 donor-only open-loop feasibility

The saved data supports the proposed fixed-control H8 diagnostic. This is a source review plus one authorized saved-byte continuity check, **not** an H8 model run, window-array construction, preregistration or execution authorization.

The existing452 matched-input artifact contains300 donor intervals,24 restored baselines and128 restored directions. For each of A0/A2, donor k0…149 exists once. All298 unique adjacent donor links satisfy `target_x[k]`→`pre_x[k+1]` exact uint32 equality across all58 features (17,284 component comparisons); there are zero breaks. All300 donor rows also retain four bit-identical executed absolute controls. Input/source hashes were checked before and after; no values were repaired. Every link and control result is retained in `contiguity.json`.

The predeclared starts k0…142 give143 windows per path,286 total. Window k consumes actual controls k…k+7 and compares predictions against observations k+1…k+8. The last window uses controls142…149 and ends at observation150; it creates no u150 or additional policy decision. The152 restored branches provide only one20ms interval each, so they cannot supply an H8 continuation and are excluded by availability, not by error or labels. No window arrays were built during this check.

The same strict105 loader is sufficient:

```
load_world_model_checkpoint(checkpoint, bundle, "A2", training_run, health_report,
    expected_health_auditor=original_pin, expected_iteration=2000)
```

Its frozen FP32 CPU wrapper accepts any positive batch size. A future run can initialize one batch286 from each window's native PRE state, then call `wm(current_prediction, actual_u_at_horizon)` eight times under `torch.no_grad()`, feeding the preceding prediction into the next call. It must not reset intermediate predictions to truth, repeat Actor/EMA/C2, change the executed target schedule, update normalization or reconstruct a native environment. The fixed xi46 and zero174 disturbance contract remain exactly those of119f; context368 is not xi.

The minimal model budget is one strict loader, internally two checkpoint reads and one validation-only Actor construction, plus eight WM forwards of286 rows (2288 dependent prediction rows). Source-accounted passive MJB cost stays3: caller bundle1, loader-internal bundle revalidation1 and Actor constant validation1. No Actor forward, AD, objective, new state conversion, native FK/query, Env, physics or GPU is needed. This is FP32CPU shadow inference; actual training GPU AMP precision is different. Runtime/memory have not been benchmarked here, and eight forwards do not imply eight times the previous whole-run wall time.

Use the same unit-separated raw58 error definitions as the completed452 diagnostic, retaining every seed/horizon: q/qd/p/R9/v/omega; raw R element/Frobenius errors without angle projection or repair. The286 overlapping windows and2288 predictions are not independent samples. A horizon-one result from these286 starts is a different population from the prior all452 aggregate and should not be presented as a change in model quality.

This would measure accumulation under **fixed actual external controls**, not native Actor imagination, policy-gradient replay or a closed-loop learned controller. It omits Actor observations, recomputed actions, residual/controller memory, failure-head survival, rewards and critic masks. Hidden simulator state is not serialized by state58. Existing physical-gradient5/9 findings and all physical/proxy qualifications remain unchanged. Finite errors should not select windows; if a future strict wrapper encounters nonfinite output, preserve the partial result rather than repair or substitute truth.

Relevant immutable source: `world_model_checkpoint.py` lines76–124 (strict wrapper/loader, SHA119fa7ce…78052); `state_prediction452105_v1.py` lines127–157 (group alignment/control source, SHA5cd15a0a…e240); `penalty_observability105_v1.py` lines160–188 (452 group source ordering/cadence); the new continuity checker is source-bound in `contiguity.json`. No implementation or new protocol is proposed for execution by this document.
