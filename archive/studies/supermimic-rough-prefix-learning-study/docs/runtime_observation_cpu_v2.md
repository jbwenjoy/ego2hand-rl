# Selected observation refresh, runtime v2

The original runtime remains frozen at `5736dd1fd678a39f87867461d30d0710e6242af10e42f6fa2d57f6c9301b1ff9`. Its first native A0 attempt failed before any probe steps because the installed ObservationManager rejects `compute(update_history=False, env_ids=ids)`. That attempt is an incomplete engineering result; physical quality was not assessed.

`runtime_v2.py` changes only the reset observation refresh to `compute(update_history=True, env_ids=ids)`, after the original native reset and the selected rows' corrected initial state. It does not repeat ObservationManager.reset. The installed manager backfills already initialized selected histories and delay storage, replaces its cached observation, and uses DelayBuffer.peek without an additional lag draw or timeline advance.

The [25-check report](../outputs/runtime_observation_cpu_v2/report.json) uses the installed ObservationManager, CircularBuffer, and DelayBuffer on CPU. It covers full and partial reset, corrected observations, unchanged unselected history/delay storage and counters, no extra correction RNG, the original assertion, and the stale-cache consequence of dropping env_ids. [The diff](../outputs/runtime_observation_cpu_v2/runtime_v1_to_v2.diff) and source snapshots preserve the executed version. The previous 40 runtime checks remain separate evidence; they did not use this real observation manager.

Env/Entity state writes remain fixtures in this proof. Observation terms are stateless, matching the native play configuration. This does not qualify private camera histories, a native physical reset, GPU execution, or training.

The [saved-only A0 v1 report](../outputs/native_a0_v1_incomplete_saved_review_v1/report.json) verifies 235 retained facts, including zero steps, the failure, actual process identity and exit, source bindings, and the actual context readbacks. No raw full-horizon state or quality result exists for that attempt. Its full initial state was not saved and is not inferred from the failed reset.
